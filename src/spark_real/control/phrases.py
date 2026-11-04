"""Blended UR10e phrases: IK every waypoint, safety-filter the rows, play one URScript path."""

import logging
import os
import time

import numpy as np

from spark_real.control.executor_types import (
    ExecutionResult,
)
from spark_real.control.waypoints import (
    UR_MAX_EPOCH,
    UR_MAX_ROW_DQ_RAD,
    JointRow,
    build_ur_blend_program,
    decode_blend_progress,
    resolve_blend_radii,
)

try:
    from spark_real.control.ur10e_ik_pyroki import solve_ik as solve_ik_ur10e
    from spark_real.control.ur10e_ik_pyroki import solve_ik_rtde as solve_ik_ur10e_rtde
except ImportError:
    solve_ik_ur10e = None
    solve_ik_ur10e_rtde = None
from spark_real.control.motion_limits import resolve_joint_limits

logger = logging.getLogger(__name__)


class PhraseMixin:
    """Blended UR10e phrases: IK every waypoint, safety-filter the rows, play one URScript path."""

    def _next_blend_epoch(self, n_rows: int) -> int:
        """Pick a progress-register epoch the CURRENT register value is not in.

        The register survives across paths and across process restarts, so a
        stale value inside this path's window would read as progress, and a
        stale terminal value would read as "path complete" before the program
        had written a byte. Reading first and stepping off the held value
        closes that: never infer from a signal this path has not been observed
        to produce.
        """
        held = self._read_blend_progress()
        ep = getattr(self, "_blend_epoch", 0) % UR_MAX_EPOCH + 1
        for _ in range(UR_MAX_EPOCH):
            if decode_blend_progress(held, ep, n_rows) is None:
                break
            ep = ep % UR_MAX_EPOCH + 1
        self._blend_epoch = ep
        return ep

    def _solve_blend_rows(self, waypoints, q_start):
        """IK every waypoint with seed chaining; None rejects the whole phrase.

        Pre-validating the ENTIRE path before the first byte is sent is the
        point: a mid-path IK failure cannot be recovered once the program is
        running, and an IK branch flip on one row becomes a single continuous
        blended wrench instead of a discrete reconfiguration between two
        stopped moves.
        """
        if solve_ik_ur10e_rtde is None:
            logger.info("[blend] UR10e pyroki IK unavailable; no blended path")
            return None
        if os.environ.get("SPARK_UR_PYROKI_IK", "1") != "1":
            return None
        tcp_off = self._ur_tcp_offset()
        if tcp_off is None:
            logger.info("[blend] TCP offset unavailable; no blended path")
            return None

        q_seed = np.asarray(q_start, dtype=float).reshape(-1)
        rows = []
        for i, wp in enumerate(waypoints):
            pos = np.asarray(wp[0], dtype=float).reshape(3)
            orient = list(wp[1])
            vel = float(wp[2]) if len(wp) > 2 and wp[2] else self.velocity
            label = str(wp[3]) if len(wp) > 3 else f"row{i}"
            q = solve_ik_ur10e_rtde(pos, orient, q_seed=q_seed, tcp_offset=tcp_off)
            if q is None:
                logger.info(
                    "[blend] IK failed at %s pos=%s; falling back to per-move",
                    label,
                    pos.round(3).tolist(),
                )
                return None
            dq = float(np.max(np.abs(np.asarray(q, dtype=float) - q_seed[: q.size])))
            if dq > UR_MAX_ROW_DQ_RAD:
                logger.info(
                    "[blend] %s needs dq_max=%.2f rad (> %.2f): IK branch flip, "
                    "refusing to blend through it",
                    label,
                    dq,
                    UR_MAX_ROW_DQ_RAD,
                )
                return None
            vel_j, acc_j = resolve_joint_limits(
                vel, dq=np.asarray(q, dtype=float) - q_seed[: q.size]
            )
            rows.append(
                JointRow(
                    q=q,
                    position=pos,
                    velocity=vel_j,
                    acceleration=acc_j,
                    label=label,
                )
            )
            q_seed = np.asarray(q, dtype=float)
        return rows

    def _clip_blend_waypoints(self, waypoints, start_pos, phrase):
        """Apply _move_to's workspace clip + oversize refusal to every row.

        Per-move dispatch gets both of these inside _move_to; a phrase must not
        bypass them just because it takes one send instead of N. Returns the
        clipped waypoint list, or None to refuse the phrase.
        """
        out = []
        prev = np.asarray(start_pos, dtype=float)
        for wp in waypoints:
            raw = np.asarray(wp[0], dtype=float).reshape(3)
            pos = raw
            if not self._check_workspace(pos):
                pos = np.clip(raw, self.WORKSPACE_MIN, self.WORKSPACE_MAX)
                logger.info(
                    "[blend] %s row CLIPPED by workspace: (%.3f,%.3f,%.3f) -> "
                    "(%.3f,%.3f,%.3f)",
                    phrase,
                    raw[0],
                    raw[1],
                    raw[2],
                    pos[0],
                    pos[1],
                    pos[2],
                )
            seg = float(np.linalg.norm(pos - prev))
            if seg > 1.0:
                logger.warning(
                    "[blend] %s segment %.3fm too large; per-move dispatch",
                    phrase,
                    seg,
                )
                return None
            out.append((pos,) + tuple(wp[1:]))
            prev = pos
        return out

    def _blend_rows_pass_safety_filter(self, rows, phrase) -> bool:
        """Run the driver wrapper's movej joint filter over EVERY row.

        Deliberately redundant with SafeRobot's own per-row movej scan, for
        two reasons:

          - It degrades to per-move dispatch (return False) instead of
            refusing the motion outright, so a single near-singular waypoint
            costs the blend, not the task.
          - It is the only joint check in the chain when self.robot is a bare
            driver with no SafeRobot wrapper.

        Blending must not buy speed with safety, and a second independent
        evaluation of the same barrier is cheap.

        Preference order: a public per-target predicate on the wrapper if one
        exists, else the wrapper's OWN epsilon applied with the same predicate
        shape SafeRobot uses (so there is still one source for the number). A
        chain with no filter in it (a bare driver) has nothing to enforce.
        """
        node = self.robot
        for _ in range(4):
            if node is None:
                return True
            check = getattr(node, "check_joint_target_safe", None)
            if callable(check):
                for row in rows:
                    if not check(row.q.tolist()):
                        logger.warning(
                            "[blend] %s row %s refused by the safety filter; "
                            "per-move dispatch",
                            phrase,
                            row.label,
                        )
                        return False
                return True
            eps = getattr(getattr(node, "_cfg", None), "eps_singularity", None)
            if eps is not None:
                for row in rows:
                    q2 = float(row.q[2])
                    if (q2**2) * ((q2 - np.pi) ** 2) - float(eps) ** 2 < 0:
                        logger.warning(
                            "[blend] %s row %s near the elbow singularity "
                            "(q2=%.3f); per-move dispatch",
                            phrase,
                            row.label,
                            q2,
                        )
                        return False
                return True
            node = getattr(node, "_robot", None)
        return True

    def _blend_transit(self, waypoints, final_radius_m=None, phrase="phrase") -> bool:
        """Fly ``waypoints`` as ONE blended URScript program. True = arrived.

        ``waypoints`` is a list of ``(position, orientation[, velocity[,
        label]])``. Every row but the last carries a blend radius, so the arm
        cuts the corner at each intermediate waypoint instead of decelerating
        to a stop; the last row is always ``r=0`` and is therefore a true stop
        the exact joint-arrival test still applies to.

        Returns False (having commanded nothing, or having stopped the arm)
        whenever the phrase cannot be trusted (blending disabled, wrong family,
        IK rejection, nothing left to blend after dropping no-op rows, a failed
        send, a stall, or a timeout). Callers must treat False as "do it the
        old way from the current pose".
        """
        if not self._blend_enabled():
            return False
        if self._robot_family() != "ur10e":
            return False
        if not hasattr(self.robot, "_send_script"):
            return False
        self._check_abort()

        cfg = self._blend_cfg()
        max_rows = int(cfg["max_rows"])
        if len(waypoints) < 2:
            return False
        if len(waypoints) > max_rows:
            logger.info(
                "[blend] %s has %d rows (> blend_max_rows=%d); per-move dispatch",
                phrase,
                len(waypoints),
                max_rows,
            )
            return False

        try:
            q_start = np.asarray(self.robot.get_joint_positions(), dtype=float)
        except Exception as exc:  # noqa: BLE001
            logger.info("[blend] cannot read joints (%s); per-move dispatch", exc)
            return False
        start_pos = self._get_current_position()

        waypoints = self._clip_blend_waypoints(waypoints, start_pos, phrase)
        if waypoints is None:
            return False

        rows = self._solve_blend_rows(waypoints, q_start)
        if not rows:
            return False
        if not self._blend_rows_pass_safety_filter(rows, phrase):
            return False

        kept, dropped = resolve_blend_radii(
            start_pos,
            rows,
            radius_m=float(cfg["radius_m"]),
            final_radius_m=(
                float(cfg["descent_radius_m"])
                if final_radius_m is None
                else float(final_radius_m)
            ),
            min_radius_m=float(cfg["min_radius_m"]),
            start_q=q_start,
        )
        if dropped:
            logger.info("[blend] %s: dropped no-op rows %s", phrase, dropped)
        if len(kept) < 2:
            # Nothing left to blend: one real move is exactly what the
            # unblended path would issue, so let the caller issue it.
            return False

        epoch = self._next_blend_epoch(len(kept))
        program = build_ur_blend_program(kept, epoch, self._blend_register())
        logger.info(
            "[blend] %s: %d rows, r=[%s], epoch=%d",
            phrase,
            len(kept),
            ", ".join(f"{r.blend:.3f}" for r in kept),
            epoch,
        )
        if not self.robot._send_script(program):
            logger.warning("[blend] program send failed; per-move dispatch")
            self._retire_motion_lease()
            return False
        return self._wait_for_blended_path(
            kept, epoch, start_pos=start_pos, q_start=q_start
        )

    def _execute_ur_blend_phrase(self, run) -> bool:
        """UR10e backend for the legato look-ahead: one blended path per phrase.

        Accepts only one shape: a run of
        ``move_relative`` notes, optionally closed by ONE terminal
        ``move_to_keypoint`` while holding. That terminal note is the transport,
        so it is handed to ``_transport_to`` as a lead-in rather than reduced to
        a bare "go to the detection", which is what the franky path above does
        and which would skip the clearance lift and the place descent entirely.

        Returns False without commanding anything on any shape it does not
        recognise, so the caller falls back to per-action dispatch.
        """
        if not self._blend_enabled():
            return False

        lead = []
        running = np.asarray(self._get_current_position(), dtype=float).copy()
        terminal = None
        for k, note in enumerate(run):
            atype = note.get("type", note.get("name", ""))
            params = note.get("params", {}) or {}
            if atype == "move_relative":
                running = running + np.array(
                    [
                        params.get("dx", 0.0),
                        params.get("dy", 0.0),
                        params.get("dz", 0.0),
                    ],
                    dtype=float,
                )
                lead.append(
                    (
                        running.copy(),
                        list(self.GRASP_ORIENTATION),
                        self.velocity,
                        "rel%d(dz=%.2f)" % (k, params.get("dz", 0.0)),
                    )
                )
            elif atype == "move_to_keypoint" and k == len(run) - 1 and self._holding:
                if not self._blend_cfg()["cross_node"]:
                    # OFF BY DEFAULT, and not for want of speed.
                    # executor_core._run_actions skips its whole failure/recovery
                    # block when a legato phrase returns True (`i = j; continue`),
                    # so a drop detected inside the phrase would be reported and
                    # then ignored: no _attempt_recovery, no re-grasp, and the
                    # following release would open on nothing. Folding the
                    # preceding move_relative into the transport is worth ~1 s,
                    # which is not worth silently losing drop recovery. Turning
                    # this on needs _run_actions to consume a phrase's per-note
                    # results; until then the
                    # transport still blends internally, inside _transport_to.
                    return False
                terminal = note
            else:
                return False

        t0 = time.time()
        if terminal is None:
            if not self._blend_transit(lead, final_radius_m=0.0, phrase="move_rel"):
                return False
        else:
            params = terminal.get("params", {}) or {}
            label = params.get("keypoint_label", "")
            det = self.detection_map.get(label)
            if det is None:
                return False
            pos = np.asarray(det["position_3d"], dtype=float).copy()
            if pos[2] < self.TABLE_Z_FLOOR:
                pos[2] = self.TABLE_Z_FLOOR
            target = pos + np.array(
                [
                    params.get("offset_x", 0.0),
                    params.get("offset_y", 0.0),
                    params.get("offset_z", 0.0),
                ],
                dtype=float,
            )
            self._last_keypoint_label = label
            ok = self._transport_to(
                target,
                target_detection=det,
                target_label=label,
                params=params,
                lead=lead,
            )
            if not ok or not self._holding:
                # Drop detected mid-transport. The notes ARE consumed (the arm
                # moved), so report them here exactly as _move_to_keypoint would
                # rather than replaying the phrase.
                self._append_phrase_results(run, time.time() - t0, failed_last=True)
                return True

        self._append_phrase_results(run, time.time() - t0)
        return True

    def _append_phrase_results(self, run, elapsed: float, failed_last: bool = False):
        """One ExecutionResult per note of a phrase that played as one motion."""
        per = elapsed / max(len(run), 1)
        for k, note in enumerate(run):
            atype = note.get("type", note.get("name", ""))
            last = k == len(run) - 1
            bad = failed_last and last
            self._results.append(
                ExecutionResult(
                    action_type=atype,
                    success=not bad,
                    message=(
                        "Object lost during transport"
                        if bad
                        else "Played as one blended phrase (%d notes)" % len(run)
                    ),
                    duration=per,
                )
            )

    def _execute_legato_phrase(self, run, run_start_idx: int, total: int):
        """
        Play a sequence of blendable actions as one joint-space phrase.

        UR10e only: one URScript blended path (_execute_ur_blend_phrase).
        Returns False when the phrase cannot play, so the caller falls back
        to per-action dispatch.
        """
        if self._robot_family() != "ur10e":
            return False
        return self._execute_ur_blend_phrase(run)
