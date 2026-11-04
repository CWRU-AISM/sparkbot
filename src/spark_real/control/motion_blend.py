"""Blended UR10e paths: config, progress register, motion lease and halt."""

import logging
import os
import time

import numpy as np

from spark_real.control.motion_common import _StillnessTracker
from spark_real.control.waypoints import (
    UR_PROGRESS_REGISTER,
    decode_blend_progress,
    joint_move_seconds,
)

logger = logging.getLogger(__name__)


class BlendMixin:
    """Blended UR10e paths: config, progress register, motion lease and halt."""

    def _blend_cfg(self) -> dict:
        """Resolve the blended-path knobs once per executor.

        Precedence: env SPARK_UR_BLEND (on/off only) > control.blend_* in the
        resolved profile / family YAML > the class defaults above.

        Default ON for the transit chains (approach + transport). Safety:
          * abort: a blended path emits URScript stopj on the same socket and
            provably brakes (_halt_arm).
          * every row goes through the workspace clip (_clip_blend_waypoints)
            and the joint/singularity filter (_blend_rows_pass_safety_filter).
          * the terminal row is forced to r=0 and a descent row keeps
            BLEND_DESCENT_RADIUS_M = 2 cm, so an over-large radius cannot
            corner-cut into the object.
        PolyScope behaviour on an over-large blend radius (silent reduction vs
        runtime error) must be read off the pendant log on the first rig run.
        control.blend_motion: false (or SPARK_UR_BLEND=0) restores per-move
        dispatch.
        """
        cached = getattr(self, "_blend_cfg_cached", None)
        if cached is not None:
            return cached
        control = self._control_config()
        enabled = bool(control.get("blend_motion", self.BLEND_MOTION_DEFAULT))
        env = os.environ.get("SPARK_UR_BLEND")
        if env is not None:
            enabled = env not in ("0", "false", "False", "")
        cfg = {
            "enabled": enabled,
            "radius_m": float(control.get("blend_radius_m", self.BLEND_RADIUS_M)),
            "descent_radius_m": float(
                control.get("blend_descent_radius_m", self.BLEND_DESCENT_RADIUS_M)
            ),
            "min_radius_m": float(control.get("blend_min_radius_m", self.BLEND_MIN_RADIUS_M)),
            "max_rows": int(control.get("blend_max_rows", self.BLEND_MAX_ROWS)),
            "register": int(control.get("blend_progress_register", UR_PROGRESS_REGISTER)),
            # Fold a preceding move_relative into the transport phrase, i.e.
            # blend ACROSS a BT node boundary. Separate switch, default off;
            # see _execute_ur_blend_phrase.
            "cross_node": bool(control.get("blend_cross_node", False)),
        }
        env_x = os.environ.get("SPARK_UR_BLEND_CROSS_NODE")
        if env_x is not None:
            cfg["cross_node"] = env_x not in ("0", "false", "False", "")
        self._blend_cfg_cached = cfg
        logger.info(
            "Blended motion: %s (r=%.3fm, descent r=%.3fm, max_rows=%d, reg=%d, " "cross_node=%s)",
            "ON" if enabled else "off",
            cfg["radius_m"],
            cfg["descent_radius_m"],
            cfg["max_rows"],
            cfg["register"],
            cfg["cross_node"],
        )
        return cfg

    def _blend_enabled(self) -> bool:
        """True when transit chains may be issued as one blended path."""
        return bool(self._blend_cfg()["enabled"])

    def _blend_register(self) -> int:
        return self._blend_cfg()["register"]

    def _rtde_receive(self):
        """The rtde_receive handle, wherever it sits in the wrapper chain."""
        obj = self.robot
        for _ in range(4):
            if obj is None:
                return None
            r = getattr(obj, "_rtde_r", None)
            if r is not None:
                return r
            obj = getattr(obj, "_robot", None)
        return None

    def _read_blend_progress(self):
        """Current value of the blend progress register, or None if unreadable.

        A pure rtde_receive read: it uploads nothing, so unlike a gripper
        publish it can never replace the running program.
        """
        reg = self._blend_register()
        fn = getattr(self.robot, "get_output_int_register", None)
        if callable(fn):
            try:
                return int(fn(reg))
            except Exception:  # noqa: BLE001
                return None
        rtde_r = self._rtde_receive()
        if rtde_r is None:
            return None
        try:
            return int(rtde_r.getOutputIntRegister(reg))
        except Exception:  # noqa: BLE001 - build may not surface this register
            return None

    def _refresh_motion_lease(self):
        """Keep the driver's motion lease alive for the length of a path.

        _MOVEJ_LEASE_S is 1.5 s, sized for ONE move's upload lag. If the lease
        lapses mid-path a single _publish_gripper_state replaces the running
        program and cancels the whole path. Re-noting a movej pushes the lease
        out by another 1.5 s; _wait_for_blended_path retires it on arrival.
        """
        note = getattr(self.robot, "_note_motion_script", None)
        if callable(note):
            try:
                note("movej(")
            except Exception:  # noqa: BLE001 - never block on bookkeeping
                pass

    def _retire_motion_lease(self):
        clear = getattr(self.robot, "clear_motion_lease", None)
        if callable(clear):
            try:
                clear()
            except Exception:  # noqa: BLE001
                pass

    def _halt_arm(self) -> bool:
        """Decelerate the arm NOW. True only if the brake left this process.

        A new send REPLACES the running program, so a URScript stop both brakes
        the arm and kills a blended path. No hardcoded ``stopj(2.0)`` here:
        overshoot is ``v**2 / 2a``, so a FIXED decel gets quadratically worse
        as commanded velocity rises (at 1.78 rad/s, ``stopj(2.0)`` is 0.79 rad
        of travel AFTER the stop, roughly 0.8 m of TCP at full reach). The
        driver sizes its decel from the speed it last commanded
        (UR10eDriver.brake_decel).

        Order:
          1. the driver's own ``brake()``, the only branch that knows what
             velocity was commanded;
          2. the executor's ``_stop_robot()``, the shared escalation ladder
             (reopened socket, then Dashboard); its URScript branch emits a
             FIXED ABORT_DECEL_FALLBACK, hence (1) first;
          3. a literal stopj, for a bare MotionMixin with neither.
        """
        ok = False
        brake = getattr(self.robot, "brake", None)
        if callable(brake):
            try:
                ok = bool(brake())
            except Exception as exc:  # noqa: BLE001 - a stop never raises
                logger.warning("[blend] driver brake failed: %s", exc)
        if not ok:
            stop = getattr(self, "_stop_robot", None)
            if callable(stop):
                try:
                    ok = bool(stop())
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[blend] brake escalation failed: %s", exc)
        if not ok:
            send = getattr(self.robot, "_send_script", None)
            if callable(send):
                try:
                    ok = bool(send("stopj(%.2f)" % self.BLEND_ABORT_DECEL))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[blend] stopj send failed: %s", exc)
        self._retire_motion_lease()
        return ok

    def _wait_for_blended_path(self, rows, epoch, start_pos=None, q_start=None) -> bool:
        """Block until a blended URScript path lands. True only on arrival.

        Arrival is the FINAL waypoint's joint target, nothing else.
        Intermediate waypoints are never reached (the arm cuts each corner at
        distance r), so a per-waypoint joint match would never fire and a
        per-waypoint Cartesian test would never come within 5 mm. The last row
        is forced to r=0 (waypoints.resolve_blend_radii) so the terminal
        waypoint is a true stop.

        Stillness is evaluated over the WHOLE path, never per waypoint, and
        may only mean "arrived" if motion has been observed; a path's upload
        lag is longer than one move's.
        """
        n = len(rows)
        final = rows[-1]
        target = final.position
        q_target = np.asarray(final.q, dtype=float)
        register_ok = self._read_blend_progress() is not None
        if not register_ok:
            logger.warning(
                "[blend] progress register %d unreadable: no mid-path stall "
                "detection, only the whole-path budget",
                self._blend_register(),
            )

        # Budget the WHOLE path, not one move: 15 s is both too long for a
        # 2 cm reposition and too short for a 5-row transport. Same shape as
        # _move_to_linear's per-move budget (3x ideal + start lag + slack).
        # Without q_start the first segment is budgeted off a half-turn rather
        # than under-budgeted.
        ideal = 0.0
        q_prev = None if q_start is None else np.asarray(q_start, dtype=float)
        for row in rows:
            dq = (
                float(np.max(np.abs(row.q - q_prev[: row.q.size]))) if q_prev is not None else np.pi
            )
            ideal += joint_move_seconds(dq, row.velocity, row.acceleration)
            q_prev = row.q
        timeout = float(
            np.clip(
                3.0 * ideal + self.URSCRIPT_START_LAG_S + 1.0,
                self._min_motion_budget_s(),
                self.BLEND_TIMEOUT_CAP_S,
            )
        )

        t0 = time.time()
        motion = _StillnessTracker(
            self._get_current_position(),
            t0,
            sample_s=self.MOTION_SAMPLE_S,
            still_samples=self.MOTION_STILL_SAMPLES,
        )
        next_record = t0
        progress = None
        progress_at = t0

        def _arrived(why: str) -> bool:
            self._retire_motion_lease()
            logger.info(
                "[blend] path complete after %.2fs (%s, %d rows)",
                time.time() - t0,
                why,
                n,
            )
            return True

        while time.time() - t0 < timeout:
            if self._abort:
                # Reach the arm BEFORE raising (see _halt_arm); otherwise an
                # aborted blended path keeps running to its last waypoint
                # after the executor has unwound.
                self._halt_arm()
            self._check_abort()
            time.sleep(self.MOTION_POLL_S)
            now = time.time()
            self._refresh_motion_lease()
            tcp = self._get_current_position()
            if self._recorder and now >= next_record:
                self._recorder.record()
                next_record = now + self.MOTION_SAMPLE_S

            # 1. Joints match the FINAL commanded row: unambiguously arrived.
            try:
                q = np.asarray(self.robot.get_joint_positions(), dtype=float)
                if np.max(np.abs(q[: q_target.size] - q_target)) < self.JOINT_ARRIVE_TOL:
                    return _arrived("joint target")
            except Exception:  # noqa: BLE001 - fall through to the other tests
                pass
            # 2. Cartesian proximity to the final waypoint.
            if np.linalg.norm(tcp - target) < 0.005:
                return _arrived("cartesian proximity")

            motion.update(tcp, now)

            # 3. Progress register. `n` means the program ran past its last
            #    movej, so the controller itself reports the path finished.
            #    Trustworthy only because _next_blend_epoch stepped off
            #    whatever value the register was already holding.
            k = decode_blend_progress(self._read_blend_progress(), epoch, n)
            if k is not None and (progress is None or k > progress):
                progress = k
                progress_at = now
                if k >= n:
                    return _arrived("progress register")
            # 4. Stall: started, still before the LAST row, register frozen,
            #    TCP still. Intermediate rows only: a stopped arm on the final
            #    row is the ordinary end of a move, handled by (5).
            if (
                progress is not None
                and progress < n - 1
                and now - progress_at > self.BLEND_STALL_S
                and motion.stopped()
            ):
                logger.warning(
                    "[blend] STALLED at row %d/%d after %.1fs (register frozen, "
                    "TCP still); stopping the path",
                    progress,
                    n,
                    now - t0,
                )
                self._halt_arm()
                return False
            # 5. Whole-path stillness, gated on observed motion (never an exit
            #    without ever_moved). Also refused while the register says the
            #    program is still inside its rows: a mid-path stop looks like a
            #    finished path from the outside. On the LAST row, or with a
            #    mute register, this is the per-move wait's test.
            settled = motion.stopped() and now - t0 > self.URSCRIPT_START_LAG_S
            if settled and motion.ever_moved and (progress is None or progress >= n - 1):
                return _arrived("settled after observed motion")

        logger.warning(
            "[blend] %.1fs timeout, no arrival signal (moved=%s, row=%s/%d, "
            "dist=%.3fm); stopping the path",
            timeout,
            motion.ever_moved,
            progress,
            n,
            float(np.linalg.norm(self._get_current_position() - target)),
        )
        self._halt_arm()
        return False
