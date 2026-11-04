"""Did the grasp hold: force, position, object flag and vision tie-break."""

import logging
import os

import numpy as np

from spark_real.control import success_verifier
from spark_real.control.grasp_verdict import (
    GRASP_FORCE_EMPTY_DEFAULT_N,
    GRASP_LIFT_DELTA_MIN_N,
    GraspVerdict,
)

logger = logging.getLogger(__name__)


class GraspVerifyMixin:
    """Did the grasp hold: force, position, object flag and vision tie-break."""

    def _force_empty_n(self) -> float:
        """Resolve the force-verify empty threshold (N).

        env > grasp.force_empty_n > module default. No floor: the configured
        value is authoritative in both directions.
        """
        env = os.environ.get("SPARK_GRASP_FORCE_EMPTY_N")
        if env is not None:
            try:
                return float(env)
            except ValueError:
                logger.warning("SPARK_GRASP_FORCE_EMPTY_N=%r ignored", env)
        return float(getattr(self, "GRASP_FORCE_EMPTY_N", GRASP_FORCE_EMPTY_DEFAULT_N))

    def _lift_delta_min_n(self) -> float:
        return float(
            os.environ.get(
                "SPARK_GRASP_LIFT_DELTA_MIN_N",
                getattr(self, "GRASP_LIFT_DELTA_MIN_N", GRASP_LIFT_DELTA_MIN_N),
            )
        )

    def _record_grasp_verdict(self, verdict: "GraspVerdict") -> "GraspVerdict":
        """Publish the G1 verdict for the success verifier's gate.

        Routed through success_verifier so the gate flag is set and a
        successful re-grasp clears an earlier transport drop; the structured
        verdict itself is stored verbatim for the trace. Also stamps the typed
        GraspOutcome (gObj / closed-stop position / force corroboration) when
        the producer did not set one.
        """
        if verdict.outcome is None:
            try:
                from spark_core.grasp_outcome import (
                    classify_grasp_endpoint,
                )

                verdict.outcome = classify_grasp_endpoint(
                    verdict.gobj,
                    verdict.gripper_pos,
                    closed_pos=float(
                        getattr(self, "GRIPPER_FULLY_CLOSED", 250)
                    ),
                    force_holding=(
                        verdict.held if verdict.source == "force" else None
                    ),
                ).value
            except Exception:  # noqa: BLE001 - annotation must never break a grasp
                pass
        success_verifier.record_grasp_verdict(
            self, verdict.held, verdict.source, verdict=verdict
        )
        return verdict

    def _vision_says_still_on_table(self, label: str):
        """Did the object stay put? True = it never left the table.

        Compares a fresh detection against the position the object had when
        the grasp was planned; an object still within VISION_MOVED_TOL_M of
        where it started was not carried.

        Returns True (still there -> NOT grasped), False (moved or gone ->
        consistent with grasped), or None (no evidence).
        """
        try:
            label = label or getattr(self, "_last_keypoint_label", "") or ""
            if not label:
                return None
            prior = ((self.detection_map or {}).get(label) or {}).get("position_3d")
            if prior is None:
                return None
            prior = np.asarray(prior, dtype=float)
            pipe = getattr(self, "_pipeline", None)
            if pipe is None:
                return None
            base_prompt = label.rsplit(" ", 1)[0] if label[-1:].isdigit() else label
            caps = pipe.capture()
            merged = pipe.merge_detections(pipe.detect(caps, prompts=[base_prompt]))
            best = None
            for d in merged or []:
                pos = getattr(d, "position_3d", None)
                if pos is None:
                    continue
                dist = float(np.linalg.norm(np.asarray(pos, float)[:2] - prior[:2]))
                if best is None or dist < best:
                    best = dist
            if best is None:
                # Not re-detected is NOT evidence of a grasp: the gripper
                # occludes it just as effectively as carrying it.
                logger.info("[grasp-check] '%s' not re-detected; no verdict", label)
                return None
            still = best <= self.VISION_MOVED_TOL_M
            logger.info(
                "[grasp-check] '%s' nearest detection is %.1f cm from where it "
                "was before the close -> %s",
                label, best * 100,
                "STILL ON THE TABLE" if still else "it moved with us",
            )
            return still
        except Exception as exc:  # noqa: BLE001 - a failed look is not a verdict
            logger.warning("[grasp-check] vision cross-check failed: %s", exc)
            return None

    def _verify_grasp(self) -> bool:
        """
        Did the last grasp actually pick up an object?

        Reads three independent physical signals: TCP external force delta,
        gripper width position, and the driver's object-detect flag.
        Returns True if evidence indicates an object is held.
        """
        # Width-fast-path
        tw = getattr(self, "_last_grasp_target_width", None)
        if tw is not None and hasattr(self.robot, "get_gripper_width"):
            try:
                achieved_w = float(self.robot.get_gripper_width())
                if achieved_w < 0.001:
                    logger.info(
                        "Verify width: target=%.3fm achieved=%.3fm "
                        "(jaws fully closed) -> NOT grasped",
                        tw,
                        achieved_w,
                    )
                    return False
                if (tw - 0.005) <= achieved_w <= (tw + 0.015):
                    logger.info(
                        "Verify width: target=%.3fm achieved=%.3fm -> HOLDING",
                        tw,
                        achieved_w,
                    )
                    return True
                logger.info(
                    "Verify width: target=%.3fm achieved=%.3fm "
                    "out of band, falling back to force/pos verify",
                    tw,
                    achieved_w,
                )
            except Exception as exc:
                logger.warning("Verify width read failed: %s", exc)

        # One publish for both registers; the gObj half is consumed by the
        # "Driver flag" block below.
        gobj_sample, pos = self._read_gripper_state()
        if pos is not None and pos >= self.GRIPPER_FULLY_CLOSED:
            logger.info(
                "Verify: gripper fully closed (pos=%d >= %d) -> NOT grasped",
                pos,
                self.GRIPPER_FULLY_CLOSED,
            )
            return False

        # Force feedback
        force_says_holding = None
        baseline_vec = getattr(self, "_pre_grasp_force", None)
        try:
            if baseline_vec is None:
                # No pre-grasp sample -> NO DELTA EXISTS, so this channel must
                # abstain. Substituting zeros would turn "unknown" into
                # "holding": the raw reading carries the tool's own weight
                # (~78 N in z). Position and object-detect decide instead.
                logger.info(
                    "Verify force: no pre-grasp baseline; force channel "
                    "abstains (position/object-detect decide)"
                )
            elif hasattr(self.robot, "get_tcp_force"):
                ft = np.asarray(self.robot.get_tcp_force(), dtype=float)
                if ft.size >= 6:
                    f_now_vec = ft[:6].copy()
                    delta_xyz = f_now_vec[:3] - baseline_vec[:3]
                    delta_mag = float(np.linalg.norm(delta_xyz))
                    delta_z = float(delta_xyz[2])
                    # One threshold for one decision, shared with _grasp_v2 via
                    # _force_empty_n(). A real grasp shows a clear delta (~10 N
                    # of contact); the ~0.6 N close-on-air transient must NOT
                    # pass, or it false-confirms and skips the descent-retry.
                    _empty_n = self._force_empty_n()
                    if delta_mag > _empty_n or delta_z < -_empty_n:
                        force_says_holding = True
                    elif delta_mag < 0.5 and abs(delta_z) < 0.5:
                        force_says_holding = False
                    logger.info(
                        "Verify force: now=(%.2f,%.2f,%.2f) baseline=(%.2f,%.2f,%.2f) "
                        "delta=(%.2f,%.2f,%.2f) |delta|=%.2f -> %s",
                        f_now_vec[0],
                        f_now_vec[1],
                        f_now_vec[2],
                        baseline_vec[0] if baseline_vec is not None else 0.0,
                        baseline_vec[1] if baseline_vec is not None else 0.0,
                        baseline_vec[2] if baseline_vec is not None else 0.0,
                        delta_xyz[0],
                        delta_xyz[1],
                        delta_xyz[2],
                        delta_mag,
                        {True: "HOLDING", False: "EMPTY", None: "ambiguous"}[
                            force_says_holding
                        ],
                    )
        except Exception as exc:
            logger.warning("Verify force read failed: %s", exc)

        # Position in the holding band
        pos_says_holding = None
        if pos is not None:
            if self.GRIPPER_EMPTY_THRESHOLD <= pos < self.GRIPPER_FULLY_CLOSED:
                pos_says_holding = True
            elif pos < self.GRIPPER_EMPTY_THRESHOLD:
                pos_says_holding = False
            logger.info(
                "Verify position: pos=%d empty<=%d closed>=%d -> %s",
                pos,
                self.GRIPPER_EMPTY_THRESHOLD,
                self.GRIPPER_FULLY_CLOSED,
                {True: "HOLDING", False: "EMPTY", None: "ambiguous"}[pos_says_holding],
            )

        # Driver flag
        flag_says_holding = gobj_sample
        if flag_says_holding is not None:
            logger.info(
                "Verify flag (driver=%s): is_object_detected=%s",
                self._gripper_type(),
                flag_says_holding,
            )

        # Trust a POSITIVE Robotiq gObj object-detect flag outright: force alone
        # reads ~empty for a light object held statically, so it must not veto a
        # confirmed hardware grasp.
        if flag_says_holding is True and pos_says_holding is False:
            # Position disagrees with the flag; ask the cameras to break the
            # tie. Vision abstains freely (the gripper occludes), so an unknown
            # answer falls through to the remaining channels.
            logger.warning(
                "Verify: flag says held but position says EMPTY (pos=%s); "
                "cross-checking with vision", pos,
            )
            still_there = self._vision_says_still_on_table(
                self._last_keypoint_label or ""
            )
            if still_there is True:
                logger.warning(
                    "Verify: NOT GRASPED -- the object is still on the table "
                    "and the jaws are open; the object-detect flag was stale"
                )
                return False
            if still_there is False:
                logger.info(
                    "Verify: GRASPED (object left its place; flag agrees)"
                )
                return True
        elif flag_says_holding is True:
            logger.info("Verify: GRASPED (Robotiq object-detect flag)")
            return True
        # A False gObj flag is NOT trusted as a veto: the flag and gripper
        # position ride the contended _send_script register path and read
        # stale/False on real grasps. Fall through to TCP force (rtde_r channel)
        # instead of hard-rejecting.
        if flag_says_holding is False:
            logger.info(
                "Verify: object-detect flag False (stale), "
                "deferring to force (force=%s pos=%s)",
                force_says_holding,
                pos_says_holding,
            )
        if force_says_holding is True:
            logger.info("Verify: GRASPED (force confirmed)")
            return True
        if force_says_holding is False:
            logger.info(
                "Verify: NOT grasped (force says empty even though "
                "pos=%s flag=%s)",
                pos_says_holding,
                flag_says_holding,
            )
            return False
        if pos_says_holding is True and flag_says_holding is True:
            logger.info("Verify: GRASPED (pos+flag both positive, force ambiguous)")
            return True
        if pos_says_holding is True and flag_says_holding is None:
            logger.info("Verify: GRASPED (pos positive, flag unavailable)")
            return True
        logger.info(
            "Verify: NOT grasped (force=%s pos=%s flag=%s)",
            force_says_holding,
            pos_says_holding,
            flag_says_holding,
        )
        return False

    def _verify_holding_during_transport(self) -> bool:
        """
        Is the object still held during a transport move?

        Force-feedback first, then position check. Conservative: returns
        True unless both force and position confirm loss.
        """
        self._check_abort()
        gt = self._gripper_type()

        force_says_holding = None
        try:
            if hasattr(self.robot, "get_tcp_force"):
                ft = np.asarray(self.robot.get_tcp_force(), dtype=float)
                if ft.size >= 6:
                    baseline = getattr(self, "_pre_grasp_force", None)
                    if baseline is not None and baseline.size >= 3:
                        delta_xyz = ft[:3] - baseline[:3]
                        delta_mag = float(np.linalg.norm(delta_xyz))
                        delta_z = float(delta_xyz[2])
                        if delta_mag > 0.25 or delta_z < -0.25:
                            force_says_holding = True
                        elif delta_mag < 0.10 and abs(delta_z) < 0.10:
                            force_says_holding = False
                        logger.info(
                            "Transport force: |delta|=%.2f F_z_delta=%.2f -> %s",
                            delta_mag,
                            delta_z,
                            {True: "HOLDING", False: "LOST", None: "ambiguous"}[
                                force_says_holding
                            ],
                        )
        except Exception as exc:
            logger.warning("Transport force read failed: %s", exc)

        if gt == "robotiq_2f85":
            self._robotiq_resqueeze(speed=50, force=100, settle=0.3)

        pos = self._get_gripper_position()
        pos_says_holding = None
        if pos is not None:
            pos_says_holding = (
                pos < self.GRIPPER_FULLY_CLOSED and pos < self.GRIPPER_EMPTY_THRESHOLD
            )
            logger.info(
                "Transport position: pos=%d empty=%d closed=%d -> %s",
                pos,
                self.GRIPPER_EMPTY_THRESHOLD,
                self.GRIPPER_FULLY_CLOSED,
                {True: "HOLDING", False: "LOST", None: "ambiguous"}[pos_says_holding],
            )

        if force_says_holding is True:
            return True
        if force_says_holding is None and pos_says_holding is True:
            return True
        if force_says_holding is False:
            logger.info("Transport: object LOST (force evidence)")
            return False
        return True
