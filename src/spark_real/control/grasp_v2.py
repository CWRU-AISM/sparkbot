"""Robotiq grasp v2: mid-height grasp with force verification."""

import logging
import os
import time
from typing import Optional, Tuple

import numpy as np

from spark_real.control.executor_types import ExecutionResult
from spark_real.control.grasp_verdict import (
    GRASP_WIDTH_MAX_M,
    GRASP_WIDTH_MIN_M,
    GraspVerdict,
    force_verdict,
)

logger = logging.getLogger(__name__)


class GraspV2Mixin:
    """Robotiq grasp v2: mid-height grasp with force verification."""

    @staticmethod
    def _derive_target_width_from_detection(
        detection: dict,
    ) -> Optional[Tuple[float, float, str]]:
        """
        Synthesize a width-targeted grasp width + force from mask geometry.

        Uses 0.6x the OBB minor axis (depth-backprojected to meters) as the
        target width. Returns (target_width_m, force_hint_N, source) or None.
        """
        if detection is None:
            return None
        try:
            obb_minor_m = float(detection.get("obb_minor_m", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        if obb_minor_m <= 0.0:
            return None

        raw_width = obb_minor_m * 0.6
        target_width = float(
            np.clip(
                raw_width,
                GRASP_WIDTH_MIN_M,
                min(0.060, GRASP_WIDTH_MAX_M),
            )
        )

        force_hint_N = 35.0
        try:
            aspect_ratio = float(detection.get("aspect_ratio", 1.0) or 1.0)
        except (TypeError, ValueError):
            aspect_ratio = 1.0
        if obb_minor_m < 0.010:
            force_hint_N = 20.0
        elif aspect_ratio >= 3.0:
            force_hint_N = 25.0
        elif obb_minor_m >= 0.050:
            force_hint_N = 45.0

        source = "mask_obb_minor_x0.6"
        if raw_width != target_width:
            source += "_clamped"
        return (target_width, force_hint_N, source)

    def _grasp_v2(self, params: dict, t0: float) -> ExecutionResult:
        """Mid-height grasp with lift-based verify (gated by SPARK_GRASP_V2=1).

        (1) descend the OPEN jaws to block MID-height so they straddle the
        block, (2) clamp with force, (3) verify by LIFTING 2cm and checking
        the jaws stayed partly open on an object. Hard-clamped to TABLE_Z_FLOOR;
        gentle descent; abort-safe.

        Tunables (env, restart to apply): SPARK_GRASP_DEPTH_M (default 0.022,
        how far below the perceived top to aim, ~half a 48mm block) and
        SPARK_GRASP_EMPTY_POS (default 205, gripper pos at/above which the jaws
        are considered fully shut = empty; a held 36mm block reads ~147).
        """
        self._check_abort()
        # Grasp force (0-100): GRASP_CLOSE_FORCE default, BT `force` param wins.
        # A FORCE cap only: the close still commands the fully-closed POSITION.
        force = float(params.get("force", self.GRASP_CLOSE_FORCE))
        # No target_width param -> full close (pos 1.0), not a partial width
        # setpoint. Only an explicit BT target_width narrows it.
        tw = params.get("target_width", None)
        self._last_grasp_target_width = float(tw) if tw is not None else None
        # Class default (overridden by the `grasp:` YAML block); env var wins.
        depth = float(os.environ.get("SPARK_GRASP_DEPTH_M", self.GRASP_DEPTH_M))

        # 1. Descend OPEN jaws to block mid-height, clamped above the table.
        self._ensure_jaws_open("grasp")
        self._wait_force_settled()
        # Baseline TCP force (rtde_r channel) with the jaws open. The gripper
        # position/OBJ REGISTER reads ride the contended _send_script path and
        # come back STALE, so they cannot verify a grasp alone.
        try:
            _bft = np.asarray(self.robot.get_tcp_force(), dtype=float)
            base_ft = _bft[:6].copy() if _bft.size >= 6 else np.zeros(6)
        except Exception:
            base_ft = np.zeros(6)
        start = self._get_current_position()
        # Reuse the approach's (possibly OBB-yawed) orientation so the jaws stay
        # aligned to the object through the descent + clamp.
        grasp_orient = getattr(self, "_active_grasp_orient", None) or self.GRASP_ORIENTATION

        # Direct descent to the grasp height computed from the detection. depth
        # is how far below the approach pose to aim; clamp to the safe
        # closed-fingertip floor. Force is only a GUARD: if the jaws feel contact
        # before reaching the target (a thick block, a mis-detected top), stop
        # early, back off a couple mm, and clamp there instead of ramming.
        target_z = max(float(start[2]) - depth, self.TABLE_Z_FLOOR + 0.002)
        contact_n = float(os.environ.get("SPARK_GRASP_CONTACT_N", "8.0"))
        backoff = float(os.environ.get("SPARK_GRASP_BACKOFF_M", "0.003"))

        def _fz_delta():
            try:
                ft = np.asarray(self.robot.get_tcp_force(), dtype=float)
                return abs(float(ft[2])) - abs(float(base_ft[2])) if ft.size >= 3 else 0.0
            except Exception:
                return 0.0

        contact_z = None
        if hasattr(self.robot, "send_velocity"):
            desc_v = max(0.015, self.velocity * 0.15)  # normal descent, not a crawl
            logger.info(
                "Grasp v2: descend from z=%.3f to target z=%.3f (v=%.3f, "
                "contact-guard>%.1fN)",
                float(start[2]), target_z, desc_v, contact_n,
            )
            t_start = time.time()
            while not self._abort:
                cur = self._get_current_position()
                if cur[2] <= target_z:            # reached computed height, no contact
                    break
                if time.time() - t_start > 15.0:
                    logger.warning("Grasp v2: descent time cap at z=%.3f", cur[2])
                    break
                d = _fz_delta()
                if abs(d) > contact_n:            # GUARD: unexpected early contact
                    contact_z = cur[2]
                    logger.info("Grasp v2: early contact at z=%.3f (|dFz|=%.1fN)", cur[2], d)
                    break
                self.robot.send_velocity([0.0, 0.0, -desc_v, 0.0, 0.0, 0.0], 0.5, 0.06)
                time.sleep(0.03)
            try:
                self.robot.send_velocity([0.0] * 6, 1.0, 0.05)
            except Exception:
                pass
            if contact_z is not None:
                # Backed off a couple mm above the contact point, then clamp there.
                back = self._get_current_position()
                back[2] = min(contact_z + backoff, float(start[2]))
                self._move_to(back, grasp_orient, velocity=self.velocity * 0.15)
                logger.info(
                    "Grasp v2: contact z=%.3f -> backed off %.0fmm to z=%.3f, clamping",
                    contact_z, backoff * 1000, back[2],
                )
        else:
            # Fallback (driver without velocity control): direct servo to target.
            descend = start.copy()
            descend[2] = target_z
            self._servo_to(descend, grasp_orient, velocity=self.velocity * 0.15)

        # 2. Clamp to the fully-closed POSITION at the capped `force`: the
        # closed position takes up slack on a soft object while the force cap
        # protects it from over-crush. The re-squeeze is CONDITIONAL: gObj
        # True at this stationary pose means the jaws stopped on an object and
        # there is no slack to take up; otherwise it runs with a settle sized
        # to the real travel from the measured position.
        self._check_abort()
        # Record the ACTUAL grasp TCP z at jaw close: the object hangs from
        # THIS height, not from perceived_top + GRIPPER_OPEN_Z_OFFSET. The
        # release computation reads it back (behind place.release_giveback).
        try:
            self._actual_grasp_tcp_z = float(self._get_current_position()[2])
        except Exception:  # noqa: BLE001 - bookkeeping must not stop a grasp
            self._actual_grasp_tcp_z = None
        self._gripper_squeeze(
            force=force, speed=60, settle=self._robotiq_travel_settle_s(60)
        )
        fast_ok = os.environ.get("SPARK_GRASP_SKIP_LIFT_VERIFY", "1") == "1"
        gobj_fast, gpos_fast = (None, None)
        # The squeeze settle above already covers rq_close_and_wait's full
        # stroke, so this read is fresh.
        if fast_ok:
            gobj_fast, gpos_fast = self._read_gripper_state()
        if gobj_fast is not True:
            self._robotiq_resqueeze(
                speed=50,
                force=int(force),
                settle=self._robotiq_travel_settle_s(
                    50, from_pos=gpos_fast, to_pos=self.ROBOTIQ_POS_MAX
                ),
            )
            if fast_ok:
                self._abort_sleep(
                    float(os.environ.get("SPARK_GRASP_VERIFY_SETTLE_S",
                                         self.GRASP_VERIFY_SETTLE_S))
                )
                gobj_fast, gpos_fast = self._read_gripper_state()

        # Fast path: a gObj flag confirming the jaws stopped on an object at
        # the stationary clamped pose skips the ~10s lift-based force verify
        # below (the fallback when gObj is unavailable or does NOT confirm).
        # SPARK_GRASP_SKIP_LIFT_VERIFY=0 forces the full lift verify.
        if fast_ok:
            # The flag alone is not enough: gObj has read True with the jaws
            # open (pos=5). A contradiction runs the full lift verify instead.
            if gobj_fast is True and self._jaws_contradict_holding(gpos_fast):
                logger.warning(
                    "Grasp v2: object-detect flag says held but the jaws are "
                    "open; NOT taking the fast path -- running the full lift "
                    "verify"
                )
                gobj_fast = None
            if gobj_fast is True:
                self._holding = True
                logger.info(
                    "Grasp v2: GRASPED (gObj fast-path, pos=%s; lift verify skipped)",
                    gpos_fast,
                )
                self._record_grasp_verdict(
                    GraspVerdict(
                        held=True,
                        source="gobj_fast",
                        gobj=True,
                        gripper_pos=gpos_fast,
                        force_empty_n=self._force_empty_n(),
                        label=self._last_keypoint_label or "",
                        strategy=getattr(self, "_active_grasp_strategy", "") or "",
                    )
                )
                return ExecutionResult(
                    action_type="grasp",
                    success=True,
                    message=f"Grasped (gObj object-detect, pos={gpos_fast})",
                    duration=time.time() - t0,
                )
            # gObj did not confirm -> fall through to the lift + force verify.

        # 3. Verify by TCP FORCE + a confirming lift, NOT the (stale) gripper
        # position register. A real object shows a clear wrist-force delta from
        # the squeeze reaction and/or its weight on the lift; empty air stays
        # near zero through both.
        def _fdelta():
            try:
                ft = np.asarray(self.robot.get_tcp_force(), dtype=float)
                if ft.size >= 6:
                    return (
                        float(np.linalg.norm(ft[:3] - base_ft[:3])),
                        float(ft[2] - base_ft[2]),
                    )
            except Exception:
                pass
            return 0.0, 0.0

        clamp_mag, clamp_dz = _fdelta()
        # Lift 2cm; a held block adds a downward pull (its weight) at the wrist.
        self._check_abort()
        lift = self._get_current_position()
        lift[2] += self.GRASP_LIFT_CHECK_M
        self._move_to(lift, grasp_orient, velocity=self.velocity * 0.3)
        self._abort_sleep(0.3)
        lift_mag, lift_dz = _fdelta()
        empty_n = self._force_empty_n()
        lift_min = self._lift_delta_min_n()
        fv = force_verdict(clamp_mag, clamp_dz, lift_mag, lift_dz, empty_n, lift_min)
        peak = fv.peak_n
        lift_delta_n = fv.lift_delta_n
        lift_holds = fv.lift_corroborates
        force_says_holding = fv.holding

        # PRIMARY signal: the Robotiq gObj object-detect flag, read at a
        # STATIONARY pose (settled after the 2cm lift) after a forced fresh
        # publish. gObj True = the jaws stopped on an object. Valid on a
        # COMPRESSIBLE object too: a soft plushie squeezes the jaws PAST the
        # "fully closed" position (pos ~252 >= 250) while genuinely held, so
        # the position veto must never override a True gObj.
        settle_verify = float(
            os.environ.get("SPARK_GRASP_VERIFY_SETTLE_S", self.GRASP_VERIFY_SETTLE_S)
        )
        self._abort_sleep(settle_verify)
        obj_detected, gpos = self._read_gripper_state()
        logger.info(
            "Grasp v2 verify: gObj=%s pos=%s force_peak=%.2fN (empty<%.2fN) "
            "clamp|df|=%.2f lift|df|=%.2f lift_delta=%.2fN (corroborates >= %.2f: %s)",
            obj_detected, gpos, peak, empty_n, clamp_mag, lift_mag,
            lift_delta_n, lift_min, lift_holds,
        )

        def _verdict(held: bool, source: str) -> None:
            self._record_grasp_verdict(
                GraspVerdict(
                    held=held,
                    source=source,
                    gobj=obj_detected,
                    gripper_pos=gpos,
                    force_peak_n=float(peak),
                    force_empty_n=float(empty_n),
                    lift_delta_n=lift_delta_n,
                    lift_corroborates=lift_holds,
                    label=self._last_keypoint_label or "",
                    strategy=getattr(self, "_active_grasp_strategy", "") or "",
                )
            )

        # 1. Positive gObj -> GRASPED. No position veto (plushie false-negative).
        if obj_detected is True:
            self._holding = True
            logger.info("Grasp v2: GRASPED (Robotiq object-detect, pos=%s)", gpos)
            _verdict(True, "gobj")
            return ExecutionResult(
                action_type="grasp",
                success=True,
                message=f"Grasped (gObj object-detect, pos={gpos})",
                duration=time.time() - t0,
            )

        # 2. gObj did NOT confirm (False or unavailable). Corroborate with TCP
        #    force. Force alone false-positives when the tool presses a flat
        #    plate / the table (large |dF| but jaws close FULLY on nothing), so
        #    here, and only where gObj has NOT confirmed, keep the pos>=CLOSED
        #    veto to reject that flat-plate/edge-miss case.
        if force_says_holding:
            if gpos is not None and gpos >= self.GRIPPER_FULLY_CLOSED:
                self._holding = False
                logger.info(
                    "Grasp v2: force peak %.2fN but jaws fully closed "
                    "(pos=%d >= %d) and gObj=%s -> empty (flat-plate/miss)",
                    peak, gpos, self.GRIPPER_FULLY_CLOSED, obj_detected,
                )
                _verdict(False, "jaws_fully_closed")
                return ExecutionResult(
                    action_type="grasp",
                    success=False,
                    message=f"Empty (jaws fully closed, pos={gpos})",
                    duration=time.time() - t0,
                )
            self._holding = True
            logger.info(
                "Grasp v2: GRASPED (force-corroborated, peak %.2fN, pos=%s, "
                "gObj=%s)",
                peak, gpos, obj_detected,
            )
            _verdict(True, "force")
            return ExecutionResult(
                action_type="grasp",
                success=True,
                message=f"Grasped (force-corroborated, {peak:.1f}N)",
                duration=time.time() - t0,
            )

        # 3. gObj not True AND force ~empty -> genuinely empty.
        self._holding = False
        logger.info(
            "Grasp v2: NOT grasped (gObj=%s, force peak %.2fN < %.2fN, "
            "lift_delta %.2fN)",
            obj_detected, peak, empty_n, lift_delta_n,
        )
        _verdict(False, "empty")
        return ExecutionResult(
            action_type="grasp",
            success=False,
            message=f"Empty (gObj={obj_detected}, force peak {peak:.1f}N)",
            duration=time.time() - t0,
        )
