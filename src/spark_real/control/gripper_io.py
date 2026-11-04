"""Gripper commands and reads: squeeze, open, jaw position, object flag."""

import logging
import time
from typing import Optional

import numpy as np

from spark_real.control.grasp_verdict import _call_force_publish, _call_maybe_passive

logger = logging.getLogger(__name__)


class GripperIOMixin:
    """Gripper commands and reads: squeeze, open, jaw position, object flag."""

    def _gripper_type(self) -> str:
        """
        Return the underlying gripper hardware class.

        Walks through SafeRobot's __getattr__ chain so wrappers are
        transparent. Defaults to "robotiq_2f85" for backward compat.
        """
        target = self.robot
        seen = set()
        while target is not None and id(target) not in seen:
            seen.add(id(target))
            gt = getattr(target, "GRIPPER_TYPE", None)
            if isinstance(gt, str):
                return gt
            target = getattr(target, "_robot", None)
        return "robotiq_2f85"

    def _robotiq_travel_settle_s(
        self, speed_norm: int, from_pos: Optional[int] = None, to_pos: float = None
    ) -> float:
        """Settle a jaw move needs, from the datasheet.

        ``speed_norm`` is the 0-100 knob the executor passes to the driver;
        ``from_pos``/``to_pos`` are jaw counts (0 open .. 255 closed). Unknown
        ``from_pos`` means "assume the full stroke", which is the conservative
        direction. The driver's own post-send sleep is subtracted, so the return
        value is only the part the caller still has to wait out; it is never
        negative and never below the margin.
        """
        frac = float(np.clip(float(speed_norm) / 100.0, 0.0, 1.0))
        mm_s = self.ROBOTIQ_SPEED_MIN_MM_S + frac * (
            self.ROBOTIQ_SPEED_MAX_MM_S - self.ROBOTIQ_SPEED_MIN_MM_S
        )
        if from_pos is None or to_pos is None:
            travel_frac = 1.0
        else:
            travel_frac = abs(float(to_pos) - float(from_pos)) / self.ROBOTIQ_POS_MAX
            travel_frac = float(np.clip(travel_frac, 0.0, 1.0))
        travel_s = self.ROBOTIQ_STROKE_MM * travel_frac / max(mm_s, 1e-6)
        return float(
            max(
                self.ROBOTIQ_SETTLE_MARGIN_S,
                travel_s + self.ROBOTIQ_SETTLE_MARGIN_S - self.ROBOTIQ_DRIVER_SETTLE_S,
            )
        )

    def _ensure_jaws_open(self, context: str = "") -> bool:
        """Open the jaws only if they are not already open. True if commanded.

        Requires a stationary arm (precondition for the register publish).
        Conservative on every unknown: a failed read, a non-Robotiq gripper,
        or a position at/above the open threshold all fall through to
        commanding the open. Descending with part-closed jaws is a collision,
        so the open is only SKIPPED when the register proves the jaws open.
        """
        pos = None
        if self._gripper_type() == "robotiq_2f85":
            try:
                _obj, pos = self._read_gripper_state()
            except Exception as exc:  # noqa: BLE001
                logger.debug("_ensure_jaws_open read failed: %s", exc)
                pos = None
        if pos is not None and pos <= self.JAWS_OPEN_MAX_POS:
            logger.info(
                "%s: jaws already open (pos=%s <= %d); skipping open_gripper",
                context or "gripper",
                pos,
                self.JAWS_OPEN_MAX_POS,
            )
            return False
        try:
            self.robot.open_gripper()
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s open_gripper failed: %s", context or "gripper", exc)
            return False
        speed = self.ROBOTIQ_OPEN_SPEED_NORM_ASSUMED
        self._abort_sleep(
            self._robotiq_travel_settle_s(speed, from_pos=pos, to_pos=0.0)
            if pos is not None
            else self._robotiq_travel_settle_s(speed)
        )
        return True

    def _wait_force_settled(
        self, timeout: float = 0.3, tol_n: float = 0.5, poll: float = 0.02
    ) -> bool:
        """Block until the TCP force channel is quiet enough to baseline off.

        Settled means two consecutive get_tcp_force samples agree within
        ``tol_n``. The rtde_r channel is independent of the URScript socket,
        so this costs no program upload and cannot cancel a gripper program.
        Returns True if it settled, False if it ran out the timeout.
        """
        t_end = time.time() + float(timeout)
        prev = None
        while True:
            self._check_abort()
            try:
                ft = np.asarray(self.robot.get_tcp_force(), dtype=float)
                cur = ft[:3].copy() if ft.size >= 3 else None
            except Exception:  # noqa: BLE001
                cur = None
            if cur is None:
                self._abort_sleep(max(0.0, t_end - time.time()))
                return False
            if prev is not None and float(np.linalg.norm(cur - prev)) < tol_n:
                return True
            prev = cur
            remaining = t_end - time.time()
            if remaining <= 0:
                return False
            time.sleep(min(poll, remaining))

    def _gripper_squeeze(self, force: int, speed: int = 50, settle: float = 0.3):
        """
        Close gripper around an object.

        Dispatches via GRIPPER_TYPE: robotiq_2f85 uses the UR fast path;
        other drivers use set_gripper_position, else close_gripper.
        """
        gt = self._gripper_type()

        # Width-targeted path
        tw = getattr(self, "_last_grasp_target_width", None)
        if tw is not None and hasattr(self.robot, "grasp_to_width"):
            try:
                self.robot.grasp_to_width(
                    width=float(tw),
                    force=min(force, 100),
                    speed=float(speed) / 100.0 if speed > 1 else float(speed),
                )
                self._abort_sleep(settle)
                return
            except Exception as exc:
                logger.warning(
                    "grasp_to_width(%.3f) raised: %s; falling back to "
                    "force-only close",
                    tw,
                    exc,
                )

        if gt == "robotiq_2f85" and hasattr(self.robot, "close_gripper"):
            # Full close to the mechanical stop (rq_close_and_wait), not the
            # norm-scaled move that halts at rPR~226 leaving a ~29/255 gap.
            self.robot.close_gripper(speed=speed, force=min(force, 100))
            self._abort_sleep(settle)
            return
        if hasattr(self.robot, "set_gripper_position"):
            try:
                self.robot.set_gripper_position(1.0, speed=speed, force=min(force, 100))
                self._abort_sleep(settle)
                return
            except Exception:
                pass
        self.robot.close_gripper()
        self._abort_sleep(settle)

    def _robotiq_resqueeze(
        self, speed: int = 50, force: int = 100, settle: float = 1.0
    ) -> bool:
        """
        Robotiq-only partial-close re-squeeze. Returns True if issued.
        """
        gt = self._gripper_type()
        if gt == "robotiq_2f85":
            if hasattr(self.robot, "close_gripper"):
                self.robot.close_gripper(speed=speed, force=min(force, 100))
                self._abort_sleep(settle)
                return True
            return False
        if gt in ("franka_hand", "ssg48", "dynamixel"):
            return False  # force-controlled grasp; no re-squeeze
        logger.warning(
            "Unknown GRIPPER_TYPE=%r; skipping partial-close "
            "re-squeeze (driver-specific safety unknown)",
            gt,
        )
        return False

    def _get_gripper_position(self, publish: bool = True) -> Optional[int]:
        """
        Jaw position on the Robotiq count scale, 0 (open) .. 255 (closed).

        Robotiq: the register read. ``publish=False`` reads the register a
        PRECEDING _force_gripper_publish already refreshed, instead of paying
        for a second publish of the same values. See _read_gripper_state.

        Other drivers: their get_gripper_position (already 0..255), else
        get_gripper_width against _GRASP_WIDTH_MAX_M. None only when there is
        no position source.
        """
        try:
            if hasattr(self.robot, "get_gripper_position"):
                return _call_maybe_passive(
                    self.robot.get_gripper_position, publish
                )
            if hasattr(self.robot, "get_gripper_width"):
                frac = 1.0 - float(self.robot.get_gripper_width()) / self._GRASP_WIDTH_MAX_M
                return float(np.clip(frac, 0.0, 1.0)) * 255.0
        except Exception:
            pass
        return None

    def _jaws_contradict_holding(self, pos) -> bool:
        """True when the gripper position is physically incompatible with
        holding an object, RE-READ once before judging.

        The gObj/position registers go stale on the contended script path, so
        a single open-looking sample is not enough to call a grasp failed, but
        two in a row against a flag that claims contact means the flag is the
        stale one. Never fabricates a pass.
        """
        try:
            if pos is None or float(pos) > self.JAWS_OPEN_POS_MAX:
                return False
        except (TypeError, ValueError):
            return False
        try:
            self._abort_sleep(0.15)
            gobj2, pos2 = self._read_gripper_state()
            if pos2 is None or float(pos2) > self.JAWS_OPEN_POS_MAX:
                logger.info(
                    "[grasp-check] first position sample %s was stale; "
                    "re-read %s is consistent with holding", pos, pos2,
                )
                return False
            logger.warning(
                "[grasp-check] jaws read OPEN twice (pos=%s then %s, open<=%d) "
                "while the object-detect flag claims contact -- the flag is "
                "stale, not the geometry",
                pos, pos2, self.JAWS_OPEN_POS_MAX,
            )
            return True
        except Exception:  # noqa: BLE001 - unknown stays non-contradictory
            return False

    def _is_object_detected(self, publish: bool = True) -> Optional[bool]:
        """Robotiq gObj flag, or None if unreadable. See _get_gripper_position."""
        try:
            if hasattr(self.robot, "is_object_detected"):
                return bool(
                    _call_maybe_passive(self.robot.is_object_detected, publish)
                )
        except Exception as exc:
            logger.warning("object-detect read failed: %s", exc)
        return None

    def _read_gripper_state(self):
        """
        One publish, then both register reads. Returns ``(gObj, position)``.

        gObj and jaw position live in two registers that a SINGLE
        _publish_gripper_state refreshes together; each extra publish uploads
        a URScript program and waits for the controller to run it.
        """
        self._force_gripper_publish()
        # _call_maybe_passive, not a direct publish=False call: subclasses and
        # test doubles may override these with a no-kwarg signature.
        return (
            _call_maybe_passive(self._is_object_detected, False),
            _call_maybe_passive(self._get_gripper_position, False),
        )

    def _force_gripper_publish(self):
        """Force ONE fresh Robotiq state publish before a stationary read.

        The driver only refreshes rq_current_pos_norm / rq_is_object_detected
        into the RTDE registers via _publish_gripper_state, and REFUSES to do
        so while the arm is moving (TCP-speed guard). A reliable gObj/position
        read requires the arm STATIONARY and a forced publish first. No-op for
        non-robotiq drivers or on any read error.
        """
        try:
            if hasattr(self.robot, "_publish_gripper_state"):
                # force=True: the caller guarantees a parked arm, so the
                # driver's motion lease must not withhold the refresh. The
                # driver still checks measured motion before it publishes.
                _call_force_publish(self.robot._publish_gripper_state)
        except Exception as exc:
            logger.debug("_force_gripper_publish failed: %s", exc)

    def _grip_intact(self) -> bool:
        """Cheap per-waypoint drop check at a STATIONARY arm pose.

        Primary signal is the driver's object-detect flag (Robotiq gObj:
        True = jaws stopped on an object). If it does not
        confirm, a still-spread jaw position (pos < GRIPPER_FULLY_CLOSED, on
        the 0..255 scale _get_gripper_position gives every driver) also counts
        as holding: a soft object keeps the jaws off the fully-closed stop.
        Only when the flag is not True AND the jaws have reached the
        fully-closed stop is it a DROP. Conservative: returns True for a
        driver with no jaw state at all and on any read failure (never
        fabricate a drop).
        """
        # Arm must be stationary here (caller's responsibility).
        obj, pos = self._read_gripper_state()
        if obj is None and pos is None and self._gripper_type() != "robotiq_2f85":
            return True
        if obj is True:
            logger.info("_grip_intact: HOLDING (object-detect flag)")
            return True
        if pos is not None and pos < self.GRIPPER_FULLY_CLOSED:
            logger.info(
                "_grip_intact: HOLDING (jaws still spread on object, "
                "pos=%d < closed=%d, gObj=%s)",
                pos,
                self.GRIPPER_FULLY_CLOSED,
                obj,
            )
            return True
        logger.warning(
            "_grip_intact: DROPPED (gObj=%s, pos=%s >= closed=%d)",
            obj,
            pos,
            self.GRIPPER_FULLY_CLOSED,
        )
        return False
