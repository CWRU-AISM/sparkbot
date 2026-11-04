"""Single Cartesian moves and the wait-for-arrival loop."""

import logging
import time

import numpy as np

from spark_real.control.executor_types import (
    AbortRequested,
)
from spark_real.control.motion_common import _StillnessTracker

logger = logging.getLogger(__name__)


class MotionIOMixin:
    """Single Cartesian moves and the wait-for-arrival loop."""

    def _move_to(self, position, orientation, velocity=None):
        """
        Move with workspace protection; uses movej for large moves.
        """
        self._check_abort()
        vel = velocity or self.velocity
        target_raw = np.array(position, dtype=float)
        target = target_raw.copy()
        if not self._check_workspace(target):
            target = np.clip(target, self.WORKSPACE_MIN, self.WORKSPACE_MAX)
            logger.info(
                "Move target CLIPPED by workspace: requested (%.3f,%.3f,%.3f) "
                "-> commanded (%.3f,%.3f,%.3f)",
                target_raw[0],
                target_raw[1],
                target_raw[2],
                target[0],
                target[1],
                target[2],
            )

        pose = list(target) + list(orientation)
        current = self._get_current_position()
        dist = np.linalg.norm(target - current)
        logger.info(
            "MOVE_TO target=(%.3f,%.3f,%.3f) orient=(%.2f,%.2f,%.2f) "
            "from=(%.3f,%.3f,%.3f) dist=%.3fm v=%.2f",
            target[0],
            target[1],
            target[2],
            orientation[0],
            orientation[1],
            orientation[2],
            current[0],
            current[1],
            current[2],
            dist,
            vel,
        )
        if dist > 1.0:
            logger.warning("Move %.3fm too large, skipping", dist)
            return
        try:
            self._movej_via_ik(target, orientation, vel)
        except AbortRequested:
            raise
        except Exception as e:
            logger.warning("_movej_via_ik failed (%s); trying _movej_to_pose", e)
            try:
                # Full velocity: on UR10e _movej_to_pose IS the primary path
                # (controller firmware IK); halving it would make every UR
                # move crawl at ~0.375 rad/s.
                self._movej_to_pose(pose, vel)
            except Exception:
                raise

    def _issue_linear(self, pose, vel):
        """
        Dispatch a Cartesian move to whichever driver method exists.
        """
        if hasattr(self.robot, "move_to_pose"):
            self.robot.move_to_pose(pose, velocity=vel, wait=False)
        elif hasattr(self.robot, "move_linear"):
            try:
                self.robot.move_linear(pose, velocity=vel, asynchronous=True)
            except TypeError:
                self.robot.move_linear(pose, velocity=vel)

    def _move_to_linear(self, position, orientation, velocity=None):
        """
        Force linear (Cartesian) move to preserve TCP orientation.
        """
        self._check_abort()
        vel = velocity or self.velocity
        target = np.array(position)
        if not self._check_workspace(target):
            target = np.clip(target, self.WORKSPACE_MIN, self.WORKSPACE_MAX)
        pose = list(target) + list(orientation)
        dist = np.linalg.norm(target - self._get_current_position())
        if dist > 1.0:
            logger.warning("Move %.3fm too large, skipping", dist)
            return
        try:
            self._issue_linear(pose, vel)
            # Budget the wait to the move actually commanded. There is no
            # q_target on this path (the driver runs its own Cartesian IK), so
            # the only exits are proximity and stillness-after-motion, and a
            # movel short enough to finish inside the URScript start lag hits
            # neither. 3x the ideal duration plus the start lag and a second
            # of slack, capped at 15 s.
            ideal = dist / max(vel, 1e-3)
            budget = float(
                np.clip(
                    3.0 * ideal + self.URSCRIPT_START_LAG_S + 1.0,
                    self._min_motion_budget_s(),
                    15.0,
                )
            )
            arrived = self._wait_for_motion(target, timeout=budget)
            if arrived is False:
                # The arm never moved and is not at the target: the command
                # was dropped (dead RTDE control script, refused URScript).
                # Raising lets callers with a fallback (e.g. the tilt release)
                # take it instead of proceeding as if the pose was reached.
                raise RuntimeError(
                    f"linear move to {list(np.round(target, 3))} produced no "
                    "motion (command dropped)"
                )
        except AbortRequested:
            raise
        except Exception as e:
            if any(k in str(e).lower() for k in ("singular", "protective", "accel")):
                try:
                    self._movej_to_pose(pose, vel * 0.5)
                except Exception:
                    raise
            else:
                raise

    def _min_motion_budget_s(self) -> float:
        """Floor for any arrival budget, tied to the program start lag.

        A fast move has a small `ideal` and collapses onto the floor; a floor
        that does not clear the upload lag by a real margin times out before
        the arm has moved and reads as a motion failure.
        """
        return float(self.URSCRIPT_START_LAG_S + self.MOTION_BUDGET_SLACK_S)

    def _wait_for_motion(self, target, timeout=15.0, q_target=None):
        """
        Block until a fire-and-forget URScript movej lands.

        ``q_target`` is the authoritative signal: the controller is done when
        the measured joints equal the commanded ones. Cartesian proximity
        alone is not, because the commanded pose and the achieved TCP
        routinely differ by more than the 5 mm gate (IK rounding, TCP offset),
        and the stillness exit cannot fire on a move that finished inside the
        start lag.
        """
        t0 = time.time()
        motion = _StillnessTracker(
            self._get_current_position(),
            t0,
            sample_s=self.MOTION_SAMPLE_S,
            still_samples=self.MOTION_STILL_SAMPLES,
        )
        next_record = t0

        def _arrived():
            """Retire the driver's motion lease: this move is provably done."""
            clear = getattr(self.robot, "clear_motion_lease", None)
            if callable(clear):
                try:
                    clear()
                except Exception:  # noqa: BLE001 - never block on bookkeeping
                    pass

        while time.time() - t0 < timeout:
            self._check_abort()
            time.sleep(self.MOTION_POLL_S)
            now = time.time()
            tcp = self._get_current_position()
            # The recorder keeps its own ~10 Hz cadence: the faster poll exists
            # to cut arrival latency, not to change what gets logged.
            if self._recorder and now >= next_record:
                self._recorder.record()
                next_record = now + self.MOTION_SAMPLE_S
            # 1. Joints match the command: unambiguously arrived.
            if q_target is not None:
                try:
                    q = np.asarray(self.robot.get_joint_positions(), dtype=float)
                    if np.max(np.abs(q - np.asarray(q_target, dtype=float))) < (
                        self.JOINT_ARRIVE_TOL
                    ):
                        _arrived()
                        return True
                except Exception:  # noqa: BLE001 - fall through to the other tests
                    pass
            # 2. Cartesian proximity (kept: works when q_target is unavailable).
            if np.linalg.norm(tcp - target) < 0.005:
                _arrived()
                return True
            # 3. Stillness, past the start lag AND only after the arm was seen
            #    to move: if the URScript upload lag runs past
            #    URSCRIPT_START_LAG_S, "still" means "not started yet", not
            #    "arrived" (returned ~20 cm short, grasp closed above the
            #    object). The joint-target test above is exact and needs no
            #    motion history.
            motion.update(tcp, now)
            settled = motion.stopped() and now - t0 > self.URSCRIPT_START_LAG_S
            if settled and motion.ever_moved:
                _arrived()
                return True
        dist = float(np.linalg.norm(self._get_current_position() - target))
        logger.warning(
            "[_wait_for_motion] %.1fs timeout with no arrival signal " "(moved=%s, dist=%.3fm)",
            timeout,
            motion.ever_moved,
            dist,
        )
        # A timeout where the arm NEVER MOVED and is still away from the
        # target is a dropped command, not a slow one. False lets callers
        # distinguish "arm is basically there" from "nothing happened".
        return bool(motion.ever_moved or dist < 0.02)
