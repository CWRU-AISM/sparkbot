"""Operator abort: epochs, brake, and abort-aware sleeps."""

import logging
import time
from typing import Optional

from spark_real.control.executor_types import (
    AbortRequested,
)

logger = logging.getLogger(__name__)


class AbortMixin:
    """Operator abort: epochs, brake, and abort-aware sleeps."""

    def note_abort_requested(self) -> int:
        """
        Raise the abort flag, advance the epoch, and return the new epoch.

        Every stop path must come through here rather than assigning
        ``_abort`` directly, or the request is invisible to execute_score's
        arming check. The epoch goes up FIRST, so a reader that sees
        ``_abort`` True has already seen the epoch move.

        The returned epoch identifies THIS request, so the primitive-timeout
        watchdog can retract its own abort without stepping on an operator
        stop that arrived in the same window.
        """
        with self._abort_lock:
            self._abort_epoch += 1
            self._abort = True
            return self._abort_epoch

    def retract_abort(self, epoch: int) -> bool:
        """
        Withdraw abort request ``epoch``. False unless it is the ONLY
        outstanding one.

        For the primitive-timeout watchdog, whose abort is a transient it must
        undo so recovery can run. TWO conditions, both needed:

        * nothing NEWER has landed (``_abort_epoch == epoch``): an operator
          stop raised while the primitive unwound;
        * nothing OLDER is still outstanding (``epoch - 1 == _armed_epoch``):
          an operator stop raised BEFORE the watchdog fired.

        Together: this request is the only thing between the task's arming
        point and now, so undoing it restores exactly the pre-timeout state.
        """
        with self._abort_lock:
            if self._abort_epoch != epoch:
                logger.warning(
                    "retract_abort(%d) refused: a newer stop (epoch %d) is "
                    "standing; leaving the abort in place",
                    epoch,
                    self._abort_epoch,
                )
                return False
            if epoch - 1 != self._armed_epoch:
                logger.warning(
                    "retract_abort(%d) refused: an earlier stop is still "
                    "outstanding (armed at epoch %d); leaving the abort in "
                    "place",
                    epoch,
                    self._armed_epoch,
                )
                return False
            self._armed_epoch = epoch
            self._abort = False
            return True

    def abort(self, epoch: Optional[int] = None) -> bool:
        """Request a stop AND brake the arm. True if the brake reached the arm.

        Setting the flag only stops this process from issuing new commands.
        URScript motion is fire-and-forget, so an unbraked abort leaves the arm
        running to its target while the executor unwinds. Brake here, at the
        single entry point every caller uses (/api/stop, /api/abort, the
        primitive-timeout watchdog).

        ``epoch`` is for a caller that has ALREADY called
        note_abort_requested() and is holding its epoch so it can retract
        exactly that request later (the watchdog). Passing it skips the bump
        so one stop is one epoch.
        """
        if epoch is None:
            self.note_abort_requested()
        self._servo.abort()
        braked = self._stop_robot()
        if braked:
            logger.warning("Abort requested; arm braked")
        else:
            logger.critical(
                "Abort requested but THE ARM WAS NOT BRAKED -- it may still be "
                "moving. Use the physical E-stop."
            )
        return braked

    def _stop_robot(self) -> bool:
        """Brake the arm. True only if a stop actually reached the driver.

        Order matters. The first branch is the UR escalation (URScript stopj on
        the primary socket, then a reopened socket, then Dashboard); it is the
        only thing that stops a fire-and-forget movej. rtde_c.stopJ is a no-op
        against a dead control script, so it must not be the first and only try.
        """
        emergency = getattr(self.robot, "emergency_stop", None)
        if callable(emergency):
            try:
                result = emergency()
                if isinstance(result, dict) and result.get("braked"):
                    return True
                logger.error("emergency_stop did not reach the arm: %s", result)
            except Exception as exc:  # noqa: BLE001 - fall through to the rest
                logger.warning("emergency_stop raised: %s", exc)
        # Any other URScript-capable driver: emit the brake directly. The
        # abort-aware _send_script wrapper lets braking scripts through.
        if getattr(self.robot, "SUPPORTS_URSCRIPT", False):
            send = getattr(self.robot, "_send_script", None)
            if callable(send):
                try:
                    if bool(send("stopj(%.2f)" % self.ABORT_DECEL_FALLBACK)):
                        return True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("stopj send failed: %s", exc)
        # Drivers without a URScript channel: try each stop method in order,
        # and do not stop at the first one that merely fails to raise.
        braked = False
        for name in ("stop_motion", "stop", "servo_stop"):
            fn = getattr(self.robot, name, None)
            if callable(fn):
                try:
                    # A driver that reports failure (False) has not braked.
                    braked = fn() is not False or braked
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s failed: %s", name, exc)
        return braked

    def _check_abort(self):
        """
        Raise AbortRequested if stop has been requested.
        """
        if self._abort:
            self._stop_robot()
            raise AbortRequested()

    def _abort_sleep(self, duration: float, tick: float = 0.05):
        """
        Abort-aware sleep. Polls the abort flag every tick seconds.
        """
        t_end = time.time() + duration
        while True:
            self._check_abort()
            remaining = t_end - time.time()
            if remaining <= 0:
                return
            time.sleep(min(tick, remaining))
