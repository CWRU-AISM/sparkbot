"""Replay of the last pick cycle after an evidenced verification failure."""

import logging
from typing import Optional, Sequence

from spark_real.control.success_predicates import FAIL, PASS
from spark_real.control.verify_retry_common import (
    RETRY_LIFT_M,
    RETRY_MIN_CLEAR_Z,
    _detection_fingerprint,
    _fail_has_evidence,
    _is_fresh,
    _keypoint_label,
    _replay_labels,
)

logger = logging.getLogger(__name__)


class ReplayMixin:
    """Replay of the last pick cycle after an evidenced verification failure."""

    def _replay_last_pick_cycle(self, score: dict, actions: list, max_attempts: int):
        """Opt-in recovery replay. Only ever reached on an explicit ``fail``.

        Bounded by ``max_attempts``. Every attempt establishes the arm state
        and re-perceives BEFORE it moves laterally; either precondition failing
        ends the loop rather than degrading into a blind replay.
        """
        last_pick_start = self._find_last_pick_cycle(actions)
        if last_pick_start is None:
            logger.info("Verification fail but no pick cycle to replay")
            return

        retry_actions = actions[last_pick_start:]
        pick_label = _keypoint_label(retry_actions[0])
        labels = _replay_labels(retry_actions)
        if not pick_label:
            logger.info("Verification fail but the pick cycle names no keypoint")
            return

        # A SUCCESSFUL release already added the object to _placed_labels and
        # cleared _last_pick_label. Both bits of bookkeeping block this retry:
        #   * executor_motion short-circuits move_to_keypoint with "already in
        #     container, skipping pick approach" for a placed label, so the
        #     replayed grasp would close on air;
        #   * the pre-approach re-detect is skipped for a placed label, so the
        #     replay would aim at the plan-time pose.
        # Un-place it for the duration of the retry; a completed retry re-marks
        # it in _run_replay_actions.
        if pick_label in self._placed_labels:
            self._placed_labels.discard(pick_label)
            logger.info(
                "Retry: un-placed '%s' so the replay re-detects it and does not "
                "skip the pick approach",
                pick_label,
            )

        for attempt in range(max_attempts):
            if self._abort:
                break

            logger.warning(
                "Verification FAILED (attempt %d/%d), replaying the last pick "
                "cycle -- this opens the gripper",
                attempt + 1,
                max_attempts,
            )

            if not self._establish_retry_ready_state():
                break
            if not self._reperceive_for_retry(pick_label, labels):
                break

            self._run_replay_actions(retry_actions, attempt, pick_label)
            if self._abort:
                break

            outcome = self._verify_task_completion(score, actions)
            if outcome.status == PASS:
                logger.info("Verification passed on replay %d", attempt + 1)
                return
            if outcome.status != FAIL or not _fail_has_evidence(outcome):
                # Same rule as the entry gate: only an evidenced fail continues.
                logger.info(
                    "Verification %s after replay %d; stopping", outcome.status, attempt + 1
                )
                return

        logger.warning("Giving up after %d replay attempt(s)", max_attempts)

    def _establish_retry_ready_state(self) -> bool:
        """Bring the arm to the ONE state a pick replay may start from.

        The replayed tree opens with ``move_to_keypoint -> grasp``: a lateral
        transit, a descent, and a close. That is safe only from jaws open, jaws
        EMPTY, tool lifted clear. After a failed place the arm may be open over
        the container, closed on nothing, still holding the object, or parked
        mid-transit, so none of it is assumed here; it is commanded and then
        checked. Returns False when the state could not be established, in
        which case the caller must not move.
        """
        self._check_abort()
        if getattr(self, "_holding", False):
            logger.warning(
                "Retry: the executor still believes it holds an object; opening "
                "at the current pose (the lowest drop available) before transit"
            )

        # Open in place. If something is still in the jaws this drops it, and
        # the current pose is the shortest fall on offer; lifting first would
        # only increase it.
        self.robot.open_gripper()
        self._abort_sleep(0.3)
        self._holding = False

        # Confirm empty. A soft object (the plushie) can stay seated in a
        # partly-open jaw; re-approaching with it aboard drags it through the
        # descent. Same signal _run_actions' own failure recovery trusts.
        verify_grasp = getattr(self, "_verify_grasp", None)
        if callable(verify_grasp):
            try:
                if verify_grasp():
                    logger.error(
                        "Retry ABANDONED: the gripper still reads HOLDING after "
                        "an open; not replaying a pick cycle with a loaded jaw"
                    )
                    self._holding = True
                    return False
            except Exception as exc:
                logger.warning(
                    "Retry: grip state unreadable (%s); treating the jaws as open",
                    exc,
                )

        # Lift clear. Vertical only, and before any lateral motion.
        self._check_abort()
        current = self._get_current_position()
        current[2] = max(current[2] + RETRY_LIFT_M, RETRY_MIN_CLEAR_Z)
        self._move_to(current, self.GRASP_ORIENTATION)
        return True

    def _reperceive_for_retry(self, pick_label: str, labels: Sequence[str]) -> bool:
        """Re-detect everything the replay drives to. False = do not move.

        The object is not where the plan said it was (that is why the place
        failed), so replaying against the plan-time detection map re-commits
        the same error. The pick target must therefore be re-detected AND the
        re-detection must have landed: ``redetect_single`` silently keeps the
        old position when the detector finds nothing, so each entry is
        fingerprinted either side of the call.
        """
        det_map = getattr(self, "detection_map", None)
        if det_map is None:
            logger.error("Retry ABANDONED: no detection map to refresh")
            return False

        fresh = False
        for label in labels:
            before = _detection_fingerprint(det_map, label)
            self._redetect_single(label)
            landed = _is_fresh(before, _detection_fingerprint(det_map, label))
            logger.info(
                "Retry re-perceive '%s': %s", label, "fresh" if landed else "STALE"
            )
            if label == pick_label:
                fresh = landed

        if not fresh:
            logger.error(
                "Retry ABANDONED: '%s' could not be re-perceived; replaying "
                "against the plan-time pose would repeat the miss",
                pick_label,
            )
            return False
        return True

    def _run_replay_actions(self, retry_actions: list, attempt: int, pick_label: str):
        """Dispatch the replayed segment. Perception is already refreshed."""
        for ri, action in enumerate(retry_actions):
            if self._abort:
                return
            atype = action.get("type", action.get("name", "unknown"))
            params = action.get("params", {})
            logger.info(
                "[RETRY-%d %d/%d] %s %s",
                attempt + 1,
                ri + 1,
                len(retry_actions),
                atype,
                params,
            )
            self._recorder.set_action_label(f"retry-{attempt+1} {atype}")

            result = self._dispatch_action(atype, params)
            self._results.append(result)

            if result.success:
                if atype == "release" and pick_label:
                    # Mirror _run_actions' post-release bookkeeping: the replay
                    # bypasses it, and pipeline_run's closed loop reads
                    # _placed_labels to decide what is still outstanding.
                    self._placed_labels.add(pick_label)
                    logger.info("Retry: re-marked '%s' as placed", pick_label)
                continue

            recovery = self._attempt_recovery(atype, params, result, retry_actions, ri)
            if recovery and recovery.success:
                self._results[-1] = recovery
                continue
            return

    @staticmethod
    def _find_last_pick_cycle(actions: list) -> Optional[int]:
        for j in range(len(actions) - 1, -1, -1):
            if actions[j].get("type") == "move_to_keypoint":
                for k in range(j, -1, -1):
                    if actions[k].get("type") == "move_to_keypoint":
                        has_grasp = any(
                            actions[m].get("type") == "grasp"
                            for m in range(k, min(k + 3, len(actions)))
                        )
                        if has_grasp:
                            return k
                break
        return None
