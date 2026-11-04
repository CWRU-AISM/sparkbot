"""Behavior-tree node handlers (sequence, selector, retry), recovery budget and leaf dispatch."""

import logging
import time
from typing import List

import numpy as np

from spark_core import bt_grammar
from spark_real.control import primitive_timeouts
from spark_real.control.executor_types import (
    BUILTIN_PRIMITIVES,
    AbortRequested,
    ExecutionResult,
)
from spark_real.skills import registry as skill_registry

logger = logging.getLogger(__name__)


class DispatchMixin:
    """Behavior-tree node handlers (sequence, selector, retry), recovery budget and leaf dispatch."""

    def _flatten_tree(self, node: dict) -> List[dict]:
        """Flatten a score tree into the action list ``_run_actions`` walks.

        Sequences collapse. A ``selector``/``fallback`` or ``retry`` node does
        NOT: it becomes ONE opaque marker action carrying its already-flattened
        branches, otherwise "try A, else B" would flatten to "do A, then do B".
        """
        node_type = node.get("type", "")

        if node_type in bt_grammar.SELECTOR_TYPES:
            branches = []
            for child in node.get("children") or []:
                branch = self._flatten_tree(child)
                if branch:
                    branches.append(branch)
            if len(branches) > bt_grammar.MAX_SELECTOR_BRANCHES:
                logger.warning(
                    "selector has %d branches; running the first %d",
                    len(branches),
                    bt_grammar.MAX_SELECTOR_BRANCHES,
                )
                branches = branches[: bt_grammar.MAX_SELECTOR_BRANCHES]
            return [
                {
                    "type": "selector",
                    "_bt_selector": True,
                    "branches": branches,
                    "params": dict(node.get("params") or {}),
                }
            ]

        if node_type in bt_grammar.RETRY_TYPES:
            body = []
            for child in node.get("children") or []:
                body.extend(self._flatten_tree(child))
            attempts = bt_grammar.clamp_retry_attempts(
                (node.get("params") or {}).get("max_attempts")
            )
            return [
                {
                    "type": "retry",
                    "_bt_retry": True,
                    "branch": body,
                    "max_attempts": attempts,
                    "params": dict(node.get("params") or {}),
                }
            ]

        if node_type == "sequence" or (
            "children" in node and node_type not in self._LEAF_TYPES
        ):
            actions = []
            for child in node.get("children", []):
                actions.extend(self._flatten_tree(child))
            return actions
        return [node]

    def _leaf_actions(self, actions: List[dict]) -> List[dict]:
        """Expand control-flow markers into the leaves they can run.

        Downstream consumers (post-task verification, the pick-cycle replay)
        reason about "which labels did this plan touch", not about control
        flow.
        """
        out: List[dict] = []
        for action in actions:
            if action.get("_bt_selector"):
                for branch in action.get("branches") or []:
                    out.extend(self._leaf_actions(branch))
            elif action.get("_bt_retry"):
                out.extend(self._leaf_actions(action.get("branch") or []))
            else:
                out.append(action)
        return out

    def _spend_recovery_budget(self, what: str) -> bool:
        """Charge one recovery re-attempt against the per-score budget.

        Returns False once the budget is gone, which makes the enclosing node
        fail cleanly. The per-node caps alone are not enough: retry(3) nested
        inside fallback(3) multiplies to nine attempts, and nine blind
        re-approaches on a real arm is not recovery, it is thrashing.
        """
        used = getattr(self, "_recovery_attempts_used", 0)
        if used >= bt_grammar.MAX_RECOVERY_ATTEMPTS_PER_SCORE:
            logger.warning(
                "recovery budget exhausted (%d/%d); refusing %s",
                used,
                bt_grammar.MAX_RECOVERY_ATTEMPTS_PER_SCORE,
                what,
            )
            return False
        self._recovery_attempts_used = used + 1
        logger.info(
            "recovery attempt %d/%d: %s",
            self._recovery_attempts_used,
            bt_grammar.MAX_RECOVERY_ATTEMPTS_PER_SCORE,
            what,
        )
        return True

    @staticmethod
    def _first_label_of(actions: List[dict]) -> str:
        for action in actions:
            label = (action.get("params") or {}).get("keypoint_label")
            if label:
                return str(label)
        return ""

    def _reset_before_recovery(self, next_actions: List[dict]) -> None:
        """Put the arm in a known state before a recovery branch runs.

        Mirrors the reset the flat path already does on a failed action: drop
        nothing that is still gripped, clear the workspace vertically, return
        to home, and re-ground the label the next branch is about to reach for.
        Every step is best-effort: a reset that raises must not mask the
        failure that triggered it.
        """
        self._check_abort()

        still_holding = False
        if self._holding:
            try:
                still_holding = self._verify_grasp()
            except Exception as exc:
                logger.warning("Recovery reset: grasp verify failed: %s", exc)
        if not still_holding:
            try:
                self.robot.open_gripper()
                self._abort_sleep(0.3)
            except AbortRequested:
                raise
            except Exception as exc:
                logger.warning("Recovery reset: open_gripper failed: %s", exc)
            self._holding = False

        try:
            cur = self._get_current_position()
            if cur[2] < 0.45:
                cur[2] = min(cur[2] + 0.10, 0.45)
                # A reset LIFT is a translation; re-commanding GRASP_ORIENTATION
                # here would unwind the task's yaw at v=2.67 rad/s.
                self._move_to(cur, self.current_orientation())
        except AbortRequested:
            raise
        except Exception as exc:
            logger.warning("Recovery reset: lift failed: %s", exc)

        go_home = getattr(self.robot, "go_home", None)
        if callable(go_home):
            try:
                go_home()
            except AbortRequested:
                raise
            except Exception as exc:
                logger.warning("Recovery reset: go_home failed: %s", exc)

        if self._pipeline is not None:
            label = self._first_label_of(self._leaf_actions(next_actions))
            if label and label not in self._placed_labels:
                try:
                    self._redetect_single(label)
                except AbortRequested:
                    raise
                except Exception as exc:
                    logger.warning("Recovery reset: re-detect failed: %s", exc)

    def _skip_failed_cycle(self, actions: List[dict], index: int) -> int:
        """Index of the next action worth running after a control-flow node
        exhausted its recovery.

        Once the acquire has genuinely failed, the transport and the place
        that follow it would move an empty gripper to a container and open it.
        Skip to the next action that can start a NEW pick cycle (one whose
        leaves contain a grasp), which for a single-object plan is the end,
        and for a multi-object plan is the next object.
        """
        j = index + 1
        while j < len(actions):
            leaves = self._leaf_actions([actions[j]])
            if any(a.get("type") in ("grasp", "grasp_se3") for a in leaves):
                return j
            self._results.append(
                ExecutionResult(
                    action_type=actions[j].get("type", "unknown"),
                    success=False,
                    message="Skipped (recovery for the preceding step failed)",
                )
            )
            j += 1
        return j

    def _run_branch(self, actions: List[dict]) -> bool:
        """Run one branch under STRICT sequence semantics.

        Strict means: stop at the first failure and report it. The enclosing
        selector's next branch, or the enclosing retry's next attempt, IS the
        reaction to the failure.
        """
        for k, action in enumerate(actions):
            self._check_abort()
            # Same release look-ahead as _run_actions: a recovery re-run must
            # steer its approach yaw by the same upcoming dip the first try did.
            self._pending_release_tilt = self._peek_release_tilt(actions[k:])
            self._pending_place_label = self._peek_place_label(actions[k + 1:])
            if action.get("_bt_selector"):
                result = self._run_selector(action)
            elif action.get("_bt_retry"):
                result = self._run_retry(action)
            else:
                result = self._run_leaf(action)
            self._results.append(result)
            self._last_action_failed = not bool(result.success)
            if not result.success:
                return False
        return True

    def _run_leaf(self, action: dict) -> ExecutionResult:
        action_type = action.get("type", action.get("name", "unknown"))
        params = action.get("params", {}) or {}
        logger.info("[branch] %s %s", action_type, params)
        try:
            self._recorder.set_action_label(action_type)
            self._recorder.mark_transition(action_type)
            self._recorder.record()
        except Exception as exc:
            logger.debug("Recorder skipped for %s: %s", action_type, exc)

        if action_type in ("move_to_keypoint", "grasp_se3") and not self._holding:
            self._last_pick_label = params.get("keypoint_label", "")

        result = self._dispatch_action(action_type, params)
        if result.success and action_type == "release" and self._last_pick_label:
            self._placed_labels.add(self._last_pick_label)
            self._last_pick_label = ""
        return result

    def _run_selector(self, marker: dict) -> ExecutionResult:
        branches = marker.get("branches") or []
        total = len(branches)
        if total == 0:
            return ExecutionResult(
                action_type="selector", success=True, message="empty selector"
            )
        for idx, branch in enumerate(branches):
            if idx > 0:
                if not self._spend_recovery_budget(
                    f"selector branch {idx + 1}/{total}"
                ):
                    return ExecutionResult(
                        action_type="selector",
                        success=False,
                        message=(
                            f"All {total} selector branches failed "
                            f"(recovery budget exhausted after {idx})"
                        ),
                    )
                self._reset_before_recovery(branch)
            if self._run_branch(branch):
                return ExecutionResult(
                    action_type="selector",
                    success=True,
                    message=f"branch {idx + 1}/{total} succeeded",
                )
            logger.warning("selector branch %d/%d failed", idx + 1, total)
        return ExecutionResult(
            action_type="selector",
            success=False,
            message=f"All {total} selector branches failed",
        )

    def _run_retry(self, marker: dict) -> ExecutionResult:
        branch = marker.get("branch") or []
        if not branch:
            return ExecutionResult(
                action_type="retry", success=True, message="empty retry"
            )
        attempts = bt_grammar.clamp_retry_attempts(marker.get("max_attempts"))
        for k in range(attempts):
            if k > 0:
                if not self._spend_recovery_budget(f"retry attempt {k + 1}/{attempts}"):
                    return ExecutionResult(
                        action_type="retry",
                        success=False,
                        message=(
                            f"retry gave up after {k}/{attempts} attempts "
                            "(recovery budget exhausted)"
                        ),
                    )
                self._reset_before_recovery(branch)
            if self._run_branch(branch):
                return ExecutionResult(
                    action_type="retry",
                    success=True,
                    message=f"succeeded on attempt {k + 1}/{attempts}",
                )
            logger.warning("retry attempt %d/%d failed", k + 1, attempts)
        return ExecutionResult(
            action_type="retry",
            success=False,
            message=f"all {attempts} retry attempts failed",
        )

    def _normalize_params(self, params: dict) -> dict:
        if not params:
            return params
        if not any(a in params for a in self._PARAM_ALIASES):
            return params
        norm = dict(params)
        for alias, canonical in self._PARAM_ALIASES.items():
            if alias in norm and canonical not in norm:
                norm[canonical] = norm[alias]
        return norm

    def _dispatch_action(self, action_type: str, params: dict) -> ExecutionResult:
        """Normalize grammar aliases, then run under the primitive's budget.

        On a budget overrun the watchdog aborts the motion and returns a
        failed result so the caller's recovery path handles it like any
        failed post-condition.
        """
        params = self._normalize_params(params)
        budget = primitive_timeouts.timeout_for(
            getattr(self, "_primitive_timeouts", None) or {}, action_type
        )
        return primitive_timeouts.run_with_timeout(
            self,
            action_type,
            budget,
            lambda: self._dispatch_action_body(action_type, params),
        )

    def _reconcile_after_protective_stop(self, action_type: str, params: dict):
        """One world-model check at the first action boundary after an
        auto-cleared protective stop.

        The unlock path (ur10e_driver._check_safety_and_recover) restores
        CONTROL, but a protective stop means a COLLISION happened: the held
        object may have been knocked from the jaws (ask the jaws) and the
        involved detection may have moved (re-bind).
        """
        ts = float(getattr(self.robot, "last_protective_recovery_ts", 0.0) or 0.0)
        if ts <= self._last_safety_reconcile_ts:
            return
        self._last_safety_reconcile_ts = ts
        logger.warning(
            "[post-stop] reconciling before '%s': the last protective stop "
            "was auto-cleared, so the collision's effects are checked now",
            action_type,
        )
        # 1. Still holding? A drop here fails the CURRENT action so recovery
        #    re-grasps instead of transporting empty jaws to the target.
        if getattr(self, "_holding", False):
            try:
                if not self._grip_intact():
                    logger.warning(
                        "[post-stop] the jaws are EMPTY; the collision cost "
                        "us the object -- marking not-holding so recovery "
                        "re-grasps"
                    )
                    self._holding = False
            except Exception:  # noqa: BLE001 - unknown grip stays assumed-held
                pass
        # 2. Re-bind the label this action is about, if it names one. The
        #    collision may have moved it (or been CAUSED by it moving).
        label = (params or {}).get("keypoint_label") or (params or {}).get(
            "container_label"
        )
        if label:
            try:
                from spark_real.control.execution_recovery import (
                    rebind_before_approach,
                )

                rebind_before_approach(self, label)
            except Exception:  # noqa: BLE001
                pass

    def _dispatch_action_body(
        self, action_type: str, params: dict
    ) -> ExecutionResult:
        t0 = time.time()
        try:
            self._reconcile_after_protective_stop(action_type, params)
            builtin = set(BUILTIN_PRIMITIVES) | {"action"}
            if action_type not in builtin and skill_registry.get(action_type) is not None:
                return skill_registry.dispatch(action_type, self, params)

            if action_type in ("move_to_keypoint", "action"):
                name = params.get("name", action_type)
                dispatch_map = {
                    "move_to_keypoint": self._move_to_keypoint,
                    "grasp": self._grasp,
                    "release": self._release,
                    "move_relative": self._move_relative,
                }
                handler = dispatch_map.get(name)
                if handler:
                    return handler(params, t0)
                if action_type == "action" and name != action_type:
                    return self._dispatch_action_body(name, params)
                return self._move_to_keypoint(params, t0)
            elif action_type == "grasp":
                return self._grasp(params, t0)
            elif action_type == "release":
                return self._release(params, t0)
            elif action_type == "move_relative":
                return self._move_relative(params, t0)
            elif action_type == "wait":
                return self._wait(params, t0)
            else:
                return ExecutionResult(
                    action_type=action_type,
                    success=False,
                    message=f"Unknown action type: {action_type}",
                    duration=time.time() - t0,
                )
        except AbortRequested:
            raise
        except Exception as e:
            return ExecutionResult(
                action_type=action_type,
                success=False,
                message=str(e),
                duration=time.time() - t0,
            )

    def _move_relative(self, params: dict, t0: float) -> ExecutionResult:
        self._check_abort()
        dx = params.get("dx", 0.0)
        dy = params.get("dy", 0.0)
        dz = params.get("dz", 0.0)
        current = self._get_current_position()
        self._move_to(current + np.array([dx, dy, dz]), self.current_orientation())
        return ExecutionResult(
            action_type="move_relative",
            success=True,
            message=f"Moved by ({dx}, {dy}, {dz})",
            duration=time.time() - t0,
        )

    def _wait(self, params: dict, t0: float) -> ExecutionResult:
        # duration defaults to 0.5s when the plan omits it.
        duration = params.get("duration", 0.5)
        self._abort_sleep(duration)
        return ExecutionResult(
            action_type="wait",
            success=True,
            message=f"Waited {duration}s",
            duration=time.time() - t0,
        )
