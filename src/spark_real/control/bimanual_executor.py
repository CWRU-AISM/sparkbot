"""
Bimanual score executor.

Runs SPARK behavior-tree scores on the bimanual rig. On top of the
single-arm grammar it adds:

* ``parallel``: children run on their own threads; the join is a barrier.
* ``sync_barrier``: a named rendezvous inside a ``parallel`` block.

Leaves:

* The typed bimanual primitives (skills/bimanual_primitives.py:
  pick_with_arm, place_with_arm, move_to_keypoint_arm, grasp_arm,
  release_arm, bimanual_grasp_points, bimanual_lift_together,
  bimanual_arc_drape, handoff, bimanual_lift, bimanual_place, and the
  bimanual_shirt_fold chain in skills/bimanual_fold.py) run on the
  rig's motion path (control/bimanual_motion.py), each with its
  post-condition. ``hold_in_place`` waits.
* Any other arm-tagged single-arm primitive goes to that arm's
  ScoreExecutor, whose robot is the arm driver with its jaw mounted.

Frames: detections arrive in the right arm's base frame (the external
camera's calibration). T_right_to_left from calibration_bimanual.json maps
them into the left arm's frame, for the primitives and for the left arm's
executor; without that calibration execute_score refuses. Every motion
target passes the inter-arm distance gate.

There is no bimanual recovery module. When one branch of a ``parallel``
block fails, the others run to completion (or to a broken barrier) and the
block reports failure once every branch has joined.
"""

from __future__ import annotations

import copy
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from spark_real.control.bimanual_motion import BimanualMotion
from spark_real.control.executor_types import AbortRequested, ExecutionResult
from spark_real.control.score_executor import ScoreExecutor
from spark_real.skills import bimanual_fold, bimanual_primitives

logger = logging.getLogger(__name__)

ARMS = ("left", "right")

# Leaves that need an arm tag; a missing tag defaults to "right" with a warning.
_PER_ARM_PRIMITIVES = {
    "move_to_keypoint", "grasp", "release", "move_relative", "wait", "wiggle", "push_object",
    "open_drawer", "push", "pull", "drag", "stack", "tool_grip_pose",
}
_TYPED = {name: getattr(bimanual_primitives, name) for name in bimanual_primitives.BIMANUAL_PRIMITIVES}
_TYPED["bimanual_shirt_fold"] = bimanual_fold.bimanual_shirt_fold
# Listed in the frozen planner prompt but not implemented in this release.
_UNSHIPPED = {"bimanual_pants_fold", "bimanual_handover_sponge"}


def entry_to_arm_frame(entry: dict, T: np.ndarray) -> dict:
    """A detection entry with its base-frame geometry mapped through T (4x4)."""
    out = dict(entry)
    pos = entry.get("position_3d")
    if pos is not None:
        out["position_3d"] = (T @ np.append(np.asarray(pos, dtype=float)[:3], 1.0))[:3].tolist()
    yaw = math.atan2(T[1, 0], T[0, 0])
    if entry.get("world_major_axis_rad") is not None:
        out["world_major_axis_rad"] = float(entry["world_major_axis_rad"]) + yaw
    if entry.get("slots"):
        out["slots"] = [
            {**s, "world_xyz": (T @ np.append(np.asarray(s["world_xyz"], dtype=float)[:3], 1.0))[:3].tolist()}
            if s.get("world_xyz") is not None else dict(s)
            for s in entry["slots"]
        ]
    return out


@dataclass
class BimanualState:
    """Per-arm state carried between leaves."""

    holding: Dict[str, bool] = field(default_factory=lambda: {"left": False, "right": False})
    last_pick_label: Dict[str, str] = field(default_factory=lambda: {"left": "", "right": ""})
    last_keypoint: Dict[str, Optional[str]] = field(default_factory=lambda: {"left": None, "right": None})
    # Last commanded jaw-tip target and jaw yaw per arm (two-arm phases chain on them).
    last_target: Dict[str, Optional[np.ndarray]] = field(default_factory=lambda: {"left": None, "right": None})
    last_yaw: Dict[str, Optional[float]] = field(default_factory=lambda: {"left": None, "right": None})


class BimanualScoreExecutor:
    """Score executor that dispatches per-arm and parallel BT nodes."""

    # Mirrored from the per-arm executor that ran verification; None until a
    # score has been verified (success_verifier.reset_run_state resets it).
    verify_outcome = None

    def __init__(
        self,
        safe,
        detection_map: Optional[Dict] = None,
        velocity: float = 0.20,
        pipeline=None,
        strict_placement_verify: bool = False,
        T_right_to_left: Optional[np.ndarray] = None,
        motion_cfg: Optional[Dict] = None,
    ):
        self.safe = safe
        self.driver = safe.driver
        self.velocity = velocity
        self._pipeline = pipeline
        self.state = BimanualState()
        self._results: List[ExecutionResult] = []
        self._abort = False
        self._running = False
        self._barriers: Dict[str, threading.Barrier] = {}
        self._branch = threading.local()  # .arm: last arm tag in this branch thread
        # Arm of the most recent per-arm leaf; verification runs on it.
        self._last_arm = "right"
        self.motion = BimanualMotion(
            self.driver, None, motion_cfg, clearance=getattr(safe, "is_motion_safe", None)
        )
        self.T_right_to_left = T_right_to_left
        self._arm_executors: Dict[str, ScoreExecutor] = {
            arm: ScoreExecutor(
                safe.for_arm(arm),
                velocity=velocity,
                pipeline=pipeline,
                strict_placement_verify=strict_placement_verify,
            )
            for arm in ARMS
        }
        for arm, ex in self._arm_executors.items():
            ex.arm = arm
            # The opt-in pick-cycle replay is single-arm only: bimanual leaves
            # carry params['arm'] and the other arm's steps.
            ex._replay_last_pick_cycle = self._skip_replay
        self.update_detections(detection_map or {})

    @property
    def T_right_to_left(self) -> Optional[np.ndarray]:
        return self.motion.T_right_to_left

    @T_right_to_left.setter
    def T_right_to_left(self, T) -> None:
        self.motion.T_right_to_left = None if T is None else np.asarray(T, dtype=float)

    def update_detections(self, detection_map: Dict) -> None:
        """Right arm: the map as given. Left arm: mapped by T_right_to_left."""
        self.detection_map = detection_map
        self._arm_executors["right"].detection_map = detection_map
        left = detection_map
        if self.T_right_to_left is not None and detection_map:
            left = copy.copy(detection_map)
            for label, entry in detection_map.items():
                left[label] = entry_to_arm_frame(entry, self.T_right_to_left)
        self._arm_executors["left"].detection_map = left

    # Per-task fields the pipeline sets on its executor; both arms need them.
    def _set_on_arms(name):
        def setter(self, value):
            for ex in self._arm_executors.values():
                setattr(ex, name, value)

        return property(lambda self: getattr(self._arm_executors["right"], name, None), setter)

    _destination_labels = _set_on_arms("_destination_labels")
    _task_verify_block = _set_on_arms("_task_verify_block")
    del _set_on_arms

    def request_abort(self) -> None:
        self._abort = True
        for ex in self._arm_executors.values():
            # note_abort_requested advances the per-arm abort epoch, so the
            # arm's next execute does not wipe the request.
            ex.note_abort_requested()

    def note_abort_requested(self) -> None:
        self.request_abort()

    def reset_task_state(self) -> None:
        """Task boundary: re-arm both arms' abort epochs, drop per-task state."""
        self.state = BimanualState()
        for ex in self._arm_executors.values():
            ex.reset_task_state()

    @staticmethod
    def _skip_replay(score: Dict, actions: List[Dict], max_attempts: int) -> None:
        logger.warning("verification.retry_on_fail: replay is single-arm only; not replaying")

    def abort(self) -> bool:
        """Request a stop and brake both arms. True only if both braked."""
        self._abort = True
        # A list, not a generator: all() must not short-circuit past an arm.
        return all([ex.abort() for ex in self._arm_executors.values()])

    @property
    def is_running(self) -> bool:
        return self._running

    def _fail(self, action: str, message: str, t0: float) -> ExecutionResult:
        res = ExecutionResult(action_type=action, success=False, message=message, duration=time.time() - t0)
        self._results.append(res)
        return res

    def execute_score(self, score: Dict) -> List[ExecutionResult]:
        """Run a bimanual score; returns the accumulated results."""
        self._results = []
        self._abort = False
        t0 = time.time()
        tree = score.get("tree")
        if tree is None:
            self._fail("<root>", "score missing 'tree'", t0)
            return self._results
        if self.T_right_to_left is None:
            self._fail(
                "<root>",
                "no bimanual calibration loaded (calibration_bimanual.json): detections "
                "are not in either arm's frame; run spark_real.calibration.bimanual_anchor_tap",
                t0,
            )
            return self._results
        self._running = True
        try:
            self._run_node(tree)
            if self._pipeline is not None and not self._abort:
                self._verify(score, tree)
        finally:
            self._running = False
        return self._results

    def _verify(self, score: Dict, tree: Dict) -> None:
        """Post-task verification on the arm that dispatched the last leaf."""
        ex = self._arm_executors[self._last_arm]
        ex._results = self._results  # the verify row lands in this run's results
        leaves = ex._leaf_actions(ex._flatten_tree(tree))
        try:
            ex._run_post_task_verification(score, leaves)
        except AbortRequested:
            logger.warning("Post-task verification aborted by user")
        self.verify_outcome = ex.verify_outcome

    def _run_node(self, node: Dict) -> bool:
        if self._abort:
            return False
        ntype = (node or {}).get("type", "sequence")
        children = node.get("children") or []
        if ntype in ("sequence", "selector"):
            # Sequence: all must succeed. Selector: first success wins.
            for child in children:
                ok = self._run_node(child)
                if not ok and ntype == "sequence":
                    return False
                if ok and ntype == "selector":
                    return True
            return ntype == "sequence"
        if ntype == "parallel":
            return self._run_parallel(children)
        return self._run_leaf(node)

    def _run_parallel(self, children: List[Dict]) -> bool:
        if not children:
            return True
        # Count how many branches reach each named barrier (params.name,
        # else node.name, else "<barrier-depth>").
        counts: Dict[str, int] = {}

        def _count(node: Dict, depth: int) -> None:
            if (node or {}).get("type") == "sync_barrier":
                name = (node.get("params") or {}).get("name") or node.get("name") or f"<barrier-{depth}>"
                counts[name] = counts.get(name, 0) + 1
                return
            for ch in node.get("children", []) or []:
                _count(ch, depth + 1)

        for branch in children:
            _count(branch, 0)
        for name, count in counts.items():
            if count != len(children):
                logger.warning(
                    "parallel block: barrier %r reached by %d of %d branches", name, count, len(children)
                )
            self._barriers[name] = threading.Barrier(min(count, len(children)) or 1)

        results: Dict[int, bool] = {}

        def _branch(idx: int, sub: Dict) -> None:
            self._branch.arm = None
            results[idx] = self._run_node(sub)

        threads = [
            threading.Thread(target=_branch, args=(i, c), name=f"bt-branch-{i}", daemon=True)
            for i, c in enumerate(children)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self._barriers.clear()
        return all(results.values())

    def _barrier_wait(self, name: str, timeout: float = 30.0) -> bool:
        b = self._barriers.get(name)
        if b is None:
            logger.warning("sync_barrier %r not found; continuing", name)
            return True
        try:
            b.wait(timeout=timeout)
            return True
        except threading.BrokenBarrierError:
            logger.error("sync_barrier %r broken (timeout or abort)", name)
            return False

    def _inter_arm_block(self, arm: str, ex, params: Dict) -> str:
        """Why a keypoint target is too close to the other arm, or ""."""
        det = ex.detection_map.get(params.get("keypoint_label") or "") if params.get("keypoint_label") else None
        gate = getattr(self.safe, "is_motion_safe", None)
        if det is None or det.get("position_3d") is None or gate is None:
            return ""
        target = np.asarray(det["position_3d"], dtype=float)[:3] + [
            float(params.get(k, 0.0)) for k in ("offset_x", "offset_y", "offset_z")
        ]
        if gate(arm, target):
            return ""
        return f"target {np.round(target, 3).tolist()} is inside the other arm's keep-out distance"

    def _run_leaf(self, node: Dict) -> bool:
        t0 = time.time()
        ntype = node.get("type", "")
        params = dict(node.get("params") or {})

        if ntype == "sync_barrier":
            ok = self._barrier_wait(params.get("name", node.get("name", "<barrier-0>")))
            self._results.append(ExecutionResult(action_type=ntype, success=ok, duration=time.time() - t0))
            return ok
        if ntype in _UNSHIPPED:
            return self._fail(ntype, f"{ntype!r} is not in this release; compose bimanual_grasp_points / "
                              "bimanual_lift_together / bimanual_arc_drape", t0).success
        if ntype in _TYPED:
            # A leaf without an arm tag inherits the arm of the last tagged
            # leaf in its branch (the paper's plans leave move_relative untagged).
            if "arm" in params:
                self._branch.arm = params["arm"]
            elif getattr(self._branch, "arm", None) and ntype in ("pick_with_arm", "place_with_arm",
                                                                  "move_to_keypoint_arm", "grasp_arm", "release_arm"):
                params["arm"] = self._branch.arm
            try:
                res = _TYPED[ntype](self, params)
            except (KeyError, ValueError, RuntimeError) as exc:
                res = ExecutionResult(ntype, False, str(exc), time.time() - t0)
            if "arm" in params:
                self._last_arm = params["arm"]
            res.action_type = f"{ntype}[{params['arm']}]" if params.get("arm") in ARMS else ntype
            self._results.append(res)
            return res.success
        if ntype == "hold_in_place":
            dwell = float(params.get("dwell", 3.0))
            time.sleep(dwell)
            self._results.append(ExecutionResult(ntype, True, f"held {dwell:.1f}s", time.time() - t0))
            return True

        arm = params.pop("arm", None) or getattr(self._branch, "arm", None)
        if arm is None and ntype in _PER_ARM_PRIMITIVES:
            logger.warning("primitive %r in bimanual mode missing 'arm'; defaulting to 'right'", ntype)
            arm = "right"
        if arm not in ARMS:
            return self._fail(ntype, f"invalid arm {arm!r} for primitive {ntype!r}", t0).success
        self._branch.arm = arm
        if ntype == "move_relative" and self.state.last_target.get(arm) is not None:
            # After a pick or move on the rig motion path, stay on it.
            goal = np.asarray(self.state.last_target[arm]) + [float(params.get(k, 0.0)) for k in ("dx", "dy", "dz")]
            ok = self.motion.move(arm, goal)
            if ok:
                self.state.last_target[arm] = goal
            self._results.append(ExecutionResult(f"move_relative[{arm}]", ok, "", time.time() - t0))
            return ok

        ex = self._arm_executors[arm]
        blocked = self._inter_arm_block(arm, ex, params)
        if blocked:
            return self._fail(f"{ntype}[{arm}]", blocked, t0).success
        self._last_arm = arm
        # Per-arm state lives here so it survives across leaves and branches.
        ex._holding = self.state.holding[arm]
        ex._last_pick_label = self.state.last_pick_label[arm]
        ex._last_keypoint_label = self.state.last_keypoint[arm]
        result = ex._dispatch_action(ntype, params)
        self.state.holding[arm] = ex._holding
        self.state.last_pick_label[arm] = ex._last_pick_label
        self.state.last_keypoint[arm] = ex._last_keypoint_label

        result.action_type = f"{ntype}[{arm}]"
        self._results.append(result)
        return result.success
