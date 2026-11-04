"""
LIBERO-PRO behavior-tree executor.

Owns the per-trial mutable state (``holding``, last pick label / position, label bindings) and
dispatches each BT leaf to a handler in ``libero_pro.primitives``. Handlers are free functions
``(executor, params)``. Binding and scene-diff logic lives in ``binding.BindingMixin``, the grasp close
and informative-moment captures in ``grasp_close.GraspCloseMixin``.
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

import mujoco
import numpy as np

from spark_bench.libero_pro.binding import BindingMixin, action_uses_keypoint
from spark_bench.libero_pro.grasp_close import GraspCloseMixin
from spark_bench.libero_pro.ik import compute_ik_6dof, get_q
from spark_bench.libero_pro.motion import (
    find_actuator_ids,
    find_ee_site,
    find_joint_ids,
    find_robot_base_world,
    gripper_action,
    gripper_action_hold,
    joint_move,
    move_to,
    move_to_pose,
)
from spark_bench.libero_pro.perception import _drawer_part_of_cabinet, redetect_agentview
from spark_bench.libero_pro.primitives import drawer as _drawer
from spark_bench.libero_pro.primitives import keypoint as _keypoint
from spark_bench.libero_pro.primitives import knob as _knob
from spark_bench.libero_pro.primitives import push as _push
from spark_bench.libero_pro.primitives import simple as _simple

__all__ = ['LiberoExecutor', 'execute_on_libero']


def _flatten_tree(tree: Any) -> list[dict]:
    """Depth-first flatten of a BT score into its leaf primitives.

    This open-loop executor has no per-subtree success signal: a ``retry`` inlines as one attempt
    (trial-level recovery supplies the re-attempts) and a ``fallback`` runs only its nominal branch.
    """
    actions: list[dict] = []

    def _walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        t = node.get('type')
        if t in ('sequence', 'selector', 'retry'):
            for c in node.get('children', []) or []:
                _walk(c)
        elif t == 'fallback':
            children = node.get('children', []) or []
            if children:
                _walk(children[0])
        else:
            actions.append(node)

    _walk(tree)
    return actions


def _handle_grasp_se3(executor, params: dict) -> None:
    """Compound pick: ``move_to_keypoint`` then ``grasp``."""
    _keypoint.handle(executor, {
        'keypoint_label': params.get('keypoint_label', ''),
        'offset_x': params.get('offset_x', 0.0),
        'offset_y': params.get('offset_y', 0.0),
        'offset_z': params.get('offset_z', 0.0),
    })
    _simple.handle_grasp(executor, {'force': params.get('force', 100)})


_HANDLERS = {
    'move_to_keypoint': _keypoint.handle,
    'grasp':            _simple.handle_grasp,
    'grasp_se3':        _handle_grasp_se3,
    'release':          _simple.handle_release,
    'move_relative':    _simple.handle_move_relative,
    'wait':             _simple.handle_wait,
    'turn_knob':        _knob.handle_turn_knob,
    'rotate':           _knob.handle_rotate,
    'screw':            _knob.handle_rotate,      # the planner emits 'screw' for knob turns
    'push_object':      _push.handle_push_object,
    'open_drawer':      _drawer.handle_open_drawer,
    'pull':             _drawer.handle_open_drawer,   # the recovery grammar's name for open_drawer
}


class LiberoExecutor(BindingMixin, GraspCloseMixin):
    """Executes a behaviour tree against a LIBERO ``OffScreenRenderEnv``."""

    def __init__(self, env, cfg, det_map: dict, *,
                 sam3=None, depth: Optional[np.ndarray] = None,
                 cam_pos: Optional[np.ndarray] = None,
                 cam_mat: Optional[np.ndarray] = None,
                 cam_fovy: Optional[float] = None,
                 cam_w: int = 640, cam_h: int = 480,
                 instruction: str = '',
                 prompts_for_refresh: Optional[list] = None,
                 trial_meta: Optional[dict] = None):
        self.env = env
        self.cfg = cfg
        self.det_map = det_map
        self.det_map_plan = dict(det_map)   # plan-time detections; refreshes may rebind a label later
        self.sam3 = sam3
        self.depth = depth
        self.cam_pos = cam_pos
        self.cam_mat = cam_mat
        self.cam_fovy = cam_fovy
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.instruction = instruction
        # Prompts the freshness gate re-detects with between primitives (no-op when empty).
        self.prompts_for_refresh = list(prompts_for_refresh or [])

        self.model = env.sim.model._model
        self.data = env.sim.data._data
        self.joint_ids = find_joint_ids(self.model)
        self.ee_site = find_ee_site(self.model)
        self.ndof = len(self.joint_ids)
        self.arm_actuator_ids, self.gripper_actuator_ids = find_actuator_ids(self.model, self.ndof)
        self.robot_base_world = find_robot_base_world(self.model, self.data)
        self._invalid = self.ee_site < 0 or self.ndof < 6

        # Per-trial state.
        self.holding: bool = False
        self.last_pick_label: Optional[str] = None
        self.last_pick_det_pos: Optional[np.ndarray] = None
        self.last_place_target: Optional[np.ndarray] = None
        # Set by open_drawer: interior of the pulled-out tray, for a follow-up place-into-drawer.
        self.last_drawer_tray_pos: Optional[np.ndarray] = None
        self.trial_meta: dict = trial_meta if trial_meta is not None else {}
        # Plan-time binding (label -> position_3d) for the scene diff; self-caused changes (retargets,
        # releases) re-bind their labels as they happen.
        self.plan_binding: dict[str, Optional[np.ndarray]] = {
            k: (np.asarray(d.position_3d, dtype=float).copy()
                if getattr(d, 'position_3d', None) is not None else None)
            for k, d in det_map.items()}
        # Set when the scene diff routes execution to the runner's recovery loop; run() stops then.
        self.abort: Optional[dict] = None
        self._finger_addrs: Optional[list[int]] = None
        self._sticky_flags: dict[str, str] = {}
        # Same-source displacement reference: the agentview-only centroid each label was bound at.
        # Re-detections are agentview-only while the plan binding may be wrist-fused, and the cameras
        # disagree by a systematic 1-2 cm, so displacement is measured agentview vs agentview.
        self.agent_ref: dict[str, np.ndarray] = {}
        for k, d in det_map.items():
            p = getattr(d, 'position_agentview', None)
            if p is None:
                p = getattr(d, 'position_3d', None)
            if p is not None:
                self.agent_ref[k] = np.asarray(p, dtype=float).copy()
        # Per-capture fresh observations (reset by each sticky merge). Verdict paths read these, never
        # the retained det_map entries.
        self._fresh_dets: dict[str, Any] = {}
        self._fresh_out_of_gate: dict[str, Any] = {}
        self._fresh_groups: dict[str, list] = {}
        # Wall-clock time of the last perception (det_map was just computed by the caller).
        self._last_perception_t: float = time.time()

    # Thin wrappers around motion/ik so handler bodies stay short.
    def _move_to(self, target, gripper_open, steps=200, **kw) -> float:
        return move_to(self.env, self.model, self.data, self.joint_ids, self.ee_site,
                       self.arm_actuator_ids, self.gripper_actuator_ids,
                       target, gripper_open, steps=steps, **kw)

    def _move_to_pose(self, target_pos, target_mat, gripper_open, steps: int = 200, **kw) -> tuple:
        """Position-and-orientation OSC move; see motion.move_to_pose."""
        return move_to_pose(self.env, self.model, self.data, self.joint_ids, self.ee_site,
                            self.arm_actuator_ids, self.gripper_actuator_ids,
                            target_pos, target_mat, gripper_open, steps=steps, **kw)

    def _joint_move(self, q_target, gripper_open, steps=200, kp=40.0, kd=12.0) -> None:
        joint_move(self.env, self.model, self.data, self.joint_ids, self.ee_site,
                   self.arm_actuator_ids, self.gripper_actuator_ids,
                   q_target, gripper_open, steps=steps, kp=kp, kd=kd)

    def _gripper(self, open_gripper, steps=60) -> None:
        gripper_action(self.env, open_gripper, steps=steps)

    def _gripper_hold(self, open_gripper, steps=60) -> None:
        """Open/close while OSC regulates the EE to its current pose (drawer pull, knob turn)."""
        gripper_action_hold(self.env, self.model, self.data, self.ee_site, open_gripper, steps=steps)

    def _ik6(self, target_pos, target_quat) -> np.ndarray:
        return compute_ik_6dof(self.model, self.data, self.joint_ids, self.ee_site, target_pos, target_quat)

    def _ee(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.ee_site].copy() if self.ee_site >= 0 else np.zeros(3)

    def _q_now(self) -> np.ndarray:
        return get_q(self.model, self.data, self.joint_ids)

    def _goal_satisfied(self) -> bool:
        """
        Best-effort BDDL goal check (False on any failure).

        SPARK_NO_ORACLE=1 (set by the protocol scripts) makes this always False: the pre-action
        short-circuit, the push stop-poll and the drawer pull poll then lose the simulator's success
        signal, which a real robot does not have. Episode-end scoring is unaffected.
        """
        if os.environ.get('SPARK_NO_ORACLE', '').strip() in ('1', 'true', 'yes'):
            return False
        try:
            return bool(self.env.check_success())
        except Exception:
            return False

    def run(self, score: dict) -> None:
        if self._invalid:
            return
        actions = _flatten_tree(score.get('tree', {}))
        if self.cfg.verbose:
            print(f"[Plan] {len(actions)} actions: {[a.get('type', '?') for a in actions]}")
            for a in actions:
                if a.get('params', {}).get('keypoint_label'):
                    print(f"{a['type']}: keypoint_label={a['params']['keypoint_label']}")
        for idx, act in enumerate(actions):
            # Goal already satisfied: an extra push after the plate lands knocks it back out.
            if self._goal_satisfied():
                if self.cfg.verbose:
                    print(f"[Exec] goal already satisfied before action {idx} ({act.get('type')}); "
                          f"short-circuit")
                break
            if not self.dispatch_action(idx, act) or self.abort is not None:
                # Scene diff aborted: let the runner's recovery loop take over with a fresh detect.
                if self.cfg.verbose and self.abort is not None:
                    print(f"[SceneDiff] abort at action {idx}: {self.abort}")
                break
        # Publish the end-of-execution binding so the recovery loop's scene diff compares against the
        # end state (otherwise the just-placed object reads as an external 'moved' forever).
        try:
            self.trial_meta['final_binding'] = {
                k: ([float(x) for x in v] if v is not None else None)
                for k, v in self.plan_binding.items()}
        except Exception:
            pass

    def dispatch_action(self, act_idx: int, act: dict) -> bool:
        """Dispatch one leaf. False when the pre-primitive scene diff aborted into recovery."""
        atype = act.get('type')
        params = act.get('params', {}) or {}
        self._maybe_refresh_perception(act)
        if not self._pre_primitive_gate(act_idx, act):
            return False
        if self.cfg.verbose:
            ee = self._ee()
            print(f"[Exec {act_idx}] {atype} params={params} "
                  f"ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) holding={self.holding}")
        handler = _HANDLERS.get(atype)
        if handler is not None:
            handler(self, params)
        return self.abort is None

    def _maybe_refresh_perception(self, act: dict) -> None:
        """Freshness gate: re-run SAM3 before a keypoint-referencing action when the detections are
        older than cfg.freshness_timeout_s (or, with cfg.scene_diff, older than 50 ms)."""
        timeout = getattr(self.cfg, 'freshness_timeout_s', None)
        scene_diff_on = bool(getattr(self.cfg, 'scene_diff', False))
        if timeout is None and not scene_diff_on:
            return
        if self.sam3 is None or not self.prompts_for_refresh or not action_uses_keypoint(act):
            return
        age = time.time() - self._last_perception_t
        if age < (0.05 if scene_diff_on else float(timeout)):
            return
        try:
            fresh = redetect_agentview(self.env, self.sam3, self.prompts_for_refresh, self.cfg,
                                       all_instances=self._sticky_enabled())
        except Exception as e:
            if self.cfg.verbose:
                print(f"[Freshness] redetect failed: {e}")
            return
        n_labels, n_updated = self._merge_fresh_detections(fresh)
        # The refresh map is built without the planning-time part-of rule: a fresh 'drawer' on the wine
        # rack overwrote the cabinet binding and the place went to the rack.
        _drawer_part_of_cabinet(self.det_map)
        self._last_perception_t = time.time()
        if self.cfg.verbose and n_labels:
            print(f"[Freshness] re-detected {n_labels} ({n_updated} updates, {n_labels-n_updated} new)")


def execute_on_libero(env, score: dict, det_map: dict, cfg, *,
                      sam3=None, depth=None,
                      cam_pos=None, cam_mat=None, cam_fovy=None,
                      cam_w: int = 640, cam_h: int = 480,
                      instruction: str = '',
                      prompts_for_refresh: Optional[list] = None,
                      trial_meta: Optional[dict] = None) -> None:
    """
    Build a ``LiberoExecutor`` and run ``score`` against ``env``.

    ``prompts_for_refresh`` is the prompt list of the initial SAM3 call, used by the freshness gate and
    the captures. ``trial_meta`` receives the per-trial records: grasp_outcomes, capture_events,
    scene_diffs, retarget_events, scene_abort, track_*.
    """
    executor = LiberoExecutor(env, cfg, det_map, sam3=sam3, depth=depth,
                              cam_pos=cam_pos, cam_mat=cam_mat, cam_fovy=cam_fovy,
                              cam_w=cam_w, cam_h=cam_h, instruction=instruction,
                              prompts_for_refresh=prompts_for_refresh, trial_meta=trial_meta)
    if executor._invalid:
        return
    executor.run(score)
