"""
Simple primitives: grasp, release, move_relative, wait.
"""
from __future__ import annotations

import numpy as np

from spark_bench.libero_pro.motion import step_env
from spark_bench.libero_pro.telemetry import GraspOutcome, classify_grasp_outcome

__all__ = [
    'handle_grasp',
    'handle_release',
    'handle_move_relative',
    'handle_wait',
]


def handle_grasp(executor, params: dict) -> None:
    # Pre-close informative moment (cfg.event_captures); with cfg.scene_diff a small target move
    # re-approaches, a large move or missing target aborts into recovery without closing on air.
    executor.pre_close_capture()
    if executor.abort is not None:
        return
    # Thin-feature picks (plate/dish rim wall) need an OSC-regulated close: the plain close lets the EE
    # drift toward the controller's neutral and slides the jaws off the 2.5 mm rim wall. Bulk objects
    # keep the compliant close (bowl rim pinches rely on the drift). The regulated close produces no
    # aperture trace, so no grasp vote.
    lbl = (executor.last_pick_label or '').lower()
    thin_feature = ('plate' in lbl or 'dish' in lbl) and 'bowl' not in lbl
    cls = None
    if thin_feature:
        executor._gripper_hold(False, steps=300)
    else:
        # Close with per-step aperture telemetry, classified as the first, instant post-grasp vote.
        cls = classify_grasp_outcome(executor.close_gripper_with_telemetry(steps=300))
        executor.record_grasp_outcome(cls)
    executor.holding = True
    if 'bowl' in lbl:
        executor.env._grip_offset = np.array([0.0, 0.025, 0.0])
    elif ('plate' in lbl or 'dish' in lbl) and executor.last_pick_det_pos is not None:
        executor.env._grip_offset = np.array([0.0, 0.04, 0.0])
    else:
        executor.env._grip_offset = np.zeros(3)
    # The object left the tabletop: cached masks at its old XY are stale.
    executor._last_perception_t = 0.0
    # Telemetry EMPTY_CLOSE -> camera second vote -> single local retry (cfg.event_captures).
    if cls is not None and cls.outcome == GraspOutcome.EMPTY_CLOSE:
        executor.post_grasp_second_vote()


def handle_release(executor, params: dict) -> None:
    # Settle with the EE held before opening, so transport wobble damps out (opening at non-zero EE
    # velocity flicks a bowl off a small receiver), then open with OSC still regulating the pose.
    executor._gripper_hold(False, steps=50)
    executor._gripper_hold(True, steps=40)
    executor.holding = False
    ee = executor._ee()
    up = ee.copy()
    up[2] += 0.03
    executor._move_to(up, True, steps=60)
    # Post-release informative moment (cfg.event_captures): when it ran, det_map is already fresh;
    # otherwise invalidate perception so the next keypoint lookup re-detects.
    if not executor.post_release_capture():
        executor._last_perception_t = 0.0


def handle_move_relative(executor, params: dict) -> None:
    ee = executor._ee()
    delta = np.array([params.get('dx', 0), params.get('dy', 0), params.get('dz', 0)])
    executor._move_to(ee + delta, not executor.holding, steps=200)
    # Re-perceive after a translation of 10 cm or more while holding (occlusions change and the held
    # object's mask moves); small adjustments skip the SAM3 call.
    if float(np.linalg.norm(delta)) >= 0.10 and executor.holding:
        executor._last_perception_t = 0.0


def handle_wait(executor, params: dict) -> None:
    try:
        duration = float(params.get('duration', 0.5))
    except Exception:
        duration = 0.5
    cmd = 1.0 if executor.holding else -1.0
    for _ in range(max(0, int(duration * 50))):
        a = np.zeros(7)
        a[6] = cmd
        if step_env(executor.env, a):
            break
