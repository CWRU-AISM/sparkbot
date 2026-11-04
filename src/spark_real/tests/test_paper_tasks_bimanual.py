"""
Bimanual paper tasks end to end on the fake rig (no hardware): the planner's
reply for the frozen bimanual prompt is parsed and validated, the bimanual
executor runs it (parallel branches on threads), every leaf lands on a typed
primitive or the rig motion path, and post-conditions decide success.

Usage:
    PYTHONPATH=src python -m pytest src/spark_real/tests/test_paper_tasks_bimanual.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_bimanual_primitives import _rig  # noqa: E402

from spark_core.planner import SPARKPlanner  # noqa: E402

SILVERWARE = """```yaml
task: place the fork and the knife into the tray, using both arms
tree:
  type: sequence
  children:
  - type: parallel
    children:
    - type: sequence
      children:
      - {type: pick_with_arm, params: {arm: left, keypoint_label: knife, target_width: 0.01, force: 40, prefer_angled: false}}
      - {type: move_relative, params: {dx: 0.0, dy: 0.0, dz: 0.2}}
    - type: sequence
      children:
      - {type: pick_with_arm, params: {arm: right, keypoint_label: fork, target_width: 0.01, force: 40, prefer_angled: false}}
      - {type: move_relative, params: {dx: 0.0, dy: 0.0, dz: 0.2}}
  - type: parallel
    children:
    - type: sequence
      children:
      - {type: move_to_keypoint_arm, params: {arm: left, keypoint_label: tray, offset_x: 0.0, offset_y: 0.05, offset_z: 0.08}}
      - {type: place_with_arm, params: {arm: left, tilt_angle: 0.0}}
    - type: sequence
      children:
      - {type: move_to_keypoint_arm, params: {arm: right, keypoint_label: tray, offset_x: 0.0, offset_y: -0.05, offset_z: 0.08}}
      - {type: place_with_arm, params: {arm: right, tilt_angle: 0.0}}
```"""

FOLD = """```yaml
task: fold the t-shirt
tree:
  type: sequence
  children:
    - type: bimanual_shirt_fold
      params: {instruction: "fold the t-shirt"}
```"""

SHIRT = {
    "left sleeve": {"position_3d": [0.45, 0.35, -0.02]},
    "right sleeve": {"position_3d": [0.45, -0.15, -0.02]},
    "shirt": {"position_3d": [0.42, 0.10, -0.02]},
    "hem": {"position_3d": [0.30, 0.10, -0.02]},
}


def _plan(reply, labels):
    planner = SPARKPlanner.__new__(SPARKPlanner)
    planner.robot_family = "bimanual_franka"
    planner.temperature = 0.0
    planner.generate = lambda contents, tag: reply
    score = planner.generate_score("task", keypoint_labels=labels)
    assert planner.validate_score(score, labels) == []
    return score


def test_silverware_sort_plan_runs_through_the_primitives():
    ex = _rig()
    results = ex.execute_score(_plan(SILVERWARE, list(ex.detection_map)))
    kinds = sorted(r.action_type for r in results)
    assert all(r.success for r in results), [(r.action_type, r.message) for r in results]
    assert kinds.count("pick_with_arm[left]") == 1 and kinds.count("pick_with_arm[right]") == 1
    assert "move_relative[left]" in kinds and "move_relative[right]" in kinds  # inherited arm tags
    assert kinds.count("place_with_arm[left]") == 1 and kinds.count("place_with_arm[right]") == 1
    # Both jaws closed on their utensil, then opened over the tray.
    assert all(j.packs == [0, 255, 0] for j in ex.driver.grippers.jaws.values())
    # The left arm reached the tray in its own frame: right-frame y 0.10 + 0.05 -> left 0.95.
    left_tray = ex.driver._drivers["left"].moves[-2]
    assert np.allclose(left_tray[:2], [0.50, 0.95])


def test_silverware_sort_fails_when_a_jaw_closes_on_air():
    ex = _rig(closed_raw=255)
    results = ex.execute_score(_plan(SILVERWARE, list(ex.detection_map)))
    picks = [r for r in results if r.action_type.startswith("pick_with_arm")]
    assert picks and not any(r.success for r in picks)


def test_shirt_fold_plan_runs_the_fold_chain():
    ex = _rig()
    ex.update_detections(SHIRT)
    results = ex.execute_score(_plan(FOLD, list(SHIRT)))
    assert [r.action_type for r in results][0] == "bimanual_shirt_fold"
    assert results[0].success, results[0].message
    # Two pinch-and-drape phases: each jaw closed and opened twice.
    assert all(j.packs == [0, 255, 0, 0, 255, 0] for j in ex.driver.grippers.jaws.values())
    right = ex.driver._drivers["right"].moves
    # Right sleeve (smaller y) landed past the shirt center line at y = 0.10.
    assert any(abs(m[1] - 0.10) < 1e-6 and abs(m[2] - (0.025 + 0.103)) < 1e-6 for m in right)
