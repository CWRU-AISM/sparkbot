"""
FR3 paper tasks end to end with a mocked driver: the planner's reply (the
BT a planner call returns for the frozen FR3 prompt) is parsed and validated
by SPARKPlanner, the executor dispatches every leaf to its registered skill,
and each skill's post-condition decides the result. No camera, no robot, no
network: SPARKPlanner.generate is replaced by the canned reply.

Usage:
    PYTHONPATH=src python -m pytest src/spark_real/tests/test_paper_tasks_fr3.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_selector_semantics import _FakePipeline, _MockRobot  # noqa: E402

import spark_real.control.executor_ik as executor_ik  # noqa: E402
from spark_core.planner import SPARKPlanner  # noqa: E402
from spark_real.control.score_executor import ScoreExecutor  # noqa: E402


class _FR3(_MockRobot):
    """FR3 + Franka Hand double: Cartesian moves land exactly; a close on an
    object stops at closed_width, on air at 0."""

    def __init__(self, closed_width=0.02):
        super().__init__()
        self.closed_width = closed_width
        self.path = []

    def move_linear(self, pose, velocity=0.1, asynchronous=False):
        super().move_linear(pose, velocity, asynchronous)
        self.path.append(np.array(pose[:3], dtype=float))

    def close_gripper(self, **kw):
        self._open = False
        self._log.append("close_gripper")

    def open_gripper(self, **kw):
        self._open = True
        self._log.append("open_gripper")

    def grasp_to_width(self, width, force=20, speed=0.1, **kw):
        self.close_gripper()
        return True

    def get_gripper_width(self):
        return 0.08 if self._open else self.closed_width


SCENE = {
    "sponge": [0.45, 0.15, 0.0], "plate": [0.50, -0.10, 0.01], "mug": [0.40, 0.20, 0.03],
    "bowl": [0.55, -0.05, 0.02], "brush": [0.35, 0.25, 0.01], "red cube": [0.45, -0.05, 0.0],
    "blue cube": [0.50, 0.00, 0.0], "dustpan": [0.48, 0.20, 0.0], "left sleeve": [0.45, 0.22, -0.02],
    "right sleeve": [0.45, -0.22, -0.02], "shirt": [0.42, 0.0, -0.02], "hem": [0.30, 0.0, -0.02],
    "fork": [0.40, -0.15, 0.0], "tray": [0.55, 0.10, 0.01],
}

PLANS = {
    "scrub the plate with the sponge": """```yaml
task: scrub the plate with the sponge
tree:
  type: sequence
  children:
    - {type: grasp_se3, params: {keypoint_label: sponge, strategy: top_down, force: 25, target_width: 0.035}}
    - {type: move_relative, params: {dx: 0, dy: 0, dz: 0.10}}
    - {type: constrained_scrub, params: {target_label: plate, force_n: 8.0, n_cycles: 2, scrub_radius_m: 0.04, pattern: circle}}
    - {type: move_to_keypoint, params: {keypoint_label: sponge, offset_z: 0.02}}
    - {type: release, params: {}}
```""",
    "pour from the mug into the bowl": """```yaml
task: pour from the mug into the bowl
tree:
  type: sequence
  children:
    - {type: grasp_se3, params: {keypoint_label: mug, strategy: horizontal, force: 60, target_width: 0.05}}
    - {type: pour, params: {source_label: mug, target_label: bowl, pour_angle_rad: 1.4, max_pour_s: 0.05}}
    - {type: move_relative, params: {dx: 0, dy: 0, dz: 0.10}}
```""",
    "sweep the cubes into the dustpan": """```yaml
task: sweep the cubes into the dustpan
tree:
  type: sequence
  children:
    - {type: grasp_se3, params: {keypoint_label: brush, force: 60, target_width: 0.020}}
    - {type: sweep, params: {object_labels: "red cube, blue cube", target_label: dustpan, n_passes: 1, sweep_height_m: 0.04, sweep_velocity: 0.08}}
```""",
    "fold the shirt": """```yaml
task: fold the shirt
tree:
  type: sequence
  children:
    - {type: cloth_fold, params: {instruction: fold the shirt}}
```""",
    "put the fork in the tray": """```yaml
task: put the fork in the tray
tree:
  type: sequence
  children:
    - {type: grasp_se3, params: {keypoint_label: fork, force: 40, target_width: 0.008}}
    - {type: move_relative, params: {dx: 0, dy: 0, dz: 0.10}}
    - {type: move_to_keypoint, params: {keypoint_label: tray, offset_z: 0.06}}
    - {type: release, params: {}}
```""",
}

# Skill leaves each plan must reach, in order (move_relative etc. are built-ins).
SKILLS = {
    "scrub the plate with the sponge": ["grasp_se3", "constrained_scrub"],
    "pour from the mug into the bowl": ["grasp_se3", "pour"],
    "sweep the cubes into the dustpan": ["grasp_se3", "sweep"],
    "fold the shirt": ["cloth_fold"],
    "put the fork in the tray": ["grasp_se3"],
}


def _plan(instruction, monkeypatch):
    planner = SPARKPlanner.__new__(SPARKPlanner)
    planner.robot_family = "franka"
    planner.temperature = 0.0
    monkeypatch.setattr(planner, "generate", lambda contents, tag: PLANS[instruction], raising=False)
    score = planner.generate_score(instruction, keypoint_labels=list(SCENE))
    assert planner.validate_score(score, list(SCENE)) == []
    return score


def _executor(robot, monkeypatch):
    monkeypatch.setattr(executor_ik, "solve_ik_fr3", None)  # Cartesian path on the double
    pipeline = _FakePipeline()
    pipeline.profile = SimpleNamespace(raw={"verification": {"enabled": False}})
    ex = ScoreExecutor(robot, detection_map={}, velocity=0.1, pipeline=pipeline)
    ex.update_detections({k: {"position_3d": v} for k, v in SCENE.items()})
    return ex


@pytest.mark.parametrize("instruction", list(PLANS))
def test_paper_task_runs_plan_to_skills_to_post_condition(instruction, monkeypatch):
    robot = _FR3()
    ex = _executor(robot, monkeypatch)
    results = ex.execute_score(_plan(instruction, monkeypatch))
    kinds = [r.action_type for r in results]
    assert all(r.success for r in results), [(r.action_type, r.message) for r in results]
    for skill in SKILLS[instruction]:
        assert skill in kinds, (skill, kinds)
    assert robot.path, "no motion reached the driver"


def test_a_pick_that_closes_on_air_fails_its_post_condition(monkeypatch):
    robot = _FR3(closed_width=0.0)
    ex = _executor(robot, monkeypatch)
    results = ex.execute_score(_plan("sweep the cubes into the dustpan", monkeypatch))
    pick = next(r for r in results if r.action_type == "grasp_se3")
    assert not pick.success and "holding=False" in pick.message
    assert robot._log.count("close_gripper") >= 4  # stepped retries before giving up


def test_sweep_path_runs_through_the_objects_into_the_dustpan(monkeypatch):
    robot = _FR3()
    ex = _executor(robot, monkeypatch)
    ex.execute_score(_plan("sweep the cubes into the dustpan", monkeypatch))
    low = [p for p in robot.path if abs(p[2] - (-0.025 + 0.04)) < 1e-6 or abs(p[2] - (ex.TABLE_Z_FLOOR + 0.04)) < 1e-6]
    assert low, "no waypoint at sweep height"
    d = np.array(SCENE["dustpan"][:2]) - np.mean([SCENE["red cube"][:2], SCENE["blue cube"][:2]], axis=0)
    d /= np.linalg.norm(d)
    proj = [float(np.dot(p[:2] - SCENE["dustpan"][:2], d)) for p in low]
    assert proj[0] < min(np.dot(np.array(SCENE[c][:2]) - SCENE["dustpan"][:2], d) for c in ("red cube", "blue cube"))
    assert proj[-1] > 0  # ends past the dustpan center
