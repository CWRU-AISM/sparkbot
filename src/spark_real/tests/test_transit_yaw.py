"""UR10e transit must not unwind the wrist: home sits on the grasp yaw, transit rows hold yaw."""

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

import spark_real.control.motion_approach as ma
from spark_real.control.grasp_strategy import measured_yaw_offset
from spark_real.control.score_executor import ScoreExecutor
from spark_real.control.ur10e_driver import UR10eDriver
from spark_real.tests.blend_plant import UrPlant, fk_rtde


def test_home_matches_config_and_sits_near_the_grasp_yaw():
    cfg = yaml.safe_load(open(ma.__file__.replace("control/motion_approach.py", "configs/ur10e_default.yaml")))
    assert UR10eDriver.HOME_CONFIG == cfg["robot"]["home_config"]
    ex = ScoreExecutor(UrPlant(), detection_map={}, velocity=0.25)
    _, M = fk_rtde(UR10eDriver.HOME_CONFIG)
    rv = Rotation.from_matrix(M).as_rotvec()
    assert abs(measured_yaw_offset(rv, ex.GRASP_ORIENTATION)) < np.deg2rad(10)


@pytest.mark.parametrize("family,refine,holds", [("ur10e", False, True), ("ur10e", True, False), ("franka", False, False)])
def test_transit_rows_hold_yaw_on_ur10e_only(monkeypatch, family, refine, holds):
    monkeypatch.setenv("SPARK_UR_BLEND", "1")
    plant = UrPlant()
    ex = ScoreExecutor(plant, detection_map={}, velocity=0.25)
    base = Rotation.from_rotvec(ex.GRASP_ORIENTATION)
    start = list((Rotation.from_euler("z", -90, degrees=True) * base).as_rotvec())
    orient = list((Rotation.from_euler("z", -30, degrees=True) * base).as_rotvec())
    plant.robot_family = family
    ex._wrist_refine_enabled = refine
    ex._last_keypoint_label = "spoon"
    monkeypatch.setattr(ex, "current_orientation", lambda fallback=None: start)
    monkeypatch.setattr(ex, "_wrist_refine_mode", lambda: "ray_plane")
    monkeypatch.setattr(ex, "_refine_with_wrist", lambda *a, **k: None)
    monkeypatch.setattr(ex, "_blend_enabled", lambda: True)
    monkeypatch.setattr(ma, "resolve_grasp_orientation", lambda *a, **k: (orient, "obb"))
    seen = {}

    def capture(rows, **k):
        seen.update({row[3]: row[1] for row in rows})
        raise AssertionError("stop after the phrase")

    monkeypatch.setattr(ex, "_blend_transit", capture)
    far = np.asarray(plant.get_tcp_pose()[:3]) + [0.7, 0.0, 0.0]
    with pytest.raises(AssertionError, match="stop after the phrase"):
        ex._approach_target(far)
    assert seen["lift"] == (start if holds else ex.GRASP_ORIENTATION)
    assert seen["above"] == (orient if holds else ex.GRASP_ORIENTATION)


def test_ik_keeps_wrist3_within_a_turn():
    pytest.importorskip("pyroki")
    from spark_real.control.ur10e_ik_pyroki import solve_ik_rtde

    q = np.array(UrPlant.HOME_CONFIG, dtype=float)
    q[5] = 5.0  # the same pose is reachable a turn lower
    pos, M = fk_rtde(q)
    sol = solve_ik_rtde(pos, Rotation.from_matrix(M).as_rotvec(), q_seed=q, tcp_offset=UrPlant().tcp_offset)
    assert sol[5] == pytest.approx(5.0 - 2 * np.pi, abs=0.02)
    got, _ = fk_rtde(sol)
    assert np.linalg.norm(got - pos) < 0.002
