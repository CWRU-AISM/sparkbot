"""Knob fin pose from mask points: centre and long axis of the top band, angle from front-back."""
import types

import numpy as np

from spark_bench.libero_pro.primitives import knob


def test_fin_axis_and_off_angle(monkeypatch):
    rng = np.random.default_rng(0)
    yaw = 0.7
    u = np.array([np.cos(yaw), np.sin(yaw)])
    v = np.array([-u[1], u[0]])
    # hub disc (low) + fin (7 cm x 2.4 cm, 2.5 cm taller) centred on (-0.19, 0.2)
    hub = np.c_[rng.uniform(-0.2, -0.18, (300, 2)), np.full(300, 0.918)]
    s, t = rng.uniform(-0.035, 0.035, 300), rng.uniform(-0.012, 0.012, 300)
    fin = np.c_[(-0.19, 0.2) + s[:, None] * u + t[:, None] * v, rng.uniform(0.945, 0.955, 300)]
    monkeypatch.setattr(knob, 'mask_points_3d', lambda ex, det: np.vstack([hub, fin]))
    centre, ax = knob._knob_fin(None, object())
    assert np.allclose(centre[:2], (-0.19, 0.2), atol=0.005)
    assert abs(abs(float(ax @ u)) - 1.0) < 0.01
    ex = types.SimpleNamespace(robot_base_world=np.array([-0.66, 0.0, 0.9]))
    assert abs(knob._fin_off_angle(ex, ax) - yaw) < 0.05
