"""Slot direction is measured from each slot floor; no tray layout is assumed."""

import numpy as np
import pytest

from spark_real.perception.mask_features import resolve_slot_direction
from spark_real.perception.slot_detector import detect_slots

FX, H, W = 600.0, 720, 1280
K = np.array([[FX, 0, W / 2], [0, FX, H / 2], [0, 0, 1.0]])
T = np.eye(4)
T[:3, :3] = np.diag([1.0, -1.0, -1.0])
T[2, 3] = 1.0  # overhead camera, 1 m up
U, V = np.meshgrid(np.arange(W), np.arange(H))
X, Y = (U - W / 2) / FX, -(V - H / 2) / FX


def tray(yaw_deg, across_long_side):
    """A 34 x 21 cm tray, 4 cm deep, with three slots. Returns (rgb, depth, mask)."""
    t = np.deg2rad(yaw_deg)
    p, q = X * np.cos(t) + Y * np.sin(t), -X * np.sin(t) + Y * np.cos(t)
    mask = (abs(p) < 0.17) & (abs(q) < 0.105)
    cut = p if across_long_side else q
    span = 0.17 if across_long_side else 0.105
    divider = abs(abs(cut) - span / 3) < 0.008
    rim = (abs(p) > 0.16) | (abs(q) > 0.095)
    z = np.where(rim | divider, 0.04, 0.0)
    return np.zeros((H, W, 3), np.uint8), np.where(mask, 1 - z, 1.05).astype(np.float32), mask


def err_deg(angle, truth_deg):
    return abs((np.rad2deg(angle) - truth_deg + 90) % 180 - 90)


@pytest.mark.parametrize("yaw", [0, 30, -70])
@pytest.mark.parametrize("across_long_side", [False, True])
def test_measured_direction_follows_the_slots_not_a_layout_rule(yaw, across_long_side):
    rgb, depth, mask = tray(yaw, across_long_side)
    slots = detect_slots(rgb, depth, mask, K, T)
    assert len(slots) == 3 and all(s["mode"] == "normals" for s in slots)
    truth = yaw + 90 if across_long_side else yaw
    for idx in (None, 0):
        angle, src = resolve_slot_direction({"slots": slots}, "tray", idx)
        assert err_deg(angle, truth) < 5, src
    # slot_idx walks across the tray, side by side
    across = np.deg2rad(truth + 90)
    proj = [s["world_xyz"][0] * np.cos(across) + s["world_xyz"][1] * np.sin(across) for s in slots]
    assert abs(proj[2] - proj[0]) > 0.05


def test_each_compartment_of_a_mixed_tray_answers_for_itself():
    slots = [
        {"slot_idx": 0, "mode": "normals", "n_pixels": 900, "axis_rad": 0.0, "axis_ar": 4.0},
        {"slot_idx": 1, "mode": "normals", "n_pixels": 900, "axis_rad": np.pi / 2, "axis_ar": 4.0},
    ]
    assert err_deg(resolve_slot_direction({"slots": slots}, "tray", 1)[0], 90) < 1
    # no single direction for the tray as a whole: use its long axis
    det = {"slots": slots, "world_major_axis_rad": 0.2, "aspect_ratio": 1.6}
    angle, src = resolve_slot_direction(det, "tray")
    assert angle == 0.2 and src.startswith("container_long_axis")


def test_unresolved_dividers_default_to_the_long_side():
    det = {"slots": [{"slot_idx": 0, "mode": "fallback"}], "world_major_axis_rad": 0.5, "aspect_ratio": 1.6}
    assert resolve_slot_direction(det, "tray", 0)[0] == 0.5
    # near-square OBB: its axis cannot name the long side
    det = {"world_major_axis_rad": 0.5, "aspect_ratio": 1.1, "plane_orientation_angle": 1.2}
    assert resolve_slot_direction(det, "tray")[0] == 1.2
