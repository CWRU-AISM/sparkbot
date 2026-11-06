"""Offline tests for the world-XY OBB: trimmed extents + obb_confidence.

Synthetic depth images only -- no camera, no SAM3. The camera looks straight
down (OpenCV convention) from `CAM_H`, so world XY is a pure scaled flip of
image XY and the expected OBB is known in closed form.
"""

import numpy as np
import pytest

from spark_core.mask_obb import _world_xy_pca_obb

F = 500.0
W = H = 320
CX = CY = 160.0
CAM_H = 1.0
# world = cam_mat @ (x_cam, y_cam, z_cam) + cam_pos, camera looking down -Z.
CAM_MAT = np.diag([1.0, -1.0, -1.0])
CAM_POS = np.array([0.0, 0.0, CAM_H])


def _render(height_fn, inside_fn, noise_m=0.0, seed=0):
    """Return (mask, depth) for a surface z=height_fn(X,Y) over inside_fn."""
    rng = np.random.default_rng(seed)
    v, u = np.mgrid[0:H, 0:W]
    # First-order backprojection at the table plane; good enough for a
    # few-cm-tall object 1 m away.
    x = (u - CX) * CAM_H / F
    y = -(v - CY) * CAM_H / F
    inside = inside_fn(x, y)
    z = np.where(inside, height_fn(x, y), 0.0)
    depth = CAM_H - z
    if noise_m:
        depth = depth + rng.normal(0.0, noise_m, depth.shape)
    return inside.astype(np.uint8), depth.astype(np.float64)


def _obb(mask, depth):
    return _world_xy_pca_obb(
        mask,
        depth,
        CAM_POS,
        CAM_MAT,
        fx=F,
        fy=F,
        cx_k=CX,
        cy_k=CY,
        use_opencv=True,
        return_confidence=True,
    )


def _rect(a, b, theta):
    """Axis-aligned a x b rectangle rotated by theta about the origin."""
    c, s = np.cos(-theta), np.sin(-theta)

    def inside(x, y):
        xr = c * x - s * y
        yr = s * x + c * y
        return (np.abs(xr) <= a / 2) & (np.abs(yr) <= b / 2)

    return inside


# the round dome: an arbitrary major axis must score low


@pytest.mark.parametrize("seed", range(20))
def test_round_dome_is_near_isotropic_and_low_confidence(seed):
    R, HH = 0.05, 0.03

    def dome(x, y):
        r2 = np.clip(1.0 - (x**2 + y**2) / R**2, 0.0, None)
        return HH * np.sqrt(r2)

    mask, depth = _render(
        dome, lambda x, y: x**2 + y**2 <= R**2, noise_m=0.004, seed=seed
    )
    orient, ar, mj, mn, conf = _obb(mask, depth)
    assert ar is not None
    # AR alone cannot gate this. The dome's SILHOUETTE is a perfect circle, so
    # trimming the extents barely moves it (measured over these 20 seeds:
    # untrimmed 1.00-1.35, trimmed 1.02-1.32) while the reported major-axis
    # angle wanders over the full -162..165 deg. obb_confidence is what
    # actually catches it: 0.004-0.104 here.
    assert ar < 1.4, f"seed {seed}: dome aspect_ratio {ar:.2f}"
    assert conf < 0.4, f"seed {seed}: dome obb_confidence {conf:.2f}"


def test_dome_major_axis_is_arbitrary():
    """The reason a round object must not be yawed: the angle is noise."""
    R, HH = 0.05, 0.03
    angles = []
    for seed in range(20):
        mask, depth = _render(
            lambda x, y: HH * np.sqrt(np.clip(1.0 - (x**2 + y**2) / R**2, 0.0, None)),
            lambda x, y: x**2 + y**2 <= R**2,
            noise_m=0.004,
            seed=seed,
        )
        angles.append(np.rad2deg(_obb(mask, depth)[0]))
    assert np.ptp(angles) > 90.0, f"angles clustered: {np.ptp(angles):.0f} deg spread"


# a genuine elongated object still reads correctly


@pytest.mark.parametrize("theta_deg", [0.0, 25.0, 60.0, 120.0])
def test_elongated_block_keeps_its_axis_and_scores_high(theta_deg):
    theta = np.deg2rad(theta_deg)
    mask, depth = _render(
        lambda x, y: 0.02, _rect(0.16, 0.04, theta), noise_m=0.001, seed=1
    )
    orient, ar, mj, mn, conf = _obb(mask, depth)
    assert ar == pytest.approx(4.0, rel=0.15)
    # Angle is defined mod 180 deg.
    err = (np.rad2deg(orient) - theta_deg + 90) % 180 - 90
    assert abs(err) < 3.0, f"angle error {err:.1f} deg"
    assert conf > 0.5, f"obb_confidence {conf:.2f}"


def test_metric_extents_are_not_biased_by_trimming():
    """The 5-95 trim is rescaled, so lengths still read true."""
    mask, depth = _render(lambda x, y: 0.02, _rect(0.16, 0.04, 0.0), seed=2)
    _, _, mj, mn = _world_xy_pca_obb(
        mask, depth, CAM_POS, CAM_MAT, fx=F, fy=F, cx_k=CX, cy_k=CY, use_opencv=True
    )
    assert mj == pytest.approx(0.16, rel=0.06)
    assert mn == pytest.approx(0.04, rel=0.10)


def test_outlier_pixels_no_longer_set_the_extent():
    """Two stray far pixels used to define max()-min(). Now they are trimmed."""
    mask, depth = _render(lambda x, y: 0.02, _rect(0.06, 0.05, 0.0), seed=3)
    clean = _world_xy_pca_obb(
        mask, depth, CAM_POS, CAM_MAT, fx=F, fy=F, cx_k=CX, cy_k=CY, use_opencv=True
    )
    # A thin 1-px whisker: a mask leak along one image axis.
    mask2 = mask.copy()
    mask2[CY_ROW := int(CY), int(CX) : int(CX) + 60] = 1
    depth2 = depth.copy()
    depth2[CY_ROW, int(CX) : int(CX) + 60] = CAM_H - 0.02
    dirty = _world_xy_pca_obb(
        mask2, depth2, CAM_POS, CAM_MAT, fx=F, fy=F, cx_k=CX, cy_k=CY, use_opencv=True
    )
    assert dirty[1] < 1.6 * clean[1], (
        f"whisker inflated aspect_ratio {clean[1]:.2f} -> {dirty[1]:.2f}"
    )


def test_major_axis_is_always_the_longer_one():
    """The axis-inversion guard: mj_len >= mn_len for every orientation."""
    for theta_deg in range(0, 180, 11):
        mask, depth = _render(
            lambda x, y: 0.02,
            _rect(0.06, 0.055, np.deg2rad(theta_deg)),
            noise_m=0.002,
            seed=7,
        )
        _, ar, mj, mn, _ = _obb(mask, depth)
        assert mj >= mn, f"theta {theta_deg}: mj {mj:.4f} < mn {mn:.4f}"
        assert ar >= 1.0


def test_degenerate_inputs_return_none_not_a_crash():
    empty = np.zeros((H, W), dtype=np.uint8)
    depth = np.full((H, W), CAM_H)
    assert _obb(empty, depth) == (None,) * 5
    assert _world_xy_pca_obb(None, depth, CAM_POS, CAM_MAT) == (None,) * 4
    one = empty.copy()
    one[10, 10] = 1
    assert _obb(one, depth) == (None,) * 5


def test_return_arity_is_unchanged_by_default():
    """Existing callers unpack four values; that must keep working."""
    mask, depth = _render(lambda x, y: 0.02, _rect(0.10, 0.04, 0.0), seed=4)
    out = _world_xy_pca_obb(
        mask, depth, CAM_POS, CAM_MAT, fx=F, fy=F, cx_k=CX, cy_k=CY, use_opencv=True
    )
    assert len(out) == 4
    assert len(_obb(mask, depth)) == 5


# --- a tilted top surface (SPARK_OBB_TILT_FIT) ---
_FX, _H, _W = 600.0, 720, 1280
_U, _V = np.meshgrid(np.arange(_W), np.arange(_H))
_X, _Y = (_U - _W / 2) / _FX, -(_V - _H / 2) / _FX


def _overhead_obb(monkeypatch, a, b, yaw_deg, zfun, noise=0.0015):
    """(axis error in deg, aspect ratio) from a camera 1 m above an a x b footprint."""
    monkeypatch.setenv("SPARK_OBB_TILT_FIT", "1")
    t = np.deg2rad(yaw_deg)
    p, q = _X * np.cos(t) + _Y * np.sin(t), -_X * np.sin(t) + _Y * np.cos(t)
    mask = (abs(p) < a / 2) & (abs(q) < b / 2)
    depth = 1 - zfun(p, q) + np.random.default_rng(0).normal(0, noise, p.shape)
    angle, ar, *_ = _world_xy_pca_obb(
        mask.astype(np.uint8), np.where(mask, depth, 0).astype(np.float32),
        np.array([0, 0, 1.0]), np.diag([1.0, -1.0, -1.0]),
        fx=_FX, fy=_FX, cx_k=_W / 2, cy_k=_H / 2, use_opencv=True,
    )
    return abs((np.rad2deg(angle) - yaw_deg + 90) % 180 - 90), ar


@pytest.mark.parametrize("tilt_deg", [0.0, 1.0, 1.5, 3.0])
def test_tilted_tray_keeps_its_long_axis(monkeypatch, tilt_deg):
    def tray(p, q):
        rim = (abs(p) > 0.162) | (abs(q) > 0.0965)
        return np.where(rim, 0.04, 0.01) + np.tan(np.deg2rad(tilt_deg)) * _X
    err, ar = _overhead_obb(monkeypatch, 0.348, 0.217, 30, tray)
    assert err < 3 and ar == pytest.approx(1.6, abs=0.1)


def test_spoon_raised_at_one_end_stays_elongated(monkeypatch):
    err, ar = _overhead_obb(monkeypatch, 0.20, 0.02, 40, lambda p, q: 0.005 + 0.02 * (p + 0.1) / 0.2)
    assert err < 3 and ar > 8


def test_box_side_face_stays_out_of_the_footprint(monkeypatch):
    # 5 x 5 cm top at 5 cm height; the camera also sees a 2 cm band of side face, far below the top.
    err, ar = _overhead_obb(monkeypatch, 0.07, 0.05, 0, lambda p, q: np.where(p > 0.015, 0.05 * (0.035 - p) / 0.02, 0.05))
    assert ar < 1.15
