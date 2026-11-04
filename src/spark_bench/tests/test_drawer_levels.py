"""Drawer level height from the cabinet mask: band from the cabinet top and the top bar."""
import numpy as np
from types import SimpleNamespace as N

from spark_bench.libero_pro.primitives.drawer_geometry import level_band_z


def _ex(z_lo, z_hi):
    span = (z_hi - z_lo) / 0.96             # uniform points whose 2nd/98th percentiles are z_lo/z_hi
    pts = np.column_stack([np.zeros(501), np.zeros(501), np.linspace(z_lo - 0.02 * span, z_hi + 0.02 * span, 501)])
    sam3 = N(_backproject_mask=lambda *a: pts)
    return N(sam3=sam3, depth=np.zeros((2, 2)), cam_pos=None, cam_mat=None, cam_fovy=None, cam_w=0, cam_h=0,
             det_map={'wooden cabinet': N(mask=np.ones((2, 2)), position_3d=np.zeros(3))})


def _cl(*zs):
    return [('drawer handle', np.array([0.0, 0.0, z]), 1.0, z) for z in zs]


def test_middle_from_top_bar_when_the_bottom_is_hidden():
    # goal swap 0: only the top bar detected, lower third of the cabinet hidden (mask 0.968-1.128)
    z = level_band_z(_ex(0.968, 1.128), 'middle', _cl(1.095))
    assert abs(z - 1.029) < 0.006          # detected-style middle bar: 1.023


def test_levels_with_the_whole_face_visible():
    ex = _ex(0.92, 1.128)
    assert abs(level_band_z(ex, 'bottom', _cl(1.095)) - 0.963) < 0.012   # bar 0.953
    assert abs(level_band_z(ex, 'top', _cl(0.953)) - 1.093) < 0.006       # no top bar: equal thirds
