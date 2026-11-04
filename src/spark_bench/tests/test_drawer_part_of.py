"""A 'drawer' detection far from every cabinet / handle is re-pointed at the cabinet."""
import numpy as np
from types import SimpleNamespace as N

from spark_bench.libero_pro.perception import _drawer_part_of_cabinet


def _d(x, y, z):
    return N(position_3d=np.array([x, y, z]))


def test_stray_drawer_goes_to_the_cabinet():
    cab = _d(-0.19, -0.18, 1.088)
    m = {'drawer': _d(0.054, -0.273, 1.206), 'wooden cabinet': cab, 'drawer handle': _d(-0.25, -0.163, 1.098)}
    _drawer_part_of_cabinet(m)
    assert m['drawer'] is cab


def test_drawer_on_the_cabinet_is_kept():
    dr = _d(-0.22, -0.17, 1.02)
    m = {'top drawer': dr, 'cabinet': _d(-0.19, -0.18, 1.088)}
    _drawer_part_of_cabinet(m)
    assert m['top drawer'] is dr
