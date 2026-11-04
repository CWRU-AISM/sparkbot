"""relation-bind 'on': the anchor decides unless its detection is one of the twins; then height decides."""
import numpy as np
from types import SimpleNamespace as N

from spark_bench.libero_pro.perception import _relation_choose


def _d(x, y, z):
    return N(position_3d=np.array([x, y, z]))


def test_bowl_on_the_stove_uses_the_anchor_under_it():
    # the stove mask sits right under the target bowl; the twin stands higher on the cabinet
    inst = [_d(-0.20, 0.20, 0.98), _d(0.08, 0.20, 1.197)]
    chosen, rule = _relation_choose(inst, 'pick the akita black bowl on the stove and place it on the plate',
                                    'akita black bowl', lambda p: np.array([-0.20, 0.20, 0.93]))
    assert rule == 'on' and chosen is inst[0]


def test_anchor_detection_on_the_twin_falls_back_to_height():
    # SAM3's 'ramekin' landed on the other bowl (same height); the target is stacked on the real ramekin
    inst = [_d(0.066, 0.212, 0.95), _d(-0.215, 0.179, 1.013)]
    chosen, rule = _relation_choose(inst, 'pick the akita black bowl on the ramekin and place it on the plate',
                                    'akita black bowl', lambda p: np.array([0.066, 0.212, 0.95]))
    assert rule == 'on:highest' and chosen is inst[1]


def test_generic_bowl_on_the_stove_burner_is_set_aside():
    # 'bowl' also matched the dark burner at the stove's own position; the target rests 9 cm off it, the
    # twin stands on the cabinet (higher)
    inst = [_d(-0.216, -0.10, 0.926), _d(-0.30, -0.15, 0.98), _d(0.0, -0.284, 1.178)]
    chosen, rule = _relation_choose(inst, 'pick the akita black bowl on the stove and place it on the plate',
                                    'akita black bowl', lambda p: np.array([-0.216, -0.10, 0.926]))
    assert rule == 'on:rests' and chosen is inst[1]
