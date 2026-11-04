"""_relation_choose picks the target black bowl on the LIBERO-spatial layouts (simulator positions)."""
from types import SimpleNamespace

import numpy as np
import pytest

from spark_bench.libero_pro.perception import _relation_choose


def det(x, y, z=0.03):
    return SimpleNamespace(position_3d=np.array([x, y, z]))


CASES = [  # (instruction, target bowl, distractor bowl, anchors)
    ("pick up the black bowl between the plate and the ramekin and place it on the plate",
     (-0.05, 0.20), (-0.19, 0.31), {"plate": (0.06, 0.19), "ramekin": (-0.19, 0.21)}),
    ("pick up the black bowl next to the cookie box and place it on the plate",
     (0.13, -0.07), (-0.27, -0.15), {"cookie box": (0.07, 0.02)}),
    ("pick up the black bowl next to the plate and place it on the plate",
     (0.01, 0.31), (-0.19, 0.31), {"plate": (0.06, 0.19)}),
    ("pick up the black bowl on the wooden cabinet and place it on the plate",
     (0.03, -0.28), (-0.28, -0.15), {"wooden cabinet": (0.03, -0.27)}),
]


@pytest.mark.parametrize("instruction,target,distractor,anchors", CASES)
def test_picks_target(instruction, target, distractor, anchors):
    t, d = det(*target), det(*distractor)
    locate = lambda phrase: next((np.array([*p, 0.0]) for k, p in anchors.items() if k in phrase), None)
    for order in ([t, d], [d, t]):
        chosen, _ = _relation_choose(order, instruction, "black bowl", locate)
        assert chosen is t


def test_top_drawer_prefers_higher_inside_not_on_top():
    # spatial task 4: target in the top drawer (z 1.11), twin resting ON the cabinet (z 1.18), cabinet centroid z 1.13
    in_drawer, on_cabinet = det(0.08, -0.15, 1.11), det(-0.01, -0.27, 1.18)
    cabinet = np.array([0.08, -0.24, 1.13])
    for text, word in [("pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate", "drawer"),
                       ("Pick the akita black bowl in the top layer of the wooden cabinet and place it on the plate", "layer")]:
        for order in ([in_drawer, on_cabinet], [on_cabinet, in_drawer]):
            chosen, rule = _relation_choose(order, text, "black bowl", lambda p: cabinet)
            assert chosen is in_drawer and word in rule


def test_no_relation_or_missing_anchor_returns_none():
    a, b = det(0, 0), det(0.2, 0.2)
    assert _relation_choose([a, b], "pick up the black bowl and place it on the plate", "black bowl", lambda p: None) is None
    assert _relation_choose([a, b], "pick up the black bowl next to the plate and place it on the plate",
                            "black bowl", lambda p: None) is None


def test_place_clause_is_not_a_pick_relation():
    a, b = det(0, 0), det(0.2, 0.2)
    locate = lambda p: np.array([0.2, 0.2, 0.0])
    assert _relation_choose([a, b], "pick up the alphabet soup and place it in the basket", "alphabet soup", locate) is None
    assert _relation_choose([a, b], "put the bowl on the plate", "bowl", locate) is None
