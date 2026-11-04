"""SPARK_PICK_REGROUND: the wrist re-ground shifts the pick by a near fix and rejects one past the gate."""
from types import SimpleNamespace

import numpy as np

from spark_bench.libero_pro.primitives import keypoint


class _Exec:
    sam3 = object()
    prompts_for_refresh = ['milk carton']
    env = None
    holding = False
    cfg = None

    def __init__(self):
        self.trial_meta = {}

    def _move_to(self, *a, **k):
        return 0.0


def test_reground_gate(monkeypatch):
    monkeypatch.setattr(keypoint, 'step_env', lambda *a, **k: False)
    monkeypatch.setattr(keypoint, 'PICK_REGROUND_GATE_M', 0.05)
    target = np.array([0.10, 0.20, 0.05])
    fix = {'xy': None}
    monkeypatch.setattr(keypoint, '_wrist_fix', lambda ex, lbl, near: fix['xy'])

    fix['xy'] = np.array([0.12, 0.20])           # 2 cm: a real fix
    ex = _Exec()
    shift = keypoint._reground_pick(ex, 'milk', target)
    assert np.allclose(shift, [0.02, 0.0])

    fix['xy'] = np.array([0.20, 0.20])           # 10 cm: a look-alike, rejected
    ex = _Exec()
    assert keypoint._reground_pick(ex, 'milk', target) is None
    assert ex.trial_meta['pick_reground'][0]['rejected'] is True


def test_handle_shifts_pick_not_shared_det(monkeypatch):
    """handle() moves the pick target and a copy of the detection; the det_map entry stays put."""
    monkeypatch.setattr(keypoint, 'PICK_REGROUND', True)
    monkeypatch.setattr(keypoint, '_reground_pick', lambda ex, lbl, t: np.array([0.02, -0.01]))
    got = {}
    monkeypatch.setattr(keypoint, '_do_pick', lambda ex, label, det, target, *a: got.update(det=det, target=target))
    det = SimpleNamespace(label='milk', position_3d=np.array([0.10, 0.20, 0.05]))
    ex = _Exec()
    ex.det_map = {'milk': det}
    keypoint.handle(ex, {'keypoint_label': 'milk'})
    assert np.allclose(got['target'][:2], [0.12, 0.19])
    assert np.allclose(got['det'].position_3d[:2], [0.12, 0.19])
    assert np.allclose(det.position_3d, [0.10, 0.20, 0.05])
