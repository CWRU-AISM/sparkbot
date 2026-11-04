"""Free-joint lookup, qpos displacement, contacts and pick-target resolution in a LIBERO MuJoCo scene."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from spark_core.binding import fuzzy_key

_HINT_STOPWORDS = {'the', 'and', 'left', 'right', 'front', 'back', 'top'}


def hint_tokens(hint: str) -> list[str]:
    # Digits are kept: LIBERO instances differ only by suffix (akita_black_bowl_1 vs _2).
    return [t for t in hint.lower().replace('_', ' ').split()
            if (len(t) >= 3 or t.isdigit()) and t not in _HINT_STOPWORDS]


def free_object_joints(model, mj):
    """(qposadr, body_id, body_name) of every free joint on a non-robot body."""
    out = []
    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) != int(mj.mjtJoint.mjJNT_FREE):
            continue
        bid = int(model.jnt_bodyid[jid])
        bn = (mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid) or '').lower()
        if not bn or 'robot' in bn or 'gripper' in bn:
            continue
        out.append((int(model.jnt_qposadr[jid]), bid, bn))
    return out


def find_free_joint_for_hint(model, hint: str, mj=None):
    """(qposadr, body_name) of the free joint whose body name best matches ``hint``, else (None, None)."""
    if mj is None:
        import mujoco as mj  # type: ignore[no-redef]
    tokens = hint_tokens(hint)
    if not tokens:
        return None, None
    best = (None, None)
    best_score = 0
    for adr, _bid, bn in free_object_joints(model, mj):
        score = sum(1 for t in tokens if t in bn)
        # Prefer more matched tokens, then the shorter name.
        if score > best_score or (score == best_score and score > 0
                                  and best[1] is not None and len(bn) < len(best[1])):
            best = (adr, bn)
            best_score = score
    return best if best_score > 0 else (None, None)


def displace_body_qpos(env, hint: str, delta: np.ndarray) -> Optional[str]:
    """Shift the free body matching ``hint`` by ``delta`` (m) in env.sim qpos; returns its name or None."""
    import mujoco
    model = env.sim.model._model
    data = env.sim.data._data
    adr, name = find_free_joint_for_hint(model, hint, mj=mujoco)
    if adr is None:
        return None
    d = np.asarray(delta, dtype=float).reshape(-1)
    data.qpos[adr] += d[0]
    data.qpos[adr + 1] += d[1]
    if len(d) > 2:
        data.qpos[adr + 2] += d[2]
    mujoco.mj_forward(model, data)
    try:
        env.sim.forward()
    except Exception:
        pass
    return name


def body_object_contact(model, data, bid, other_bids) -> bool:
    """True if body ``bid`` penetrates any body in ``other_bids``."""
    for i in range(int(data.ncon)):
        c = data.contact[i]
        if c.dist > -1e-4:      # only real penetration
            continue
        b1 = int(model.geom_bodyid[c.geom1])
        b2 = int(model.geom_bodyid[c.geom2])
        if (b1 == bid and b2 in other_bids) or (b2 == bid and b1 in other_bids):
            return True
    return False


def ee_reader(env) -> Callable[[], Optional[np.ndarray]]:
    """Reader of the gripper grip-site position (MID_APPROACH progress)."""
    import mujoco
    model = env.sim.model._model
    data = env.sim.data._data
    site = -1
    for s in range(model.nsite):
        if 'grip_site' in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s) or '').lower():
            site = s
            break

    def _read() -> Optional[np.ndarray]:
        if site < 0:
            return None
        mujoco.mj_forward(model, data)
        return data.site_xpos[site].copy()

    return _read


def _iter_leaves(tree):
    """Depth-first leaves of a BT score dict."""
    if not isinstance(tree, dict):
        return
    if tree.get('type') in ('sequence', 'selector'):
        for c in tree.get('children', []) or []:
            yield from _iter_leaves(c)
    else:
        yield tree


def resolve_pick_target_pos(det_map, pick_hint: str, score=None, _fuzzy=fuzzy_key) -> Optional[np.ndarray]:
    """Pick-target position in ``det_map``: the score's first move_to_keypoint label, then fuzzy_key on
    the BDDL hint, then token overlap ('akita_black_bowl_1' vs 'dark bowl' share no substring)."""
    def _pos(key):
        if key is None or key not in det_map:
            return None
        p = getattr(det_map[key], 'position_3d', None)
        return None if p is None else np.asarray(p, dtype=float)

    try:
        for act in _iter_leaves((score or {}).get('tree', {})):
            lbl = (act.get('params', {}) or {}).get('keypoint_label')
            if lbl:
                p = _pos(_fuzzy(det_map, str(lbl)))
                if p is not None:
                    return p
                break
    except Exception:
        pass
    p = _pos(_fuzzy(det_map, pick_hint))
    if p is not None:
        return p
    toks = set(hint_tokens(pick_hint))
    if toks:
        best_key, best_n = None, 0
        for k in det_map:
            n = len(toks & set(str(k).lower().replace('_', ' ').split()))
            if n > best_n:
                best_key, best_n = k, n
        return _pos(best_key)
    return None
