"""Wrist verification: re-ground a weakly detected pick target from the home-pose wrist image."""
from __future__ import annotations

import logging
import re
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from spark_core.planner import SPARKPlanner
from spark_core.sam3 import ObjectDetection
from spark_bench.libero_pro.bddl import _OBJ_VISUAL_PROMPTS
from spark_bench.libero_pro.perception.detect import DetectionResult, redetect_wrist

log = logging.getLogger(__name__)

_VERIFY_GENERIC = ('grocery item', 'packaged food', 'container')
_VERIFY_STOP = {'the', 'a', 'an', 'of', 'with', 'and', 'box', 'bottle', 'can', 'cup', 'container', 'package',
                'small', 'large', 'short', 'tall', 'red', 'white', 'dark', 'brown', 'yellow', 'cylinder'}


def _crop_grid(crops: list, size: int = 200) -> Image.Image:
    """Numbered, enlarged (rgb, mask) crops, 4 per row: printed labels stay legible to the VLM, unlike the full frame."""
    tiles = []
    for k, (rgb, mm) in enumerate(crops, 1):
        ys, xs = np.where(mm)
        y0, y1 = max(0, ys.min() - 12), min(rgb.shape[0], ys.max() + 12)
        x0, x1 = max(0, xs.min() - 12), min(rgb.shape[1], xs.max() + 12)
        t = cv2.resize(np.ascontiguousarray(rgb[y0:y1, x0:x1]), (size, size), interpolation=cv2.INTER_CUBIC)
        t = cv2.copyMakeBorder(t, 34, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(t, str(k), (size // 2 - 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        tiles.append(t)
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 255))
    return Image.fromarray(np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))


def wrist_verify_picks(env, sam3, det: DetectionResult, picks: list, cfg,
                       trial_meta: Optional[dict] = None, conf_ok: float = 0.5) -> None:
    """
    Re-ground a weakly detected pick target from the wrist camera (cfg.wrist_verify).

    A short object behind a tall one is invisible to the agentview, and SAM3 then puts
    the task's words on a look-alike neighbour. From the home pose the wrist camera sees
    the whole table from above. When the pick name has no agentview detection at
    conf >= ``conf_ok``, SAM3 proposes generic object masks on the wrist image, the VLM
    picks the numbered enlarged crop that is the named object, and that detection
    replaces every pick-related label (they were bound to the neighbour).
    Fail-open: any miss leaves det untouched. Mutates det.det_map / det.dets.
    """
    meta = trial_meta.setdefault('wrist_verify', []) if trial_meta is not None else []
    for pick in picks or []:
        name = pick.replace('_', ' ')
        cur = det.det_map.get(name)
        if cur is not None and float(cur.confidence) >= conf_ok:
            continue
        rec = {'name': name, 'agent_conf': None if cur is None else round(float(cur.confidence), 3)}
        meta.append(rec)
        fresh = redetect_wrist(env, sam3, list(_VERIFY_GENERIC), cfg, all_instances=True)
        if fresh is None:
            rec['error'] = 'no_wrist_dets'
            continue
        cands = []
        for d in fresh.dets:
            mm = getattr(d, 'mask', None)
            if (d.position_3d is None or mm is None or float(d.confidence) < 0.2
                    or not 150 <= int(np.asarray(mm).sum()) <= 0.25 * np.asarray(mm).size):
                continue
            if any(np.linalg.norm(np.asarray(d.position_3d) - np.asarray(c.position_3d)) < 0.03 for c in cands):
                continue
            cands.append(d)
        cands = cands[:7]
        # The agentview's own binding competes as crop 1: a weak but correct detection (the object is
        # under the parked gripper, out of the wrist view) must be keepable, else the VLM picks a look-alike.
        crops = [(fresh.rgb, np.asarray(c.mask).astype(bool)) for c in cands]
        if cur is not None and getattr(cur, 'mask', None) is not None and np.asarray(cur.mask).any():
            cands = [cur] + cands
            crops = [(det.rgb, np.asarray(cur.mask).astype(bool))] + crops
        rec['n_cands'] = len(cands)
        if not cands:
            continue
        ask = (f"The image is a grid of numbered close-up crops of objects on a table. "
               f"Which number is the {name}? Read printed labels on the packages. "
               f"Finish with a line ANSWER: <number>, using 0 if none of them is the {name}.")
        try:
            txt = SPARKPlanner().generate(
                [ask, _crop_grid(crops)],
                tag='wrist_verify', temperature=0)
        except Exception as e:  # noqa: BLE001 - verification must never break a trial
            rec['error'] = f'vlm: {e}'
            continue
        m = re.findall(r'ANSWER:\s*(\d+)', txt or '')
        k = int(m[-1]) if m else 0
        rec['answer'] = k
        if not 1 <= k <= len(cands):
            continue
        chosen = cands[k - 1]
        if chosen is cur:
            rec['kept_agentview'] = True
            continue
        pos = np.asarray(chosen.position_3d, dtype=float).copy()
        new = ObjectDetection(label=name, confidence=float(chosen.confidence),
                              centroid_2d=chosen.centroid_2d, mask_area=chosen.mask_area,
                              depth_meters=chosen.depth_meters, position_3d=pos, camera='wrist')
        new.position_agentview = pos.copy()
        words = {w for ph in [name] + list(_OBJ_VISUAL_PROMPTS.get(pick, []))
                 for w in ph.lower().split() if w not in _VERIFY_STOP}
        stale = [lbl for lbl in det.det_map if lbl != name and words & set(lbl.lower().split())]
        for lbl in stale:
            det.det_map.pop(lbl, None)
        det.dets = [d for d in det.dets if d.label not in stale and d.label != name] + [new]
        det.det_map[name] = new
        rec.update(pos=[round(float(x), 3) for x in pos], dropped=stale)
        if cfg.verbose:
            log.info("wrist verify %r -> crop %d at %s (dropped %s)", name, k, rec['pos'], stale)

