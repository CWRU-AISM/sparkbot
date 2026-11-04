"""Relation binding: among twin pick targets, take the one the instruction's spatial relation names."""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

import numpy as np

from spark_core.binding import fuzzy_key
from spark_bench.libero_pro.bddl import _OBJ_VISUAL_PROMPTS
from spark_bench.libero_pro.perception.detect import DetectionResult, _detect_one_cam

log = logging.getLogger(__name__)

_REL_CLAUSE = re.compile(
    r"\b(between\s+(?:the\s+)?(?P<a>.+?)\s+and\s+(?:the\s+)?(?P<b>.+?)"
    r"|(?P<rel>next to|beside|on top of|on|in the (?:top|middle|bottom) (?:drawer|layer) of"
    r"|in|inside|under|from)\s+(?:the\s+)?(?P<x>.+?))"
    r"(?=\s+and\s+(?:place|put))")  # a pick relation is always followed by its own place clause


def _relation_choose(instances: list, instruction: str, pick_noun: str, locate) -> Optional[tuple]:
    """
    Pick the instance that satisfies the instruction's spatial relation, from 3D positions only.

    ``instances``: detections of the pick object (>= 2). ``locate(phrase)`` returns an anchor's 3D
    position or None. Returns (instance, rule) or None when the clause or an anchor is not found.
    """
    low = instruction.lower()
    i = low.find(pick_noun.lower())
    if i < 0:
        return None
    # the relation must describe the PICK object, i.e. follow its noun directly ("the bowl between ...");
    # searching further would read the place clause ("... and place it in the basket") as a pick relation
    m = _REL_CLAUSE.match(low[i + len(pick_noun):].lstrip())
    if not m:
        return None
    xy = lambda d: np.asarray(d.position_3d, float)[:2]
    if m.group('a'):
        pa, pb = locate(m.group('a')), locate(m.group('b'))
        if pa is None or pb is None:
            return None
        mid = (np.asarray(pa, float)[:2] + np.asarray(pb, float)[:2]) / 2
        return min(instances, key=lambda d: np.linalg.norm(xy(d) - mid)), 'between'
    rel, x = m.group('rel'), m.group('x')
    px = locate(x)
    if rel.strip() in ('on', 'on top of'):
        # "the bowl ON the ramekin / cookie box / cabinet". Trust the anchor unless it is missing or sits on
        # one of the pick instances themselves (SAM3's "ramekin" landed on the other black bowl in the
        # spatial swap layout); then the stacked instance is the one standing clear above its twins.
        # Height alone regressed the task-perturbed cell, where both bowls stand raised (spatial 5 task 0/10).
        # an anchor detection that IS a twin sits at the twin's own height; an object resting ON the anchor
        # stands above it (the bowl on the stove: the stove mask is right under it and must not count)
        # (5 cm / 2.5 cm: at 3 cm / 1.5 cm a 'ramekin' mask on the twin bowl still read as a real anchor and
        # bound the twin in 4/10 spatial 5 position trials)
        on_twin = px is not None and any(
            np.linalg.norm(xy(d) - np.asarray(px, float)[:2]) < 0.05
            and abs(float(d.position_3d[2]) - float(px[2])) < 0.025 for d in instances)
        if on_twin:
            # Either the anchor's mask landed on a twin (ramekin case), or a generic 'bowl' phrase found the
            # anchor itself (the dark stove burner): set the coinciding instances aside, and take one that
            # rests on the anchor (within 15 cm, above it) if there is one.
            pa = np.asarray(px, float)
            rest = [d for d in instances if np.linalg.norm(xy(d) - pa[:2]) >= 0.05]
            on_it = [d for d in rest if np.linalg.norm(xy(d) - pa[:2]) < 0.15
                     and float(d.position_3d[2]) > float(pa[2]) + 0.015]
            if on_it:
                return min(on_it, key=lambda d: np.linalg.norm(xy(d) - pa[:2])), 'on:rests'
        # a missing anchor leaves the binding alone: 'highest' then took the bowl on the ramekin for 'the bowl
        # on the cookie box' (the box is hidden under its bowl) in 10/10 spatial 5 task trials
        if on_twin:
            zs = sorted(instances, key=lambda d: float(d.position_3d[2]), reverse=True)
            if len(zs) >= 2 and float(zs[0].position_3d[2]) - float(zs[1].position_3d[2]) > 0.025:
                return zs[0], 'on:highest'
    if px is None:
        return None
    px = np.asarray(px, float)
    near = instances
    if rel in ('next to', 'beside'):  # "next to" is never the anchor's own spot; "on X" is exactly there
        near = [d for d in instances if np.linalg.norm(xy(d) - px[:2]) > 0.02] or instances
    if 'drawer' in rel or 'layer' in rel:  # LIBERO-PRO says 'top layer of the cabinet'
        # drawers stack vertically inside the cabinet: an object resting ON the cabinet is above the
        # cabinet's (mask-centroid) height and is not "in" it; among the ones inside, take by height
        close = sorted(instances, key=lambda d: np.linalg.norm(xy(d) - px[:2]))[:3]
        inside = [d for d in close if float(d.position_3d[2]) < float(px[2]) + 0.02] or close
        pick = max if 'top' in rel else min
        return pick(inside, key=lambda d: float(d.position_3d[2])), rel
    return min(near, key=lambda d: np.linalg.norm(xy(d) - px[:2])), rel


def relation_bind_pick(det: DetectionResult, target_hint: str, instruction: str, cfg, sam3=None,
                       trial_meta: Optional[dict] = None) -> None:
    """
    Bind the pick target to the instance that satisfies the instruction's spatial relation (cfg.relation_bind).

    With twin objects (the spatial suite's two black bowls) position-sticky binding holds an identity but
    cannot know which instance the instruction means ("the black bowl between the plate and the ramekin").
    This collects every instance of the pick target (re-running SAM3 on the frame in hand when the first
    detection kept one), then resolves the relation from 3D positions (``_relation_choose``). Leaves the
    binding alone when there is one instance, no relation clause or no anchor. Mutates ``det.det_map`` and
    records ``trial_meta['instance_disambig']``.
    """
    if not getattr(cfg, 'relation_bind', False) or not target_hint:
        return
    key = fuzzy_key(det.det_map, target_hint)
    if key is None:
        # BDDL hints ('akita_black_bowl_1') often share no substring with
        # prompt phrases; retry on normalized token overlap.
        toks = {t for t in target_hint.lower().replace('_', ' ').split()
                if len(t) >= 3}
        best, best_n = None, 0
        for k in det.det_map:
            n = len(toks & set(str(k).lower().split()))
            if n > best_n:
                best, best_n = k, n
        key = best
    if key is None:
        # Detected under the object's curated SAM3 phrase rather than its name: try those phrases too.
        obj = re.sub(r'_\d+$', '', target_hint.lower().replace(' ', '_'))
        for ph in _OBJ_VISUAL_PROMPTS.get(obj, []):
            key = fuzzy_key(det.det_map, ph)
            if key is not None:
                break
    if key is None:
        return
    label = getattr(det.det_map[key], 'label', key)

    instances = [d for d in det.dets
                 if d.label == label and d.position_3d is not None
                 and getattr(d, 'mask', None) is not None]
    if len(instances) < 2 and det.depth is not None:
        # Initial detection kept only top-1: re-run this ONE prompt in
        # multi-instance mode against the frame already in hand.
        try:
            # relation-bind chooses among instances by geometry, so keep weak ones as candidates: the
            # 0.50 secondary-instance floor dropped most of a row of look-alike items
            dets2 = _detect_one_cam(
                sam3, det.rgb, det.depth, [label], det.cam,
                cfg.cam_width, cfg.cam_height, {label}, secondary_score=0.15)
            instances = [d for d in dets2
                         if d.position_3d is not None
                         and getattr(d, 'mask', None) is not None]
        except Exception as e:
            log.warning("relation bind: multi-instance redetect failed: %s", e)
            return
    # Twin objects often land under different phrasings ("black bowl" on one, "bowl" on the other):
    # every detection sharing the head noun is an instance, merged by 3D position.
    head = label.split()[-1].lower() if label.split() else label.lower()
    for d in det.dets:
        if (head in str(d.label).lower().split() and d.position_3d is not None
                and all(np.linalg.norm(np.asarray(d.position_3d) - np.asarray(i.position_3d)) > 0.04
                        for i in instances)):
            instances.append(d)
    if len(instances) < 2 and det.depth is not None:
        # The instruction names a relation, so twins exist, but the specific phrase ("akita black bowl")
        # found only one: retry the object's other phrases and the head noun ("black bowl", "bowl"), which
        # find both. Real-PRO spatial task cells: 1 instance in 8/10 trials -> wrong bowl, 2/10.
        obj = re.sub(r'_\d+$', '', target_hint.lower().replace(' ', '_'))
        for ph in dict.fromkeys(_OBJ_VISUAL_PROMPTS.get(obj, []) + [head]):
            if len(instances) >= 2:
                break
            if ph == label:
                continue
            try:
                more = _detect_one_cam(sam3, det.rgb, det.depth, [ph], det.cam, cfg.cam_width, cfg.cam_height,
                                       {ph}, secondary_score=0.15)
            except Exception:
                continue
            for d in more:
                if (d.position_3d is not None and getattr(d, 'mask', None) is not None
                        and all(np.linalg.norm(np.asarray(d.position_3d) - np.asarray(i.position_3d)) > 0.04
                                for i in instances)):
                    instances.append(d)
    if len(instances) < 2:
        if trial_meta is not None:
            trial_meta.setdefault('instance_disambig', []).append(
                {'label': key, 'n_instances': len(instances), 'how': 'geometry:skipped (one instance)'})
        return  # nothing to disambiguate

    # LLM-free: resolve "between A and B" / "next to X" / "on X" / "in the top drawer of X" from 3D positions.
    def locate(phrase):
        k = fuzzy_key(det.det_map, phrase)
        if k is None:
            alias = getattr(cfg, '_adaptive_concept_to_phrase', None) or {}
            ck = fuzzy_key(alias, phrase)
            if ck is not None:
                k = fuzzy_key(det.det_map, alias[ck])
        if k is not None and getattr(det.det_map[k], 'position_3d', None) is not None:
            return det.det_map[k].position_3d
        try:
            found = _detect_one_cam(sam3, det.rgb, det.depth, [phrase], det.cam, cfg.cam_width,
                                    cfg.cam_height, set())
            found = [f for f in found if f.position_3d is not None]
            return max(found, key=lambda f: float(f.confidence)).position_3d if found else None
        except Exception:
            return None
    noun = label if label.lower() in instruction.lower() else target_hint.replace('_', ' ')
    _seen = {}

    def locate_logged(phrase):
        p_ = locate(phrase)
        _seen[phrase] = None if p_ is None else [round(float(v), 3) for v in p_]
        return p_
    res = _relation_choose(instances, instruction, noun, locate_logged)
    if res is None and trial_meta is not None:
        trial_meta.setdefault('instance_disambig', []).append({
            'label': key, 'n_instances': len(instances), 'how': 'geometry:skipped (no relation or anchor not found)'})
    if res is not None:
        chosen, rule = res
        # the planner may name the twin by any of its phrasings ("black bowl", "bowl"): point them all at
        # the chosen instance, so no label is left on the distractor
        for k in list(det.det_map):
            if head in str(k).lower().split():
                det.det_map[k] = chosen
        det.det_map[key] = chosen
        if trial_meta is not None:
            trial_meta.setdefault('instance_disambig', []).append({
                'label': key, 'n_instances': len(instances), 'how': f'geometry:{rule}', 'anchors': _seen,
                'instances': [[round(float(v), 3) for v in d.position_3d] for d in instances],
                'pos': [round(float(x), 3) for x in chosen.position_3d], 't': time.time()})
        if cfg.verbose:
            log.info("relation bind %r: %d instances, %s -> %s", key, len(instances), rule,
                     np.round(chosen.position_3d, 3))

