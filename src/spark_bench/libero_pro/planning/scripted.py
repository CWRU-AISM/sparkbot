"""Scripted behaviour-tree plans from the BDDL pick/place hints: the LLM-free smoke and fixed-plan path."""
from __future__ import annotations

from typing import Optional

import numpy as np

_PLACE_KEYWORDS = {'basket', 'plate', 'table', 'cabinet', 'stove', 'rack', 'bowl'}


def _scripted_plan(labels, pick_hint, place_hint, instruction,
                    picks=None, places=None, det_map=None, prompts=None):
    """
    Scripted plan for the no-LLM path (--no-gemini) and for an empty LLM plan.

    When ``picks`` is supplied with >1 entry (libero_10 compound goals),
    emit a chained pick/place sequence - one pick->grasp->place->release
    cycle per pick.  The shared ``place`` is reused with a small
    increasing z-offset so successive drops don't stack.
    """
    il = (instruction or '').lower()

    # Push tasks are not pick-place: emit push_object with source +
    # destination labels so the primitive derives direction/distance from
    # perception (same shape the LLM planner emits).  Generic:
    # source = BDDL pick hint (or the first non-place label), target =
    # BDDL place hint matched against the detected labels.
    # Best-overlap label binding: argmax on (hint-word hits, prompt-table
    # priority).
    def _conf_of(l: str) -> float:
        if not det_map or l not in det_map:
            return 1.0
        try:
            return float(getattr(det_map[l], 'confidence', 1.0))
        except Exception:
            return 1.0

    def _kw_match(hint: str) -> Optional[str]:
        kws = [w for w in (hint or '').lower().replace('_', ' ').split()
                if len(w) > 2]
        if not kws:
            return None
        # Two-pass eligibility gate: a sub-noise detection must not win the
        # binding on word overlap alone. If nothing passes the gate, fall
        # back to the full list.
        for min_conf in (0.2, 0.0):
            best = None
            best_key = (0, 0.0, 0)
            for idx, l in enumerate(labels):
                if _conf_of(l) < min_conf:
                    continue
                lw = [w for w in l.lower().split() if w]
                hits = sum(1 for w in kws if w in l.lower())
                # Coverage: fraction of the LABEL's words the hint accounts
                # for.  Breaks hits-ties toward the label that is fully
                # explained by the hint ('stove' beats 'stove knob' for
                # hint 'flat_stove' - the knob is a different part).
                cover = (sum(1 for w in lw
                              if any(w in k or k in w for k in kws))
                         / max(len(lw), 1))
                key = (hits, cover, -idx)
                if hits and key > best_key:
                    best, best_key = l, key
            if best is not None:
                return best
        return None

    # Drawer-opening tasks: emit open_drawer (executor derives level +
    # pull direction from the instruction and perception).  A compound
    # "open ... and put X inside" additionally chains the pick + a place
    # bound to the drawer handle label - the executor's drawer-aware
    # placement retargets it to the pulled-out tray.
    if 'open' in il and any(k in il for k in ('drawer', 'layer', 'cabinet')):
        handle_lbl = next((l for l in labels if 'handle' in l.lower()),
                           (labels[0] if labels else None))
        children: list = [{'type': 'open_drawer',
                            'params': ({'keypoint_label': handle_lbl}
                                        if handle_lbl else {})}]
        pick_lbl = _kw_match(pick_hint) if pick_hint else None
        if pick_lbl is not None and handle_lbl is not None:
            children.extend([
                {'type': 'move_to_keypoint',
                 'params': {'keypoint_label': pick_lbl}},
                {'type': 'grasp', 'params': {'force': 100}},
                {'type': 'move_relative', 'params': {'dz': 0.20}},
                {'type': 'move_to_keypoint',
                 'params': {'keypoint_label': handle_lbl, 'offset_z': 0.10}},
                {'type': 'release'},
            ])
        return {'tree': {'type': 'sequence', 'children': children}}

    if 'push' in il:
        src = _kw_match(pick_hint)
        if src is None:
            for l in labels:
                if not any(kw in l.lower() for kw in _PLACE_KEYWORDS):
                    src = l
                    break
        tgt = _kw_match(place_hint)
        if src is not None:
            p: dict = {'keypoint_label': src}
            if tgt is not None:
                p['target_label'] = tgt
            return {'tree': {'type': 'sequence', 'children': [
                {'type': 'push_object', 'params': p},
            ]}}

    # Multi-pick chained plan.
    if picks and len(picks) > 1:
        # Match each pick keyword to the best label in the detection set.
        def _match(hint: str) -> Optional[str]:
            kws = [w for w in hint.lower().replace('_', ' ').split()
                    if len(w) > 2]
            if not kws:
                return None
            best = None
            best_hits = 0
            for l in labels:
                ll = l.lower()
                hits = sum(1 for w in kws if w in ll)
                if hits > best_hits:
                    best = l
                    best_hits = hits
            return best if best_hits else None

        pick_lbls = [_match(p) for p in picks]
        place_lbl = (_match(places[0]) if places else None) or (
            _match(place_hint) if place_hint else None)
        if not place_lbl:
            for l in labels:
                if any(kw in l.lower() for kw in _PLACE_KEYWORDS):
                    place_lbl = l
                    break
        children: list = []
        for i, pl in enumerate(pick_lbls):
            if pl is None or place_lbl is None:
                continue
            # Stagger z-offset across pairs so drops don't stack on top
            # of each other (basket interior is ~5cm deep - 0.05/0.08).
            z = 0.05 + 0.03 * i
            children.extend([
                {'type': 'move_to_keypoint',
                 'params': {'keypoint_label': pl}},
                {'type': 'grasp', 'params': {'force': 100}},
                {'type': 'move_relative', 'params': {'dz': 0.20}},
                {'type': 'move_to_keypoint',
                 'params': {'keypoint_label': place_lbl, 'offset_z': z}},
                {'type': 'release'},
            ])
        if children:
            return {'tree': {'type': 'sequence', 'children': children}}

    # Best-overlap label binding with a confidence-eligibility gate and
    # prompt-priority tie-break:
    #
    # * a low-confidence detection is not binding-grade evidence - a
    #   2-hit label at conf 0.25 (SAM3 ghost on a lookalike) must not
    #   outrank a 1-hit label at conf 0.89 that latched the real object,
    #   so labels below _BIND_MIN_CONF are ineligible while any eligible
    #   match exists;
    # * confidence is NOT a valid tie-break either - visually confusable
    #   distractors produce HIGH-confidence false positives ('Campbell
    #   soup can' at 0.95 on the tomato-sauce can).  The prompt table is
    #   curated most-discriminative-first, so among equal-hit labels the
    #   one EARLIEST in the emitted prompt list wins.  Priority comes
    #   from ``prompts`` (the ordered list the runner detected with),
    #   NOT det_map insertion order - wrist-camera-only detections are
    #   APPENDED by the merge, which silently demotes the most
    #   discriminative phrasing whenever only the wrist saw it.
    _BIND_MIN_CONF = 0.3
    # A word hit on a near-threshold detection is not evidence of an
    # object: a 0.03-confidence ghost phrase bound the scripted plan to
    # empty table (0 cm control, object task 2) while the real bottle sat
    # under a phrase with no hint words. The confidence-free pass keeps
    # a floor so the caller's own fallback wins over a ghost.
    _BIND_FALLBACK_MIN_CONF = 0.1
    prompt_rank = {p: i for i, p in enumerate(prompts or [])}

    def _rank(i, l):
        # Lower is better: prompt-table index, falling back to det_map
        # position for labels not in the prompt list.
        return prompt_rank.get(l, len(prompt_rank) + i)

    def _best_label(hint, exclude=()):
        if not hint:
            return None
        words = [w for w in hint.lower().replace('_', ' ').split()
                 if len(w) > 2]
        if not words:
            return None

        def _conf(l):
            if det_map is None:
                return None
            return float(getattr(det_map.get(l), 'confidence', 0.0) or 0.0)

        def _pass(min_conf):
            best, best_key = None, None
            for i, l in enumerate(labels):
                if l in exclude:
                    continue
                c = _conf(l)
                if min_conf is not None and c is not None and c < min_conf:
                    continue
                hits = sum(1 for w in words if w in l.lower())
                if hits == 0:
                    continue
                key = (-hits, _rank(i, l))
                if best_key is None or key < best_key:
                    best_key, best = key, l
            return best

        return _pass(_BIND_MIN_CONF) or _pass(_BIND_FALLBACK_MIN_CONF)

    def _hits(hint, l):
        words = [w for w in hint.lower().replace('_', ' ').split()
                 if len(w) > 2]
        return sum(1 for w in words if w in l.lower())

    def _trackable_alias(hint, l):
        # Adaptation-observability guard: binding
        # to a WRIST-ONLY detection (no agentview mask) makes the target
        # invisible to the agentview-based refresh loop - every pre-close
        # retarget silently disappears and only blind recovery remains.
        # If a CO-LOCATED (<=5 cm) hint-matching label exists WITH an
        # agentview mask, bind that label instead: same physical object,
        # chosen via the discriminative phrasing's position, tracked via
        # the phrasing the verification camera can actually re-detect.
        if det_map is None or l is None:
            return l
        d = det_map.get(l)
        if d is None or getattr(d, 'mask', None) is not None:
            return l
        p = getattr(d, 'position_3d', None)
        if p is None:
            return l
        p = np.asarray(p, dtype=float)[:3]
        best_alt, best_r = None, None
        for i, l2 in enumerate(labels):
            if l2 == l or _hits(hint, l2) == 0:
                continue
            d2 = det_map.get(l2)
            if d2 is None or getattr(d2, 'mask', None) is None:
                continue
            p2 = getattr(d2, 'position_3d', None)
            if p2 is None:
                continue
            if float(np.linalg.norm(
                    np.asarray(p2, dtype=float)[:3] - p)) > 0.05:
                continue
            r = _rank(i, l2)
            if best_r is None or r < best_r:
                best_r, best_alt = r, l2
        return best_alt or l

    pick = _trackable_alias(pick_hint, _best_label(pick_hint))
    place = _trackable_alias(place_hint, _best_label(place_hint))
    # A single label must not serve as both pick and place: prefer the pick
    # binding (BDDL pick hints are more specific) and rebind the place to
    # the next-best distinct label with the same ranking.
    if pick is not None and place == pick:
        alt = _best_label(place_hint, exclude={pick})
        place = _trackable_alias(place_hint, alt)
        if place == pick:
            place = alt

    if pick is None:
        for l in labels:
            if not any(kw in l.lower() for kw in _PLACE_KEYWORDS):
                pick = l; break
    if place is None:
        for l in labels:
            if l != pick:
                place = l; break

    if not pick or not place:
        return None

    return {
        'tree': {'type': 'sequence', 'children': [
            {'type': 'move_to_keypoint', 'params': {'keypoint_label': pick}},
            {'type': 'grasp', 'params': {'force': 100}},
            {'type': 'move_relative', 'params': {'dz': 0.20}},
            {'type': 'move_to_keypoint',
             'params': {'keypoint_label': place, 'offset_z': 0.05}},
            {'type': 'release'},
        ]}
    }
