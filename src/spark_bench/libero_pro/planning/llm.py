"""LLM behaviour-tree plan: the instruction plus detected 3D positions and the agentview image go to
SPARKPlanner. A planner error fails the trial (returns None); it is never replaced by a scripted plan."""
from __future__ import annotations

import logging
from typing import Optional

import yaml
from PIL import Image

from spark_core.planner import SPARKPlanner
from spark_bench.libero_pro.planning.scripted import _scripted_plan

log = logging.getLogger(__name__)

_COMPOUND_MARKERS = (' and ', ' then ', ' both ', ' two ', ' three ', ' each ', 'first ')

_PUSH_HINT = (
    '\nFor push_object: when the task is to push one object TO/'
    'TOWARD another (e.g. "push the plate to the front of the '
    'stove"), emit BOTH keypoint_label (the source to push) and '
    'target_label (a SAM3 detection label of the destination). '
    'The primitive infers push_direction and push_distance from '
    'the two 3D positions, so push_direction/push_distance are '
    'optional. Example:\n'
    '  - type: push_object\n'
    '    params:\n'
    '      keypoint_label: "plate"\n'
    '      target_label: "stove"\n'
    'Only fall back to push_direction/push_distance when no '
    'destination label exists in the detected objects.')


def _count_actions(node) -> int:
    if node.get('type') in ('sequence', 'selector'):
        return sum(_count_actions(c) for c in node.get('children', []))
    return 1


def _hint_instruction(instruction: str, pick_hint: str, place_hint: str, det_map: Optional[dict],
                      no_bddl_hints: bool, picks, places, pick_counts) -> tuple[str, bool]:
    """The planner's task text: instruction, BDDL hints (unless no_bddl_hints), 3D positions, and the
    compound and push guidance. Returns (text, has_compound)."""
    hint = instruction
    has_compound = (picks is not None and places is not None
                    and (len(picks) > 1 or bool(pick_counts and any(c > 1 for c in pick_counts))))
    if not no_bddl_hints and has_compound:
        pcs = pick_counts or [1] * len(picks)
        place_for_pick = places[0] if len(places) == 1 else None
        step_lines = []
        for i, p in enumerate(picks):
            cnt = pcs[i] if i < len(pcs) else 1
            q = place_for_pick if place_for_pick is not None else (
                places[i] if i < len(places) else (places[-1] if places else 'none'))
            q_clean = q.replace('_', ' ') if q != 'none' else 'destination'
            step_lines.append(f'  {i + 1}. {p.replace("_", " ")}{f" (x{cnt})" if cnt > 1 else ""} -> {q_clean}')
        hint += ('\nCOMPOUND GOAL - emit a sequence covering EVERY pair:\n'
                 + '\n'.join(step_lines)
                 + '\nFor each pair: move_to_keypoint(pick) -> grasp -> '
                 'move_to_keypoint(place, offset_z>=0.10, vary z_offset '
                 'across pairs to avoid stacking) -> release. Chain ALL '
                 'pairs in a single sequence (do NOT stop at the first).')
    elif not no_bddl_hints:
        for name, prefix in ((pick_hint, 'Pick object'), (place_hint, 'Place target')):
            if name and name != 'none':
                clean = name.replace('_', ' ')
                if clean not in instruction.lower():
                    hint += f'. {prefix}: {clean}'
    if det_map:
        pos_lines = [f'  {lbl}: ({d.position_3d[0]:.3f}, {d.position_3d[1]:.3f}, {d.position_3d[2]:.3f})'
                     for lbl, d in det_map.items() if d.position_3d is not None]
        if pos_lines:
            hint += '\nDetected object positions (x,y,z meters):\n' + '\n'.join(pos_lines)
    # Clause count from the instruction text only (not BDDL), added even with no_bddl_hints: without it a
    # perception miss can drop one object and the planner emits only the visible-object phase.
    n_markers = sum(1 for m in _COMPOUND_MARKERS if m in instruction.lower())
    if n_markers >= 1 or has_compound:
        hint += (f'\nCOMPOUND TASK ({max(n_markers + 1, 2)}+ clauses detected from '
                 'instruction text). You MUST emit ONE sequence containing '
                 'EVERY clause as its own sub-chain. Even if perception '
                 'currently shows only one of the mentioned objects, the BT '
                 'must include ALL clauses - the executor re-perceives '
                 'between primitives. Emitting only one phase (e.g. just '
                 'pick, or just close) guarantees BDDL failure.')
    if 'push' in instruction.lower():
        hint += _PUSH_HINT
    return hint, has_compound


def plan_llm(instruction: str, labels: list, pick_hint: str = '', place_hint: str = '',
             det_map: Optional[dict] = None, rgb_image=None, no_bddl_hints: bool = False,
             picks: Optional[list] = None, places: Optional[list] = None,
             pick_counts: Optional[list] = None) -> Optional[dict]:
    """YAML BT from the planner. An empty plan falls back to the scripted plan; an error returns None."""
    try:
        planner = SPARKPlanner()
        hint, has_compound = _hint_instruction(instruction, pick_hint, place_hint, det_map, no_bddl_hints,
                                               picks, places, pick_counts)
        detection_details = []
        for lbl, det in (det_map or {}).items():
            dd = {'label': lbl, 'confidence': det.confidence}
            if det.position_3d is not None:
                p = det.position_3d
                dd['position_3d'] = p.tolist() if hasattr(p, 'tolist') else list(p)
            detection_details.append(dd)
        image = None if rgb_image is None else (
            rgb_image if isinstance(rgb_image, Image.Image) else Image.fromarray(rgb_image))
        score = planner.generate_score(hint, annotated_image=image, keypoint_labels=labels,
                                       detection_details=detection_details, temperature=0.0)
        for u in getattr(planner, 'usage_log', []):
            log.info("llm usage %s", u)
        tree = score.get('tree', {})
        if _count_actions(tree) == 0:
            return _scripted_plan(labels, pick_hint, place_hint, instruction,
                                  picks=picks, places=places, det_map=det_map)
        raw_yaml = yaml.safe_dump(score, sort_keys=False)
        score.setdefault('__raw_yaml', raw_yaml)
        score.setdefault('__planner', f"{planner.llm_backend}:{planner.model}")
        log.info("bt actions=%d compound=%s picks=%s places=%s yaml=%s", _count_actions(tree), has_compound,
                 picks, places, raw_yaml[:240].replace('\n', ' | '))
        return score
    except Exception as e:  # noqa: BLE001 - a planner error fails this trial, not the run
        log.error("planner error, trial fails: %s", e)
        return None
