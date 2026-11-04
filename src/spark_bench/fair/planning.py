"""The runner's planner call and the post-parse label validator."""
from __future__ import annotations

import logging
from typing import Optional

from spark_core.bt_validator import validate_and_repair_bt
from spark_bench.fair.config import FairConfig
from spark_bench.libero_pro.planning import _scripted_plan, plan_llm

log = logging.getLogger(__name__)


def _plan(cfg: FairConfig, instruction: str, det_map: dict, rgb, pick_hint: str, place_hint: str,
          picks: Optional[list] = None, places: Optional[list] = None, pick_counts: Optional[list] = None,
          prompts: Optional[list] = None):
    """The LLM plan, or the scripted plan under --no-gemini. ``picks`` / ``places`` / ``pick_counts``
    carry compound goals ("put both X and Y in Z")."""
    labels = list(det_map.keys())
    if cfg.no_gemini:
        return _scripted_plan(labels, pick_hint, place_hint, instruction,
                              picks=picks, places=places, det_map=det_map, prompts=prompts)
    return plan_llm(instruction, labels, pick_hint, place_hint, det_map=det_map, rgb_image=rgb,
                    no_bddl_hints=cfg.no_bddl_hints, picks=picks, places=places, pick_counts=pick_counts)


def _run_label_validator(cfg: FairConfig, score: dict, det_map: dict, instruction: str, rgb,
                         pick_hint: str, place_hint: str, trial_meta: Optional[dict] = None,
                         slot_key: str = 'validator_actions', picks: Optional[list] = None,
                         places: Optional[list] = None, pick_counts: Optional[list] = None) -> dict:
    """
    Snap BT labels that are not in ``det_map`` to the nearest detected label (Levenshtein cutoff 0.4);
    any label left over triggers ONE replan with a STRICT preamble listing the allowed labels. Records the
    snap / retry events in ``trial_meta[slot_key]``.
    """
    if not isinstance(score, dict) or not det_map:
        return score
    allowed = set(det_map.keys()) | {'none'}

    def _retry_fn(suffix: str) -> Optional[dict]:
        return _plan(cfg, suffix + "\n\n" + instruction, det_map, rgb, pick_hint, place_hint,
                     picks=picks, places=places, pick_counts=pick_counts)

    repaired, actions = validate_and_repair_bt(
        score, allowed, retry_fn=_retry_fn, log=log.info if cfg.verbose else (lambda _msg: None))
    if trial_meta is not None and actions:
        trial_meta.setdefault(slot_key, []).extend(actions)
    return repaired
