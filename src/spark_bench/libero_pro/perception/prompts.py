"""SAM3 prompt selection: VLM-proposed prompts and adaptive phrasings scored by SAM3."""
from __future__ import annotations

import json
import logging
from typing import Optional

import numpy as np
import torch
from PIL import Image

from spark_core.planner import SPARKPlanner
from spark_bench.libero_pro.perception.detect import get_sam3

log = logging.getLogger(__name__)


def _vlm_prompts(rgb: np.ndarray, instruction: str, cfg, prompts: list[str]) -> list[str]:
    """SAM3 prompts proposed by the planner VLM from the instruction and the agentview image."""
    try:
        gp = SPARKPlanner().generate_prompts(instruction, scene_image=Image.fromarray(rgb))
        if getattr(cfg, 'language_only', False) and gp and isinstance(gp, list):
            # language-only: the VLM's phrases plus the instruction's own noun phrases (relation anchors
            # the VLM may drop, e.g. "wooden cabinet"); both come from the language, nothing from BDDL
            merged = list(dict.fromkeys([g for g in gp if isinstance(g, str)] + list(prompts)))[:10]
            if cfg.verbose:
                log.info("vlm prompts (language-only) %s", merged)
            return merged
        if gp and isinstance(gp, list) and len(gp) >= 2:
            if cfg.verbose:
                log.info("vlm prompts %s", gp)
            return gp[:10]
    except Exception as e:
        log.warning("vlm prompts failed, using static prompts: %s", e)
    return prompts


def _adaptive_variants(rgb: np.ndarray, instruction: str, cfg,
                        prompts: list[str]) -> list[str]:
    """
    Self-consistency: ask the planner VLM for K variants per concept, pick the
    variant whose SAM3 score (cardinality * confidence) is highest.
    """
    if not prompts:
        return prompts
    try:
        k = int(getattr(cfg, 'adaptive_k', 3))
        planner = SPARKPlanner()
        variants_prompt = (
            f"You are helping an open-vocabulary object detector. For each "
            f"concept below (one per line), output {k} distinct short "
            f"detection phrases (1-4 words each) that single out THAT object "
            f"among similar ones; the first phrase must be the object's name "
            f"exactly as written in the task text when the task names it. "
            f"Vary colors, shapes, materials, printed labels. "
            f"Task context: \"{instruction}\"\n\n"
            "Concepts:\n" + "\n".join(f"- {p}" for p in prompts[:10]) +
            "\n\nOutput JSON only, no code fences:\n"
            "{\"concept1\": [\"phrase1\", \"phrase2\", ...], ...}"
        )
        try:
            # same model as the planner (SPARK_GEMINI_MODEL, else the planner default): the preview
            # model this used to pin drifts between weeks and stopped naming objects distinctly
            txt = planner.generate(variants_prompt, tag='adaptive_variants',
                                   temperature=0, json_only=True).strip()
            if '```' in txt:
                txt = txt.split('```', 2)[1]
                if txt.startswith('json'):
                    txt = txt[4:]
            variants_by_concept = json.loads(txt.strip())
        except Exception as e:
            log.warning("adaptive prompts: variant generation failed: %s", e)
            return prompts

        if not variants_by_concept:
            return prompts
        chosen = _score_variants(rgb, variants_by_concept, k, instruction=instruction)
        # remember which phrase stands for each instruction concept, so relation anchors named in the
        # instruction ("the ramekin") can be found under the phrase that was detected ("small ceramic cup")
        cfg._adaptive_concept_to_phrase = {
            c: v for c, (_, v, _) in zip([c for c, vs in variants_by_concept.items() if isinstance(vs, list)], chosen)}
        chosen_phrases = _dedupe_chosen(chosen, instruction=instruction)
        # the phrase sweeps are the first thing to read when a task binds the wrong object
        log.info("adaptive variants=%s chosen=%s kept=%s", json.dumps(variants_by_concept)[:400],
                 json.dumps([(round(s, 2), v) for s, v, _ in chosen]), chosen_phrases)
        return chosen_phrases[:10] if chosen_phrases else prompts
    except Exception as e:
        log.warning("adaptive prompts failed, keeping the input prompts: %s", e)
        return prompts


def _score_variants(rgb: np.ndarray, variants_by_concept: dict, k: int,
                    instruction: str = ''):
    """
    Return [(score, phrase, centroid)] one entry per concept.

    The curated concept phrase (the task's own object name) wins whenever SAM3 detects
    anything for it; the variants are only a fallback for a name that does not fire.
    Descriptive variants ("Campbell soup can") score higher on a look-alike neighbour
    than the true name scores on a half-occluded target, and nothing in the image can
    tell those apart without ground truth (object task 0: 42/50 with the name, 0/5
    with the best-scoring variant).
    """
    preflight = get_sam3()
    preflight.load_models(load_da3=False)
    # bf16 autocast as in sim_detect: SAM3 main crashes on mixed dtypes without it.
    def bf16():
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    with bf16():
        state = preflight._sam3.set_image(Image.fromarray(rgb))
    chosen = []
    for concept, variants in variants_by_concept.items():
        if not isinstance(variants, list):
            continue
        best = (-1.0, concept, None)
        for v in [concept] + [x for x in variants[:k] if x != concept]:
            if not isinstance(v, str):
                continue
            with bf16():
                st = preflight._sam3.set_text_prompt(prompt=v, state=state)
            m = st.get('masks', torch.tensor([]))
            s = st.get('scores', torch.tensor([]))
            if m.numel() == 0:
                score, centroid = 0.0, None
            else:
                n = m.shape[0]
                conf = float(s.max().item()) if s.numel() else 0.0
                # Prefer exactly 1 detection; penalise >2 (e.g. "knob" -> many).
                card_mult = 1.0 if n == 1 else (0.6 if n == 2 else 0.1)
                score = card_mult * conf
                best_idx = int(s.argmax().item())
                msk = m[best_idx].cpu().numpy().squeeze().astype(bool)
                if msk.any():
                    ys, xs = np.where(msk)
                    centroid = (float(xs.mean()), float(ys.mean()))
                else:
                    centroid = None
            if score > best[0]:
                best = (score, v, centroid)
            if v == concept and centroid is not None:
                break  # ponytail: the name fired, keep it; a confidence floor here would need GT to tune
        chosen.append(best)
    return chosen


def _dedupe_chosen(chosen: list, px_tol: float = 30.0, instruction: str = '') -> list[str]:
    """
    Drop duplicates whose mask centroids are within ``px_tol`` of each other.

    Of two phrases on one object, keep the one written in the task text, else the
    higher score: the planner looks for the task's words, and when "bbq sauce" is
    dropped for "dark brown sauce bottle" it takes "bbq sauce bottle" on the ketchup.
    """
    named = lambda v: v.lower() in instruction.lower()
    deduped: list[tuple[float, str, Optional[tuple[float, float]]]] = []
    for s_i, v_i, c_i in chosen:
        if c_i is None:
            deduped.append((s_i, v_i, c_i))
            continue
        is_dup = False
        for j, (s_j, v_j, c_j) in enumerate(deduped):
            if c_j is not None and np.hypot(c_i[0] - c_j[0],
                                              c_i[1] - c_j[1]) < px_tol:
                is_dup = True
                if (named(v_i), s_i) > (named(v_j), s_j):
                    deduped[j] = (s_i, v_i, c_i)
                break
        if not is_dup:
            deduped.append((s_i, v_i, c_i))
    return [v for _, v, _ in deduped]


def select_prompts(rgb: np.ndarray, instruction: str, cfg, prompts: list[str]) -> list[str]:
    """The SAM3 prompt list: the task prompts, optionally replaced by VLM proposals, then narrowed to one
    adaptive phrasing per object."""
    if getattr(cfg, 'use_gemini_prompts', False):
        prompts = _vlm_prompts(rgb, instruction, cfg, prompts)
    if getattr(cfg, 'adaptive_prompts', False):
        prompts = _adaptive_variants(rgb, instruction, cfg, prompts)
    return prompts
