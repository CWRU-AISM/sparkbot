"""Task prompt contracts: count-gated detection with prompt escalation (cache, alternates, LLM)."""

import logging
from typing import Dict, List

from spark_real.perception.prompt_registry import (
    TaskSpec,
    base_label,
    load_registry,
)

logger = logging.getLogger(__name__)


class TaskPromptMixin:
    """Task prompt contracts: count-gated detection with prompt escalation (cache, alternates, LLM)."""

    def prompt_registry(self):
        """
        Lazily-built, process-wide task prompt registry.

        Built once and cached on the pipeline instance. The directory is
        `perception.task_prompts_dir` from the family YAML when set, else
        $SPARK_TASK_PROMPTS, else the packaged configs/tasks.
        """
        registry = getattr(self, "_prompt_registry", None)
        if registry is None:
            config = getattr(self, "config", None)
            registry = load_registry(getattr(config, "task_prompts_dir", None))
            self._prompt_registry = registry
        return registry

    def task_spec(self, instruction: str):
        """
        Resolve an instruction to its frozen perception contract, or None.

        A hit means the prompts, the expected object counts and the instance
        ordering for this task are all pinned in version control, so no LLM
        is consulted and the labelling is reproducible run to run. A miss is
        not an error: the caller falls through to its existing prompt path.
        """
        if not instruction:
            return None
        return self.prompt_registry().lookup(instruction)

    def detect_for_task(
        self,
        captures: Dict,
        instruction: str,
        spec: "TaskSpec" = None,
        prompts: List[str] = None,
    ):
        """
        Detect + merge under a task's registered perception contract.

        Returns ``(merged, all_detections, resolution, spec, prompts_used)``.
        ``resolution`` is None when the instruction is not registered, in which
        case this is exactly the old detect-then-merge with no count checking.

        Escalation when the declared object counts are not met, cheapest
        first, stopping as soon as the gate passes:

        1. Re-run with the task's relaxed secondary-mask score. The usual
           cause of a missing second instance is the global 0.50 floor, and
           this costs one more SAM3 pass over an already-encoded image.
        2. Re-run with the group's alt_prompts appended. Covers the case where
           the vocabulary, not the threshold, is wrong. Alt prompts are
           relabelled to their canonical group name during resolution, so
           downstream names are unchanged.
        3. Ask the planner LLM to look at the frame and propose prompts for
           the still-missing groups (``fallback.llm_propose_prompts``). Covers
           the viewpoint-dependent case the task author could not anticipate.
           Proposals are folded into the spec as alt_prompts, so they are
           relabelled to the canonical name exactly like rung 2's.
        4. Apply ``fallback.on_mismatch``: ``abort`` raises
           :class:`PromptCountMismatch`, ``operator_click`` and
           ``best_effort`` return with ``resolution.ok`` False for the caller
           to route to a point prompt or to proceed anyway.

        ``abort`` is the default for the demo task set: a mislabelled episode
        silently poisons a training set.
        """
        if spec is None:
            spec = self.task_spec(instruction)

        if spec is None:
            used = prompts or self._extract_prompts(instruction)
            all_dets = self.detect(captures, used, multi_instance=True)
            return self.merge_detections(all_dets), all_dets, None, None, used

        base_prompts = list(prompts or spec.prompts)
        multi = spec.multi_instance_prompts

        def _pass(prompt_list, secondary):
            all_dets = self.detect(
                captures,
                prompt_list,
                multi_instance_prompts=multi | (set(prompt_list) & multi),
                secondary_score=secondary,
            )
            merged = self.merge_detections(all_dets, spec=spec)
            return merged, all_dets, self._last_resolution

        merged, all_dets, resolution = _pass(base_prompts, None)
        used = base_prompts

        relax = spec.fallback.relax_secondary_score
        if not resolution.ok and relax is not None:
            logger.warning(
                "task %r: count gate failed, retrying with secondary score %.2f",
                spec.task,
                relax,
            )
            merged, all_dets, resolution = _pass(base_prompts, relax)

        if not resolution.ok and resolution.retry_prompts:
            retry = base_prompts + [p for p in resolution.retry_prompts if p not in base_prompts]
            logger.warning(
                "task %r: count gate still failed, retrying with alt prompts %s",
                spec.task,
                resolution.retry_prompts,
            )
            # Alt prompts can themselves be the multi-instance ones.
            multi = multi | {alt for g in spec.groups if g.multi_instance for alt in g.alt_prompts}
            merged, all_dets, resolution = _pass(retry, relax)
            used = retry

        # Learned-prompts rung: phrases that rescued THIS task+group on an
        # earlier run (spark_bench's tuned-prompts tier; see
        # perception/prompt_cache.py). Costs one SAM3 pass and no API call,
        # so it sits between alt_prompts and the LLM. Cached phrases are
        # re-validated by the count gate like any alt prompt -- a stale one
        # simply fails through to the LLM rung below.
        if not resolution.ok:
            cached = self._cached_prompt_proposals(spec, resolution, used)
            if cached:
                spec = spec.with_alt_prompts(cached)
                extra = [p for ps in cached.values() for p in ps if p not in used]
                retry = used + extra
                logger.warning(
                    "task %r: count gate still failed, retrying with " "learned prompts %s",
                    spec.task,
                    extra,
                )
                multi = multi | {
                    alt for g in spec.groups if g.multi_instance for alt in g.alt_prompts
                }
                merged, all_dets, resolution = _pass(retry, relax)
                used = retry

        if not resolution.ok and spec.fallback.llm_propose_prompts:
            proposed = self._llm_propose_prompts(captures, spec, resolution, used)
            if proposed:
                # Folding the proposals in as alt_prompts is what makes them
                # relabel to the canonical group name -- the same mechanism
                # rung 2 uses, not a parallel one. `_pass` closes over `spec`,
                # so rebinding it here is what the retry merges against.
                spec = spec.with_alt_prompts(proposed)
                extra = [p for ps in proposed.values() for p in ps if p not in used]
                retry = used + extra
                logger.warning(
                    "task %r: count gate still failed, retrying with " "LLM-proposed prompts %s",
                    spec.task,
                    extra,
                )
                multi = multi | {
                    alt for g in spec.groups if g.multi_instance for alt in g.alt_prompts
                }
                merged, all_dets, resolution = _pass(retry, relax)
                used = retry
                # Persist ONLY what the gate just accepted: each proposed
                # group that is no longer mismatched earned its phrases a
                # place in the cache. A rejected proposal never touches it.
                self._record_accepted_prompts(spec, resolution, proposed)

        if not resolution.ok:
            logger.error("task %r: %s", spec.task, resolution.describe())
            resolution.raise_if_abort()

        return merged, all_dets, resolution, spec, used

    def _cached_prompt_proposals(self, spec, resolution, tried) -> Dict[str, List[str]]:
        """Learned prompts for the still-missing groups, minus anything tried.

        Fail-open on absence: a pipeline built without a LearnedPrompts store
        (tests, sim, bimanual) simply has an empty cache rung.
        """
        store = getattr(self, "_learned_prompts", None)
        if store is None:
            return {}
        out: Dict[str, List[str]] = {}
        tried_set = set(tried or [])
        for missing in resolution.mismatched_groups:
            group = spec.group_for(missing)
            key = group.text if group is not None else missing
            fresh = [p for p in store.lookup(spec.task, key) if p not in tried_set]
            if fresh:
                out[key] = fresh
        return out

    def _record_accepted_prompts(self, spec, resolution, proposed) -> None:
        """Write each proposed group's phrases to the store iff its gate now passes."""
        store = getattr(self, "_learned_prompts", None)
        if store is None or not proposed:
            return
        still_missing = set(resolution.mismatched_groups) if not resolution.ok else set()
        for key, phrases in proposed.items():
            if key in still_missing:
                continue
            try:
                store.record(spec.task, key, phrases)
            except Exception as exc:  # noqa: BLE001 - cache loss must not fail detect
                logger.warning("learned-prompt record failed for %r: %s", key, exc)

    def _llm_propose_prompts(self, captures, spec, resolution, tried) -> Dict[str, List[str]]:
        """
        Rung 3: ask the planner LLM to name the groups SAM3 could not find.

        Returns ``{group_text: [prompt, ...]}``, or ``{}`` for every failure
        mode (no planner, no frame, the call raised, junk answer, or every
        suggestion already tried). Nothing here propagates: the caller sees an
        empty dict and falls through to ``fallback.on_mismatch``.

        Accepted proposals persist: _record_accepted_prompts writes gate-
        passing phrases into the pipeline's LearnedPrompts store, and the
        cached rung above tries them on the next failing run before this one
        spends an API call. Pasting a winner into the task's ``alt_prompts:``
        YAML is the stronger fix (rung 2 instead of rung 3.5).
        """
        planner = getattr(self, "_planner", None)
        if planner is None or not hasattr(planner, "propose_alt_prompts"):
            logger.warning(
                "task %r: no planner available for prompt proposal; "
                "falling through to on_mismatch",
                spec.task,
            )
            return {}

        scene_img = None
        for cam in list(self._PROPOSAL_CAM_ORDER) + list(captures or {}):
            data = (captures or {}).get(cam)
            if data is not None and data.get("rgb") is not None:
                scene_img = data["rgb"]
                break
        if scene_img is None:
            logger.warning("task %r: no frame to show the planner", spec.task)
            return {}

        tried_norm = {base_label(p) for p in (tried or [])}
        out: Dict[str, List[str]] = {}
        for missing in resolution.mismatched_groups:
            group = spec.group_for(missing)
            known = set(tried_norm)
            if group is not None:
                known |= {base_label(group.text)}
                known |= {base_label(a) for a in group.alt_prompts}
            try:
                raw = planner.propose_alt_prompts(
                    missing,
                    sorted(known),
                    scene_image=scene_img,
                    max_prompts=spec.fallback.llm_max_prompts,
                )
            except Exception as exc:  # noqa: BLE001 - never fail the recovery
                logger.warning(
                    "task %r: prompt proposal for %r failed (%s); "
                    "falling through to on_mismatch",
                    spec.task,
                    missing,
                    exc,
                )
                return {}
            if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
                logger.warning(
                    "task %r: planner returned %s for %r, not a list of prompts",
                    spec.task,
                    type(raw).__name__,
                    missing,
                )
                continue
            clean: List[str] = []
            for item in raw:
                if not isinstance(item, str):
                    continue
                text = item.strip()
                if not text or base_label(text) in known:
                    continue
                known.add(base_label(text))
                clean.append(text)
                if len(clean) >= spec.fallback.llm_max_prompts:
                    break
            if clean:
                out[missing] = clean
                logger.warning(
                    "task %r: planner proposes %s for missing %r -- add the "
                    "one that works to alt_prompts in %s",
                    spec.task,
                    clean,
                    missing,
                    spec.source_path or "the task YAML",
                )
        return out

    def _extract_prompts(self, instruction: str) -> List[str]:
        """
        Extract object names from instruction for SAM3 detection.

        Fold tasks expand to sub-part keypoints: SAM3 is open-vocab and
        can ground "shirt sleeve" / "shirt hem" / "shirt collar"
        directly, and the fold planner sequence operates over those
        parts (pinch a sleeve, arc to centre, release, repeat).
        """
        instr_l = instruction.lower()
        instr_words = instr_l.replace(",", " ").split()

        # Fold-task expansion: keypoints are garment-specific parts.
        fold_verbs = ("fold", "folding")
        if any(v in instr_words for v in fold_verbs):
            if "t-shirt" in instr_l or "tshirt" in instr_l or "shirt" in instr_l:
                return [
                    "shirt left sleeve",
                    "shirt right sleeve",
                    "shirt hem",
                    "shirt collar",
                    "shirt body",
                ]
            if "towel" in instr_l:
                return [
                    "towel left edge",
                    "towel right edge",
                    "towel top edge",
                    "towel bottom edge",
                    "towel",
                ]
            if "napkin" in instr_l:
                return ["napkin corner", "napkin edge", "napkin"]
            if "pants" in instr_l or "trousers" in instr_l:
                return ["pants left leg", "pants right leg", "pants waistband", "pants"]
            # Generic cloth fallback: corners + edges of the named item.
            # The word(s) after "fold" usually name the cloth.
            try:
                idx = next(i for i, w in enumerate(instr_words) if w in fold_verbs)
                name = " ".join(instr_words[idx + 1 : idx + 4])
                name = name.replace("the ", "").strip()
                if name:
                    return [f"{name} corner", f"{name} edge", name]
            except StopIteration:
                pass

        stop_words = {
            "pick",
            "up",
            "the",
            "and",
            "place",
            "it",
            "on",
            "put",
            "in",
            "into",
            "a",
            "an",
            "to",
            "from",
            "move",
            "grab",
            "take",
            "get",
            "set",
            "drop",
            "both",
            "all",
            "each",
        }
        prompts = []
        current = []
        for w in instr_words:
            if w in stop_words:
                if current:
                    prompts.append(" ".join(current))
                    current = []
            else:
                current.append(w)
        if current:
            prompts.append(" ".join(current))
        return prompts if prompts else [instruction]
