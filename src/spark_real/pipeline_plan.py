"""Plan resolution: BT-library cache first, then the planner, with a label check."""

import logging
import os
from typing import Dict, List

from spark_core.sam3 import ObjectDetection
from spark_real.bt_library import BTCacheMiss, BTMatch
from spark_real.plan_inputs import (
    LEGACY_DISABLE_LLM_ENV,
    PLAN_MODE_ENV,
    _bt_frozen,
    verify_block_labels,
)
from spark_real.routes import state as _routes_state

logger = logging.getLogger(__name__)


class PlanMixin:
    """Plan resolution: BT-library cache first, then the planner, with a label check."""

    def plan_mode(self) -> str:
        """
        Which plan source is currently allowed: ``auto``, ``cache_only`` or
        ``llm``.

        Precedence, highest first:
          1. the persisted library toggle (``/api/bt/toggle_cache``, stored
             in ``<library>/runtime_state.json`` so it survives a restart),
          2. the in-process ``state.use_cached_bt`` (older UI builds),
          3. ``config.bt_plan_mode``,
          4. ``SPARK_BT_PLAN_MODE`` / legacy ``SPARK_DISABLE_LLM=1``.
        """
        lib = self._bt_library
        if lib is not None and lib.use_cached_bt is not None:
            return "cache_only" if lib.use_cached_bt else "llm"
        try:
            if _routes_state.use_cached_bt is not None:
                return "cache_only" if _routes_state.use_cached_bt else "llm"
        except Exception:
            pass
        mode = getattr(self.config, "bt_plan_mode", None)
        if not mode:
            mode = os.environ.get(PLAN_MODE_ENV) or ""
        if not mode and os.environ.get(LEGACY_DISABLE_LLM_ENV, "0") == "1":
            mode = "cache_only"
        mode = str(mode).strip().lower() or "auto"
        if mode not in ("auto", "cache_only", "llm"):
            logger.warning("[plan] unknown plan mode %r; using 'auto'", mode)
            return "auto"
        return mode

    def resolve_cached_bt(self, instruction: str, require_cached=None):
        """
        Resolve ``instruction`` against the BT library without planning.

        Returns a ``BTMatch`` or None. Raises ``BTCacheMiss`` on a miss when
        strict cache-only mode is active, so a demo-collection run fails
        loudly instead of quietly minting a fresh Gemini plan nobody froze.

        Shared by ``plan()`` and by the detection-prompt harvest in
        ``run_task``, so both see exactly the same tree.
        """
        lib = self._bt_library
        if lib is None:
            return None
        mode = self.plan_mode()
        strict = (mode == "cache_only") if require_cached is None else require_cached
        if mode == "llm" and require_cached is None:
            return None

        # Legacy single-shot UI pin ("select this BT for the next execute").
        # Consumed here; the persisted per-task pin inside the library is
        # the durable mechanism and is NOT consumed.
        try:
            one_shot = _routes_state.pinned_bt_hash
        except Exception:
            one_shot = None
        if one_shot:
            try:
                _routes_state.pinned_bt_hash = None
            except Exception:
                pass
            entry = lib.get(one_shot)
            if entry is not None:
                logger.info(
                    "[plan] single-shot pin -> BT %s (%r)",
                    entry.hash,
                    entry.instruction,
                )
                return BTMatch(entry=entry, source="pin", similarity=1.0)
            logger.warning(
                "[plan] single-shot pinned hash %s is not in the library; "
                "falling through to the normal resolver",
                one_shot,
            )

        match = lib.resolve(instruction, require_cached=strict)
        if match is not None:
            source = "seed" if match.entry.seed and match.source != "pin" else match.source
            logger.info(
                "[plan] cache %s: BT %s (%r, success=%d fail=%d, sim=%.2f)",
                source,
                match.entry.hash,
                match.entry.instruction,
                match.entry.success,
                match.entry.fail,
                match.similarity,
            )
            match.source = source
        return match

    def plan(
        self,
        instruction: str,
        detections: List[ObjectDetection],
        scene_image=None,
        detection_details: list = None,
        temperature: float = None,
        scene_camera: str = None,
    ) -> Dict:
        """
        Generate a SPARK score from instruction and detections.

        ``scene_camera`` names the camera ``scene_image`` was taken with
        (recorded with the plan; not otherwise read).

        Resolves the BT library first (pin -> exact -> near-exact ->
        similarity) and only calls the planner on a miss, unless
        ``plan_mode()`` says otherwise. The chosen source and entry hash are
        left on ``self._last_plan_source`` / ``self._last_bt_hash`` so the
        caller can record the run's provenance and bump the right entry
        without re-resolving.
        """
        self._last_plan_source = None
        self._last_bt_hash = None

        # An explicit sampling temperature is an explicit request for a NEW
        # sample: the /scores page generates alternative trees for the same
        # task that way. Serving the cache there would silently return the
        # tree the operator is trying to find an alternative to.
        if temperature is None:
            match = self.resolve_cached_bt(instruction)
            if match is not None:
                self._last_plan_source = match.source
                self._last_bt_hash = match.hash
                # Detached copy: execute() pops label_corrections off the
                # score, which would otherwise mutate the library entry.
                return match.score
        elif self.plan_mode() == "cache_only":
            raise BTCacheMiss(
                f"plan_mode=cache_only forbids the LLM, but planning "
                f"{instruction!r} at temperature={temperature} is a request "
                f"for a fresh sample. Clear cache-only mode first."
            )

        if self.plan_mode() == "cache_only":
            # resolve_cached_bt already raises in this mode; belt and braces
            # for a library that is None.
            raise BTCacheMiss(
                f"plan_mode=cache_only and no BT library is loaded, so "
                f"{instruction!r} cannot be planned without the LLM"
            )
        if self._planner is None:
            raise RuntimeError(
                f"no cached BT for {instruction!r} and no planner is "
                f"configured; seed a BT into the library first"
            )
        self._last_plan_source = "llm"

        labels = [d.label for d in detections]
        # Few-shot context: similarity-retrieved entries PLUS any
        # promoted ones (which are always included regardless of
        # jaccard similarity, so they act as canonical recipes).
        _bt_examples = []
        if self._bt_library is not None and not _bt_frozen(self):
            promoted = self._bt_library.promoted_entries()
            similar = self._bt_library.top_k_for(instruction, k=3)
            seen = {p[0] for p in promoted}
            _bt_examples = promoted + [s for s in similar if s[0] not in seen]

        logger.info(
            "[plan] cache MISS for %r (%d entries, floor=%.2f); calling the "
            "planner. This run's tree will be cached on success.",
            instruction,
            len(self._bt_library) if self._bt_library is not None else 0,
            (
                getattr(self._bt_library, "min_similarity", 0.0)
                if self._bt_library is not None
                else 0.0
            ),
        )

        # Per-run sampling override: the scores page generates BT variations
        # by re-planning at nonzero temperature. The planner default is
        # restored afterwards so normal runs stay deterministic.
        _old_temp = None
        if temperature is not None and self._planner is not None:
            _old_temp = self._planner.temperature
            self._planner.temperature = float(temperature)
        try:
            score = self._planner.generate_score(
                instruction=instruction,
                annotated_image=scene_image,
                scene_camera=scene_camera,
                keypoint_labels=labels,
                detection_details=detection_details,
                fewshot_examples=_bt_examples,
            )
            score = self._validate_plan_labels(
                score,
                labels,
                instruction,
                scene_image,
                detection_details,
                _bt_examples,
                scene_camera=scene_camera,
            )
        finally:
            if _old_temp is not None:
                self._planner.temperature = _old_temp
        return score

    def _validate_plan_labels(
        self,
        score,
        labels,
        instruction,
        scene_image,
        detection_details,
        fewshot,
        scene_camera=None,
    ):
        """
        Refuse to return a BT that references labels nobody detected.

        Gemini is told to use the exact detected labels, but when the
        few-shot vocabulary diverges it can invent labels that only fail
        mid-run as 'not found in detections'. Collect every label the tree
        references, allow the detection labels plus label_corrections
        targets, re-prompt once with the exact valid set on mismatch, and
        raise rather than hand back an unexecutable plan.
        """
        label_keys = (
            "keypoint_label",
            "container_label",
            "target_label",
            "dustpan_label",
        )

        def used_labels(s):
            out = set()

            def walk(node):
                if not isinstance(node, dict):
                    return
                p = node.get("params") or {}
                for k in label_keys:
                    v = p.get(k)
                    if v:
                        out.add(str(v))
                for c in node.get("children") or []:
                    walk(c)

            walk(s.get("tree") if isinstance(s, dict) else None)
            # Predicate labels live outside the tree but must be detectable
            # too, or the run executes fine and then cannot be verified.
            out |= verify_block_labels(s)
            return out

        def unresolved(s):
            corr = (s.get("label_corrections") or {}) if isinstance(s, dict) else {}
            allowed = set(labels) | {str(v) for v in corr.values()}
            return used_labels(s) - allowed

        bad = unresolved(score)
        if not bad:
            return score
        logger.warning(
            "[plan] BT references unknown labels %s (valid: %s); "
            "re-prompting once with the exact set",
            sorted(bad),
            sorted(labels),
        )
        retry_instruction = (
            f"{instruction}\n\nIMPORTANT: your previous plan referenced "
            f"labels {sorted(bad)} which DO NOT EXIST in this scene. The "
            f"only valid keypoint labels are exactly: {sorted(labels)}. "
            f"Rewrite the plan using ONLY these labels. If a detection is "
            f"visually mislabeled you may rename it via label_corrections, "
            f"then use the corrected name."
        )
        score2 = self._planner.generate_score(
            instruction=retry_instruction,
            annotated_image=scene_image,
            scene_camera=scene_camera,
            keypoint_labels=labels,
            detection_details=detection_details,
            fewshot_examples=fewshot,
        )
        bad2 = unresolved(score2)
        if bad2:
            raise ValueError(
                f"Gemini plan references labels {sorted(bad2)} that are not "
                f"in the detections {sorted(labels)} even after a corrective "
                f"retry; refusing to execute an unresolvable plan."
            )
        logger.info("[plan] corrective re-prompt resolved the labels")
        return score2
