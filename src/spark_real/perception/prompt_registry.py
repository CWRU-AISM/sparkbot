"""
Declarative per-task SAM3 prompt registry and same-instance disambiguation.

The same instruction must produce the same prompts and the same instance
labels on every run, or a cached behaviour tree that says ``blue block 1``
grasps a different physical block each episode. A registry hit skips the LLM
prompt generation (which runs at non-zero temperature), and ``order_by``
replaces SAM3 confidence-rank instance numbering with a geometric total order.

Data lives in ``configs/tasks/<slug>.yaml``, one file per task, glob
discovered. The directory is overridable by the ``SPARK_TASK_PROMPTS`` env var
or by passing ``path=`` (a relative path resolves against ``configs/``, which
is how a ``perception.task_prompts_dir`` key in the family YAML is threaded
in).

Schema (every key optional except ``task`` and ``groups``)::

    task: "stack the blocks of same color"
    aliases: ["stack the same colored blocks"]
    bt: "bt_seeds/stack_the_blocks_of_same_color.yaml"
    ordering_camera: birdview       # frame whose image axes define the index
    per_instance_z: true            # disable merge-time median-Z flattening
    groups:
      - text: "blue block"
        multi_instance: true
        expect: {min: 0, max: 2}
        min_conf: 0.15
        order_by: [centroid_x, centroid_y]
        alt_prompts: ["blue cube"]
    require: {any_group_count: 2, alias_as: "same color block"}
    disambiguate: {hsv_cluster: false, hue_tol_deg: 15.0}
    fallback:
      alt_prompts: {"blue block": ["blue cube"]}
      relax_secondary_score: 0.30
      llm_propose_prompts: true     # ask the planner LLM to name the missing
      llm_max_prompts: 4            # group, AFTER the two rungs above fail
      on_mismatch: abort            # abort | operator_click | best_effort

Entry points
------------
``load_registry(path=None)``          -> :class:`Registry`
``Registry.lookup(instruction)``      -> :class:`TaskSpec` | ``None``
``TaskSpec.order(detections)``        -> ordered + labelled detections
``TaskSpec.resolve(detections)``      -> :class:`LabelResolution` (adds the
                                         count gate and the mismatch policy)

Instruction matching
--------------------
Every key derived from an instruction comes from ``spark_real.bt_library``:
``normalize_instruction`` (exact tier), ``near_exact_key``
(spelling/plural/filler tier) and ``_tokens`` / ``_jaccard`` (similarity
tier). This module must never grow its own dialect: an instruction that
resolves in the BT cache but not here executes a cached tree against
un-grounded prompts.

The tiers mirror ``BTLibrary.lookup``'s 2-4 (the pin tier has no analogue).
This module is stricter at the last tier: a near-tie between two specs is
refused, because ``stack the blue block on the gray block`` and its mirror
tokenize identically. The near-exact tier preserves content-word order, so it
separates that pair.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from spark_real.bt_keys import count_differs
from spark_real.bt_library import (
    _jaccard,
    _tokens,
    near_exact_key,
    normalize_instruction,
)
from spark_real.config import CONFIGS_DIR
from spark_real.perception.label_resolution import (  # noqa: F401
    GroupResult,
    LabelResolution,
    _hsv_cluster,
    _hue_distance_deg,
    resolve_labels,
)
from spark_real.perception.task_spec import (  # noqa: F401
    _DESCENDING_BY_DEFAULT,
    _INSTANCE_SUFFIX_RE,
    _ON_MISMATCH_VALUES,
    _ORDER_KEYS,
    _SLUG_RE,
    _TIEBREAK_KEYS,
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_ORDER_BY,
    DEFAULT_SECONDARY_MASK_SCORE,
    DEFAULT_TASK_PROMPTS_DIR,
    ON_MISMATCH_ABORT,
    ON_MISMATCH_BEST_EFFORT,
    ON_MISMATCH_OPERATOR_CLICK,
    SIMILARITY_TIE_EPS,
    TASK_PROMPTS_ENV,
    DisambiguateSpec,
    FallbackSpec,
    PromptCountMismatch,
    PromptGroup,
    TaskSpec,
    _centroid,
    _parse_order_key,
    _sort_key,
    _world,
    base_label,
    order_detections,
    slugify,
)

logger = logging.getLogger(__name__)


@dataclass
class Registry:
    """
    All task specs discovered under one directory.
    """

    specs: Tuple[TaskSpec, ...] = ()
    root: Optional[Path] = None
    min_similarity: float = DEFAULT_MIN_SIMILARITY

    def __len__(self) -> int:
        return len(self.specs)

    @property
    def tasks(self) -> List[str]:
        return [s.task for s in self.specs]

    def get(self, task: str) -> Optional[TaskSpec]:
        """
        Exact (normalised) task-string or alias lookup. No similarity.
        """
        want = normalize_instruction(task)
        if not want:
            return None
        for spec in self.specs:
            if want in spec.keys:
                return spec
        return None

    def get_near(self, task: str) -> Optional[TaskSpec]:
        """
        Near-exact lookup: spelling variants, plurals and filler words folded,
        content-word order preserved. ``None`` when nothing matches and when
        two specs match (never silently pick one).
        """
        want = near_exact_key(task)
        if not want:
            return None
        hits = [
            spec
            for spec in self.specs
            if want in spec.near_keys and not count_differs(task, (spec.task, *spec.aliases))
        ]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            logger.warning(
                "prompt registry: refusing ambiguous near-exact match for %r "
                "(key %r shared by %s); falling through",
                task,
                want,
                [s.task for s in hits],
            )
        return None

    def lookup(self, instruction: str) -> Optional[TaskSpec]:
        """
        Resolve an instruction to a spec, or None to fall through to the
        caller's existing prompt path.

        Three tiers, mirroring ``BTLibrary.lookup``'s 2-4 so that the prompt
        registry and the BT cache never disagree about whether two instruction
        spellings are the same task:

            1. exact    -- normalised string equality (case, punctuation,
                           underscores and whitespace folded).
            2. near     -- spelling variants, plurals and filler folded, with
                           content-word ORDER preserved.
            3. similar  -- Jaccard over the shared tokenizer, with a floor.

        A near-tie at tier 3 is refused -- for this corpus a bag-of-words score
        genuinely cannot separate "stack the blue block on the gray block" from
        its mirror, and answering anyway would silently execute the wrong plan.
        Tier 2 keeps order, so it resolves that pair rather than refusing it.
        """
        if not instruction:
            return None
        exact = self.get(instruction)
        if exact is not None:
            logger.info("prompt registry: exact hit %r", exact.task)
            return exact

        near = self.get_near(instruction)
        if near is not None:
            logger.info("prompt registry: near-exact hit %r for %r", near.task, instruction)
            return near
        key = near_exact_key(instruction)
        if key and any(key in spec.near_keys for spec in self.specs):
            # Same task, different count. The similarity tier would answer with some other tool.
            logger.info("prompt registry: no contract for the count in %r", instruction)
            return None

        query = _tokens(instruction)
        if not query:
            return None
        scored: List[Tuple[float, TaskSpec]] = []
        for spec in self.specs:
            candidates = [spec.task] + list(spec.aliases)
            if count_differs(instruction, candidates):
                continue
            best = max(_jaccard(query, _tokens(c)) for c in candidates)
            if best >= self.min_similarity:
                scored.append((best, spec))
        if not scored:
            return None
        scored.sort(key=lambda x: (-x[0], normalize_instruction(x[1].task)))
        if len(scored) > 1 and (scored[0][0] - scored[1][0]) < SIMILARITY_TIE_EPS:
            tied = [s.task for sim, s in scored if (scored[0][0] - sim) < SIMILARITY_TIE_EPS]
            logger.warning(
                "prompt registry: refusing ambiguous match for %r "
                "(%s all score %.3f); falling through",
                instruction,
                tied,
                scored[0][0],
            )
            return None
        logger.info(
            "prompt registry: fuzzy hit %r (sim=%.3f) for %r",
            scored[0][1].task,
            scored[0][0],
            instruction,
        )
        return scored[0][1]


def resolve_task_prompts_dir(path: Optional[Any] = None) -> Path:
    """
    Directory holding the task YAMLs.

    Precedence: explicit argument, then $SPARK_TASK_PROMPTS, then
    configs/tasks. A relative value resolves against configs/, which is how a
    ``perception.task_prompts_dir`` key from the family YAML is threaded in.
    """
    raw = path or os.environ.get(TASK_PROMPTS_ENV) or DEFAULT_TASK_PROMPTS_DIR
    candidate = Path(str(raw)).expanduser()
    if not candidate.is_absolute():
        candidate = Path(CONFIGS_DIR) / candidate
    return candidate


def load_registry(
    path: Optional[Any] = None,
    min_similarity: Optional[float] = None,
) -> Registry:
    """
    Glob-load every ``*.yaml`` under the task-prompts directory.

    A malformed file is logged and skipped rather than taking the server down;
    the registry is an optimisation over the LLM path, and a missing entry
    degrades to the existing behaviour instead of a boot failure.
    """
    root = resolve_task_prompts_dir(path)
    floor = DEFAULT_MIN_SIMILARITY if min_similarity is None else float(min_similarity)
    if not root.is_dir():
        logger.warning("prompt registry: %s is not a directory; registry empty", root)
        return Registry(specs=(), root=root, min_similarity=floor)

    specs: List[TaskSpec] = []
    for yaml_path in sorted(root.glob("*.yaml")):
        try:
            with open(yaml_path) as handle:
                raw = yaml.safe_load(handle) or {}
            specs.append(TaskSpec.from_yaml(raw, source=yaml_path.name))
        except Exception as exc:
            logger.error("prompt registry: skipping %s (%s)", yaml_path.name, exc)

    by_task: Dict[str, TaskSpec] = {}
    for spec in specs:
        key = normalize_instruction(spec.task)
        if key in by_task:
            logger.error(
                "prompt registry: duplicate task %r in %s and %s; keeping the first",
                spec.task,
                by_task[key].source_path,
                spec.source_path,
            )
            continue
        by_task[key] = spec

    # Two specs sharing a near-exact key are unresolvable at lookup tier 2 and
    # would fall through to the tie-refusing similarity tier, i.e. both tasks
    # would silently lose their registry entry. Say so at boot, not mid-demo.
    by_near: Dict[str, TaskSpec] = {}
    for spec in by_task.values():
        for near in spec.near_keys:
            other = by_near.get(near)
            if other is not None and other is not spec:
                logger.error(
                    "prompt registry: %s and %s share near-exact key %r; "
                    "instructions matching it will not resolve",
                    other.source_path,
                    spec.source_path,
                    near,
                )
            by_near.setdefault(near, spec)

    logger.info("prompt registry: loaded %d task(s) from %s", len(by_task), root)
    return Registry(specs=tuple(by_task.values()), root=root, min_similarity=floor)
