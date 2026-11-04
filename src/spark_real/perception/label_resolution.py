"""Deterministic instance labels for a merged detection set, with the per-group count gate."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from spark_real.perception.task_spec import (
    ON_MISMATCH_ABORT,
    DisambiguateSpec,
    PromptCountMismatch,
    TaskSpec,
    order_detections,
)

logger = logging.getLogger(__name__)


@dataclass
class GroupResult:
    text: str
    found: int
    expect_min: Optional[int]
    expect_max: Optional[int]
    ok: bool
    labels: List[str] = field(default_factory=list)


@dataclass
class LabelResolution:
    """
    Outcome of applying a :class:`TaskSpec` to a merged detection set.

    ``detections`` are the accepted detections with canonical, ordered labels.
    ``ok`` is the count gate; ``on_mismatch`` is the policy the caller must
    apply when it is False.
    """

    task: str
    detections: List[Any]
    groups: List[GroupResult]
    extras: List[Any] = field(default_factory=list)
    ok: bool = True
    on_mismatch: str = ON_MISMATCH_ABORT
    acting_group: Optional[str] = None
    aliases_added: List[str] = field(default_factory=list)
    dropped_low_conf: int = 0
    dropped_hsv: int = 0
    retry_prompts: List[str] = field(default_factory=list)

    @property
    def counts(self) -> Dict[str, int]:
        return {g.text: g.found for g in self.groups}

    @property
    def labels(self) -> List[str]:
        return [str(getattr(d, "label", "")) for d in self.detections]

    @property
    def mismatched_groups(self) -> List[str]:
        return [g.text for g in self.groups if not g.ok]

    def describe(self) -> str:
        parts = [
            f"{g.text}: {g.found} (expect {'*' if g.expect_min is None else g.expect_min}"
            f"..{'*' if g.expect_max is None else g.expect_max})"
            + ("" if g.ok else "  <-- MISMATCH")
            for g in self.groups
        ]
        head = f"task {self.task!r} count gate {'OK' if self.ok else 'FAILED'}"
        return head + "\n  " + "\n  ".join(parts)

    def raise_if_abort(self) -> "LabelResolution":
        """
        Enforce the strict policy. Returns self so it can be chained.
        """
        if not self.ok and self.on_mismatch == ON_MISMATCH_ABORT:
            raise PromptCountMismatch(self)
        return self


def _hue_distance_deg(a: float, b: float) -> float:
    d = abs(float(a) - float(b)) % 360.0
    return min(d, 360.0 - d)


def _hsv_cluster(dets: Sequence[Any], spec: DisambiguateSpec) -> List[Any]:
    """
    Keep the largest mutually-close-hue cluster. Detections without an
    ``hsv_median`` or below the saturation/value floors are kept unconditionally
    (a gray block has no meaningful hue, so hue must not be allowed to reject
    it). Clusters, never names: colour naming belongs in the YAML, not here.
    """
    hued: List[Tuple[int, float]] = []
    for i, d in enumerate(dets):
        hsv = getattr(d, "hsv_median", None)
        if hsv is None or len(hsv) < 3:
            continue
        h, s, v = float(hsv[0]), float(hsv[1]), float(hsv[2])
        if s < spec.min_saturation or v < spec.min_value:
            continue
        hued.append((i, h))
    if len(hued) < 2:
        return list(dets)

    best: List[int] = []
    for _, anchor in hued:
        members = [i for i, h in hued if _hue_distance_deg(h, anchor) <= spec.hue_tol_deg]
        if len(members) > len(best):
            best = members
    if not best:
        return list(dets)
    keep = set(best)
    # Anything that never entered the hue vote (no colour, or too dark/gray)
    # is not a cluster outlier and stays.
    voted = {i for i, _ in hued}
    return [d for i, d in enumerate(dets) if i in keep or i not in voted]


def resolve_labels(spec: TaskSpec, detections: Sequence[Any]) -> LabelResolution:
    """
    Assign stable, deterministic instance labels to a merged detection set.

    Labels are written onto the detection objects in place (the merged set is
    already a set of copies owned by the caller) and are also returned on the
    result. Resolution is idempotent: an already-numbered set re-resolves to
    the same labels.

    Numbering rule: ``base`` when a group yielded exactly one instance,
    ``base 1 .. base N`` when it yielded several. Arity aliasing (registering
    both forms) happens downstream in the detection map, not here.
    """
    result_groups: List[GroupResult] = []
    accepted: List[Any] = []
    extras: List[Any] = []
    dropped_low_conf = 0
    dropped_hsv = 0

    claimed: set = set()
    for group in spec.groups:
        members = []
        for det in detections:
            if id(det) in claimed:
                continue
            if group.accepts_label(getattr(det, "label", "") or ""):
                claimed.add(id(det))
                members.append(det)

        kept = []
        for det in members:
            if float(getattr(det, "confidence", 0.0) or 0.0) < group.min_conf:
                dropped_low_conf += 1
                continue
            kept.append(det)

        if spec.disambiguate.hsv_cluster and len(kept) > 1:
            before = len(kept)
            kept = _hsv_cluster(kept, spec.disambiguate)
            dropped_hsv += before - len(kept)

        # Trim to the declared maximum using the group's own ordering, so the
        # instances that survive are the geometrically-first ones rather than
        # whichever SAM3 happened to score highest.
        ordered = order_detections(kept, group.order_by)
        if group.expect_max is not None and len(ordered) > group.expect_max:
            extras.extend(ordered[group.expect_max :])
            ordered = ordered[: group.expect_max]

        labels = []
        for idx, det in enumerate(ordered):
            label = group.text if len(ordered) == 1 else f"{group.text} {idx + 1}"
            det.label = label
            labels.append(label)
        accepted.extend(ordered)

        result_groups.append(
            GroupResult(
                text=group.text,
                found=len(ordered),
                expect_min=group.expect_min,
                expect_max=group.expect_max,
                ok=group.count_ok(len(ordered)),
                labels=labels,
            )
        )

    # Detections for prompts the task never declared (e.g. an operator typed an
    # extra prompt). Left untouched and passed through.
    for det in detections:
        if id(det) not in claimed:
            extras.append(det)

    ok = all(g.ok for g in result_groups)

    # require: pick the acting group and optionally alias it to a role name so
    # a cached BT can address "the pair" without knowing which colour it is.
    acting: Optional[str] = None
    aliases_added: List[str] = []
    want = spec.require.get("any_group_count")
    if want is not None:
        want = int(want)
        matches = [g for g in result_groups if g.found == want]
        if len(matches) == 1:
            acting = matches[0].text
        elif len(matches) > 1:
            ok = False
            logger.warning(
                "task %r: require.any_group_count=%d is ambiguous, groups %s all match",
                spec.task,
                want,
                [g.text for g in matches],
            )
        else:
            ok = False
            logger.warning(
                "task %r: require.any_group_count=%d unsatisfied, counts=%s",
                spec.task,
                want,
                {g.text: g.found for g in result_groups},
            )

    alias_as = spec.require.get("alias_as")
    if acting is not None and alias_as:
        acting_group = next(g for g in result_groups if g.text == acting)
        by_label = {str(getattr(d, "label", "")): d for d in accepted}
        for idx, label in enumerate(acting_group.labels):
            det = by_label.get(label)
            if det is None:
                continue
            alias = str(alias_as) if len(acting_group.labels) == 1 else f"{alias_as} {idx + 1}"
            existing = list(getattr(det, "role_labels", None) or [])
            if alias not in existing:
                existing.append(alias)
            det.role_labels = existing
            aliases_added.append(alias)

    min_total = spec.require.get("min_total")
    if min_total is not None and len(accepted) < int(min_total):
        ok = False

    return LabelResolution(
        task=spec.task,
        detections=accepted,
        groups=result_groups,
        extras=extras,
        ok=ok,
        on_mismatch=spec.fallback.on_mismatch,
        acting_group=acting,
        aliases_added=aliases_added,
        dropped_low_conf=dropped_low_conf,
        dropped_hsv=dropped_hsv,
        retry_prompts=spec.alt_prompts_for([g.text for g in result_groups if not g.ok]),
    )
