"""Fusion math: SAM3 confidence pooled with a second-opinion box detector (agree, miss, abstain)."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from spark_core.util import det_field
from spark_real.perception.box_proposals import (
    BoxProposal,
    ProposalSet,
    _canonical_class,
    _strip_instance,
)
from spark_real.perception.mask_quality import MaskQuality, QualityThresholds, measure_mask

logger = logging.getLogger(__name__)

AGREE_CLASS = "agree_class"

AGREE_BOX = "agree_box"

MISS = "miss"

ABSTAIN = "abstain"

_EPS = 1e-4


@dataclass
class FusionConfig:
    """Config-gated. Every default here is the OFF / no-change behaviour."""

    enabled: bool = False  # master switch for the whole gate
    # False => the gate annotates low_quality but never rewrites confidence.
    write_confidence: bool = False
    assoc_iou: float = 0.40
    w_class: float = 0.60
    w_box: float = 0.25
    box_only_ceiling: float = 0.85
    # MEASURED, not guessed. Log-odds cost of "the second detector saw nothing
    # here". The correct value is log(1 - recall) for that detector on THIS
    # rig's objects, and the corroboration scan in tests/fusion_plushie_demo.py
    # measures it: a COCO fasterrcnn boxed a real, visible plushie in 3 of 8
    # real frames, so 1-recall = 0.625 and the penalty is -log(0.625) = 0.47.
    # Re-run the scan to recalibrate for a different backend.
    miss_penalty: float = 0.47  # logits
    low_quality_conf: float = 0.40  # mirrors perception.reprompt_min_conf
    reprompt_max_attempts: int = 2
    # A reprompt result must be the SAME OBJECT. Without this, "teddy bear"
    # returning a mask on the far side of the table gets adopted because it
    # scored better, and the arm goes to the wrong place.
    reprompt_min_iou: float = 0.30
    proposer: dict = field(default_factory=dict)
    quality: QualityThresholds = field(default_factory=QualityThresholds)

    def proposer_labels(self) -> tuple:
        """`proposer.labels`, normalised. Empty tuple = gate on vocabulary only.

        Lives here rather than in build_proposer because it is a FUSION
        decision, not a detector one: the model still runs and still reports
        every box it sees -- orphan_boxes() needs the whole set to answer
        "what else is on the table"; the gate just declines to SCORE the
        labels it is bad at.
        """
        raw = self.proposer.get("labels") if isinstance(self.proposer, dict) else None
        if raw is None:
            return ()
        if isinstance(raw, str):
            raw = [raw]
        out = []
        for entry in raw:
            e = _strip_instance(entry)
            if e and e not in out:
                out.append(e)
        return tuple(out)

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "FusionConfig":
        cfg = dict(cfg or {})
        qcfg = cfg.pop("quality", None) or {}
        known = {f for f in cls.__dataclass_fields__ if f != "quality"}
        kwargs = {k: v for k, v in cfg.items() if k in known}
        out = cls(**kwargs)
        for k, v in qcfg.items():
            if hasattr(out.quality, k):
                setattr(out.quality, k, v)
        return out


@dataclass
class FusionResult:
    """Per-detection verdict. Carries the whole audit trail."""

    label: str
    sam3_confidence: float
    fused_confidence: float
    agreement: str = ABSTAIN
    partner: Optional[BoxProposal] = None
    iou: float = 0.0
    quality: MaskQuality = field(default_factory=MaskQuality)
    low_quality: bool = False
    angle_trustworthy: bool = False
    reprompt_attempts: int = 0
    notes: list = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"{self.label}: sam3={self.sam3_confidence:.2f} -> "
            f"fused={self.fused_confidence:.2f} [{self.agreement}] "
            f"low_quality={self.low_quality} yaw_ok={self.angle_trustworthy} "
            f"| {self.quality.describe()}" + (f" | {'; '.join(self.notes)}" if self.notes else "")
        )


def _logit(p: float) -> float:
    p = min(max(float(p), _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, x))))


def mask_bbox(mask) -> Optional[tuple]:
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.ndim != 2 or not m.any():
        return None
    ys, xs = np.nonzero(m > 0)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(ua, 1e-9)


def _set_field(det, key, value) -> bool:
    """Write a field on a detection given as a dict OR a dataclass.

    The detection_map the executor actually reads is a plain dict.
    """
    if det is None:
        return False
    try:
        if hasattr(det, "__setitem__") and hasattr(det, "get"):
            det[key] = value
        else:
            setattr(det, key, value)
        return True
    except Exception as exc:  # noqa: BLE001 - a read-only detection is not fatal
        logger.warning("[gate] could not stamp %s on %r (%s)", key, type(det).__name__, exc)
        return False


# Perception-measured scalars copied verbatim onto the executor's
# `detection_map` entry, in ADDITION to the gate's own two. `rim_z_m` /
# `interior_z_m` / `height_samples` are the world-Z profile of the mask's depth
# cloud (perception.mask_geometry.mask_height_profile); control.release_height
# reads them to place the jaws relative to a container's rim and interior floor
# instead of at a planner-guessed `offset_z`. They ride this channel because it
# already implements the exact convention they need -- emit only when measured,
# so an absent key means "not measured" and the caller falls back.
MAP_GEOMETRY_FIELDS = (
    "rim_z_m",
    "interior_z_m",
    "height_samples",
    # The MASK-derived twins of detection_map's depth-derived orientation_angle
    # and aspect_ratio, so the executor's "depth disagrees with the mask"
    # fallback has the mask numbers to read.
    "mask_aspect_ratio",
    "plane_orientation_angle",
    "plane_aspect_ratio",
)


def fusion_map_fields(det) -> Dict[str, float]:
    """Perception's channel into the executor's `detection_map` dict.

    Emits a key ONLY when perception actually measured it. grasp_strategy reads
    an absent key as unknown and a present low value as a veto, so publishing
    a default 0.0 for a detection the gate never saw would kill every oriented
    grasp -- the same convention `obb_confidence` already follows, and the same
    one release_height depends on for its fallback.
    """
    out: Dict[str, float] = {}
    for key in ("fused_confidence", "axis_trust") + MAP_GEOMETRY_FIELDS:
        v = det_field(det, key, None)
        if v is None:
            continue
        try:
            out[key] = float(v)
        except (TypeError, ValueError):
            logger.warning("[gate] %s=%r on %r is not a number; dropped", key, v, det)
    return out


def _canon_or_raw(label: str, vocabulary) -> str:
    """Canonical vocabulary class for `label`, else the bare label.

    The SAME `_canonical_class` the vocabulary check runs, so synonyms, regular
    plurals, instance suffixes and sub-part prompts resolve identically on both
    sides of an allowlist comparison. The raw fallback keeps a non-vocabulary
    allowlist entry ('tray') meaningful as plain text instead of collapsing to
    None and matching every other out-of-vocabulary label.
    """
    return _canonical_class(label, vocabulary) or _strip_instance(label)


def proposer_trusts(label: str, proposals: ProposalSet, cfg: FusionConfig) -> bool:
    """Is this a label the proposer is allowed to have an opinion about?

    True for everything when no allowlist is configured, so a deployment that
    does not opt in is unchanged.
    """
    allow = cfg.proposer_labels()
    if not allow:
        return True
    canon = _canon_or_raw(label, proposals.vocabulary)
    return any(_canon_or_raw(entry, proposals.vocabulary) == canon for entry in allow)


def fuse_one(det, proposals: ProposalSet, cfg: FusionConfig) -> FusionResult:
    """Apply the fusion rule to a single detection. Pure, no I/O."""
    label = str(det_field(det, "label", "?"))
    sam3_conf = float(det_field(det, "confidence", 0.0))
    res = FusionResult(label=label, sam3_confidence=sam3_conf, fused_confidence=sam3_conf)

    res.quality = measure_mask(
        det_field(det, "mask", None),
        reported_aspect_ratio=det_field(det, "aspect_ratio", None),
        th=cfg.quality,
    )
    res.angle_trustworthy = res.quality.angle_trustworthy

    bbox = mask_bbox(det_field(det, "mask", None)) or det_field(det, "bbox", None)
    # Competence check FIRST. A label the proposer is not trusted for gets no
    # association at all: its boxes are not evidence here, so they must not
    # reach AGREE_BOX (a bonus) any more than its silence reaches MISS (a
    # penalty). Skipping the scan is also what leaves `partner` None, which is
    # what keeps ASPIRE from reprompting off an untrusted box.
    trusted = proposer_trusts(label, proposals, cfg)
    best, best_iou = None, 0.0
    if trusted and proposals.available and bbox is not None:
        for p in proposals.proposals:
            i = box_iou(bbox, p.box)
            if i > best_iou:
                best, best_iou = p, i

    in_vocab = trusted and proposals.covers(label)
    if best is not None and best_iou >= cfg.assoc_iou:
        res.partner, res.iou = best, best_iou
        same = _canonical_class(label, proposals.vocabulary) == _strip_instance(best.label)
        if same:
            res.agreement = AGREE_CLASS
            res.fused_confidence = _sigmoid(
                _logit(sam3_conf) + cfg.w_class * _logit(best.confidence)
            )
        else:
            res.agreement = AGREE_BOX
            res.fused_confidence = min(
                cfg.box_only_ceiling,
                _sigmoid(_logit(sam3_conf) + cfg.w_box * _logit(best.confidence)),
            )
            res.notes.append(f"box says '{best.label}', sam3 says '{label}'")
    elif proposals.available and in_vocab:
        res.agreement = MISS
        res.fused_confidence = _sigmoid(_logit(sam3_conf) - cfg.miss_penalty)
        res.notes.append(f"'{label}' is in the proposer vocabulary but no box overlapped")
    else:
        res.agreement = ABSTAIN
        if not proposals.available:
            res.notes.append(f"no proposer ({proposals.note or 'unavailable'})")
        elif not trusted:
            res.notes.append(
                f"'{label}' is not a label this proposer is trusted for "
                f"(proposer.labels); confidence untouched"
            )
        else:
            res.notes.append(f"'{label}' out of proposer vocabulary; confidence untouched")

    res.low_quality = bool(
        res.fused_confidence < cfg.low_quality_conf or not res.quality.angle_trustworthy
    )
    return res


def orphan_boxes(dets, proposals: ProposalSet, cfg: FusionConfig) -> List[BoxProposal]:
    """Boxes that matched no mask. Candidate geometric prompts for ASPIRE."""
    if not proposals.available:
        return []
    boxes = [mask_bbox(det_field(d, "mask", None)) or det_field(d, "bbox", None) for d in dets]
    boxes = [b for b in boxes if b is not None]
    out = []
    for p in proposals.proposals:
        if all(box_iou(b, p.box) < cfg.assoc_iou for b in boxes):
            out.append(p)
    return out
