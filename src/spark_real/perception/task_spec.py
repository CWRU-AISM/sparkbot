"""Per-task SAM3 prompt contracts: groups, fallbacks, disambiguation and instance ordering."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

from spark_real.bt_library import (
    near_exact_key,
    normalize_instruction,
)

if TYPE_CHECKING:
    from spark_real.perception.label_resolution import LabelResolution

# Directory of task YAMLs, relative to configs/. Overridable per deployment.
DEFAULT_TASK_PROMPTS_DIR = "tasks"

TASK_PROMPTS_ENV = "SPARK_TASK_PROMPTS"

# Jaccard floor for a fuzzy registry hit. Below this, no spec is returned and
# the caller falls through to its existing prompt path.
DEFAULT_MIN_SIMILARITY = 0.60

# Two specs whose similarity differs by less than this are treated as
# indistinguishable and the lookup is refused.
SIMILARITY_TIE_EPS = 1e-6

# SAM3 score a non-best mask must clear to be considered a second instance.
# A task may relax it via fallback.relax_secondary_score.
DEFAULT_SECONDARY_MASK_SCORE = 0.50

# Mismatch policies, in the order of decreasing strictness.
ON_MISMATCH_ABORT = "abort"

ON_MISMATCH_OPERATOR_CLICK = "operator_click"

ON_MISMATCH_BEST_EFFORT = "best_effort"

_ON_MISMATCH_VALUES = (
    ON_MISMATCH_ABORT,
    ON_MISMATCH_OPERATOR_CLICK,
    ON_MISMATCH_BEST_EFFORT,
)

_INSTANCE_SUFFIX_RE = re.compile(r"\s+\d+$")

_SLUG_RE = re.compile(r"[^a-z0-9]+")


class PromptCountMismatch(RuntimeError):
    """
    Raised when a registered task's detected object counts violate its
    declared ``expect`` ranges and the policy is ``abort``.

    Recording a mislabelled episode poisons a training set far worse than
    losing one episode, so this is the default for demo collection.
    """

    def __init__(self, resolution: "LabelResolution"):
        super().__init__(resolution.describe())
        self.resolution = resolution


def slugify(task: str) -> str:
    """
    Filename slug for a task string: "put the knife in the tray" ->
    "put_the_knife_in_the_tray".

    Used for configs/tasks/<slug>.yaml. Note this is NOT the same convention
    configs/bt_seeds uses (those drop articles), and nothing matches the two
    directories by filename -- task specs are discovered by glob and matched
    on the `task:` field inside, so the filename is documentation only.
    """
    return _SLUG_RE.sub("_", (task or "").strip().lower()).strip("_")


def base_label(label: str) -> str:
    """
    Strip a trailing instance index: "blue block 2" -> "blue block".

    Makes label resolution idempotent, so re-resolving an already-numbered
    detection set (a re-detect during recovery) does not compound suffixes.

    Normalises through the shared instruction dialect, so a prompt written
    "blue_block" in YAML and a detection labelled "Blue Block 2" reduce to the
    same key. NOTE: ``bt_label_resolver.base_label`` is a *different* function
    with the same name -- it strips the same suffix but preserves case and
    internal spacing, because its output is fed back to the operator and to
    detection-map lookups keyed on the label SAM3 actually produced. Do not
    assume the two are interchangeable.
    """
    return _INSTANCE_SUFFIX_RE.sub("", normalize_instruction(label)).strip()


def _centroid(det: Any) -> Tuple[float, float]:
    c = getattr(det, "centroid_2d", None)
    if c is None or len(c) < 2:
        return (0.0, 0.0)
    return (float(c[0]), float(c[1]))


def _world(det: Any) -> Tuple[float, float, float]:
    p = getattr(det, "position_3d", None)
    if p is None or len(p) < 3:
        return (0.0, 0.0, 0.0)
    return (float(p[0]), float(p[1]), float(p[2]))


# order_by vocabulary. Each entry maps a detection to a float.
_ORDER_KEYS = {
    "centroid_x": lambda d: _centroid(d)[0],
    "centroid_y": lambda d: _centroid(d)[1],
    "world_x": lambda d: _world(d)[0],
    "world_y": lambda d: _world(d)[1],
    "world_z": lambda d: _world(d)[2],
    "depth": lambda d: float(getattr(d, "depth_meters", 0.0) or 0.0),
    "mask_area": lambda d: float(getattr(d, "mask_area", 0) or 0),
    "confidence": lambda d: float(getattr(d, "confidence", 0.0) or 0.0),
}

# Keys whose natural direction is "largest first". Everything else ascends.
# confidence descending is the numbering when a task declares no order_by.
_DESCENDING_BY_DEFAULT = frozenset({"confidence", "mask_area"})

DEFAULT_ORDER_BY: Tuple[str, ...] = ("confidence",)

# Appended to every comparator so the order is total and therefore identical
# across runs even when the declared keys tie exactly.
_TIEBREAK_KEYS: Tuple[str, ...] = ("centroid_x", "centroid_y", "-mask_area")


def _parse_order_key(key: str) -> Tuple[str, bool]:
    """
    ("centroid_x", False) ascending, ("confidence", True) descending.
    A leading '-' forces descending, '+' forces ascending.
    """
    k = str(key).strip()
    if k.startswith("-"):
        return k[1:], True
    if k.startswith("+"):
        return k[1:], False
    return k, k in _DESCENDING_BY_DEFAULT


def _sort_key(det: Any, keys: Sequence[str]) -> tuple:
    out: List[Any] = []
    for raw in list(keys) + list(_TIEBREAK_KEYS):
        name, descending = _parse_order_key(raw)
        getter = _ORDER_KEYS.get(name)
        if getter is None:
            continue
        value = float(getter(det))
        out.append(-value if descending else value)
    # Final string tiebreak: camera role, then the base label itself. Both are
    # stable strings, so identical scenes sort identically across processes.
    out.append(str(getattr(det, "camera", "") or ""))
    out.append(base_label(getattr(det, "label", "") or ""))
    return tuple(out)


def order_detections(dets: Sequence[Any], order_by: Sequence[str]) -> List[Any]:
    """
    Deterministic total order over a set of same-label detections.
    """
    return sorted(dets, key=lambda d: _sort_key(d, order_by))


@dataclass(frozen=True)
class PromptGroup:
    """
    One SAM3 text prompt plus everything needed to label its instances.
    """

    text: str
    multi_instance: bool = False
    expect_min: Optional[int] = None
    expect_max: Optional[int] = None
    min_conf: float = 0.0
    order_by: Tuple[str, ...] = DEFAULT_ORDER_BY
    alt_prompts: Tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return normalize_instruction(self.text)

    def accepts_label(self, label: str) -> bool:
        b = base_label(label)
        return b == self.key or b in {normalize_instruction(a) for a in self.alt_prompts}

    def count_ok(self, n: int) -> bool:
        if self.expect_min is not None and n < self.expect_min:
            return False
        if self.expect_max is not None and n > self.expect_max:
            return False
        return True

    @classmethod
    def from_yaml(cls, raw: Dict[str, Any], source: str) -> "PromptGroup":
        text = str(raw.get("text") or "").strip()
        if not text:
            raise ValueError(f"{source}: prompt group missing 'text'")
        expect = raw.get("expect") or {}
        if not isinstance(expect, dict):
            raise ValueError(f"{source}: 'expect' must be a mapping for {text!r}")
        order_by = raw.get("order_by") or list(DEFAULT_ORDER_BY)
        if isinstance(order_by, str):
            order_by = [order_by]
        for k in order_by:
            name, _ = _parse_order_key(k)
            if name not in _ORDER_KEYS:
                raise ValueError(
                    f"{source}: unknown order_by key {k!r} for {text!r}; "
                    f"known: {sorted(_ORDER_KEYS)}"
                )
        alts = raw.get("alt_prompts") or []
        if isinstance(alts, str):
            alts = [alts]
        return cls(
            text=text,
            multi_instance=bool(raw.get("multi_instance", False)),
            expect_min=None if expect.get("min") is None else int(expect["min"]),
            expect_max=None if expect.get("max") is None else int(expect["max"]),
            min_conf=float(raw.get("min_conf", 0.0) or 0.0),
            order_by=tuple(str(k) for k in order_by),
            alt_prompts=tuple(str(a) for a in alts),
        )


@dataclass(frozen=True)
class FallbackSpec:
    """
    What to do when the scene does not match the declared arity.
    """

    alt_prompts: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    relax_secondary_score: Optional[float] = None
    on_mismatch: str = ON_MISMATCH_ABORT
    # Last rung before on_mismatch: show the frame to the planner LLM and let
    # it name the missing group in SAM3's vocabulary. Default ON: it costs an
    # API call and ~1s of latency, but is only reached after the two free
    # rungs (relax the secondary-mask score, then alt_prompts) have failed,
    # when the alternative is on_mismatch (abort for the shipped task set).
    llm_propose_prompts: bool = True
    # Cap on proposals accepted per missing group. Each one is another SAM3
    # pass over every camera, so a chatty model must not be able to turn one
    # retry into twenty. Four is enough room for a couple of genuinely
    # different guesses at what the top-down blob is.
    llm_max_prompts: int = 4

    @classmethod
    def from_yaml(cls, raw: Optional[Dict[str, Any]], source: str) -> "FallbackSpec":
        raw = raw or {}
        alt = {}
        for k, v in (raw.get("alt_prompts") or {}).items():
            alt[normalize_instruction(k)] = tuple(
                str(x) for x in (v if isinstance(v, list) else [v])
            )
        policy = str(raw.get("on_mismatch", ON_MISMATCH_ABORT)).strip().lower()
        if policy not in _ON_MISMATCH_VALUES:
            raise ValueError(
                f"{source}: on_mismatch must be one of {_ON_MISMATCH_VALUES}, got {policy!r}"
            )
        relax = raw.get("relax_secondary_score")
        max_prompts = int(raw.get("llm_max_prompts", 4) or 0)
        if max_prompts < 0:
            raise ValueError(f"{source}: llm_max_prompts must be >= 0")
        return cls(
            alt_prompts=alt,
            relax_secondary_score=None if relax is None else float(relax),
            on_mismatch=policy,
            llm_propose_prompts=bool(raw.get("llm_propose_prompts", True)),
            llm_max_prompts=max_prompts,
        )


@dataclass(frozen=True)
class DisambiguateSpec:
    """
    Optional HSV cross-check. Off by default: for the shipped task set the
    count gate plus geometric ordering is sufficient, and colour clustering is
    only worth enabling if the count gate proves flaky on the rig.
    """

    hsv_cluster: bool = False
    hue_tol_deg: float = 15.0
    min_saturation: float = 40.0
    min_value: float = 40.0

    @classmethod
    def from_yaml(cls, raw: Optional[Dict[str, Any]]) -> "DisambiguateSpec":
        raw = raw or {}
        return cls(
            hsv_cluster=bool(raw.get("hsv_cluster", False)),
            hue_tol_deg=float(raw.get("hue_tol_deg", 15.0)),
            min_saturation=float(raw.get("min_saturation", 40.0)),
            min_value=float(raw.get("min_value", 40.0)),
        )


@dataclass(frozen=True)
class TaskSpec:
    """
    Frozen perception contract for one task string.
    """

    task: str
    groups: Tuple[PromptGroup, ...]
    aliases: Tuple[str, ...] = ()
    bt: Optional[str] = None
    ordering_camera: Optional[str] = None
    per_instance_z: bool = False
    require: Dict[str, Any] = field(default_factory=dict)
    fallback: FallbackSpec = field(default_factory=FallbackSpec)
    disambiguate: DisambiguateSpec = field(default_factory=DisambiguateSpec)
    source_path: Optional[str] = None
    # Optional `verify:` block, same schema as the planner's (see
    # control/success_predicates.parse_verify_block). This is the operator's
    # no-code success override and the ONLY way to set occlusion_ok on a
    # cached tree, which never re-plans. Passed through verbatim; it is
    # validated at verify time so a typo rejects the block rather than the task.
    verify: Optional[Dict[str, Any]] = None

    # prompt surface

    @property
    def slug(self) -> str:
        return slugify(self.task)

    @property
    def keys(self) -> Tuple[str, ...]:
        """
        Every normalised instruction string that resolves to this spec by
        exact match. Same construction as ``BTEntry.keys``.
        """
        out = [normalize_instruction(t) for t in (self.task, *self.aliases)]
        return tuple(k for k in out if k)

    @property
    def near_keys(self) -> Tuple[str, ...]:
        """
        Every near-exact key (spelling, plurals and filler folded, content-word
        order preserved) that resolves to this spec. Same construction as
        ``BTEntry.near_keys``, so an instruction the BT cache resolves at its
        ``normalized`` tier resolves here too.
        """
        out = [near_exact_key(t) for t in (self.task, *self.aliases)]
        return tuple(k for k in out if k)

    @property
    def prompts(self) -> List[str]:
        """
        SAM3 prompts in declaration order. Declaration order is the order the
        registry author intended and is stable across runs.
        """
        return [g.text for g in self.groups]

    @property
    def multi_instance_prompts(self) -> set:
        """
        Prompts allowed to return more than the top-1 mask. multi_instance
        defaults to False everywhere in spark_real, so any group that can have
        two instances must say so explicitly or the second is never seen.
        """
        return {g.text for g in self.groups if g.multi_instance}

    def group_for(self, label: str) -> Optional[PromptGroup]:
        for g in self.groups:
            if g.accepts_label(label):
                return g
        return None

    def alt_prompts_for(self, labels: Iterable[str]) -> List[str]:
        """
        Retry prompts for the named groups: the group's own alt_prompts plus
        anything the task-level fallback map declares for it.
        """
        out: List[str] = []
        for label in labels:
            g = self.group_for(label)
            key = g.key if g is not None else base_label(label)
            for alt in (g.alt_prompts if g is not None else ()):
                if alt not in out:
                    out.append(alt)
            for alt in self.fallback.alt_prompts.get(key, ()):
                if alt not in out:
                    out.append(alt)
        return out

    # resolution

    def order(self, detections: Sequence[Any]) -> List[Any]:
        """
        Ordered, canonically-labelled detections. Convenience wrapper around
        :meth:`resolve` for callers that do not care about the count gate.
        """
        return self.resolve(detections).detections

    def resolve(self, detections: Sequence[Any]) -> "LabelResolution":
        # label_resolution imports this module; import at call time.
        from spark_real.perception.label_resolution import resolve_labels

        return resolve_labels(self, detections)

    def with_alt_prompts(self, extra: Dict[str, Sequence[str]]) -> "TaskSpec":
        """
        Copy of this spec whose named groups also answer to ``extra`` prompts.

        ``extra`` maps a group's declared ``text`` (or any label that group
        already accepts) to additional prompt strings. This is the ONLY
        mechanism by which a prompt discovered at run time (an LLM proposal)
        becomes indistinguishable from a hand-written ``alt_prompts`` entry:
        :meth:`PromptGroup.accepts_label` consults ``alt_prompts``, so a
        detection returned under the new prompt is claimed by the group and
        relabelled to the canonical name in :func:`resolve_labels`.

        The spec itself is frozen and shared (the registry caches it for the
        process), so this returns a copy and never mutates the registered one.
        """
        if not extra:
            return self
        wanted = {}
        for key, prompts in extra.items():
            wanted[base_label(key)] = [str(p) for p in prompts if str(p).strip()]
        groups = []
        for group in self.groups:
            adds = wanted.get(group.key, [])
            known = {group.key} | {normalize_instruction(a) for a in group.alt_prompts}
            adds = [a for a in adds if normalize_instruction(a) not in known]
            if adds:
                group = replace(group, alt_prompts=group.alt_prompts + tuple(adds))
            groups.append(group)
        return replace(self, groups=tuple(groups))

    # construction

    @classmethod
    def from_yaml(cls, raw: Dict[str, Any], source: str) -> "TaskSpec":
        task = str(raw.get("task") or "").strip()
        if not task:
            raise ValueError(f"{source}: missing 'task'")
        groups_raw = raw.get("groups") or []
        if not groups_raw:
            raise ValueError(f"{source}: missing 'groups'")
        groups = tuple(PromptGroup.from_yaml(g, source) for g in groups_raw)
        seen = set()
        for g in groups:
            if g.key in seen:
                raise ValueError(f"{source}: duplicate prompt {g.text!r}")
            seen.add(g.key)
        aliases = raw.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        require = raw.get("require") or {}
        if not isinstance(require, dict):
            raise ValueError(f"{source}: 'require' must be a mapping")
        verify = raw.get("verify")
        if verify is not None and not isinstance(verify, dict):
            raise ValueError(f"{source}: 'verify' must be a mapping")
        return cls(
            task=task,
            groups=groups,
            aliases=tuple(str(a) for a in aliases),
            bt=(str(raw["bt"]) if raw.get("bt") else None),
            ordering_camera=(str(raw["ordering_camera"]) if raw.get("ordering_camera") else None),
            per_instance_z=bool(raw.get("per_instance_z", False)),
            require=dict(require),
            fallback=FallbackSpec.from_yaml(raw.get("fallback"), source),
            disambiguate=DisambiguateSpec.from_yaml(raw.get("disambiguate")),
            source_path=source,
            verify=dict(verify) if verify else None,
        )
