"""Parse a plan's verify block, or derive a default one from the actions it ran."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from spark_core.util import as_bool, det_field
from spark_real.control.verify_types import _SCHEMA, Predicate, VerifyBlockError, VerifySpec


def _norm_bool(val: Any) -> bool:
    if isinstance(val, str):
        return as_bool(val, False)
    return bool(val)


def parse_verify_block(
    block: Any,
    *,
    label_resolver: Optional[Callable[[str], Optional[str]]] = None,
    default_min_conf: float = 0.35,
    source: str = "planner",
) -> VerifySpec:
    """Parse a ``verify:`` block. Raises :class:`VerifyBlockError` on anything
    malformed; the whole block is rejected, never partially honoured."""
    if not isinstance(block, Mapping):
        raise VerifyBlockError(f"verify block must be a mapping, got {type(block)}")
    items = block.get("all")
    if not isinstance(items, (list, tuple)) or not items:
        raise VerifyBlockError("verify block needs a non-empty 'all' list")

    preds: List[Predicate] = []
    for raw in items:
        if not isinstance(raw, Mapping):
            raise VerifyBlockError(f"predicate entry must be a mapping: {raw!r}")
        name = str(raw.get("pred", "")).strip()
        if name not in _SCHEMA:
            raise VerifyBlockError(f"unknown predicate '{name}'")
        required, optional = _SCHEMA[name]
        params: Dict[str, Any] = dict(optional)
        for key in required:
            if key not in raw or raw[key] is None:
                raise VerifyBlockError(f"'{name}' missing required param '{key}'")
            params[key] = raw[key]
        for key in optional:
            if key in raw and raw[key] is not None:
                params[key] = raw[key]
        # types
        if name == "held":
            params["value"] = _norm_bool(params["value"])
        if name == "inside":
            params["occlusion_ok"] = _norm_bool(params["occlusion_ok"])
        for key in (
            "xy_margin_m",
            "z_tol_m",
            "z_max_m",
            "xy_tol_m",
            "dz_min_m",
            "dz_max_m",
            "max_dist_m",
        ):
            if key in params:
                try:
                    params[key] = float(params[key])
                except (TypeError, ValueError):
                    raise VerifyBlockError(f"'{name}.{key}' is not a number")
        pred = Predicate(name, params)
        if label_resolver is not None:
            for lab in pred.labels:
                if not label_resolver(lab):
                    raise VerifyBlockError(f"'{name}' references unknown label '{lab}'")
        preds.append(pred)

    min_conf = block.get("min_conf", default_min_conf)
    try:
        min_conf = float(min_conf)
    except (TypeError, ValueError):
        raise VerifyBlockError("min_conf is not a number")
    return VerifySpec(
        predicates=preds,
        min_conf=min_conf,
        require_two_views=_norm_bool(block.get("require_two_views", False)),
        source=source,
    )


_GRASP_TYPES = frozenset(
    ("grasp", "grasp_se3", "grasp_top_down", "grasp_cgn", "grasp_se3_flow")
)

# Primitives that push, pull or drag the object they NAME (as opposed to the
# object already in the jaws). A label one of these touched has a stale
# plan-time pose, so it can never serve as a plan-time reference frame.
_DISPLACING_TYPES = frozenset(
    (
        "open_drawer",
        "close_drawer",
        "pull",
        "push",
        "push_object",
        "drag",
        "wiggle",
        "screw",
        "compliant_push",
        "compliant_insert",
        "throw_to",
        "grasp_perturb",
        "hold_in_place",
        "tilt_wrist",
    )
)

_CONTAINER_MIN_MINOR_M = 0.08


def _labels_touched_by(actions: Sequence[Mapping], types_) -> List[str]:
    """Labels the given action types name, in order, deduplicated.

    A primitive that omits its own label inherits the one from the preceding
    ``move_to_keypoint``.
    """
    out: List[str] = []
    pending: Optional[str] = None
    for act in actions or []:
        atype = str(act.get("type", act.get("name", "")))
        params = act.get("params", {}) or {}
        label = params.get("keypoint_label") or params.get("label")
        if atype == "move_to_keypoint":
            pending = label
            continue
        if atype in types_:
            lab = label or params.get("joint_name") or pending
            if lab and lab not in out:
                out.append(str(lab))
    return out


def _is_container(det: Any) -> bool:
    if det is None:
        return False
    if det_field(det, "slots"):
        return True
    return float(det_field(det, "obb_minor_m", 0.0) or 0.0) >= _CONTAINER_MIN_MINOR_M


def grasped_labels(actions: Sequence[Mapping]) -> List[str]:
    """Labels the tree actually closed the jaws on, in order.

    These are the only objects allowed to have moved during the run, which is
    what lets the verifier re-identify them geometrically after a re-detection
    (see ``success_verifier.IdentityBinder``).
    """
    return _labels_touched_by(actions, _GRASP_TYPES)


def displaced_labels(actions: Sequence[Mapping]) -> List[str]:
    """Every label the tree physically moved, carried OR shoved.

    Superset of :func:`grasped_labels`. Decides whether a label's plan-time
    detection may be reused as a reference frame: everything NOT in this set is
    scenery whose plan-time pose is still valid at verify time.
    """
    out = _labels_touched_by(actions, _GRASP_TYPES | _DISPLACING_TYPES)
    for act in actions or []:
        if str(act.get("type", act.get("name", ""))) != "sweep":
            continue
        # sweep's target_label is the container swept INTO (scenery). Only the
        # object_labels move.
        raw = (act.get("params", {}) or {}).get("object_labels") or ""
        objs = (
            list(raw)
            if isinstance(raw, (list, tuple))
            else [o.strip() for o in str(raw).split(",") if o.strip()]
        )
        for obj in objs:
            if obj and obj not in out:
                out.append(str(obj))
    return out


def derive_default_verify(
    actions: Sequence[Mapping],
    detections: Optional[Mapping[str, Any]] = None,
    *,
    min_conf: float = 0.35,
) -> Optional[VerifySpec]:
    """Derive a predicate block from a flattened action list.

    Returns None when nothing is derivable (pour/scrub/cloth/bimanual); the
    caller must then report ``unverified``, never ``pass``.
    """
    detections = detections or {}
    preds: List[Predicate] = []
    seen = set()

    def add(pred: Predicate):
        key = pred.describe()
        if key not in seen:
            seen.add(key)
            preds.append(pred)

    last_grasped: Optional[str] = None
    pending_keypoint: Optional[str] = None
    place_label: Optional[str] = None
    holding = False

    for act in actions:
        atype = str(act.get("type", act.get("name", "")))
        params = act.get("params", {}) or {}
        label = params.get("keypoint_label") or params.get("label")

        if atype == "move_to_keypoint":
            pending_keypoint = label
            if holding and label:
                place_label = label
            continue

        if atype in _GRASP_TYPES:
            last_grasped = label or pending_keypoint or last_grasped
            holding = True
            place_label = None
            continue

        if atype == "place_in_slot":
            cont = params.get("container_label")
            if last_grasped and cont:
                add(
                    Predicate(
                        "inside",
                        {
                            "obj": last_grasped,
                            "container": cont,
                            "xy_margin_m": -0.02,
                            "z_tol_m": 0.06,
                            "occlusion_ok": False,
                        },
                    )
                )
            holding = False
            place_label = None
            continue

        if atype == "stack":
            base = params.get("target_label")
            if last_grasped and base:
                add(
                    Predicate(
                        "stacked",
                        {
                            "obj": last_grasped,
                            "base": base,
                            "xy_tol_m": 0.035,
                            "dz_min_m": 0.010,
                            "dz_max_m": 0.120,
                        },
                    )
                )
            holding = False
            continue

        if atype == "sweep":
            target = params.get("target_label") or params.get("area_label")
            raw_objs = params.get("object_labels") or ""
            objs = (
                list(raw_objs)
                if isinstance(raw_objs, (list, tuple))
                else [o.strip() for o in str(raw_objs).split(",") if o.strip()]
            )
            for obj in objs:
                if target:
                    add(
                        Predicate(
                            "inside",
                            {
                                "obj": obj,
                                "container": target,
                                "xy_margin_m": -0.02,
                                "z_tol_m": 0.06,
                                "occlusion_ok": False,
                            },
                        )
                    )
            continue

        if atype == "release":
            if last_grasped and place_label:
                cont_det = detections.get(place_label)
                if _is_container(cont_det):
                    add(
                        Predicate(
                            "inside",
                            {
                                "obj": last_grasped,
                                "container": place_label,
                                "xy_margin_m": -0.02,
                                "z_tol_m": 0.06,
                                "occlusion_ok": False,
                            },
                        )
                    )
                else:
                    add(
                        Predicate(
                            "on",
                            {
                                "obj": last_grasped,
                                "surface": place_label,
                                "xy_margin_m": 0.0,
                                "z_max_m": 0.10,
                            },
                        )
                    )
            holding = False
            place_label = None
            continue

    if not preds:
        return None
    if last_grasped:
        add(Predicate("held", {"obj": last_grasped, "value": False}))
    return VerifySpec(predicates=preds, min_conf=min_conf, source="derived")
