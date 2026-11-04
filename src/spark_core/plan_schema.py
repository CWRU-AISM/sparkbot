"""Plan schema: parse an LLM YAML reply into a score and check its optional extension fields."""

import copy
import logging

import yaml

from spark_core import bt_grammar
from spark_core.util import is_num, label_known, walk_nodes

logger = logging.getLogger(__name__)


class PlanSchemaError(ValueError):
    """
    Raised when a plan cannot be parsed, or when a caller asks for strict
    validation and an optional field is malformed.
    """


# Optional plan extensions. A plan that omits all of them parses and executes
# unchanged; the cached trees in a BT library depend on that.

GRASP_STRATEGIES = ("auto", "topdown", "obb", "cgn", "se3")

# Node types that may carry grasp_strategy / grasp_yaw_deg. `grasp` wins on
# conflict (resolved downstream in control/grasp_strategy.py).
STRATEGY_NODE_TYPES = ("move_to_keypoint", "grasp")

# Extension param -> the value that means "not specified". Params equal to
# their default are dropped before hashing so a plan that spells out the
# default hashes identically to a cached tree that omits it.
EXTENSION_DEFAULTS = {"grasp_strategy": "auto", "grasp_yaw_deg": None}

# Score keys that carry no plan semantics and must not shift the BT hash.
# `verify` is a success predicate, not an action -- two plans that differ only
# in how success is judged are the same tree.
PLAN_NONSEMANTIC_KEYS = ("label_corrections", "verify")

_VERIFY_TOP_KEYS = frozenset({"all", "min_conf", "require_two_views"})

# Predicate vocabulary. Mirrors control/success_predicates.py, which is the
# runtime authority; this copy exists so the planner can reject a bad block at
# emit time (that module is numpy+stdlib only and cannot import from here).
_VERIFY_SPEC = {
    "inside": {
        "required": ("obj", "container"),
        "optional": {
            "xy_margin_m": "num",
            "z_tol_m": "num",
            "occlusion_ok": "bool",
        },
    },
    "on": {
        "required": ("obj", "surface"),
        "optional": {"xy_margin_m": "num", "z_max_m": "num"},
    },
    "stacked": {
        "required": ("obj", "base"),
        "optional": {"xy_tol_m": "num", "dz_min_m": "num", "dz_max_m": "num"},
    },
    "near": {"required": ("obj", "target", "max_dist_m"), "optional": {}},
    "removed_from": {"required": ("obj", "container"), "optional": {}},
    "held": {"required": ("obj", "value"), "optional": {}},
    "absent": {"required": ("obj",), "optional": {}},
}

# Predicate fields whose value is a detection label.
_VERIFY_LABEL_FIELDS = ("obj", "container", "surface", "base", "target")

# Sanity ranges. Wide on purpose: these catch unit errors and hallucinated
# values, not borderline choices.
_RANGES = {
    "min_conf": (0.0, 1.0),
    "grasp_yaw_deg": (-360.0, 360.0),
    "xy_margin_m": (-0.5, 0.5),
    "z_tol_m": (0.0, 1.0),
    "z_max_m": (0.0, 1.0),
    "xy_tol_m": (0.0, 1.0),
    "dz_min_m": (0.0, 1.0),
    "dz_max_m": (0.0, 1.0),
    "max_dist_m": (0.0, 5.0),
}

def _range_issue(name: str, value, path: str) -> str:
    lo, hi = _RANGES.get(name, (None, None))
    if lo is None or lo <= float(value) <= hi:
        return ""
    return f"{path}: {name}={value} out of range [{lo}, {hi}]"


def validate_grasp_params(node_type: str, params: dict, path: str) -> list:
    """
    Validate the optional grasp-strategy fields on one node.

    Returns a list of human-readable issues; empty means the node is fine
    (including the common case where neither field is present).
    """
    issues = []
    if not isinstance(params, dict):
        return issues

    for key in EXTENSION_DEFAULTS:
        if key not in params:
            continue
        if node_type not in STRATEGY_NODE_TYPES:
            issues.append(
                f"{path}: '{key}' is only meaningful on "
                f"{'/'.join(STRATEGY_NODE_TYPES)}, not '{node_type}'"
            )

    if "grasp_strategy" in params:
        value = params["grasp_strategy"]
        if not isinstance(value, str) or value not in GRASP_STRATEGIES:
            issues.append(
                f"{path}: unknown grasp_strategy {value!r}; "
                f"expected one of {list(GRASP_STRATEGIES)}"
            )

    if "grasp_yaw_deg" in params:
        value = params["grasp_yaw_deg"]
        if value is not None:
            if not is_num(value):
                issues.append(f"{path}: grasp_yaw_deg must be a number, got {value!r}")
            else:
                bad = _range_issue("grasp_yaw_deg", value, path)
                if bad:
                    issues.append(bad)
        strategy = params.get("grasp_strategy", EXTENSION_DEFAULTS["grasp_strategy"])
        if value is not None and strategy != "obb":
            issues.append(
                f"{path}: grasp_yaw_deg is only read when grasp_strategy is "
                f"'obb' (got {strategy!r})"
            )
    return issues


def validate_verify_block(block, keypoint_labels=None) -> list:
    """
    Validate a top-level `verify:` block against the predicate vocabulary.

    Any issue invalidates the WHOLE block: a partially
    honoured predicate is worse than none, because it silently verifies
    something other than the goal.
    """
    issues = []
    if not isinstance(block, dict):
        return [f"verify: must be a mapping, got {type(block).__name__}"]

    unknown = set(block) - _VERIFY_TOP_KEYS
    if unknown:
        issues.append(f"verify: unknown keys {sorted(unknown)}")

    if "min_conf" in block:
        if not is_num(block["min_conf"]):
            issues.append("verify.min_conf: must be a number")
        else:
            bad = _range_issue("min_conf", block["min_conf"], "verify")
            if bad:
                issues.append(bad)

    if "require_two_views" in block and not isinstance(block["require_two_views"], bool):
        issues.append("verify.require_two_views: must be a boolean")

    preds = block.get("all")
    if preds is None:
        issues.append("verify: missing 'all' (the only combinator in v1)")
        return issues
    if not isinstance(preds, list) or not preds:
        issues.append("verify.all: must be a non-empty list")
        return issues

    known = set(keypoint_labels or [])
    for i, item in enumerate(preds):
        path = f"verify.all[{i}]"
        if not isinstance(item, dict):
            issues.append(f"{path}: must be a mapping")
            continue
        name = item.get("pred")
        spec = _VERIFY_SPEC.get(name) if isinstance(name, str) else None
        if spec is None:
            issues.append(
                f"{path}: unknown pred {name!r}; expected one of {sorted(_VERIFY_SPEC)}"
            )
            continue

        for key in spec["required"]:
            if key not in item:
                issues.append(f"{path}: {name} requires '{key}'")

        allowed = set(spec["required"]) | set(spec["optional"]) | {"pred"}
        for key in set(item) - allowed:
            issues.append(f"{path}: unknown param '{key}' for pred '{name}'")

        for key, kind in spec["optional"].items():
            if key not in item:
                continue
            value = item[key]
            if kind == "bool" and not isinstance(value, bool):
                issues.append(f"{path}: '{key}' must be a boolean")
            elif kind == "num":
                if not is_num(value):
                    issues.append(f"{path}: '{key}' must be a number")
                else:
                    bad = _range_issue(key, value, path)
                    if bad:
                        issues.append(bad)

        if name == "near" and "max_dist_m" in item:
            if not is_num(item["max_dist_m"]) or item["max_dist_m"] <= 0:
                issues.append(f"{path}: near.max_dist_m must be a positive number")
            else:
                bad = _range_issue("max_dist_m", item["max_dist_m"], path)
                if bad:
                    issues.append(bad)
        if name == "held" and "value" in item and not isinstance(item["value"], bool):
            issues.append(f"{path}: held.value must be a boolean")

        for key in _VERIFY_LABEL_FIELDS:
            if key not in item:
                continue
            value = item[key]
            if not isinstance(value, str) or not value.strip():
                issues.append(f"{path}: '{key}' must be a non-empty label string")
            elif not label_known(value, known):
                issues.append(
                    f"{path}: '{key}' references label {value!r} which is not "
                    f"in the detected keypoints {sorted(known)}"
                )
    return issues


def validate_plan_extensions(score, keypoint_labels=None) -> list:
    """
    Validate ONLY the optional extension fields of a score.

    An old-style plan produces an empty list.
    """
    issues = []
    if not isinstance(score, dict):
        return ["score: must be a mapping"]
    if "verify" in score:
        issues.extend(validate_verify_block(score["verify"], keypoint_labels))
    for path, node in walk_nodes(score.get("tree")):
        issues.extend(validate_grasp_params(node.get("type"), node.get("params") or {}, path))
    # Control-flow structure (fallback / retry) and its attempt caps. Adds
    # nothing for a flat sequence, which is every tree in the BT library.
    issues.extend(bt_grammar.control_flow_issues(score.get("tree")))
    return issues


def sanitize_score(score, keypoint_labels=None, strict: bool = False):
    """
    Return ``(clean_score, issues)``.

    Malformed extension fields are LOUD: every issue is logged at WARNING and
    returned to the caller. In non-strict mode the offending field is dropped
    so the run degrades to today's behaviour (blind heuristics) rather than
    executing a half-understood instruction; a malformed `verify:` block is
    dropped whole, never partially honoured. ``strict=True`` raises instead.
    """
    if not isinstance(score, dict):
        raise PlanSchemaError(f"score must be a mapping, got {type(score).__name__}")

    issues = validate_plan_extensions(score, keypoint_labels)
    if not issues:
        return score, []

    if strict:
        raise PlanSchemaError("invalid plan extensions: " + "; ".join(issues))

    for issue in issues:
        logger.warning("[plan-schema] %s", issue)

    clean = copy.deepcopy(score)
    if "verify" in clean and validate_verify_block(clean["verify"], keypoint_labels):
        logger.warning(
            "[plan-schema] rejecting the whole verify block; success will fall "
            "back to the derived predicate"
        )
        clean.pop("verify", None)

    for path, node in walk_nodes(clean.get("tree")):
        params = node.get("params")
        if not isinstance(params, dict):
            continue
        if validate_grasp_params(node.get("type"), params, path):
            for key in EXTENSION_DEFAULTS:
                params.pop(key, None)

    # Control flow is CLAMPED, not dropped: dropping a `retry` would silently
    # reduce the attempts and dropping a `fallback` would run every branch in
    # sequence. An out-of-range attempt count becomes the cap, and an
    # over-wide selector loses its tail branches.
    bt_grammar.normalize(clean.get("tree"))
    return clean, issues


def normalize_plan_for_hash(score):
    """
    Strip everything that must not shift the BT cache key.

    Drops the non-semantic top-level keys at every level, plus any extension
    param that equals its documented default -- so a new-style plan that spells
    out ``grasp_strategy: auto`` hashes identically to the cached tree that
    predates the field. A non-default strategy DOES change the hash: it is a
    different motion.
    """
    if isinstance(score, dict):
        out = {}
        for key, value in score.items():
            if key in PLAN_NONSEMANTIC_KEYS or str(key).startswith("__"):
                continue
            if key in EXTENSION_DEFAULTS and value == EXTENSION_DEFAULTS[key]:
                continue
            out[key] = normalize_plan_for_hash(value)
        return out
    if isinstance(score, list):
        return [normalize_plan_for_hash(v) for v in score]
    return score


def _repair_yaml_text(text: str) -> str:
    """
    One cheap local repair pass over an LLM YAML response.

    Fixes what the model actually gets wrong -- tabs, stray fences, prose
    before the document -- without another API call.
    """
    lines = []
    for line in text.replace("\t", "  ").splitlines():
        if line.strip().startswith("```") or line.strip() == "yaml":
            continue
        lines.append(line)
    for i, line in enumerate(lines):
        if line.startswith("task:") or line.startswith("tree:"):
            return "\n".join(lines[i:]).strip()
    return "\n".join(lines).strip()


def parse_plan_yaml(text: str) -> dict:
    """
    Parse an LLM YAML response into a score dict.

    Raises PlanSchemaError -- never a bare YAMLError, never a 500 out of
    /api/execute -- when the response is not a usable plan.
    """
    if not isinstance(text, str) or not text.strip():
        raise PlanSchemaError("planner returned an empty response")

    parsed = None
    for attempt, candidate in enumerate((text, None)):
        if candidate is None:
            candidate = _repair_yaml_text(text)
            logger.warning("[plan-schema] retrying YAML parse after local repair")
        try:
            parsed = yaml.safe_load(candidate)
        except yaml.YAMLError as exc:
            parsed = None
            last = exc
            continue
        if isinstance(parsed, dict):
            break
        last = PlanSchemaError(f"plan must be a mapping, got {type(parsed).__name__}")
        parsed = None
        if attempt == 0:
            continue

    if not isinstance(parsed, dict):
        raise PlanSchemaError(f"could not parse plan YAML: {last}")
    if "tree" not in parsed:
        raise PlanSchemaError("plan has no 'tree' key")
    return parsed
