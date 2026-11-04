"""Small hardware-free helpers: env flags, LLM reply fences, detection fields, score-tree walks."""

import os
import re
from typing import Any

_TRUE = ("1", "true", "yes", "on")
_INSTANCE_SUFFIX = re.compile(r"\s+(\d+)\s*$")


def as_bool(value, default: bool) -> bool:
    """None -> default; str by the 1/true/yes/on vocabulary (case/space-insensitive); else bool()."""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in _TRUE
    return bool(value)


def env_flag(name: str, default: bool = False) -> bool:
    return as_bool(os.environ.get(name), default)


def det_field(det, key, default=None):
    """Read `key` from `det`; None (missing det or missing/None value) -> default."""
    if det is None:
        return default
    v = det.get(key, default) if hasattr(det, "get") else getattr(det, key, default)
    return default if v is None else v


def base_label(label: Any) -> str:
    """
    Strip a trailing instance number: ``"fork 2"`` -> ``"fork"``.
    """
    return _INSTANCE_SUFFIX.sub("", str(label)).strip()



def is_num(value) -> bool:
    # bool is an int subclass; a bool where a number belongs is a schema error.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def walk_nodes(node, path: str = "tree"):
    """Yield (path, node) for every dict node in a tree, depth first."""
    if not isinstance(node, dict):
        return
    yield path, node
    for i, child in enumerate(node.get("children") or []):
        yield from walk_nodes(child, f"{path}/{i}")


def label_known(label: str, known) -> bool:
    """True if `label` names one of the detected keypoints.

    Tolerates instance numbering in either direction ("fork 1" vs "fork"),
    matching how LabelResolvingDetectionMap binds labels at execution time.
    """
    if not known:
        return True
    if label in known:
        return True
    base = base_label(label)
    return base in known or any(base_label(k) == base for k in known)
