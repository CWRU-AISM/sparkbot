"""Verification vocabulary: statuses, the verify-block schema, predicates and outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple


class VerifyBlockError(ValueError):
    """A ``verify:`` block is malformed; the whole block must be rejected."""


_SCHEMA: Dict[str, Tuple[Tuple[str, ...], Dict[str, Any]]] = {
    "inside": (
        ("obj", "container"),
        {"xy_margin_m": -0.02, "z_tol_m": 0.06, "occlusion_ok": False},
    ),
    "on": (("obj", "surface"), {"xy_margin_m": 0.0, "z_max_m": 0.10}),
    "stacked": (
        ("obj", "base"),
        {"xy_tol_m": 0.035, "dz_min_m": 0.010, "dz_max_m": 0.120},
    ),
    "near": (("obj", "target", "max_dist_m"), {}),
    "removed_from": (("obj", "container"), {}),
    "held": (("obj", "value"), {}),
    "absent": (("obj",), {}),
}

# which params name a detection label
_LABEL_KEYS = ("obj", "container", "surface", "base", "target")

# ... and which of those name the REFERENCE FRAME rather than the object under
# test. The tree moves ``obj``; the container/surface/base/target it is judged
# against is scenery the tree never touched.
_REFERENCE_KEYS = ("container", "surface", "base", "target")


@dataclass(frozen=True)
class Predicate:
    pred: str
    params: Mapping[str, Any]

    @property
    def labels(self) -> List[str]:
        """Detection labels this predicate needs (``held`` needs none)."""
        return self._labels_for(_LABEL_KEYS)

    @property
    def reference_labels(self) -> List[str]:
        """Labels naming the reference frame this predicate measures against."""
        return self._labels_for(_REFERENCE_KEYS)

    @property
    def object_labels(self) -> List[str]:
        """Labels naming the object under test."""
        return self._labels_for(("obj",))

    def _labels_for(self, keys: Sequence[str]) -> List[str]:
        if self.pred == "held":
            return []
        out = []
        for key in keys:
            val = self.params.get(key)
            if isinstance(val, str) and val:
                out.append(val)
        return out

    def describe(self) -> str:
        if self.pred == "held":
            return f"held({self.params.get('obj', '?')}, {bool(self.params.get('value'))})"
        args = [str(self.params[k]) for k in _LABEL_KEYS if k in self.params]
        if self.pred == "near":
            args.append(f"{float(self.params['max_dist_m']):.3f}m")
        return f"{self.pred}({', '.join(args)})"


@dataclass
class VerifySpec:
    predicates: List[Predicate]
    min_conf: float = 0.35
    require_two_views: bool = False
    source: str = "planner"

    @property
    def labels(self) -> List[str]:
        return self._union("labels")

    @property
    def reference_labels(self) -> List[str]:
        return self._union("reference_labels")

    @property
    def object_labels(self) -> List[str]:
        return self._union("object_labels")

    def _union(self, attr: str) -> List[str]:
        seen, out = set(), []
        for p in self.predicates:
            for lab in getattr(p, attr):
                if lab not in seen:
                    seen.add(lab)
                    out.append(lab)
        return out

    def describe(self) -> List[str]:
        return [p.describe() for p in self.predicates]
