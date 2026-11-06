"""Geometric re-binding of predicate labels to fresh detections (never by instance number)."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import numpy as np

from spark_real.bt_label_resolver import LabelResolvingDetectionMap, canon_label
from spark_real.control.success_predicates import (
    container_obb_xy,
    detection_field,
    detection_xyz,
)

logger = logging.getLogger(__name__)

# How far a re-detected object may sit from its plan-time position and still
# count as the same physical object that never moved. Covers centroid jitter
# and depth noise; well under the spacing of distinct utensils on a table.
STATIC_MATCH_TOL_M = 0.06

# Detection fields a predicate reads off a reference object (container /
# surface / base / target). These are the ones taken from the PLAN-TIME
# detection when the reference is plan-anchored; ``mask`` and ``camera`` are
# deliberately NOT here; see _plan_anchored_reference.
_REFERENCE_FIELDS = (
    "label",
    "position_3d",
    "aspect_ratio",
    "world_major_axis_rad",
    "obb_minor_m",
    "slots",
)


def _plan_anchored_reference(plan_det: Any, observed: Any) -> Dict[str, Any]:
    """A reference detection whose POSE is plan-time and whose MASK is now.

    World-frame geometry (position, OBB, slots) comes from the plan-time
    detection: that is the pose the arm actually serviced, and it predates the
    occlusion the placement itself created.

    ``mask`` and ``camera`` must NOT come from plan time. The 2D containment
    fallback compares two masks pixel-for-pixel in ONE camera's image, and the
    plan-time map is a cross-camera merge; its mask may belong to a different
    view entirely. So the image-space fields stay with the fresh detection in
    this camera, and the world-frame fields come from plan time.

    ``confidence`` likewise comes from the fresh detection: it answers "did
    this camera see the container just now", which is what licenses reusing the
    plan-time pose at all. Falls back to the plan-time value when the
    re-detection has none.
    """
    ref: Dict[str, Any] = {name: detection_field(plan_det, name) for name in _REFERENCE_FIELDS}
    conf = detection_field(observed, "confidence") if observed is not None else None
    ref["confidence"] = conf if conf is not None else detection_field(plan_det, "confidence")
    # ``observed`` is None when verify_scope dropped this reference's prompt
    # altogether (verification.scope.anchor_references). There is then no
    # image-space evidence in this camera at all, so the 2D containment
    # fallback abstains, which is why the prompt is only dropped when that
    # fallback is switched off. The world-frame fields, which is all the 3D
    # test reads, are unaffected.
    ref["mask"] = detection_field(observed, "mask") if observed is not None else None
    ref["camera"] = detection_field(observed, "camera") if observed is not None else None
    ref["plan_anchored"] = True
    return ref


def _plan_key(plan_map: Any, label: Any) -> Optional[str]:
    """The plan-time detection key a predicate label refers to, or None.

    NOT ``LabelResolvingDetectionMap.resolve_label``. That resolver exists to
    keep a cached tree DRIVING when labels drift, so it deliberately collapses
    ``"fork"`` (and even ``"fork 9"``) onto the lowest-numbered instance. That
    is the right call for dispatching a primitive and the wrong one here: this
    key decides WHICH physical object a success predicate is about, and a
    silent collapse binds the predicate to a fork the tree never touched.

    So: an exact key, a declared role alias, or a canonical base with exactly
    one plan-time instance. More than one instance is ambiguous and returns
    None, which the binder and the predicates read as abstain.
    """
    key = str(label or "").strip()
    if not key or not plan_map:
        return None
    # Iteration yields the real detected labels for both map types.
    keys = list(plan_map)
    if key in keys:
        return key
    # A role alias is an authoritative binding published by the task's
    # perception contract, so it outranks the ambiguity rule below.
    alias = (getattr(plan_map, "aliases", None) or {}).get(key)
    if alias in keys:
        return alias
    want = canon_label(key)
    hits = [k for k in keys if canon_label(k) == want]
    if len(hits) > 1:
        return None
    if hits:
        return hits[0]
    # No instance of this base at all: sub-part / super-part resolution
    # ("knife" -> "knife handle") is unambiguous by construction.
    if isinstance(plan_map, LabelResolvingDetectionMap):
        return plan_map.resolve_label(key)
    return None


def _xy_dist(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a)[:2] - np.asarray(b)[:2]))


class IdentityBinder:
    """Bind a predicate label to the ONE physical object it was written about.

    Instance numbers are assigned per detection pass and permute freely between
    passes: SAM3 orders them geometrically, so the moment an object moves the
    numbering re-sorts. Re-resolving ``"fork 1"`` against the verify-time
    detections therefore binds the predicate to whatever object happens to be
    first NOW. Identity must come from geometry, not the number.

    Two rules, both anchored on the plan-time detection map the tree was
    written against:

      * a label the tree never grasped did not move, so it binds to the
        re-detection nearest its plan-time position, and only when that
        nearest match is unique inside the tolerance;
      * a label the tree DID grasp is the only object allowed to have moved,
        so it binds to the re-detection left over once every *other* plan-time
        instance of the same base label has claimed its own, still-stationary
        re-detection.

    ``plan_anchored`` is a third, narrower rule for REFERENCE objects (the
    container/surface/base a predicate measures against, when the tree never
    touched it). Those bind to their PLAN-TIME pose and are not re-located at
    all; see :meth:`_bind_reference`.

    Every other case (a sibling that vanished, two candidates on top of each
    other, more than one unaccounted detection, no plan-time anchor) returns
    None, which the predicates read as abstain. Abstaining is a fine outcome;
    binding to the wrong object is the failure this class prevents.
    """

    def __init__(
        self,
        plan_map: Any,
        detections: Iterable[Any],
        manipulated: Optional[Set[str]] = None,
        tol: float = STATIC_MATCH_TOL_M,
        plan_anchored: Optional[Set[str]] = None,
        pose_only: Optional[Set[str]] = None,
    ):
        self._plan = plan_map if isinstance(plan_map, Mapping) else {}
        self._manipulated = set(manipulated or ())
        self._plan_anchored = set(plan_anchored or ())
        # Plan keys whose prompt was never issued (verify_scope dropped it), so
        # "no candidate in this camera" is the scope's doing and must not abstain.
        self._pose_only = set(pose_only or ())
        self._tol = float(tol)
        self._by_base: Dict[str, List[Tuple[str, Any, Optional[np.ndarray]]]] = {}
        for det in detections or ():
            label = str(getattr(det, "label", "") or "")
            self._by_base.setdefault(canon_label(label), []).append(
                (label, det, detection_xyz(det))
            )
        self._cache: Dict[str, Tuple[Optional[Any], str]] = {}

    # public surface consumed by SceneView

    def lookup(self, label: Any) -> Optional[Any]:
        return self._resolve(label)[0]

    def explain(self, label: Any) -> str:
        return self._resolve(label)[1]

    def _resolve(self, label: Any) -> Tuple[Optional[Any], str]:
        key = str(label or "")
        if key not in self._cache:
            self._cache[key] = self._bind(key)
            det, why = self._cache[key]
            if det is None:
                logger.info("Identity binder: %r unbound (%s)", key, why)
            else:
                logger.info("Identity binder: %r -> %s", key, why)
        return self._cache[key]

    # binding

    def _plan_siblings(self, base: str, exclude: Optional[str]):
        """Plan-time instances of the same base label, minus the bound one."""
        return [
            (k, detection_xyz(v))
            for k, v in self._plan.items()
            if canon_label(k) == base and k != exclude
        ]

    def _bind(self, label: str) -> Tuple[Optional[Any], str]:
        base = canon_label(label)
        cands = list(self._by_base.get(base, ()))
        plan_key = _plan_key(self._plan, label)
        if not cands:
            if plan_key is not None and plan_key in self._pose_only:
                plan_det = dict.get(self._plan, plan_key)
                if detection_xyz(plan_det) is None:
                    return None, f"reference {plan_key!r} has no plan-time 3d anchor"
                return _plan_anchored_reference(plan_det, None), (
                    f"plan-time pose of {plan_key!r} (prompt not issued: "
                    f"verification.scope.anchor_references)"
                )
            return None, "not detected in this camera"

        # A sibling the tree also moved has no anchor left to claim a detection with.
        siblings = [
            s for s in self._plan_siblings(base, plan_key) if s[0] not in self._manipulated
        ]
        if plan_key is None:
            # No anchor to check against. A label the scene has exactly one of
            # is still the only candidate there is; anything else is ambiguous.
            if len(cands) == 1 and not siblings:
                return cands[0][1], f"sole {base!r} in view (no plan-time anchor)"
            return None, f"{len(cands)} {base!r} detections and no plan-time anchor"
        # No "sole candidate" shortcut past here. Being the only detection in
        # view is not evidence of identity: an object the tree never grasped
        # that is now 28cm from where it was planned is either a different
        # object or one that moved unobserved, and binding to it is exactly the
        # wrong-object pass this class exists to stop. Geometry decides.
        if plan_key in self._manipulated:
            return self._bind_moved(base, cands, siblings)
        if plan_key in self._plan_anchored:
            return self._bind_reference(cands, plan_key)
        return self._bind_static(base, cands, plan_key)

    def _bind_reference(self, cands, plan_key) -> Tuple[Optional[Any], str]:
        """A reference frame the tree never touched: use its plan-time pose.

        Re-detecting the container of a containment predicate is a pure loss.
        The tree does not touch it, a bowl on a table does not move on its own,
        and the object JUST PLACED occludes its interior, so the surviving
        mask is a partial rim and the depth behind those pixels is sampled off
        the placed object, which slides the backprojected centroid along the
        camera ray. Measured: 8.5cm (birdview), 13.7cm and 17.3cm (sideview) of
        apparent motion on a bowl of radius 8cm, on places that physically
        succeeded. A successful place is exactly what corrupts this measurement.

        The re-detection is still REQUIRED to exist: this method is only
        reached with a non-empty candidate list, so a container that vanished
        from a camera's view still abstains there. What the re-detection may no
        longer do is supply the pose.

        Disagreement policy: a re-detection far from the plan-time anchor is
        LOGGED (the distance goes into the bind explanation and the trace) but
        does not change the binding and does not force an abstain. There is no
        honest threshold available: the drift induced by occlusion is
        unbounded along the camera ray, so any cutoff loose enough to admit the
        17.3cm case is loose enough to admit a container that genuinely moved.
        What guards that case instead is presence (above), the exclusion of any
        container the tree itself displaced (``displaced_labels``), and the
        degenerate-extent abstain in success_predicates.
        """
        plan_det = dict.get(self._plan, plan_key)
        anchor = detection_xyz(plan_det)
        if anchor is None:
            return None, f"reference {plan_key!r} has no plan-time 3d anchor"
        scored = sorted(
            ((_xy_dist(c[2], anchor), c) for c in cands if c[2] is not None),
            key=lambda t: t[0],
        )
        if scored:
            nearest = scored[0][1]
            seen = f"re-detected {scored[0][0] * 100:.1f}cm away"
        else:
            nearest = cands[0]
            seen = "re-detected without 3d"
        ref = _plan_anchored_reference(plan_det, nearest[1])
        return ref, (
            f"plan-time pose of {plan_key!r} ({seen}; the re-detection is not "
            f"used as the reference frame)"
        )

    def _bind_moved(self, base, cands, siblings) -> Tuple[Optional[Any], str]:
        """The grasped object is whatever the stationary siblings do not claim."""
        unclaimed = list(cands)
        for name, anchor in siblings:
            if anchor is None:
                return None, f"un-manipulated {name!r} has no plan-time 3d anchor"
            near = []
            for c in unclaimed:
                d = _xy_dist(c[2], anchor)  # None (no 3d) is not a match; 0.0 is
                if d is not None and d <= self._tol:
                    near.append(c)
            if len(near) != 1:
                return None, (
                    f"{len(near)} detections within {self._tol * 100:.0f}cm of un-manipulated "
                    f"{name!r}; identity ambiguous"
                )
            unclaimed.remove(near[0])
        if len(unclaimed) != 1:
            return None, f"{len(unclaimed)} unaccounted {base!r} detections after elimination"
        if unclaimed[0][2] is None and len(cands) > 1:
            return None, f"leftover {base!r} carries no 3d; identity unconfirmable"
        return unclaimed[0][1], f"{unclaimed[0][0]!r} by elimination ({len(siblings)} sibling(s))"

    def _bind_static(self, base, cands, plan_key) -> Tuple[Optional[Any], str]:
        """An object the tree never touched must still be where it was."""
        anchor = detection_xyz(dict.get(self._plan, plan_key))
        if anchor is None:
            return None, f"{plan_key!r} has no plan-time 3d anchor and {len(cands)} candidates"
        scored = sorted(
            ((_xy_dist(c[2], anchor), c) for c in cands if c[2] is not None),
            key=lambda t: t[0],
        )
        if not scored:
            return None, f"no {base!r} candidate carries 3d"
        if len(scored) < len(cands):
            return None, f"{len(cands) - len(scored)} {base!r} candidate(s) lack 3d; ambiguous"
        tol = self._static_tol(dict.get(self._plan, plan_key))
        best_d, best = scored[0]
        if best_d > tol:
            return None, f"nearest {base!r} is {best_d * 100:.1f}cm from its plan-time position"
        if len(scored) > 1 and scored[1][0] <= tol:
            return None, f"two {base!r} candidates within {tol * 100:.0f}cm; ambiguous"
        return best[1], f"{best[0]!r} {best_d * 100:.1f}cm from its plan-time position"

    def _static_tol(self, plan_det: Any) -> float:
        """Drift a stationary object's *measured centroid* may show.

        The flat tolerance assumes the mask is stable, which holds for an
        object nothing happened to. It does NOT hold for the container just
        placed into: the object now occupying it occludes its interior, so the
        surviving mask is a partial rim whose centroid sits well off the true
        centre (measured: mask fill 0.75 -> 0.37 and a 7.3cm centroid shift on
        a bowl that did not move).

        A stationary container's centroid cannot drift beyond its own radius,
        so that radius is the physically meaningful bound. Objects with no
        measured extent (no slots, no obb_minor_m) keep the flat tolerance;
        widening those is what would let a utensil dropped BESIDE the tray
        bind to the tray's own anchor.

        Never shrinks below the flat value.
        """
        obb = container_obb_xy(plan_det) if plan_det is not None else None
        if obb is None:
            return self._tol
        _, half_a, half_b, _ = obb
        return max(self._tol, float(max(half_a, half_b)))
