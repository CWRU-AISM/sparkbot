"""Task-success verification: cheap gates, per-camera votes, ENPIRE fusion.

Order of work, cheapest first:

  G1 grasp     did the jaws ever hold anything (gObj / TCP wrench / jaw pos)
  G2 transport was the object still held at every stationary waypoint
  G3 release   was it held immediately before opening and gone right after

A gate that is explicitly False short-circuits to ``fail`` with NO capture and
NO SAM3; a drop mid-transport is caught for free. Only if the gates are clean
does the vision path run: every camera detects independently (no cross-camera
merge, no shared masks) and judges the whole predicate list on its own; the
per-camera verdicts are then fused.

Predicate labels are re-bound to detections GEOMETRICALLY, never by instance
number; see :class:`IdentityBinder`. Instance numbering is not stable across
detection passes, so re-resolving ``"fork 1"`` after the verify re-detection
could bind a predicate to a different physical fork than the tree manipulated.

Fail-closed everywhere: a missing check, a throwing check, or no predicate at
all yields ``unverified``, never ``pass``. ``unverified`` must neither reinforce
nor suppress a BT-library row.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from spark_core.util import as_bool
from spark_real.bt_label_resolver import LabelResolvingDetectionMap, base_label, canon_label
from spark_real.control import verify_scope
from spark_real.control.identity_binder import (  # noqa: F401
    _REFERENCE_FIELDS,
    STATIC_MATCH_TOL_M,
    IdentityBinder,
    _plan_anchored_reference,
    _plan_key,
    _xy_dist,
)
from spark_real.control.success_predicates import (
    ABSTAIN,
    FAIL,
    PASS,
    UNVERIFIED,
    CameraVote,
    EvalConfig,
    SceneView,
    VerifyBlockError,
    VerifyOutcome,
    VerifySpec,
    camera_verdict,
    derive_default_verify,
    detection_xyz,
    displaced_labels,
    evaluate_predicate,
    parse_verify_block,
)
from spark_real.control.verify_gates import (  # noqa: F401
    GATE_NAMES,
    ReleaseWitness,
    _confirm_jaws_open,
    _gates,
    _has_jaw_source,
    _is_robotiq,
    _jaw_sample,
    _jaws_opened,
    _read_holding,
    _verdict_holding,
    begin_release_witness,
    collect_gates,
    finish_release_witness,
    note_transport_drop,
    record_grasp_verdict,
    reset_run_state,
    set_gate,
)

logger = logging.getLogger(__name__)

CAMERA_ORDER = ("birdview", "sideview", "wrist")


@dataclass
class VerifyConfig:
    enabled: bool = True
    min_conf: float = 0.35
    require_two_views: bool = False
    inside_xy_margin_m: float = -0.02
    inside_z_tol_m: float = 0.06
    # > 1.0 disables the 2D fallback (votes abstain instead). Shipped OFF:
    # replayed over 70 corpus episodes x 3 cameras the image-space test passed
    # 6 temporal NEGATIVES and only 1 true positive, and 2D-failed 42 true
    # positives; as a veto it would suppress real successes. See
    # tests/test_verify_replay.py. Set 0.90 to re-enable.
    fallback_2d_containment: float = 1.01
    on_missing: str = UNVERIFIED

    @classmethod
    def from_raw(cls, raw: Any) -> "VerifyConfig":
        block = (raw or {}).get("verification") if hasattr(raw, "get") else None
        cfg = cls()
        if not isinstance(block, dict):
            return cfg
        for key in (
            "enabled",
            "min_conf",
            "require_two_views",
            "inside_xy_margin_m",
            "inside_z_tol_m",
            "fallback_2d_containment",
            "on_missing",
        ):
            if block.get(key) is None:
                continue
            cur, val = getattr(cfg, key), block[key]
            if isinstance(cur, bool):
                val = (
                    as_bool(val, False)
                    if isinstance(val, str)
                    else bool(val)
                )
            elif isinstance(cur, float):
                val = float(val)
            else:
                val = str(val)
            setattr(cfg, key, val)
        if cfg.on_missing == PASS:
            logger.warning("verification.on_missing=pass is not allowed; using unverified")
            cfg.on_missing = UNVERIFIED
        return cfg

    @classmethod
    def from_pipeline(cls, pipeline: Any) -> "VerifyConfig":
        profile = getattr(pipeline, "profile", None)
        return cls.from_raw(getattr(profile, "raw", None))


# gate bookkeeping: executors call these; the verifier only reads


# identity binding


# fusion


def fuse_votes(
    votes: Sequence[CameraVote],
    require_two_views: bool = False,
    silent_cameras: Sequence[str] = (),
) -> Tuple[str, str]:
    """ENPIRE fusion. Returns (status, reason).

    ``silent_cameras`` captured but resolved nothing, so they emit no vote.
    They are named in the reason as ``silent`` rather than simply omitted:
    with ``require_two_views`` off a lone camera carries the whole verdict,
    and "birdview was unplugged" must not read the same as "birdview agreed".
    """
    by_cam: Dict[str, List[CameraVote]] = {}
    for v in votes:
        by_cam.setdefault(v.camera, []).append(v)
    if not by_cam:
        return UNVERIFIED, "no camera produced a vote"

    verdicts = {cam: camera_verdict(vs) for cam, vs in by_cam.items()}
    labelled = dict(verdicts)
    for cam in silent_cameras:
        labelled.setdefault(cam, "silent")
    summary = ", ".join(f"{c}={labelled[c]}" for c in sorted(labelled))

    failing = [c for c, v in verdicts.items() if v == FAIL]
    if failing:
        return FAIL, f"camera(s) {sorted(failing)} vote fail ({summary})"

    passing = [c for c, v in verdicts.items() if v == PASS]
    # A 2D pass corroborates only: a fused pass needs a 3D pass vote.
    passing_3d = [c for c in passing if any(v.mode == "3d" and v.vote == PASS for v in by_cam[c])]
    if not passing_3d:
        return UNVERIFIED, f"no 3d pass vote ({summary})"
    if require_two_views and len(passing) < 2:
        return UNVERIFIED, f"require_two_views but only {len(passing)} pass ({summary})"
    return PASS, f"{len(passing)} camera(s) pass ({summary})"


class SuccessVerifier:
    """Produces the single authoritative verdict for a run."""

    def __init__(self, executor=None, pipeline=None, config: Optional[VerifyConfig] = None):
        self.executor = executor
        self.pipeline = pipeline if pipeline is not None else getattr(executor, "_pipeline", None)
        self.config = config or VerifyConfig.from_pipeline(self.pipeline)
        # How much perception this verification may ask for. See verify_scope
        # for the four cuts and the evidence each one trades away.
        self.scope = verify_scope.ScopeConfig.from_pipeline(self.pipeline)
        # Set by _verify: the capture timestamp the detections were computed
        # from, so the memo can only serve detections of the same frames.
        self._capture_t = 0.0

    # spec resolution (planner -> task yaml -> derived)

    def resolve_spec(self, score: Any, actions: Sequence[dict]) -> Optional[VerifySpec]:
        detections = getattr(self.executor, "detection_map", None) or {}
        resolver = None
        if isinstance(detections, LabelResolvingDetectionMap):
            resolver = lambda lab: lab if detections.get(lab) is not None else None  # noqa: E731

        for block, source in (
            ((score or {}).get("verify") if isinstance(score, dict) else None, "planner"),
            (getattr(self.executor, "_task_verify_block", None), "task-yaml"),
        ):
            if not block:
                continue
            try:
                spec = parse_verify_block(
                    block,
                    label_resolver=resolver,
                    default_min_conf=self.config.min_conf,
                    source=source,
                )
                logger.info("Verify spec from %s: %s", source, spec.describe())
                return self._apply_config_floor(spec)
            except VerifyBlockError as exc:
                logger.warning("Rejecting %s verify block: %s", source, exc)

        spec = derive_default_verify(actions or [], detections, min_conf=self.config.min_conf)
        if spec is None:
            logger.warning("No verify predicate derivable from the tree")
            return None
        logger.info("Verify spec derived from tree: %s", spec.describe())
        return self._apply_config_floor(spec)

    def _apply_config_floor(self, spec: VerifySpec) -> VerifySpec:
        """Config is a floor, never a ceiling: it can only tighten a spec."""
        if self.config.require_two_views:
            spec.require_two_views = True
        spec.min_conf = max(spec.min_conf, self.config.min_conf)
        return spec

    # vision

    def _capture(self) -> Dict[str, dict]:
        if self.pipeline is None:
            return {}
        captures, stamp, _reused = verify_scope.captures_for_verify(self.pipeline, self.scope)
        self._capture_t = stamp
        return captures

    def _pose_only_keys(self, anchored: Set[str]) -> Set[str]:
        """Plan-anchored references whose prompt the scope may drop entirely.

        Only when the 2D containment fallback is OFF. That fallback is the one
        consumer of a reference's fresh MASK, and a reference bound from plan
        time alone has none. With the fallback on, the prompt is still issued
        so the image-space branch keeps its evidence.
        """
        if not (self.scope.enabled and self.scope.anchor_references):
            return set()
        if self.config.fallback_2d_containment <= 1.0:
            logger.info(
                "Verify: not dropping reference prompts (the 2D containment "
                "fallback is on and needs a fresh mask)"
            )
            return set()
        return set(anchored or ())

    def _detect_per_camera(
        self,
        captures: Dict[str, dict],
        labels: Sequence[str],
        multi_bases: Optional[Set[str]] = None,
        pose_only: Optional[Set[str]] = None,
    ) -> Dict[str, List[Any]]:
        """Detect independently in every camera. No cross-camera merge, ever.

        EVERY detection is kept, duplicates included: collapsing repeated
        labels would hide a second instance and make an ambiguous scene look
        unambiguous to the identity binder.

        ``multi_bases`` are promoted to multi-instance prompts, because the
        default top-1-per-prompt would hand the binder one fork in a two-fork
        scene; an ambiguous scene the binder cannot see is ambiguous, so it
        would bind to the survivor and pass on the wrong object. Reference
        labels (containers, surfaces) stay top-1: a phantom second tray only
        costs an abstain, and abstaining on every container is worse.

        ``pose_only`` names references the binder will serve from the plan-time
        map, so their prompt is not issued at all, one SAM3 pass per camera
        saved per reference. See verify_scope.
        """
        plan = verify_scope.plan_prompts(labels, pose_only or (), self.scope, base_label)
        if plan.dropped:
            logger.info(
                "Verify: %d prompt(s) dropped (%s); detecting %s",
                plan.saved,
                plan.reason,
                plan.prompts,
            )
        if not plan.prompts or self.pipeline is None:
            return {}
        multi = {p for p in plan.prompts if p in (multi_bases or set())}
        dets = verify_scope.detect_scoped(
            self.pipeline,
            captures,
            plan.prompts,
            self.scope,
            multi_instance_prompts=multi,
            capture_time=self._capture_t,
        )
        return verify_scope.per_camera(dets)

    @staticmethod
    def _multi_instance_bases(spec: VerifySpec, plan_map: Any, manipulated: Set[str]) -> Set[str]:
        """Base labels the verify re-detection must see EVERY instance of.

        The objects under test (predicate ``obj``, anything the tree grasped)
        and any base the plan already saw twice. Hiding a duplicate of one of
        those is what lets a predicate bind to the wrong physical object.
        """
        out = {base_label(lab) for lab in manipulated}
        for pred in spec.predicates:
            obj = pred.params.get("obj")
            if isinstance(obj, str) and obj:
                out.add(base_label(obj))
        counts: Dict[str, int] = {}
        for key in plan_map or ():
            counts[canon_label(key)] = counts.get(canon_label(key), 0) + 1
        for key in plan_map or ():
            if counts.get(canon_label(key), 0) > 1:
                out.add(base_label(key))
        return {b for b in out if b}

    @staticmethod
    def _plan_anchored_keys(spec: VerifySpec, plan_map: Any, displaced: Set[str]) -> Set[str]:
        """Plan-time detection keys whose pose is the verify-time reference frame.

        A key qualifies only when all three hold:

          * it is used as a REFERENCE (container / surface / base / target),
            never as the ``obj`` under test in any predicate; the object is
            the thing that moved, and it must always be re-detected;
          * the tree did not displace it (``displaced_labels`` covers carrying
            it and shoving it: open_drawer, push, pull, drag, sweep, ...), so
            its plan-time pose is still the truth;
          * the predicate label resolves to exactly one plan-time detection
            (``_plan_key``); an ambiguous label anchors nothing and falls
            through to the ordinary geometric binding, which abstains.
        """
        objects = {
            key
            for key in (_plan_key(plan_map, lab) for lab in spec.object_labels)
            if key is not None
        }
        out: Set[str] = set()
        for lab in spec.reference_labels:
            key = _plan_key(plan_map, lab)
            if key is None or key in objects or key in displaced:
                continue
            out.add(key)
        return out

    @staticmethod
    def _depth_source(captures: Dict[str, dict], pipeline) -> str:
        use_hw = bool(getattr(getattr(pipeline, "config", None), "use_hardware_depth", True))
        kinds = set()
        for cam in captures.values():
            has_depth = (cam or {}).get("depth") is not None
            kinds.add("hardware" if (has_depth and use_hw) else "da3")
        if not kinds:
            return "none"
        return kinds.pop() if len(kinds) == 1 else "mixed"

    # main entry

    def verify(self, score: Any, actions: Sequence[dict]) -> VerifyOutcome:
        t0 = time.time()
        try:
            return self._verify(score, actions)
        except Exception:
            logger.exception("Verification raised; reporting unverified (fail-closed)")
            gates, _ = collect_gates(self.executor)
            return VerifyOutcome(
                status=UNVERIFIED,
                gates=gates,
                reason="verification raised an exception",
            )
        finally:
            logger.info("Verification took %.2fs", time.time() - t0)

    def _verify(self, score: Any, actions: Sequence[dict]) -> VerifyOutcome:
        gates, failures = collect_gates(self.executor)
        if not self.config.enabled:
            return VerifyOutcome(status=UNVERIFIED, gates=gates, reason="verification disabled")
        if failures:
            return VerifyOutcome(
                status=FAIL,
                gates=gates,
                reason=f"gate(s) failed before any capture: {failures}",
            )

        spec = self.resolve_spec(score, actions)
        if spec is None:
            return VerifyOutcome(
                status=self.config.on_missing, gates=gates, reason="no predicate available"
            )
        names = spec.describe()

        captures = self._capture()
        if not captures:
            return VerifyOutcome(
                status=UNVERIFIED, predicates=names, gates=gates, reason="no camera captured"
            )

        # Which labels the tree moved: the only ones whose plan-time position
        # is allowed to be stale when the binder re-identifies them.
        plan_map = getattr(self.executor, "detection_map", None) or {}
        manipulated = {
            key
            for key in (_plan_key(plan_map, lab) for lab in displaced_labels(actions))
            if key is not None
        }
        anchored = self._plan_anchored_keys(spec, plan_map, manipulated)
        if anchored:
            logger.info("Verify: reference frame(s) taken from plan time: %s", sorted(anchored))
        pose_only = self._pose_only_keys(anchored)

        # Cameras the labels under test cannot project into can only abstain,
        # so they are not detected on at all. Falls back to every camera when
        # the plan-time geometry does not support the test.
        anchor_keys = {_plan_key(plan_map, lab) for lab in spec.labels}
        anchors = [
            detection_xyz(dict.get(plan_map, key)) for key in anchor_keys if key is not None
        ]
        keep, dropped_cams = verify_scope.plan_cameras(captures, anchors, self.scope)
        if dropped_cams:
            logger.info(
                "Verify: %s cannot see %s; not detecting there",
                sorted(dropped_cams),
                spec.labels,
            )
        captures = {c: captures[c] for c in keep}
        depth_source = self._depth_source(captures, self.pipeline)

        per_cam = self._detect_per_camera(
            captures,
            spec.labels,
            self._multi_instance_bases(spec, plan_map, manipulated),
            pose_only=pose_only,
        )

        witness = getattr(self.executor, "_release_witness", None)
        held = witness.held_after if isinstance(witness, ReleaseWitness) else None
        if held is None:
            held = getattr(self.executor, "_holding", None)

        cfg = EvalConfig(
            min_conf=spec.min_conf,
            fallback_2d_containment=self.config.fallback_2d_containment,
            inside_xy_margin_m=self.config.inside_xy_margin_m,
            inside_z_tol_m=self.config.inside_z_tol_m,
            z_tol_scale=2.0 if depth_source in ("da3", "mixed") else 1.0,
        )

        cams = [c for c in CAMERA_ORDER if c in captures]
        cams += [c for c in captures if c not in CAMERA_ORDER]
        votes: List[CameraVote] = []
        silent: List[str] = []
        for cam in cams:
            cam_dets = per_cam.get(cam)
            if not cam_dets:
                logger.info("Verify: %s detected nothing, contributes no vote", cam)
                silent.append(cam)
                continue
            binder = IdentityBinder(
                plan_map,
                cam_dets,
                manipulated,
                plan_anchored=anchored,
                pose_only=pose_only,
            )
            view = SceneView(
                camera=cam,
                lookup=binder.lookup,
                held=None if held is None else bool(held),
                depth_source=depth_source,
                explain=binder.explain,
            )
            for pred in spec.predicates:
                vote = evaluate_predicate(pred, view, cfg)
                votes.append(vote)
                logger.info(
                    "Verify [%s] %s -> %s (%s, conf=%.2f) %s",
                    cam,
                    vote.predicate,
                    vote.vote,
                    vote.mode,
                    vote.confidence,
                    vote.detail,
                )

        status, reason = fuse_votes(
            votes, require_two_views=spec.require_two_views, silent_cameras=silent
        )
        status, reason = self._maybe_occlusion_escape(status, reason, spec, votes, gates)

        outcome = VerifyOutcome(
            status=status,
            predicates=names,
            votes=votes,
            gates=gates,
            reason=reason,
            depth_source=depth_source,
        )
        logger.info("VERIFY %s: %s", status.upper(), reason)
        return outcome

    def _maybe_occlusion_escape(self, status, reason, spec, votes, gates):
        """``occlusion_ok`` annotates an all-abstain result. It cannot pass it.

        The release witness proves the jaws opened inside the container's
        footprint and let go. It does NOT prove the object stayed: a bounce out
        of the tray leaves the object somewhere the cameras did not resolve it,
        which produces exactly the same all-abstain vision result as an object
        genuinely swallowed by an opaque bin. Upgrading that to ``pass`` would
        manufacture success from no visual evidence.

        So the witness is recorded in the reason and the status stays
        ``unverified``. Proving placement into an opaque container needs
        evidence the object is still there (a second viewpoint into the bin, a
        wrist look-down, tactile/weight), not more proprioception.
        """
        if status != UNVERIFIED or not votes:
            return status, reason
        inside_preds = [p for p in spec.predicates if p.pred == "inside"]
        if not inside_preds or not all(bool(p.params.get("occlusion_ok")) for p in inside_preds):
            return status, reason
        if any(v.vote == FAIL for v in votes) or not all(
            v.vote == ABSTAIN for v in votes if v.mode != "none"
        ):
            return status, reason
        witness = getattr(self.executor, "_release_witness", None)
        if not isinstance(witness, ReleaseWitness):
            return status, reason
        dz = witness.tcp_dz_to_container
        if gates.get("release") and witness.inside_region and dz is not None and dz <= 0.15:
            return status, (
                f"{reason}; occlusion_ok: release witness was inside the container "
                f"(dz={dz:.3f}m) but proprioception cannot prove the object stayed"
            )
        return status, reason


def task_success(outcome: Optional[VerifyOutcome]) -> bool:
    """The ONLY definition of task success. A missing outcome is not success."""
    return outcome is not None and outcome.status == PASS


__all__ = [
    "CAMERA_ORDER",
    "STATIC_MATCH_TOL_M",
    "IdentityBinder",
    "task_success",
    "ReleaseWitness",
    "SuccessVerifier",
    "VerifyConfig",
    "begin_release_witness",
    "collect_gates",
    "finish_release_witness",
    "fuse_votes",
    "note_transport_drop",
    "record_grasp_verdict",
    "reset_run_state",
    "set_gate",
]

