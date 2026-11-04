"""Per-release two-anchor placement verdict and label rebind."""

import logging
import os

import numpy as np

from spark_core.binding import (
    dedup_candidate_indices,
)
from spark_core.util import det_field
from spark_real.control.container_geometry import (  # noqa: E402,F401
    TRAY_FALLBACK_HALF_EXTENT_M,
    container_region,
    xy_inside_container,
)
from spark_real.control.primitive_trace import (
    PrimitiveTrace,
    TraceWriter,
    detection_summary,
)
from spark_real.control.sticky import _measure_pos, _write_adopted, sticky_settings

logger = logging.getLogger(__name__)


def release_verdict_enabled(executor) -> bool:
    """verification.release_verdict in the family YAML (default OFF = the
    trace-only behavior). SPARK_RELEASE_VERDICT env overrides."""
    env = os.environ.get("SPARK_RELEASE_VERDICT")
    if env is not None:
        return env.strip() not in ("", "0", "false", "no")
    try:
        profile = getattr(getattr(executor, "_pipeline", None), "profile", None)
        raw = getattr(profile, "raw", None) or {}
        block = raw.get("verification") or {}
        return bool(block.get("release_verdict", False))
    except Exception:  # noqa: BLE001
        return False


def _release_anchor_xy(executor):
    """The last commanded release point's XY: the ReleaseWitness TCP if one
    was recorded, else the current TCP (the arm is still at the release pose
    when verify_placement runs)."""
    witness = getattr(executor, "_release_witness", None)
    tcp = getattr(witness, "tcp_xyz", None) if witness is not None else None
    if tcp is not None:
        return np.asarray(tcp[:2], dtype=float)
    try:
        return np.asarray(executor._get_current_position()[:2], dtype=float)
    except Exception:  # noqa: BLE001
        return None


def _release_verdict(executor, pick_label, dets):
    """Two-anchor, evidence-graded per-release verdict plus unconditional
    rebind.

    Verdict (asserted only from the fresh same-capture observation):
      * candidate within gate of the release point -> ok, with xy error;
      * candidate within gate of the pre-place origin -> still_at_origin
        (the object never left; the release demonstrably failed);
      * neither -> abstain. Never a synthesized fail.

    Binding is separate from the verdict: detection_map[pick_label] is
    rebound to the nearest fresh instance to the release point at any
    distance, since its displacement is self-caused and
    _replay_last_pick_cycle must re-aim at where the object actually fell.
    The rebind fires even when the verdict abstains.
    """
    gate = sticky_settings(executor)["gate_m"]
    release_xy = _release_anchor_xy(executor)
    origin = (executor.detection_map.get(pick_label) or {})
    origin_pos = det_field(origin, "position_3d")
    origin_xy = (
        None if origin_pos is None
        else np.asarray(origin_pos[:2], dtype=float)
    )

    positions = [_measure_pos(d) for d in dets]
    confs = [float(det_field(d, "confidence", 0.0) or 0.0) for d in dets]
    kept = dedup_candidate_indices(positions, confs, merge_radius_m=0.02)
    cands = [(dets[i], positions[i]) for i in kept if positions[i] is not None]

    verdict = {"verdict": "abstain", "gate_m": gate}
    if not cands:
        verdict["detail"] = "object not visible after release"
        return verdict

    # Nearest fresh instance to the release point (any distance): binding.
    if release_xy is not None:
        d_rel, det, pos = min(
            (
                (float(np.linalg.norm(p[:2] - release_xy)), d, p)
                for d, p in cands
            ),
            key=lambda t: t[0],
        )
        _write_adopted(
            executor, pick_label, det, pos, pos, "post-release",
            note=f"rebind d_release={d_rel * 100:.1f}cm",
        )
        verdict["rebind_xy"] = [float(pos[0]), float(pos[1])]
        if d_rel <= gate:
            verdict["verdict"] = "ok"
            verdict["xy_error_m"] = round(d_rel, 4)
            return verdict
        if origin_xy is not None:
            d_org = float(np.linalg.norm(pos[:2] - origin_xy))
            if d_org <= gate:
                verdict["verdict"] = "still_at_origin"
                verdict["origin_dist_m"] = round(d_org, 4)
                # Evidence, not a synthesized fail: the object demonstrably
                # never left its pre-place position, which is exactly what
                # _fail_has_evidence wants behind a replay. The task verdict
                # stays SuccessVerifier's.
                from spark_real.control import success_verifier

                success_verifier.set_gate(
                    executor, "release", False,
                    f"post-release: '{pick_label}' still at origin "
                    f"({d_org * 100:.1f}cm from pre-place position)",
                )
                return verdict
        verdict["detail"] = (
            f"nearest instance {d_rel * 100:.1f}cm from the release point "
            "and not at the origin"
        )
    return verdict


def verify_placement(executor, pick_label, place_label):
    """Post-release trace emitter plus optional per-release verdict.

    The task verdict comes from control.success_verifier at end of task. The
    per-release verdict here (verification.release_verdict, default off) is a
    per-primitive attribution signal, never the run verdict. It records where
    the object appears right after the release and, when enabled, rebinds the
    placed label to where it actually fell so a replay re-aims correctly.
    """
    if executor._pipeline is None:
        return
    try:
        captures = executor._pipeline.capture()
        prompt = (
            pick_label.rsplit(" ", 1)[0]
            if pick_label and pick_label[-1].isdigit()
            else pick_label
        )
        dets = executor._pipeline.detect(captures, prompts=[prompt]) or []
        records = [detection_summary(d) for d in dets]
        for rec in records:
            logger.info(
                "Release trace: '%s' seen by %s conf=%.2f at %s",
                rec.get("label"),
                rec.get("camera"),
                float(rec.get("confidence") or 0.0),
                rec.get("position_3d"),
            )
        if not records:
            logger.info("Release trace: '%s' not visible after release", pick_label)

        params = {"pick_label": pick_label, "place_label": place_label}
        if release_verdict_enabled(executor):
            verdict = _release_verdict(executor, pick_label, dets)
            params["release_verdict"] = verdict
            logger.info(
                "Release verdict for '%s': %s", pick_label, verdict
            )

        writer = TraceWriter.from_pipeline(executor._pipeline)
        writer.write(
            PrimitiveTrace(
                index=writer.next_index(),
                primitive="release_placement",
                params=params,
                detections_used=records,
            )
        )
    except Exception as e:
        logger.warning("Release trace failed: %s", e)
