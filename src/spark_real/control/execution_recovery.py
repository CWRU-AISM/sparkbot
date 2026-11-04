"""
Recovery and re-detection logic for score execution.

All functions take the executor instance as their first argument. Success
verdicts live in control.success_verifier.
"""

import concurrent.futures as _cf
import logging
import os
import time
from typing import Optional

import numpy as np

from spark_core.util import det_field
from spark_real.control.executor_types import ExecutionResult
from spark_real.skills import registry as skill_registry

logger = logging.getLogger(__name__)


# Sticky (gated, held-suppressed) re-detect adoption
#
# A fresh frame never rewrites detection_map directly:
#   * a HELD label is skipped entirely (birdview cannot relocate an object
#     inside the Robotiq jaws; the mask is the object-in-hand or a phantom);
#   * candidates are deduped (SAM3 multi-instance emits offset twins);
#   * the nearest candidate within the gate wins; two in-gate candidates
#     closer than ambiguity_sep to each other are undecidable -> hold;
#   * out-of-gate is never adopted (keep-old-position fallback);
#   * displacement is measured same-source (position_agentview) and applied
#     as a delta to the fused position, so the sideview Z-override survives
#     adoption.
#
# Config: perception.sticky in the family YAML (enabled defaults False).
# The 3 cm ambiguity_sep and 12 cm gate are sim-tuned and must be re-measured
# on the real table. SPARK_STICKY=0/1 overrides the enable bit.


# container_region / xy_inside_container / TRAY_FALLBACK_HALF_EXTENT_M live in
# the leaf control.container_geometry (numpy-only) so skills can use them
# without a circular import through skills.registry. Re-exported here.
from spark_real.control.container_geometry import (  # noqa: E402,F401
    TRAY_FALLBACK_HALF_EXTENT_M,
    container_region,
    xy_inside_container,
)
from spark_real.control.redetect import (  # noqa: F401
    rebind_before_approach,
    rebind_settings,
    redetect_all,
    redetect_single,
)
from spark_real.control.release_verdict import (  # noqa: F401
    _release_anchor_xy,
    _release_verdict,
    release_verdict_enabled,
    verify_placement,
)
from spark_real.control.sticky import (  # noqa: F401
    _STICKY_DEFAULTS,
    _held_label_of,
    _measure_pos,
    _write_adopted,
    sticky_adopt,
    sticky_settings,
)

# A destination only acts as a "container" for the in-tray grasp guard if it is
# plausibly a receptacle: detected slots, or a label that reads like one.
# Otherwise the fallback radius (TRAY_FALLBACK_HALF_EXTENT_M = 10cm) makes a
# bare stack target look like a bin and the grasp is falsely skipped. A block
# is placed ON, not IN.
_CONTAINER_LABEL_HINTS = (
    "tray", "bin", "bowl", "dustpan", "pan", "box", "cup", "container",
    "basket", "mug", "pot", "plate", "drawer", "tin", "crate", "caddy",
    "holder", "sink", "bucket", "cart", "jar", "can",
)


def _is_container_like(det, label: str) -> bool:
    """True only for genuine receptacles (detected slots or a bin-like label)."""
    if det is not None:
        get = det.get if hasattr(det, "get") else (lambda k, d=None: getattr(det, k, d))
        if get("slots", None):
            return True
    lab = (label or "").lower()
    return any(h in lab for h in _CONTAINER_LABEL_HINTS)


def grasp_target_in_container(executor, label: str) -> Optional[str]:
    """
    If the grasp target ``label`` currently sits inside any known
    destination/container region, return that container's label; else None.

    Containers come from the executor's ``_destination_labels`` (derived
    from the plan's place targets in pipeline_execution). The target's xy
    is read from the live detection_map at call time so re-detection moves
    are honored. Only genuine receptacles count (see ``_is_container_like``);
    a bare stack target (another block) is placed ON, not IN, so it never
    triggers the guard.
    """
    if not label or executor is None:
        return None
    det_map = getattr(executor, "detection_map", None) or {}
    dest_labels = getattr(executor, "_destination_labels", None) or set()
    if not dest_labels:
        return None
    target_det = det_map.get(label)
    if target_det is None:
        return None
    get = (
        target_det.get
        if hasattr(target_det, "get")
        else (lambda k, d=None: getattr(target_det, k, d))
    )
    pos = get("position_3d", None)
    if pos is None:
        return None
    try:
        target_xy = np.asarray(pos[:2], dtype=float)
    except Exception:
        return None
    for dest_label in dest_labels:
        if dest_label == label:
            continue
        dest_det = det_map.get(dest_label)
        if not _is_container_like(dest_det, dest_label):
            continue
        if xy_inside_container(target_xy, dest_det):
            return dest_label
    return None


def find_preceding_label(actions: list, index: int) -> Optional[str]:
    for j in range(index - 1, -1, -1):
        if actions[j].get("type") == "move_to_keypoint":
            return actions[j].get("params", {}).get("keypoint_label", "")
    return None


def attribution_enabled(executor) -> bool:
    """verification.attribution in the family YAML (default OFF = current
    dispatch). SPARK_ATTRIBUTION env overrides for a quick A/B."""
    env = os.environ.get("SPARK_ATTRIBUTION")
    if env is not None:
        return env.strip() not in ("", "0", "false", "no")
    try:
        profile = getattr(getattr(executor, "_pipeline", None), "profile", None)
        raw = getattr(profile, "raw", None) or {}
        block = raw.get("verification") or {}
        return bool(block.get("attribution", False))
    except Exception:  # noqa: BLE001
        return False


def build_verify_result(executor, label, failed_result=None) -> dict:
    """The attribute_failure input, from live executor state: typed
    GraspOutcome + detection_map presence/confidence for ``label``."""
    det = executor.detection_map.get(label) if label else None
    v = getattr(executor, "_last_grasp_verdict", None)
    outcome = (
        v.get("outcome") if isinstance(v, dict)
        else getattr(v, "outcome", None)
    )
    missing = bool(label) and (
        det is None or det_field(det, "position_3d") is None
    )
    out = {
        "semantic_failure": False,
        "detection_missing": missing,
        "grasp_outcome": outcome,
    }
    conf = det_field(det, "confidence")
    if conf is not None:
        out["detection_confidence"] = float(conf)
    return out


def attempt_recovery(
    executor, action_type, params, failed_result, actions, action_index
):
    """
    Dispatch to the right recovery strategy based on failure type.

    With verification.attribution enabled, the lowest-responsible-layer
    table (control.attribution) picks the tier: PERCEPTION -> re-detect and
    re-bind (skip the tier-1 in-place perturb); EXECUTION -> the local
    ladder; PLAN -> stop and surface to the operator, never an autonomous
    LLM replan.
    """
    layer = None
    if attribution_enabled(executor):
        from spark_core.attribution import Layer, attribute_failure

        label = (params or {}).get("keypoint_label", "") or find_preceding_label(
            actions, action_index
        )
        _enabled, _retry, max_attempts = (True, False, 1)
        try:
            _enabled, _retry, max_attempts = executor._verify_settings()
        except Exception:  # noqa: BLE001
            pass
        # Local retries already spent on this label (executor_core's
        # per-label grasp retry counter).
        retry_count = 0
        try:
            retry_count = int(
                (getattr(executor, "_grasp_retry_count", {}) or {}).get(
                    label, 0
                )
            )
        except Exception:  # noqa: BLE001
            pass
        layer = attribute_failure(
            build_verify_result(executor, label, failed_result),
            None,  # scene_diff: wired in once the pre-primitive diff lands
            retry_count,
            max_local_retries=max(1, max_attempts),
        )
        logger.info(
            "Attribution: %s failure on %s attributed to %s",
            action_type, label or "?", layer.value,
        )
        if layer == Layer.PLAN:
            logger.warning(
                "Attribution: PLAN layer -> stopping recovery and surfacing "
                "to the operator (no autonomous replan)"
            )
            return None

    if action_type == "grasp":
        skip_tier1 = False
        if layer is not None:
            from spark_core.attribution import Layer

            skip_tier1 = layer == Layer.PERCEPTION
        return recover_grasp(
            executor, params, actions, action_index, skip_tier1=skip_tier1
        )
    if action_type == "move_to_keypoint":
        label = params.get("keypoint_label", "")
        if "not found" in failed_result.message:
            return recover_object_not_found(executor, label)
        return recover_approach_failed(executor, label)
    return None


# Recover from a failed grasp with progressive Z bias
def recover_grasp(executor, grasp_params, actions, action_index, skip_tier1=False):
    target_label = find_preceding_label(actions, action_index)
    force = grasp_params.get("force", 50)
    logger.info(
        "Grasp recovery for '%s': up to %d retries",
        target_label or "unknown",
        executor.MAX_GRASP_RETRIES,
    )

    # Config table_height overrides only when explicitly set; None falls
    # back to the executor's per-family TABLE_Z_FLOOR (config-fed).
    failed_z = executor._get_current_position()[2]
    recovery_z_floor = -0.27

    # Tier-1: in-place reseat before the expensive perception re-grounding. A
    # barely-missed grasp is often fixed by a small nudge (down / lateral) and
    # a re-close, with no re-detect and no LLM call. Only if these cheap
    # perturbations fail does recovery drop to tier-2 (re-ground).
    # skip_tier1: attribution said PERCEPTION, so the detection itself is the
    # suspect and an in-place perturb around a wrong position is wasted motion.
    if skip_tier1:
        logger.info(
            "Grasp recovery: tier-1 perturb skipped (attributed to PERCEPTION); "
            "going straight to re-grounding"
        )
    else:
        try:
            from spark_real.skills.recovery import grasp_perturb as _grasp_perturb
            for _t1 in range(2):
                if executor._abort:
                    break
                r1 = _grasp_perturb(executor, {"attempt": _t1, "force": force})
                if getattr(r1, "success", False):
                    logger.info(
                        "Grasp recovery: tier-1 perturb reseated the grasp "
                        "(attempt %d)", _t1,
                    )
                    return r1
            logger.info(
                "Grasp recovery: tier-1 perturb did not reseat; "
                "falling back to perception re-grounding"
            )
        except Exception as exc:
            logger.warning("Grasp recovery: tier-1 perturb skipped (%s)", exc)

    for attempt in range(1, executor.MAX_GRASP_RETRIES + 1):
        z_bias = executor.RECOVERY_Z_BIAS[
            min(attempt - 1, len(executor.RECOVERY_Z_BIAS) - 1)
        ]
        logger.info(
            "Recovery attempt %d/%d (Z bias: -%.1fmm)",
            attempt,
            executor.MAX_GRASP_RETRIES,
            z_bias * 1000,
        )

        executor.robot.open_gripper()
        time.sleep(0.5)

        # Lift before re-perceiving: the failed descent may have nudged
        # the object, so always clear+redetect rather than re-aim at a
        # stale XY.
        current = executor._get_current_position()
        clear_pos = current.copy()
        clear_pos[2] = max(current[2] + 0.05, failed_z + 0.03)
        executor._move_to(clear_pos, executor.GRASP_ORIENTATION)

        new_target = _redetect_for_recovery(executor, target_label)
        # Guard: a recovery re-detect that lands inside the tray / container
        # means the object is already placed (or got knocked in). Never
        # re-grasp from inside the container; report success-with-note.
        if new_target is not None and target_label:
            existing = executor.detection_map.get(target_label, {}) or {}
            probe = dict(existing)
            probe["position_3d"] = new_target.tolist()
            executor.detection_map[target_label] = probe
            _in = grasp_target_in_container(executor, target_label)
            if _in is not None:
                executor._placed_labels.add(target_label)
                executor._holding = False
                msg = (
                    f"'{target_label}' already in container '{_in}', "
                    f"skipping recovery grasp"
                )
                logger.info(msg)
                return ExecutionResult(action_type="grasp", success=True, message=msg)
        det = executor.detection_map.get(target_label)
        if new_target is None:
            if det is None:
                continue
            new_target = np.array(det["position_3d"])
            logger.info(
                "Attempt %d: redetect missed, falling back to " "stale XY=[%.3f,%.3f]",
                attempt,
                new_target[0],
                new_target[1],
            )
        else:
            logger.info(
                "Attempt %d: redetected at [%.3f,%.3f]",
                attempt,
                new_target[0],
                new_target[1],
            )

        new_target[2] = max(min(new_target[2], failed_z) - z_bias, recovery_z_floor)
        logger.info("Attempt %d: approaching [%.3f,%.3f,%.3f]", attempt, *new_target)
        existing = executor.detection_map.get(target_label, {}) or {}
        executor.detection_map[target_label] = {
            "position_3d": new_target.tolist(),
            "orientation_angle": existing.get("orientation_angle", 0.0),
            "aspect_ratio": existing.get("aspect_ratio", 1.0),
        }
        executor._approach_target(
            new_target, detection=executor.detection_map[target_label]
        )

        # Compliant descent
        if skill_registry.get("compliant_push") is not None:
            skill_registry.dispatch(
                "compliant_push",
                executor,
                {
                    "direction": [0, 0, -1],
                    "max_distance": 0.03,
                    "force_threshold": 10.0,
                    "step_size": 0.002,
                },
            )

        # Re-grasp
        try:
            if hasattr(executor.robot, "get_tcp_force"):
                ft = executor.robot.get_tcp_force()
                executor._pre_grasp_force_magnitude = np.linalg.norm(ft[:3])
        except Exception:
            executor._pre_grasp_force_magnitude = 0.0

        if hasattr(executor.robot, "_send_gripper_command"):
            executor.robot._send_gripper_command(1.0, speed=80, force=min(force, 100))
            time.sleep(1.5)
            executor.robot._send_gripper_command(1.0, speed=50, force=100)
            time.sleep(1.0)
        else:
            executor.robot.close_gripper()
            time.sleep(1.5)

        failed_z = min(failed_z, executor._get_current_position()[2])

        if executor._verify_grasp():
            executor._holding = True
            # The recovery re-grasp closed at this height (progressive
            # RECOVERY_Z_BIAS included); record it so the release give-back
            # accounts for the extra descent.
            try:
                executor._actual_grasp_tcp_z = float(
                    executor._get_current_position()[2]
                )
            except Exception:  # noqa: BLE001
                pass
            msg = f"Grasp recovery succeeded on attempt {attempt}"
            logger.info(msg)
            return ExecutionResult(action_type="grasp", success=True, message=msg)

        logger.info("Recovery attempt %d: verification failed", attempt)

    logger.warning(
        "Grasp recovery failed after %d attempts", executor.MAX_GRASP_RETRIES
    )
    return None


def _redetect_for_recovery(executor, label):
    """
    Re-detect during recovery; returns new 3D position or None.

    Logs each substep so silent stalls in capture/detect are visible. Runs
    the SAM3 detect on a threadpool with a 30 s timeout; on timeout returns
    None and the caller falls back to the stale XY.
    """
    if executor._pipeline is None or not label:
        return None
    try:
        t0 = time.time()
        logger.info("Re-detect '%s': capture...", label)
        captures = executor._pipeline.capture()
        logger.info(
            "Re-detect '%s': capture done (%.2fs, %d cams) -> detect...",
            label,
            time.time() - t0,
            len(captures),
        )

        def _do_detect():
            return executor._pipeline.detect(captures, prompts=[label])

        with _cf.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="redetect"
        ) as pool:
            fut = pool.submit(_do_detect)
            try:
                new_dets = fut.result(timeout=30.0)
            except _cf.TimeoutError:
                logger.warning(
                    "Re-detect '%s': detect timed out after 30s; "
                    "aborting redetect (fallback to stale XY)",
                    label,
                )
                return None
        logger.info(
            "Re-detect '%s': detect done (%.2fs total, %d dets)",
            label,
            time.time() - t0,
            len(new_dets),
        )

        # Anchor to the detection nearest the label's prior known position, not
        # the first re-assigned instance label: a fresh multi-instance detect
        # can renumber "blue block 1" / "blue block 2", and recovery would
        # grasp the wrong one.
        prior = executor.detection_map.get(label, {}) or {}
        prior_pos = prior.get("position_3d") if hasattr(prior, "get") else None
        best, best_d = None, float("inf")
        for det in new_dets:
            det_label = det.label if hasattr(det, "label") else det.get("label", "")
            pos = (
                det.position_3d
                if hasattr(det, "position_3d")
                else det.get("position_3d")
            )
            if pos is None:
                continue
            if prior_pos is not None:
                d = float(
                    np.linalg.norm(
                        np.asarray(pos[:2], float) - np.asarray(prior_pos[:2], float)
                    )
                )
            else:
                # No prior anchor: fall back to exact-label match.
                d = 0.0 if det_label == label else float("inf")
            if d < best_d:
                best, best_d = np.array(pos), d
        # 0.15 m gate: a genuine re-detect of the same block barely moves; a jump
        # to the other block is ~0.3 m and must not be accepted as "the same one".
        if best is not None and best_d <= 0.15:
            logger.info(
                "Re-detected '%s' at [%.3f,%.3f,%.3f] (nearest prior, d=%.3fm)",
                label, *best, best_d,
            )
            return best
        logger.info(
            "Re-detect '%s': no detection within 0.15m of prior (best_d=%.3fm, %d dets)",
            label, best_d, len(new_dets),
        )
    except Exception as e:
        logger.warning("Re-detection failed: %s", e)
    return None


def recover_object_not_found(executor, label):
    if skill_registry.get("search_keypoint") is None:
        return None
    logger.info("Search recovery for '%s'", label)
    result = skill_registry.dispatch(
        "search_keypoint",
        executor,
        {"keypoint_label": label, "search_radius": 0.15, "num_loops": 2},
    )
    if not result.success:
        return None
    det = executor.detection_map.get(label)
    if det is None:
        return None
    executor._approach_target(np.array(det["position_3d"]), detection=det)
    return ExecutionResult(
        action_type="move_to_keypoint",
        success=True,
        message=f"Found '{label}' via search",
    )


def recover_approach_failed(executor, label):
    if skill_registry.get("retract_retry") is None:
        return None
    logger.info("Retract-retry recovery for '%s'", label)
    result = skill_registry.dispatch(
        "retract_retry",
        executor,
        {
            "keypoint_label": label,
            "retract_distance": 0.08,
            "offset_range": 0.015,
            "num_retries": 3,
        },
    )
    if result.success:
        return ExecutionResult(
            action_type="move_to_keypoint",
            success=True,
            message=f"Re-approached '{label}' via retract_retry",
        )
    return None


