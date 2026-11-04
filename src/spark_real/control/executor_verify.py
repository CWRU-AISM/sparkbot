"""
VerifyMixin: task verification, recovery, re-detection, failure snapshots.

The verdict itself lives in control.success_verifier; this mixin only wires it
into the executor and decides whether anything moves afterwards.

Retry policy:

  * ``verification.enabled: false`` is a kill switch. The verifier does not
    run, nothing is captured, nothing moves.
  * A replay re-opens the gripper and drops whatever is held, so it is opt-in:
    ``verification.retry_on_fail: true``. The module default is off; a profile
    that does not mention the key must not silently acquire new motion. The
    shipped UR10e profile (configs/ur10e_default.yaml) opts in explicitly and
    pins the cap.
  * Even when opted in, a replay needs an explicit ``fail`` backed by evidence:
    a failed gate or a camera voting fail. ``unverified`` means no verdict
    was reached, and ``on_missing: fail`` re-labels a structural abstain as
    one; neither is grounds for motion, entering the loop or continuing it.

A replay re-runs ``move_to_keypoint -> grasp`` (a descent that closes the
jaws), so each of these is a precondition, not an assumption:

  * ``_establish_retry_ready_state`` commands the jaws open at the current
    (lowest) pose, confirms they are empty, and lifts clear before any lateral
    motion. If the grip does not read empty, the retry is abandoned.
  * ``_reperceive_for_retry`` re-detects every keypoint the replay drives to
    and proves the pick target's entry actually changed. A re-detection that
    found nothing leaves the stale plan-time pose behind, and driving to that
    would re-commit the miss being retried, so it abandons instead.
  * ``retry_max_attempts`` bounds the loop; there is no unbounded path.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2

from spark_core.util import as_bool
from spark_real.control import execution_recovery
from spark_real.control.executor_types import ExecutionResult
from spark_real.control.primitive_trace import TraceWriter
from spark_real.control.success_predicates import FAIL, PASS, UNVERIFIED, VerifyOutcome
from spark_real.control.success_verifier import SuccessVerifier, VerifyConfig
from spark_real.control.verify_replay import ReplayMixin  # noqa: F401
from spark_real.control.verify_retry_common import (  # noqa: F401
    _KEYPOINT_ACTIONS,
    RETRY_LIFT_M,
    RETRY_MIN_CLEAR_Z,
    _detection_fingerprint,
    _fail_has_evidence,
    _is_fresh,
    _keypoint_label,
    _replay_labels,
)

logger = logging.getLogger(__name__)

# Replaying the last pick cycle is physical motion WITH a gripper release, so
# it stays off unless the operator asks for it. Config: verification.retry_on_fail.
RETRY_ON_FAIL_DEFAULT = False


class VerifyMixin(ReplayMixin):
    """
    Post-task verification, recovery, re-detection, failure snapshots.
    """

    # Last VerifyOutcome produced by this executor. Consumers (pipeline_run,
    # routes, the episode recorder, the BT library) must read THIS, not
    # ExecutionResult.success; execution is not achievement.
    verify_outcome: Optional[VerifyOutcome] = None

    def _verify_settings(self) -> Tuple[bool, bool, int]:
        """``(enabled, retry_on_fail, max_attempts)`` from the verification block."""
        enabled = VerifyConfig.from_pipeline(self._pipeline).enabled
        raw = getattr(getattr(self._pipeline, "profile", None), "raw", None) or {}
        block = raw.get("verification") if hasattr(raw, "get") else None
        block = block if isinstance(block, dict) else {}
        retry = as_bool(block.get("retry_on_fail"), RETRY_ON_FAIL_DEFAULT)
        default_max = 1 if self._strict_placement_verify else 2
        try:
            max_attempts = int(block.get("retry_max_attempts", default_max))
        except (TypeError, ValueError):
            max_attempts = default_max
        return enabled, retry, max(0, max_attempts)


    def _unocclude_for_verify(self):
        """Lift to safe height before the verify capture ("zoom out").

        An arm hovering centimeters over the placement occludes exactly the
        region every camera rung needs to see. One lift, current orientation
        preserved (a wrist reorientation here would be pure risk), skipped
        when already high enough or when anything about the read/move fails:
        failure to lift must never fail the verify.
        """
        try:
            pos = self._get_current_position()
            safe_z = float(getattr(self, "SAFE_HEIGHT_Z", 0.35))
            if pos is None or float(pos[2]) >= safe_z - 0.02:
                return
            orient = None
            cur = getattr(self, "current_orientation", None)
            if callable(cur):
                orient = cur()
            if orient is None:
                orient = list(self.GRASP_ORIENTATION)
            logger.info(
                "[verify] lifting %.0fmm to unocclude the scene before the "
                "verify capture", (safe_z - float(pos[2])) * 1000,
            )
            self._move_to([float(pos[0]), float(pos[1]), safe_z], orient)
        except Exception as exc:  # noqa: BLE001 - see docstring
            logger.warning("[verify] unocclude lift failed: %s", exc)

    def _fold_in_spatial_rung(self, outcome, score):
        """Rung 2: the remembered-region check, run only when the predicate
        verifier abstained. A filled slot defeats re-detection and identical
        sockets defeat label matching, but geometry still has an answer: does
        the placed object's mask now sit where the approved target region
        was? A pass here upgrades the abstain (with the evidence in the
        reason); a fail downgrades it to FAIL only when the object was
        confidently re-detected somewhere else. PASS/FAIL from the predicates
        are left alone; this rung breaks ties, it does not outvote
        measurements.
        """
        if outcome.status != UNVERIFIED:
            return outcome
        try:
            from spark_real.control.spatial_verify import spatial_rung

            v = spatial_rung(self, score)
        except Exception as exc:  # noqa: BLE001 - never let the rung crash verify
            logger.warning("[spatial-verify] rung failed: %s", exc)
            return outcome
        if not isinstance(v, dict) or v.get("status") in (None, "abstain"):
            if isinstance(v, dict):
                outcome.gates["spatial"] = True  # abstain is not a failure
            return outcome
        outcome.gates["spatial"] = v["status"] == "pass"
        joiner = "; " if outcome.reason else ""
        if v["status"] == "pass":
            outcome.status = PASS
            outcome.reason = (
                outcome.reason + joiner + "spatial rung: " + v["detail"]
            )
        else:
            outcome.status = FAIL
            outcome.reason = (
                outcome.reason + joiner + "spatial rung: " + v["detail"]
            )
        logger.info("[spatial-verify] folded into verify: status=%s", outcome.status)
        return outcome

    def _fold_in_release_proprio(self, outcome):
        """Rung 0: the arm's own release-time evidence, folded into the
        camera verdict. Proprioception here only ever removes a pass, never
        grants one:

        - jaws never opened -> FAIL outright. The object was not released;
          whatever the cameras matched, the task did not happen.
        - commanded tilt not achieved -> a PASS downgrades to UNVERIFIED. The
          motion the plan asked for was dropped, so a visually-plausible
          result must not bank a BT that never executed as written.
        - proprio pass with cameras abstaining stays abstained: jaw + force
          agreement is necessary evidence, not sufficient.
        """
        v = getattr(self, "_release_proprio", None)
        if not isinstance(v, dict) or "verdict" not in v:
            return outcome
        outcome.gates["proprio"] = v["verdict"] != "fail"
        if v["verdict"] != "fail":
            return outcome
        if v.get("jaws_opened") is False:
            outcome.status = FAIL
            outcome.reason = (
                (outcome.reason + "; " if outcome.reason else "")
                + "proprio: jaws never opened -- object was not released"
            )
        elif v.get("commanded_pitch_achieved") is False and outcome.status == PASS:
            outcome.status = UNVERIFIED
            outcome.reason = (
                (outcome.reason + "; " if outcome.reason else "")
                + "proprio: commanded tilt was not achieved "
                f"(orient_err={v.get('orient_err_rad')}); not banking this run"
            )
        logger.info("[proprio-verdict] folded into verify: status=%s (%s)",
                    outcome.status, outcome.reason)
        return outcome

    def _run_post_task_verification(self, score: dict, actions: list):
        enabled, retry_on_fail, max_attempts = self._verify_settings()
        if not enabled:
            # Kill switch: no verifier, no capture, no retry, no motion.
            self.verify_outcome = VerifyOutcome(
                status=UNVERIFIED, reason="verification disabled by config"
            )
            logger.info("Verification disabled by config; no verify, no retry")
            return

        self._unocclude_for_verify()
        outcome = self._verify_task_completion(score, actions)
        outcome = self._fold_in_release_proprio(outcome)
        outcome = self._fold_in_spatial_rung(outcome, score)

        if outcome.status == PASS:
            return
        if not retry_on_fail or max_attempts <= 0:
            logger.info(
                "Verification %s; retry is off (verification.retry_on_fail), "
                "no replay",
                outcome.status,
            )
            return
        if outcome.status != FAIL:
            # Abstain, not a failure. Replaying here would drop a held object
            # on no evidence at all.
            logger.info("Verification %s (no verdict); not replaying", outcome.status)
            return
        if not _fail_has_evidence(outcome):
            logger.info("Verification fail with no failing gate or vote (%s); not replaying",
                        outcome.reason)
            return

        # Attribution consult (verification.attribution, default off): the
        # lowest-responsible-layer table decides whether a replay is even the
        # right repair. PLAN (semantic fail / retries exhausted) means stop
        # and surface to the operator, never an autonomous LLM replan.
        # PERCEPTION/EXECUTION both proceed into the replay, whose
        # preconditions (re-perceive + ready-state) implement the
        # re-detect+re-bind and local-retry repairs.
        if execution_recovery.attribution_enabled(self):
            from spark_core.attribution import Layer, attribute_failure

            actions_list = actions or []
            start = self._find_last_pick_cycle(actions_list)
            pick_label = (
                _keypoint_label(actions_list[start]) if start is not None else ""
            )
            layer = attribute_failure(
                execution_recovery.build_verify_result(self, pick_label),
                None,
                0,
                max_local_retries=max(1, max_attempts),
            )
            logger.info(
                "Attribution: task-level fail on '%s' attributed to %s",
                pick_label or "?", layer.value,
            )
            if layer == Layer.PLAN:
                logger.warning(
                    "Attribution: PLAN layer -> not replaying; surfacing the "
                    "failure to the operator"
                )
                return

        self._replay_last_pick_cycle(score, actions, max_attempts)


    def _snapshot_failure_state(self, action_type: str, params: dict, result) -> None:
        """
        Save per-camera JPEGs + manifest when an action goes red.
        """
        if self._pipeline is None:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = Path(self._pipeline.config.output_dir) / "failures" / f"{ts}_{action_type}"
        out_root.mkdir(parents=True, exist_ok=True)

        try:
            caps = self._pipeline.capture()
        except Exception:
            caps = {}
        cam_files = []
        for cam_name, cd in (caps or {}).items():
            rgb = (cd or {}).get("rgb")
            if rgb is None:
                continue
            jpg = out_root / f"{cam_name}.jpg"
            try:
                cv2.imwrite(str(jpg), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cam_files.append(str(jpg))
            except Exception:
                pass

        robot_state = {}
        try:
            if hasattr(self.robot, "get_tcp_pose"):
                p = self.robot.get_tcp_pose()
                if hasattr(p, "tolist"):
                    robot_state["tcp_pose"] = p.tolist()
        except Exception:
            pass
        try:
            if hasattr(self.robot, "get_joint_positions"):
                j = self.robot.get_joint_positions()
                if hasattr(j, "tolist"):
                    robot_state["joints"] = j.tolist()
        except Exception:
            pass

        manifest = {
            "timestamp": ts,
            "action": action_type,
            "params": params,
            "message": getattr(result, "message", ""),
            "duration": getattr(result, "duration", 0.0),
            "robot": robot_state,
            "cameras": cam_files,
        }
        (out_root / "failure.json").write_text(json.dumps(manifest, indent=2, default=str))
        logger.info("Failure snapshot saved to %s (%d cameras)", out_root, len(cam_files))

    def _attempt_recovery(self, action_type, params, failed_result, actions, action_index):
        return execution_recovery.attempt_recovery(
            self, action_type, params, failed_result, actions, action_index
        )

    def _verify_task_completion(self, score, actions):
        """Run the predicate verifier and record its outcome.

        Appends an ExecutionResult so existing UI plumbing still sees a
        "verify" row, but the authoritative value is ``self.verify_outcome``:
        the row's ``success`` collapses the tri-state and must not be used to
        decide task success.
        """
        t0 = time.time()
        verifier = SuccessVerifier(executor=self)
        outcome = verifier.verify(score, actions)
        self.verify_outcome = outcome
        try:
            TraceWriter.from_pipeline(self._pipeline).write_verify(outcome, time.time() - t0)
        except Exception as exc:
            logger.warning("Verify trace not written: %s", exc)
        self._results.append(
            ExecutionResult(
                action_type="verify",
                success=(outcome.status == PASS),
                message=f"{outcome.status}: {outcome.reason}",
                duration=time.time() - t0,
            )
        )
        return outcome

    def _redetect_all(self, actions, current_index):
        execution_recovery.redetect_all(self, actions, current_index)

    def _redetect_single(self, label):
        execution_recovery.redetect_single(self, label)
