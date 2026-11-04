"""
ScoreExecutorCore: init, config/tuning, workspace bounds and task state.

Composes AbortMixin (executor_abort), BtRunnerMixin (bt_runner) and
DispatchMixin (bt_dispatch).
"""

import logging
import math
import os
import re
import threading

import numpy as np

from spark_real.calibration_table import load_table_plane
from spark_real.config import family_block
from spark_real.control import primitive_timeouts
from spark_real.control.bt_dispatch import DispatchMixin  # noqa: F401
from spark_real.control.bt_runner import BtRunnerMixin  # noqa: F401
from spark_real.control.cartesian_servo import CartesianServo
from spark_real.control.family_bounds import FRANKA_FAMILIES, apply_franka_bounds
from spark_real.control.executor_abort import AbortMixin  # noqa: F401
from spark_real.control.executor_types import (
    AbortRequested,
)
from spark_real.grasp_depth_memory import GraspCalibration

logger = logging.getLogger(__name__)

# Statements that can only ADD speed. A script containing any of them is not a
# brake, whatever else it contains.
_ACCELERATING_STATEMENTS = ("movej(", "movel(", "movep(", "movec(", "servoj(", "speedj(")
_SPEEDL_VEC_RE = re.compile(r"speedl\(\[([^\]]*)\]")


def is_braking_script(script: str) -> bool:
    """True if every statement in ``script`` can only reduce arm speed.

    The abort-aware _send_script wrapper lets the brake through after the
    abort flag is set. A prefix test on "stopj"/"stopl" is too narrow: the
    Cartesian servo brakes with ``speedl([0,...])`` followed by ``stopl(1.0)``.
    """
    s = script.strip()
    if not s or not ("stopj(" in s or "stopl(" in s):
        return False
    if any(tok in s for tok in _ACCELERATING_STATEMENTS):
        return False
    for m in _SPEEDL_VEC_RE.finditer(s):
        try:
            vec = [float(x) for x in m.group(1).split(",")]
        except ValueError:
            return False  # unparseable speed: assume it accelerates
        if any(abs(v) > 1e-9 for v in vec):
            return False
    return True


class ScoreExecutorCore(AbortMixin, BtRunnerMixin, DispatchMixin):
    """Init, orchestration loop, dispatch, abort, and workspace logic.

    Base mixin for ScoreExecutor: __init__, the execute_score loop, action
    dispatch, abort machinery, and workspace bounds.
    """

    SAFE_HEIGHT_Z = 0.35
    APPROACH_OFFSET_Z = 0.0
    GRIPPER_OPEN_Z_OFFSET = -0.005
    # Same grasp as [2.103, -2.329, 0.059], expressed in the half-turn nearest
    # HOME_CONFIG (a parallel jaw is symmetric under 180 deg about the tool
    # axis); that value sat -177.8 deg in yaw from home and wrapped the
    # wrist-camera cable. Must match configs/ur10e_default.yaml.
    GRASP_ORIENTATION = [2.3038, 2.0802, -0.0048]
    WRIST_CAM_OFFSET_DEG = 81.0
    GRIPPER_EMPTY_THRESHOLD = 235
    GRIPPER_FULLY_CLOSED = 250
    # Release-side jaw thresholds (0=open .. 255=closed). GRIPPER_FULLY_CLOSED
    # answers "are the jaws still spread on something" AFTER a close; it says
    # nothing after an OPEN, where a released gripper reads ~0. These two are
    # the open-side pair: jaws at/near the open stop, or enough travel toward
    # open to prove the object let go. Bounds, not calibrated values; the
    # release witness logs the measured numbers so a rig run can tighten them.
    GRIPPER_RELEASED_MAX_POS = 60
    GRIPPER_RELEASE_MIN_TRAVEL = 40
    # Extra window the release witness may poll for the jaws to read open,
    # AFTER the driver's own open settle. Zero cost on a normal release (the
    # first sample already reads open).
    RELEASE_CONFIRM_TIMEOUT_S = 0.6
    # Brake decel for a URScript-capable driver that has no brake() of its own
    # (rad/s^2). UR10eDriver sizes its own from the commanded velocity; this is
    # only the last-resort literal.
    ABORT_DECEL_FALLBACK = 2.0
    MAX_GRASP_RETRIES = 3
    TABLE_Z_FLOOR = -0.276
    WORKSPACE_MIN = np.array([-1.1, -0.5, -0.27])
    WORKSPACE_MAX = np.array([-0.5, 0.7, 0.50])
    MAX_REACH = 1.15
    RECOVERY_Z_BIAS = [0.010, 0.020, 0.030]

    # Grasp / stack tuning. Overridable per-family via the `grasp:` and `stack:`
    # YAML blocks (see _apply_family_tuning); a family that omits them keeps
    # these defaults. Env vars remain as optional debug overrides.
    GRASP_DEPTH_M = 0.022            # descend this far below the perceived top
                                     # before clamping (TCP open->closed offset).
    GRASP_FORCE_EMPTY_N = 2.0        # force-verify: peak |dF| below this = empty.
                                     # gObj still leads.
    GRASP_LIFT_CHECK_M = 0.02        # confirming lift after clamp.
    GRASP_LIFT_DELTA_MIN_N = 0.5     # extra pull the confirming lift must add
                                     # for it to corroborate "holding".
    GRASP_YAW_AR_GATE = 1.8          # only do an oriented (yaw) grasp when the
                                     # object's aspect ratio exceeds this. Cubes
                                     # read ~1.1-1.5 from mask noise; real tools
                                     # ~2+ (1.4 fired 85 deg yaws on cubes).
    # Input-validity floors on the SAME oriented-grasp decision. AR cannot tell
    # a tool from a soft blob: the plushie read AR 3.14 at conf 0.24.
    GRASP_YAW_MIN_CONF = 0.45        # detection confidence floor for an OBB yaw.
    GRASP_YAW_MIN_OBB_CONF = 0.40    # OBB-axis confidence floor. Absent = unknown.
    STACK_CLEARANCE_M = 0.12         # begin place descent this far above target top.
    STACK_MAX_DESCENT_M = 0.15       # descent safety cap.
    STACK_CONTACT_FORCE_N = 5.0      # force-DELTA that counts as contact.
    STACK_HARD_STOP_DELTA_N = 25.0   # force-DELTA emergency stop.
    STACK_STEP_M = 0.004             # slow final-approach step size.
    STACK_APPROACH_GAP_M = 0.02      # fast approach stops this far above contact.
    STACK_DESCENT_VEL_FRAC = 0.10    # slow final-approach velocity fraction.

    def __init__(
        self,
        robot,
        detection_map=None,
        velocity=0.25,
        grasp_calibration: "GraspCalibration" = None,
        safe_height=None,
        pipeline=None,
        strict_placement_verify: bool = False,
    ):
        self.robot = robot
        self.detection_map = detection_map or {}
        try:
            from spark_real.control.spatial_verify import remember_targets
            remember_targets(self)
        except Exception:  # noqa: BLE001
            pass
        self.velocity = velocity
        self.safe_height = safe_height or self.SAFE_HEIGHT_Z
        self._results = []
        self._abort = False
        # Abort epochs. _abort_epoch is bumped by every abort request;
        # _armed_epoch is stamped at the TASK boundary (reset_task_state).
        # execute_score clears the flag only when the two agree, so a stop
        # pressed during capture -> detect -> plan survives into the run.
        self._abort_epoch = 0
        self._armed_epoch = 0
        # Serialises bump-and-read against retract-if-unchanged, so the
        # watchdog cannot retract an operator stop that landed between its
        # own request and its retraction.
        self._abort_lock = threading.Lock()
        self._running = False
        self._wrist_servo_enabled = True
        # Wrist-camera refinement during approach/place/search; off by
        # default. Resolved from control.wrist_refine below.
        self._wrist_refine_enabled = False
        self._pipeline = pipeline
        self._recorder = None
        self._servo = CartesianServo(robot, rate_hz=125.0)
        # Everything scoped to ONE task lives in reset_task_state.
        self.reset_task_state()
        self._strict_placement_verify = strict_placement_verify
        self.grasp_calibration = grasp_calibration
        self._install_abort_aware_send_script()
        self._apply_family_workspace_bounds()
        self._resolve_wrist_refine_flag()
        self._apply_family_tuning()
        self._load_primitive_timeouts()

    def _apply_family_tuning(self):
        """Override grasp/stack tuning constants from the `grasp:`/`stack:` YAML
        blocks (profile.raw preferred, else <family>_default.yaml). Any
        missing key keeps the class default.
        """
        cfg = getattr(self._pipeline, "config", None) if self._pipeline else None
        if cfg is None:
            return
        family = (getattr(cfg, "robot_family", "ur10e") or "ur10e").lower()
        try:
            profile = getattr(self._pipeline, "profile", None)
            grasp = family_block(profile, family, "grasp")
            stack = family_block(profile, family, "stack")
            mapping = [
                (grasp, "depth_m", "GRASP_DEPTH_M"),
                (grasp, "force_empty_n", "GRASP_FORCE_EMPTY_N"),
                (grasp, "lift_check_m", "GRASP_LIFT_CHECK_M"),
                (grasp, "lift_delta_min_n", "GRASP_LIFT_DELTA_MIN_N"),
                (grasp, "yaw_aspect_ratio_gate", "GRASP_YAW_AR_GATE"),
                (grasp, "yaw_min_conf", "GRASP_YAW_MIN_CONF"),
                (grasp, "yaw_min_obb_conf", "GRASP_YAW_MIN_OBB_CONF"),
                (grasp, "released_max_pos", "GRIPPER_RELEASED_MAX_POS"),
                (grasp, "release_min_travel", "GRIPPER_RELEASE_MIN_TRAVEL"),
                (grasp, "release_confirm_timeout_s", "RELEASE_CONFIRM_TIMEOUT_S"),
                (grasp, "max_yaw_offset_deg", "_MAX_YAW_OFFSET_DEG"),
                (stack, "clearance_m", "STACK_CLEARANCE_M"),
                (stack, "max_descent_m", "STACK_MAX_DESCENT_M"),
                (stack, "contact_force_n", "STACK_CONTACT_FORCE_N"),
                (stack, "hard_stop_delta_n", "STACK_HARD_STOP_DELTA_N"),
                (stack, "step_m", "STACK_STEP_M"),
                (stack, "approach_gap_m", "STACK_APPROACH_GAP_M"),
                (stack, "descent_vel_frac", "STACK_DESCENT_VEL_FRAC"),
            ]
            applied = {}
            for block, key, attr in mapping:
                if isinstance(block, dict) and block.get(key) is not None:
                    setattr(self, attr, float(block[key]))
                    applied[attr] = getattr(self, attr)
            if applied:
                logger.info(
                    "ScoreExecutor grasp/stack tuning from config (%s): %s",
                    family, applied,
                )
            # Env override (quick experiment): SPARK_YAW_AR_GATE sets the oriented
            # (OBB-yaw) grasp gate; set it very high to force a plain top-down
            # grasp on objects whose aspect ratio would otherwise trip the yaw.
            _g = os.environ.get("SPARK_YAW_AR_GATE")
            if _g is not None:
                self.GRASP_YAW_AR_GATE = float(_g)
                logger.info(
                    "GRASP_YAW_AR_GATE=%.2f (SPARK_YAW_AR_GATE override)",
                    self.GRASP_YAW_AR_GATE,
                )
            # Cable limit for the oriented (OBB-yaw) grasp, in degrees.
            _y = os.environ.get("SPARK_MAX_YAW_OFFSET_DEG") or getattr(
                self, "_MAX_YAW_OFFSET_DEG", None
            )
            if _y is not None:
                self.MAX_YAW_OFFSET = math.radians(float(_y))
                logger.info(
                    "MAX_YAW_OFFSET=%.0f deg (oriented-grasp wrist-3 cable limit)",
                    float(_y),
                )
        except Exception as exc:
            logger.warning(
                "Could not load grasp/stack tuning: %s; keeping defaults", exc
            )

    def _resolve_wrist_refine_flag(self):
        """Read control.wrist_refine from the profile (or family YAML).

        Mirrors how _apply_family_workspace_bounds sources its values;
        defaults to False on any missing key or read error.
        """
        cfg = getattr(self._pipeline, "config", None) if self._pipeline else None
        if cfg is None:
            return
        family = (getattr(cfg, "robot_family", "ur10e") or "ur10e").lower()
        try:
            profile = getattr(self._pipeline, "profile", None)
            control = family_block(profile, family, "control")
            if "wrist_refine" in control:
                self._wrist_refine_enabled = bool(control["wrist_refine"])
                logger.info(
                    "ScoreExecutor wrist_refine=%s (control.wrist_refine)",
                    self._wrist_refine_enabled,
                )
        except Exception as exc:
            logger.warning(
                "Could not load control.wrist_refine: %s; "
                "keeping wrist refine disabled",
                exc,
            )
        # Env override: SPARK_WRIST_REFINE=1 forces wrist refinement on.
        _env = os.environ.get("SPARK_WRIST_REFINE")
        if _env is not None:
            self._wrist_refine_enabled = _env not in ("0", "false", "False", "")
            logger.info(
                "ScoreExecutor wrist_refine=%s (SPARK_WRIST_REFINE override)",
                self._wrist_refine_enabled,
            )

    def _load_primitive_timeouts(self):
        """Resolve the per-primitive wall-clock budgets.

        Starts from the defaults and overlays any timeouts: block in the
        resolved RobotProfile (preferred) or family YAML. Re-run from
        pipeline_init once _profile is wired so the profile override wins.
        """
        cfg = getattr(self._pipeline, "config", None) if self._pipeline else None
        family = (
            (getattr(cfg, "robot_family", None) or "").lower() if cfg is not None
            else ""
        )
        profile = getattr(self, "_profile", None)
        try:
            self._primitive_timeouts = primitive_timeouts.load_timeouts(
                profile=profile, family=family or None
            )
            logger.info(
                "ScoreExecutor primitive timeouts loaded (%d entries, default=%.1fs)",
                len(self._primitive_timeouts),
                self._primitive_timeouts.get(
                    "default", primitive_timeouts.DEFAULT_TIMEOUT_S
                ),
            )
        except Exception as exc:
            logger.warning(
                "Could not load primitive timeouts (%s); using paper defaults", exc
            )
            self._primitive_timeouts = primitive_timeouts.default_timeouts()
            self._primitive_timeouts["default"] = primitive_timeouts.DEFAULT_TIMEOUT_S

    def _apply_family_workspace_bounds(self):
        """Replace UR-default workspace bounds from the resolved profile.

        Sources configs/<family>_default.yaml plus any per-machine overlay.
        Falls back to class defaults if no pipeline/config is present, and
        re-reads the family YAML directly when the profile is missing/empty.
        """
        cfg = getattr(self._pipeline, "config", None) if self._pipeline else None
        if cfg is None:
            return
        family = (getattr(cfg, "robot_family", "ur10e") or "ur10e").lower()
        if family == "ur10e":
            # UR10e keeps its box, reach, table floor, and grasp rotvec in
            # the family YAML control: block.
            self._apply_ur10e_control_bounds()
        elif family in FRANKA_FAMILIES:
            apply_franka_bounds(self, family)

    def _apply_ur10e_control_bounds(self):
        """Load the UR10e executor box from the control: block.

        Sources the resolved profile (or ur10e_default.yaml directly);
        every key defaults to the class constant. Reads control.* (the
        executor/SafeRobot box), not the workspace: block.
        """
        try:
            control = family_block(getattr(self, "_profile", None), "ur10e", "control")
            if not control:
                return
            grasp = control.get("grasp_orientation")
            if grasp is not None:
                self.GRASP_ORIENTATION = [float(v) for v in grasp]
            ws_min = control.get("workspace_min")
            if ws_min is not None:
                self.WORKSPACE_MIN = np.array([float(v) for v in ws_min], dtype=float)
            ws_max = control.get("workspace_max")
            if ws_max is not None:
                self.WORKSPACE_MAX = np.array([float(v) for v in ws_max], dtype=float)
            if control.get("max_reach") is not None:
                self.MAX_REACH = float(control["max_reach"])
            # Table floor precedence: calibration file > YAML control.table_z_floor
            # > class default. The YAML value / class const stay as the FALLBACK.
            if control.get("table_z_floor") is not None:
                self.TABLE_Z_FLOOR = float(control["table_z_floor"])
                floor_src = "config: %.4f" % self.TABLE_Z_FLOOR
            else:
                floor_src = "class default: %.4f" % self.TABLE_Z_FLOOR
            cal = load_table_plane("ur10e")
            if cal is not None:
                surface_z = float(cal["surface_z"])
                # Margin is added so the floor sits a few mm ABOVE (less negative
                # than) the measured surface. Prefer the cal file's own margin.
                margin = float(
                    cal.get(
                        "safety_margin_m",
                        control.get("table_safety_margin_m", 0.003),
                    )
                )
                cal_floor = surface_z + margin
                # Sanity clamp: a cal recorded with a bad TCP reading must not
                # LOWER the safety floor below the config/class fallback by more
                # than 2 cm (that is how a bumped calibration drives the gripper
                # into the table). Raising the floor is always accepted.
                if cal_floor < self.TABLE_Z_FLOOR - 0.02:
                    logger.warning(
                        "ScoreExecutor REJECTING table calibration floor %.4f: "
                        "more than 2cm below fallback %.4f (suspect bad cal); "
                        "keeping fallback",
                        cal_floor,
                        self.TABLE_Z_FLOOR,
                    )
                else:
                    self.TABLE_Z_FLOOR = cal_floor
                    floor_src = "calibration: surface=%.4f +%dmm -> %.4f" % (
                        surface_z,
                        round(margin * 1000),
                        self.TABLE_Z_FLOOR,
                    )
            logger.info("ScoreExecutor table floor from %s", floor_src)
            logger.info(
                "ScoreExecutor UR10e control bounds from config: "
                "X=[%.2f,%.2f] Y=[%.2f,%.2f] Z=[%.2f,%.2f] reach=%.2f "
                "table_z_floor=%.3f grasp_rotvec=%s",
                self.WORKSPACE_MIN[0],
                self.WORKSPACE_MAX[0],
                self.WORKSPACE_MIN[1],
                self.WORKSPACE_MAX[1],
                self.WORKSPACE_MIN[2],
                self.WORKSPACE_MAX[2],
                self.MAX_REACH,
                self.TABLE_Z_FLOOR,
                self.GRASP_ORIENTATION,
            )
        except Exception as exc:
            logger.warning(
                "Could not load UR10e control bounds: %s; " "keeping class defaults",
                exc,
            )

    def _install_abort_aware_send_script(self):
        """
        Wrap robot._send_script (UR-only) to respect the abort flag.
        """
        executor_ref = self

        def make_wrapper(orig):
            def abort_aware_send(script):
                if (
                    executor_ref._running
                    and executor_ref._abort
                    and not is_braking_script(script)
                ):
                    raise AbortRequested()
                return orig(script)

            abort_aware_send._spark_abort_wrapped = True
            return abort_aware_send

        target = self.robot
        seen = set()
        while target is not None and id(target) not in seen:
            seen.add(id(target))
            if hasattr(target, "_send_script"):
                original_send = target._send_script
                if not getattr(original_send, "_spark_abort_wrapped", False):
                    try:
                        target._send_script = make_wrapper(original_send)
                    except AttributeError:
                        pass
            target = getattr(target, "_robot", None)

    def update_detections(self, detection_map):
        self.detection_map = detection_map
        # Snapshot the approved scene for the end-of-task spatial rung here,
        # not at verify time: mid-run re-detections mutate detection_map in
        # place.
        try:
            from spark_real.control.spatial_verify import remember_targets
            remember_targets(self)
        except Exception:  # noqa: BLE001 - memory is an aid, never a blocker
            pass

    def _join_redetect(self, timeout: float = 5.0):
        """Bounded wait for a live post-release re-detect thread.

        Called before the executor starts any new detect so two detection
        passes never run concurrently against the stateful SAM3 predictor
        or mutate detection_map from two threads. Bounded so a wedged detect
        cannot hang a run.
        """
        thread = getattr(self, "_redetect_thread", None)
        if thread is None or not thread.is_alive():
            return
        try:
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning(
                    "_join_redetect: re-detect thread still running after "
                    "%.1fs; proceeding (perception lock serializes SAM3)",
                    timeout,
                )
            else:
                self._redetect_thread = None
        except Exception as exc:
            logger.warning("_join_redetect failed: %s", exc)

    # Fields whose lifetime is ONE task. Attribute -> factory for a fresh value.
    # Anything a run writes and a later run reads belongs here.
    _TASK_SCOPED_STATE = {
        "_holding": bool,
        "_placed_labels": set,
        "_last_pick_label": str,
        "_last_place_label": str,
        "_last_keypoint_label": lambda: None,
        "_last_action_failed": bool,
        "_grasp_retry_count": dict,
        "_destination_labels": set,
        # Depth memory written by _approach_target and read by the grasp's
        # calibration record.
        "_grasp_perception_target_z": lambda: None,
        # Actual TCP z at the moment the jaws closed (grasp v2 / recovery
        # re-grasp). Read by the release give-back. Reset per pick in
        # _approach_target, per task here.
        "_actual_grasp_tcp_z": lambda: None,
        "_last_grasp_target_width": lambda: None,
        "_active_grasp_label": lambda: None,
        "_active_grasp_orient": lambda: None,
        "_active_grasp_strategy": lambda: None,
        # Release look-ahead written per-action by _run_actions/_run_branch
        # and read by the place transport's deep-dip side choice
        # (ReleaseMixin._dip_side_orient).
        "_pending_release_tilt": lambda: None,
    }

    def reset_task_state(self):
        """Drop every field scoped to ONE task.

        The executor is a server-lifetime singleton: one instance runs every
        task. Called from __init__ and from success_verifier.reset_run_state
        (the task boundary). NOT from execute_score: the closed loop calls
        that once per pass and _placed_labels must survive the passes of one
        task.
        """
        for attr, factory in self._TASK_SCOPED_STATE.items():
            setattr(self, attr, factory())
        # The task boundary is the ONE place an abort is legitimately
        # forgotten. Arming here lets execute_score distinguish a stale flag
        # from a stop pressed while this task was being planned.
        self._armed_epoch = getattr(self, "_abort_epoch", 0)
        self._abort = False
        # A post-release re-detection thread mutates detection_map in place,
        # so an unjoined one would rewrite the NEXT task's detections. Daemon
        # threads cannot be killed; wait briefly and say so if still running.
        thread = getattr(self, "_redetect_thread", None)
        if thread is not None:
            try:
                thread.join(timeout=2.0)
                if thread.is_alive():
                    logger.warning(
                        "reset_task_state: re-detect thread from the previous "
                        "task is still running; it may overwrite detections"
                    )
            except Exception as exc:
                logger.warning("reset_task_state: re-detect join failed: %s", exc)
            self._redetect_thread = None


    def _check_workspace(self, pos):
        if np.linalg.norm(pos) > self.MAX_REACH:
            return False
        return bool(
            np.all(pos >= self.WORKSPACE_MIN) and np.all(pos <= self.WORKSPACE_MAX)
        )

    def _get_current_position(self):
        if hasattr(self.robot, "get_tcp_pose"):
            pose = self.robot.get_tcp_pose()
            if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
                return pose[:3, 3].copy()
            return np.array(pose[:3])
        return np.zeros(3)

    _LEGATO_BLENDABLE = {"move_relative", "move_to_keypoint"}

    _LEAF_TYPES = {
        "move_to_keypoint",
        "grasp",
        "release",
        "move_relative",
        "open_drawer",
        "push_object",
        "turn_knob",
        "wait",
        "pour",
    }


    # Grammar slot aliases accepted everywhere the canonical name is expected.
    # The canonical name wins if both are present. Applied once at the dispatch
    # choke point so both builtin and skill primitives see normalized params.
    _PARAM_ALIASES = {
        "label": "keypoint_label",
        "target": "target_label",
    }


    _last_safety_reconcile_ts = 0.0


