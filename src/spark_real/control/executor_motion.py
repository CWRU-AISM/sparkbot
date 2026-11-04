"""MotionMixin: movement primitives for ScoreExecutor.

Composes the motion mixins (motion_keypoint, motion_wrist, motion_approach,
motion_place, motion_place_yaw, motion_io, motion_blend) and holds the
motion constants they share through ``self``. The IK-backed joint-space
helpers live in executor_ik.IkMixin.
"""

import logging

import numpy as np

from spark_real.control.motion_approach import ApproachMixin  # noqa: F401
from spark_real.control.motion_blend import BlendMixin  # noqa: F401
from spark_real.control.motion_common import _StillnessTracker  # noqa: F401
from spark_real.control.motion_io import MotionIOMixin  # noqa: F401
from spark_real.control.motion_keypoint import KeypointMixin  # noqa: F401
from spark_real.control.motion_place import PlaceMixin  # noqa: F401
from spark_real.control.motion_place_yaw import PlaceYawMixin  # noqa: F401
from spark_real.control.motion_wrist import WristRefineMixin  # noqa: F401

logger = logging.getLogger(__name__)


class MotionMixin(
    KeypointMixin,
    WristRefineMixin,
    ApproachMixin,
    PlaceMixin,
    PlaceYawMixin,
    MotionIOMixin,
    BlendMixin,
):
    """
    Movement primitives mixed into ScoreExecutor.
    """

    # Hard wrist-3 cable limit, not a tuning knob: the wrist-camera USB cable
    # will not take more travel than this. Config: grasp.max_yaw_offset_deg
    # (env SPARK_MAX_YAW_OFFSET_DEG). Enforced on the COMPOSED rotation by
    # grasp_strategy.check_yaw_limit, not just on the requested angle.
    # 100, not 90: a two-jaw gripper is symmetric mod 180, and at a 90 cap the
    # nearer equivalent of a yaw near +-90 (e.g. +92.9 for -87.1) is refused
    # by a few degrees, so the wrist takes the LONGER path. 10 deg of headroom.
    MAX_YAW_OFFSET = np.deg2rad(100)
    CLEARANCE_MARGIN = 0.12
    OBSTACLE_XY_RADIUS = 0.12

    # How much of the place descent the Cartesian servo keeps (m). Everything
    # above this is flown as a movej/blend row; the servo owns every millimetre
    # where the held object can contact the container.
    PLACE_SERVO_STANDOFF_M = 0.02
    # Extra clearance above the first possible rim contact, covering the
    # held-extent estimate's own error. See _place_standoff_m.
    PLACE_CONTACT_MARGIN_M = 0.015
    # Ceiling on the slow phase. The servo crawls near convergence, so an
    # unbounded standoff would turn a bad held-extent number into a 10 s place.
    PLACE_SERVO_STANDOFF_MAX_M = 0.12

    WRIST_REFINE_MODES = ("off", "depth", "ray_plane")
    WRIST_REFINE_MODE_DEFAULT = "ray_plane"
    # A plane prior nearer the camera than this is rejected: the ray grazes
    # the plane and the intersection runs away laterally.
    WRIST_RAY_MIN_STANDOFF_M = 0.03

    # How far OUTSIDE a container's own planar footprint the TCP may sit and
    # still count as having arrived over it. The ONLY tolerance the place path
    # judges arrival with (see _place_arrival); not the servo's pos_threshold
    # (3 mm), which says nothing about whether the object is over the container.
    # +2 cm mirrors the verifier's containment margin
    # (success_predicates.EvalConfig.inside_xy_margin_m = -0.02): anything
    # refused here is at least 4 cm outside a region that could have passed
    # verify, so the gate is never the reason a good task refuses to let go.
    PLACE_ARRIVAL_XY_MARGIN_M = 0.02


    # A shape fit must beat every orientation >=90 deg away by this much
    # before it is allowed to decide direction; below it the object is too
    # symmetric to call and the heavy-end test gets a turn.
    JIGSAW_MIN_MARGIN = 0.05
    JIGSAW_MIN_IOU = 0.20


    # Above this residual orientation error, the wrist is corrected at
    # clearance height before any contact-zone motion (5 deg: well above
    # servo convergence residuals ~0.06 deg, well below anything that sweeps
    # a held tool into a rim).
    PLACE_ORI_FIX_GATE_RAD = np.deg2rad(5.0)
    # Clearance above the place target for wrist rotations with a held
    # object: longer than the tool extent below the jaws (~10 cm screwdriver)
    # plus margin.
    ROTATE_CLEARANCE_M = 0.15


    # Joint tolerance counting a URScript movej as arrived (rad, per joint).
    JOINT_ARRIVE_TOL = 0.01
    # URScript upload/start lag on the primary socket: the arm is stationary for
    # up to ~1 s after the send, so stillness before this means "not started".
    URSCRIPT_START_LAG_S = 1.0
    # Arrival-poll cadence. The exact tests (joint target, Cartesian proximity)
    # are cheap register reads, so polling at 25 ms rather than 100 ms removes
    # up to 75 ms of dead time per program (~0.5 s over a pick-and-place).
    # MOTION_SAMPLE_S stays the window the displacement test is evaluated
    # over; see _StillnessTracker.
    MOTION_POLL_S = 0.025
    MOTION_SAMPLE_S = 0.1
    MOTION_STILL_SAMPLES = 3
    # Slack added on top of 3x the ideal move duration when budgeting a wait.
    MOTION_BUDGET_SLACK_S = 1.5

    # Blended-path defaults. Config: control.blend_motion / blend_radius_m /
    # blend_descent_radius_m / blend_min_radius_m / blend_max_rows; env
    # SPARK_UR_BLEND overrides the on/off switch. See _blend_cfg.
    BLEND_MOTION_DEFAULT = True
    BLEND_RADIUS_M = 0.05
    # A phrase that ends in a grasp/place descent keeps that descent
    # essentially vertical: 2 cm of corner-cut into a 10-15 cm descent, not 5.
    BLEND_DESCENT_RADIUS_M = 0.02
    BLEND_MIN_RADIUS_M = 0.01
    BLEND_MAX_ROWS = 8
    BLEND_TIMEOUT_CAP_S = 45.0
    # No progress-register advance AND a still TCP for this long, mid-path, is
    # the stall signature (vs a 15.0 s grind into a mechanical stop ending as
    # a silent full-path timeout).
    BLEND_STALL_S = 3.0
    # Last-resort brake decel (rad/s^2) for a driver with no velocity-sized
    # brake() of its own. Only reached by bare-mixin test doubles; the real
    # driver computes this from the speed it last commanded.
    BLEND_ABORT_DECEL = 4.0


