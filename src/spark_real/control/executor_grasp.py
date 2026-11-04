"""
GraspMixin: gripper control and grasp verification for ScoreExecutor.
"""

import logging

from spark_real.control.grasp_exec import GraspExecMixin  # noqa: F401
from spark_real.control.grasp_v2 import GraspV2Mixin  # noqa: F401
from spark_real.control.grasp_verdict import (  # noqa: F401
    _SUBPART_WORDS,
    GRASP_FORCE_EMPTY_DEFAULT_N,
    GRASP_LIFT_DELTA_MIN_N,
    GRASP_WIDTH_MAX_M,
    GRASP_WIDTH_MIN_M,
    ForceVerdict,
    GraspVerdict,
    _call_force_publish,
    _call_maybe_passive,
    _parent_object_label,
    force_verdict,
)
from spark_real.control.grasp_verify import GraspVerifyMixin  # noqa: F401
from spark_real.control.gripper_io import GripperIOMixin  # noqa: F401

logger = logging.getLogger(__name__)


class GraspMixin(GripperIOMixin, GraspV2Mixin, GraspExecMixin, GraspVerifyMixin):
    """
    Gripper squeeze, grasp execution, and grasp verification.
    """

    _GRASP_WIDTH_MIN_M = GRASP_WIDTH_MIN_M
    _GRASP_WIDTH_MAX_M = GRASP_WIDTH_MAX_M

    # Robotiq close / re-squeeze force (0-100 percent). The closed POSITION
    # (pos 1.0) takes up slack on a compressible object; this force CAP
    # protects it from over-crush (100 crushed soft plushies). A BT `force`
    # param still wins; env SPARK_GRASP_CLOSE_FORCE overrides here.
    GRASP_CLOSE_FORCE = 60
    # Settle pause (s) after the clamp / at a stationary transport waypoint,
    # BEFORE the forced gObj publish+read, so the read is fresh rather than a
    # mid-move stale value. Env: SPARK_GRASP_VERIFY_SETTLE_S.
    GRASP_VERIFY_SETTLE_S = 0.4

    # Robotiq 2F-85 jaw-travel model
    # The settle must be a sleep, not a condition wait: reading the jaw
    # registers uploads a publish program, and a new program REPLACES the
    # rq_close_and_wait still running on the controller, so polling the close
    # would cancel the close. The sleep is sized from the datasheet upper
    # bound on travel time.
    #
    # 2F-85: 85 mm stroke, closing speed 20-150 mm/s over the speed_norm range.
    ROBOTIQ_STROKE_MM = 85.0
    ROBOTIQ_SPEED_MIN_MM_S = 20.0
    ROBOTIQ_SPEED_MAX_MM_S = 150.0
    # UR10eDriver.close_gripper / open_gripper already sleep this much after the
    # send, so the caller's settle only has to cover what is left.
    ROBOTIQ_DRIVER_SETTLE_S = 0.5
    # Margin on top of the datasheet travel time (send + compile + the
    # controller finishing rq_close_and_wait's own bookkeeping).
    ROBOTIQ_SETTLE_MARGIN_S = 0.15
    # Jaw counts, 0 (open) .. 255 (closed).
    ROBOTIQ_POS_MAX = 255.0
    # UNMEASURED ASSUMPTION: UR10eDriver.open_gripper passes no speed_norm, so
    # the URCap's own open speed is unknown on this rig. The open settle is
    # sized as if it ran at this speed_norm (the one the grasp closes at).
    # Assuming the datasheet's slowest (20 mm/s) would mean a 3.9 s settle per
    # open; at 60 the open pays 0.5 s (driver) + ~0.52 s.
    ROBOTIQ_OPEN_SPEED_NORM_ASSUMED = 60
    # Jaw position at/below which the jaws count as ALREADY OPEN, so
    # _ensure_jaws_open may skip commanding an open. Much tighter than the
    # release witness's GRIPPER_RELEASED_MAX_POS (60): this one gates whether
    # it is safe to descend, so it demands the jaws be within ~5 mm of the
    # open stop (15/255 of an 85 mm stroke). Too permissive here means
    # descending with part-closed jaws.
    JAWS_OPEN_MAX_POS = 15


    # A gripper position at/below this means the jaws are essentially open
    # (open reads ~2.5, fully closed 250+). gObj has read True at pos=5 with
    # the jaws never closed; open jaws cannot hold anything, whatever a flag
    # says.
    JAWS_OPEN_POS_MAX = 60


    # An object that has not moved further than this since the grasp was
    # planned is still sitting where it was.
    VISION_MOVED_TOL_M = 0.04


