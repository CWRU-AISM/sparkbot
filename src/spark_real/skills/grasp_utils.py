"""Shared grasp-skill helpers: yaw folding and orientation logging."""

import logging

import numpy as np
from scipy.spatial.transform import Rotation as _R

logger = logging.getLogger(__name__)


def canonical_symmetric_yaw(yaw: float) -> float:
    """Fold a yaw into the two-jaw canonical range via modulo pi.

    A two-jaw gripper grasps identically at yaw and yaw+pi, so any angle
    can be reduced modulo pi and then folded into (-pi/2, pi/2]. This is
    the modulo-pi form used by the OBB-derived grasp yaws; place_in_slot
    uses a no-modulo single fold on an already-bounded input. Distinct from `_nearest_symmetric_yaw`, which instead
    minimizes wrist travel against the current pose after this fold.
    """
    yaw = yaw % np.pi
    if yaw > np.pi / 2:
        yaw -= np.pi
    return yaw


def orient_diag(tag: str, target_rotvec, achieved_R=None) -> None:
    """
    Log TCP-Z direction and tilt error for orientation diagnostics.
    """
    try:
        target_R = _R.from_rotvec(np.asarray(target_rotvec, dtype=float)).as_matrix()
        tz = target_R[:, 2]
        tilt_deg = float(np.degrees(np.arccos(np.clip(-tz[2], -1.0, 1.0))))
        msg = (
            "[diag/%s] target TCP-Z=(%+.3f,%+.3f,%+.3f) "
            "tilt_from_down=%.2f deg" % (tag, tz[0], tz[1], tz[2], tilt_deg)
        )
        if achieved_R is not None:
            az = np.asarray(achieved_R)[:, 2]
            dot = float(np.clip(np.dot(tz, az), -1.0, 1.0))
            err_deg = float(np.degrees(np.arccos(dot)))
            ach_tilt_deg = float(np.degrees(np.arccos(np.clip(-az[2], -1.0, 1.0))))
            msg += (
                " | achieved TCP-Z=(%+.3f,%+.3f,%+.3f) "
                "achieved_tilt_from_down=%.2f deg orient_err=%.2f deg"
                % (az[0], az[1], az[2], ach_tilt_deg, err_deg)
            )
        logger.info(msg)
    except Exception as exc:
        logger.debug("[diag/%s] orient log failed: %s", tag, exc)
