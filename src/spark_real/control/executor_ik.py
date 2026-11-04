"""
IkMixin: IK-backed joint-space motion primitives for ScoreExecutor.

Splits the inverse-kinematics motion path out of MotionMixin so the
solver-bound helpers live next to their lazy IK imports. Provides
_movej_via_ik, _movej_to_pose, _wait_stationary_after_servo, and
_execute_legato_phrase. These are mixed into ScoreExecutor alongside
MotionMixin and rely on the same instance attributes (self.robot,
self._servo, self.detection_map, ...) that the rest of the executor
sets up.
"""

import logging
import os
import time

import numpy as np

try:
    from spark_real.control.ur10e_ik_pyroki import solve_ik as solve_ik_ur10e
    from spark_real.control.ur10e_ik_pyroki import solve_ik_rtde as solve_ik_ur10e_rtde
except ImportError:
    solve_ik_ur10e = None
    solve_ik_ur10e_rtde = None

try:
    from spark_real.control.fr3_ik_pyroki import solve_ik as solve_ik_fr3
except ImportError:
    solve_ik_fr3 = None

# IK seed when the joint read fails (not the ready pose; J5 off the
# wrist singularity).
_FR3_IK_SEED = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

from spark_real.control.motion_limits import (  # noqa: F401
    _LEGACY,
    _LEGACY_LIMITS_ENV,
    _SPEED_FRACTION_ENV,
    JOINT_ACC_CAP_RAD_S2,
    JOINT_VEL_FLOOR_RAD_S,
    JOINT_VEL_MARGIN,
    JOINT_VEL_REF_LINEAR_M_S,
    LINEAR_ACC_CAP_M_S2,
    LINEAR_VEL_CAP_M_S,
    UR10E_JOINT_VEL_RATED_RAD_S,
    joint_speed_fraction,
    leading_axis_vel_cap,
    legacy_joint_limits,
    resolve_joint_limits,
    resolve_linear_limits,
)
from spark_real.control.phrases import PhraseMixin  # noqa: F401

logger = logging.getLogger(__name__)


class IkMixin(PhraseMixin):
    """
    IK-backed joint-space motion primitives mixed into ScoreExecutor.
    """

    def _robot_family(self):
        """Resolve the robot family off the driver."""
        fam = getattr(self.robot, "robot_family", None) or getattr(
            getattr(self.robot, "_robot", None), "robot_family", None
        )
        return str(fam or "").lower()

    def _movej_via_ik(self, position, orientation, velocity):
        """
        Solve IK in-process (ur10e_ik_pyroki), then movej over _send_script.

        Raises when the IK path is unavailable (SPARK_UR_PYROKI_IK=0, pyroki
        missing, non-UR family, no TCP offset) so the caller falls back to
        _movej_to_pose (firmware IK). The single-arm Franka solves with
        fr3_ik_pyroki and moves through the driver's move_to_joint_config.
        """
        self._check_abort()
        family = self._robot_family()
        pos = np.array(position, dtype=float)
        orient = list(orientation)

        if family == "franka":
            return self._movej_via_fr3_ik(pos, orient, velocity)
        if family != "ur10e":
            raise RuntimeError(f"joint-IK path is UR10e-only (family={family!r})")
        if os.environ.get("SPARK_UR_PYROKI_IK", "1") != "1":
            raise RuntimeError("SPARK_UR_PYROKI_IK disabled; using legacy path")
        if solve_ik_ur10e_rtde is None:
            raise RuntimeError("UR10e pyroki IK not available")
        # The executor targets are RTDE-frame gripper-TIP poses; pyroki
        # solves URDF-frame tool0. solve_ik_rtde bridges the Rz(180) base
        # rotation + the pendant TCP offset. Without the offset that bridge
        # is unsafe, so fall back to the (frame-safe) legacy path.
        ur_tcp_off = None
        for src in (self.robot, getattr(self.robot, "_robot", None)):
            if src is not None and hasattr(src, "get_tcp_offset"):
                try:
                    ur_tcp_off = np.asarray(src.get_tcp_offset(), dtype=float)
                    break
                except Exception as exc:
                    logger.warning("[movej_via_ik:ur10e] get_tcp_offset failed: %s", exc)
        if ur_tcp_off is None:
            raise RuntimeError("UR10e TCP offset unavailable; using legacy path")
        default_seed = np.array(
            getattr(self.robot, "HOME_CONFIG",
                    [1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]),
            dtype=float,
        )

        try:
            q_current = np.array(self.robot.get_joint_positions(), dtype=float)
        except Exception:
            q_current = default_seed

        q_target = solve_ik_ur10e_rtde(
            pos, orient, q_seed=q_current, tcp_offset=ur_tcp_off
        )
        if q_target is None:
            raise RuntimeError(
                f"pyroki IK failed for pos={pos.round(3)} orient={np.array(orient).round(3)}"
            )

        dq = np.abs(q_target - q_current[: q_target.size])
        logger.info(
            "[movej_via_ik:%s] pos=(%.3f,%.3f,%.3f) dq_max=%.2frad j%d",
            family or "?",
            pos[0],
            pos[1],
            pos[2],
            dq.max(),
            dq.argmax(),
        )

        # velocity is Cartesian m/s but joint moves want joint velocity (rad/s).
        # resolve_joint_limits owns that mapping AND the envelope; dq is passed
        # so a wrist-led move gets the wrist rating rather than the base's.
        vel_j, acc_j = resolve_joint_limits(velocity, dq=dq)
        logger.info(
            "[movej_via_ik:%s] v=%.2f rad/s a=%.2f rad/s^2 (lead j%d, cap %.2f)",
            family or "?",
            vel_j,
            acc_j,
            int(dq.argmax()),
            leading_axis_vel_cap(dq),
        )

        # movej over the URScript primary channel (_send_script): rtde_c.moveJ
        # silently no-ops after any speedl/gripper _send_script, so route the
        # joint target through the conflict-free URScript socket.
        # Fire-and-forget, then block on _wait_for_motion.
        self.robot.move_to_joint_config_urscript(
            q_target.tolist(), velocity=vel_j, acceleration=acc_j
        )
        # Hand the wait the joint target it just commanded: that is the
        # exact arrival test. Without it the wait falls back to Cartesian
        # proximity plus a velocity heuristic and burns the full timeout on
        # any move short enough to finish inside the URScript start lag.
        self._wait_for_motion(pos, q_target=q_target)

    # Blended multi-waypoint motion (UR10e only)

    def _ur_tcp_offset(self):
        """Pendant TCP offset (gripper tip relative to tool0), or None.

        Cached at connect by the driver; solve_ik_rtde cannot bridge the
        RTDE-frame tip target to a URDF tool0 target without it.
        """
        for src in (self.robot, getattr(self.robot, "_robot", None)):
            if src is not None and hasattr(src, "get_tcp_offset"):
                try:
                    return np.asarray(src.get_tcp_offset(), dtype=float)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[blend] get_tcp_offset failed: %s", exc)
        return None


    def _movej_to_pose(self, pose, velocity):
        """
        Joint-space move to a Cartesian pose (legacy fallback).

        UR fast path: emit movej(p[...]) URScript. Non-UR drivers fall back
        to move_linear or CartesianServo PD.
        """
        self._check_abort()
        # No solved joint target on this path (the controller runs its own IK),
        # so dq is unknown and the cap collapses to the slowest joint's rating.
        vel_j, acc_j = resolve_joint_limits(velocity, pose_fallback=True)
        if getattr(self.robot, "SUPPORTS_URSCRIPT", False) and hasattr(
            self.robot, "_send_script"
        ):
            ps = f"p[{pose[0]},{pose[1]},{pose[2]}," f"{pose[3]},{pose[4]},{pose[5]}]"
            if os.environ.get("SPARK_UR_MOVEL", "1") == "1":
                # Straight-line movel via the PRIMARY interface (_send_script),
                # NOT rtde_c.moveL: the gripper/speedl _send_script calls stop
                # the rtde_c control script, so rtde_c.moveL no-ops after a grasp
                # (arm never moves -> stack guard bails at full xy_err). movel
                # over _send_script works script-stopped AND keeps the TCP on a
                # straight line, killing the joint-space "wrenching" of movej.
                # movel v/a are Cartesian (m/s, m/s^2).
                vel_l, acc_l = resolve_linear_limits(velocity)
                sent = self.robot._send_script(f"movel({ps}, a={acc_l}, v={vel_l})")
            else:
                sent = self.robot._send_script(f"movej({ps}, a={acc_j}, v={vel_j})")
            if not sent:
                # Dropped command = failed primitive, never a silent pass
                # while _wait_for_motion times out on an arm that was never
                # told to move.
                raise RuntimeError(
                    "URScript move dropped (channel down); primitive failed"
                )
            self._wait_for_motion(np.array(pose[:3]))
            return

        if hasattr(self.robot, "move_linear"):
            try:
                self.robot.move_linear(list(pose), velocity=velocity)
                return
            except Exception as exc:
                logger.warning("move_linear failed (%s); trying CartesianServo PD", exc)
        if self._servo is not None and hasattr(self._servo, "move_to_pose"):
            try:
                self._servo.move_to_pose(list(pose), velocity=velocity)
                self._wait_stationary_after_servo()
                return
            except Exception as exc:
                logger.warning(
                    "CartesianServo PD fallback failed (%s); " "trying move_to_pose",
                    exc,
                )

        if hasattr(self.robot, "move_to_pose"):
            self.robot.move_to_pose(pose, velocity=velocity, wait=False)
            self._wait_for_motion(np.array(pose[:3]))
        elif hasattr(self.robot, "move_linear"):
            try:
                self.robot.move_linear(list(pose), velocity=velocity, asynchronous=True)
            except TypeError:
                self.robot.move_linear(list(pose), velocity=velocity)
            self._wait_for_motion(np.array(pose[:3]))

    def _wait_stationary_after_servo(
        self, max_wait: float = 2.0, vel_tol: float = 2e-3
    ):
        """
        Block until FR3 has fully decelerated after velocity-mode motion.
        """
        raw = getattr(self.robot, "_robot", self.robot)
        if hasattr(raw, "_robot"):
            raw = raw._robot
        try:
            t0 = time.time()
            prev_ok = False
            while time.time() - t0 < max_wait:
                try:
                    qdot = np.asarray(raw.state.dq, dtype=float)
                except Exception:
                    time.sleep(0.20)
                    return
                if np.max(np.abs(qdot)) < vel_tol:
                    if prev_ok:
                        return
                    prev_ok = True
                else:
                    prev_ok = False
                time.sleep(0.01)
        except Exception:
            time.sleep(0.20)

    def _movej_via_fr3_ik(self, pos, orient, velocity):
        """FR3 joint-space move: pyroki IK for the hand TCP, then a franky JointMotion."""
        if solve_ik_fr3 is None:
            raise RuntimeError("FR3 pyroki IK not available")
        try:
            q_current = np.asarray(self.robot.get_joint_positions(), dtype=float)
        except Exception:
            q_current = np.array(_FR3_IK_SEED)
        q_target = solve_ik_fr3(pos, orient, q_seed=q_current)
        if q_target is None:
            raise RuntimeError(f"FR3 IK failed for pos={pos.round(3)}")
        dq = np.abs(np.asarray(q_target) - q_current[:7])
        # Cartesian m/s -> joint rad/s, the same mapping the UR path uses.
        vel_j, _acc = resolve_joint_limits(velocity, dq=dq)
        logger.info(
            "[movej_via_ik:franka] pos=(%.3f,%.3f,%.3f) dq_max=%.2frad v=%.2frad/s",
            pos[0], pos[1], pos[2], float(dq.max()), vel_j,
        )
        self._check_abort()
        self.robot.move_to_joint_config(np.asarray(q_target).tolist(), velocity=vel_j)
