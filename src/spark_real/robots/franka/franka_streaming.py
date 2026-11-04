"""
Realtime streaming for FrankaDriver: joint servo and Cartesian velocity.

Each servo or velocity call issues an asynchronous franky motion that
preempts the previous one; franky's Ruckig planner re-plans at the 1 kHz
control loop while Python sends updates at 200-500 Hz. The host class
provides self._robot (franky Robot), the session locks and flags set in
FrankaDriver.__init__, _check_connected, _dispatch_motion and
_velocity_to_rdf.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

try:
    from franky import (
        CartesianVelocityMotion,
        CartesianVelocityStopMotion,
        Frame,
        JointStopMotion,
        JointVelocityStopMotion,
        JointWaypoint,
        JointWaypointMotion,
        RelativeDynamicsFactor,
        RobotVelocity,
        Twist,
    )
except ImportError:
    pass

logger = logging.getLogger(__name__)


class FrankyStreamingMixin:
    """servo_joint / send_velocity sessions and their teardown."""

    def servo_joint(
        self,
        q: Sequence[float],
        velocity: float = 0.5,
        acceleration: float = 0.5,
        dt: float = 0.002,
        lookahead_time: float = 0.1,
        gain: int = 300,
    ):
        """
        Real-time joint servo: stream a new target to the running motion.

        Each call issues an asynchronous JointWaypointMotion; franky's Ruckig
        planner preempts and re-plans on the fly, producing smooth motion at
        the 1 kHz control loop while accepting Python-side updates at 200-500 Hz.
        """
        self._check_connected()
        q_arr = list(q)
        if len(q_arr) != 7:
            raise ValueError(f"Franka expects 7 joints, got {len(q_arr)}.")
        self._preempt_velocity_if_active()

        vel_rdf = self._velocity_to_rdf(
            velocity, self._max_joint_vel, self.DEFAULT_JOINT_VEL
        )
        rdf = RelativeDynamicsFactor(vel_rdf, vel_rdf * 0.5, vel_rdf * 0.25)
        target = np.asarray(q_arr, dtype=float)

        with self._servo_lock:
            waypoint = JointWaypoint(q_arr)
            motion = JointWaypointMotion(
                [waypoint],
                relative_dynamics_factor=rdf,
                return_when_finished=False,
            )
            self._dispatch_motion(motion, asynchronous=True)
            self._servo_active = True
            self._servo_last_target = target

    def servo_stop(self):
        # Tear down an active servo streaming session.
        self._check_connected()
        with self._servo_lock:
            try:
                self._robot.move(JointStopMotion(), asynchronous=False)
            except Exception as exc:
                logger.warning("servo_stop: %s", exc)
            finally:
                self._servo_active = False
                self._servo_last_target = None

    def _preempt_servo_if_active(self):
        if self._servo_active:
            logger.debug("Preempting active servo session")
            self.servo_stop()


    def send_velocity(
        self,
        linear: Sequence[float],
        angular: Optional[Sequence[float]] = None,
        acceleration: float = 0.5,
        duration: float = 0.0,
    ):
        # Stream a Cartesian velocity to the robot.
        self._check_connected()

        linear_arr = np.asarray(linear, dtype=float).ravel()
        if angular is None:
            if linear_arr.size != 6:
                raise ValueError(
                    f"send_velocity: expected 6-vec, got {linear_arr.size}."
                )
            twist6 = linear_arr
        else:
            angular_arr = np.asarray(angular, dtype=float).ravel()
            if linear_arr.size != 3 or angular_arr.size != 3:
                raise ValueError("send_velocity: expected linear=3, angular=3.")
            twist6 = np.concatenate([linear_arr, angular_arr])

        self._preempt_servo_if_active()

        # Auto-recover from reflex and suppress this tick
        if self._robot.has_errors:
            try:
                self._robot.recover_from_errors()
                logger.debug("send_velocity: auto-recovered from reflex")
            except Exception as _err:
                logger.warning("send_velocity: recover failed: %s", _err)
            self._velocity_active = False
            self._velocity_last_target = None
            return

        # Joint-limit-aware Cartesian velocity shaping
        shaped_twist = twist6
        try:
            qdot = self._safe_joint_velocity(twist6)
            J = np.asarray(
                self._robot.model.zero_jacobian(Frame.EndEffector, self._robot.state),
                dtype=float,
            )
            shaped_twist = J @ qdot
        except Exception as _exc:
            logger.warning("send_velocity: twist shaping unavailable (%s)", _exc)

        twist = Twist(shaped_twist[:3].copy(), shaped_twist[3:].copy())
        robot_vel = RobotVelocity(twist)
        rdf = 0.15

        motion = CartesianVelocityMotion(
            robot_vel,
            relative_dynamics_factor=rdf,
        )

        with self._velocity_lock:
            try:
                self._dispatch_motion(motion, asynchronous=True)
            except Exception as exc:
                self._velocity_active = False
                self._velocity_last_target = None
                msg = str(exc).lower()
                if any(
                    k in msg
                    for k in (
                        "motion finished commanded",
                        "velocity_discontinuity",
                        "acceleration_discontinuity",
                        "reflex",
                        "current mode",
                    )
                ):
                    try:
                        if self._robot.has_errors:
                            self._robot.recover_from_errors()
                    except Exception:
                        pass
                    logger.debug("send_velocity swallowed: %s", exc)
                    return
                raise
            self._velocity_active = True
            self._velocity_last_target = twist6.copy()

    def _safe_joint_velocity(self, twist6: np.ndarray) -> np.ndarray:
        """
        Convert a base-frame twist into a 7-DOF joint velocity
        with joint-limit repulsion via gradient projection (Liegeois 1977).
        """
        if self._robot.model is None:
            raise RuntimeError("franky model not loaded")

        state = self._robot.state
        q = np.asarray(state.q, dtype=float)
        J = np.asarray(
            self._robot.model.zero_jacobian(Frame.EndEffector, state), dtype=float
        )

        # Damped pseudo-inverse
        lam2 = 0.05**2
        Jpinv = J.T @ np.linalg.inv(J @ J.T + lam2 * np.eye(6))
        qdot_task = Jpinv @ np.asarray(twist6, dtype=float)

        # Joint-limit repulsion in the null space
        margin_frac = 0.10
        repel_gain = 1.5
        qdot_repel = np.zeros(7)
        for i, (lo, hi) in enumerate(self.JOINT_LIMITS):
            rng = hi - lo
            if rng <= 0:
                continue
            m = margin_frac * rng
            if q[i] > hi - m:
                qdot_repel[i] = -((q[i] - (hi - m)) / m) * repel_gain
            elif q[i] < lo + m:
                qdot_repel[i] = (((lo + m) - q[i]) / m) * repel_gain

        N = np.eye(7) - Jpinv @ J
        qdot_null = N @ qdot_repel
        qdot = qdot_task + qdot_null

        # Hard clamp near limits
        for i, (lo, hi) in enumerate(self.JOINT_LIMITS):
            rng = hi - lo
            if rng <= 0:
                continue
            m = margin_frac * rng
            if q[i] > hi - m and qdot[i] > 0:
                qdot[i] *= max(0.0, (hi - q[i]) / m)
            elif q[i] < lo + m and qdot[i] < 0:
                qdot[i] *= max(0.0, (q[i] - lo) / m)

        return qdot

    def stop_velocity(self):
        # Tear down an active velocity streaming session.
        self._check_connected()
        with self._velocity_lock:
            stopped = False
            for stop_cls in (CartesianVelocityStopMotion, JointVelocityStopMotion):
                try:
                    self._robot.move(stop_cls(), asynchronous=False)
                    stopped = True
                    break
                except Exception:
                    continue
            if not stopped:
                logger.warning("stop_velocity: neither stop type accepted")
            self._velocity_active = False
            self._velocity_last_target = None

    def _preempt_velocity_if_active(self):
        if self._velocity_active:
            logger.debug("Preempting active velocity session")
            self.stop_velocity()

