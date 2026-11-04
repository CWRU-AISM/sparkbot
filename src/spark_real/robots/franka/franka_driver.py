"""
Franka FR3 / Panda driver via franky (motion generators).

Thin translation layer over franky (TimSchneider42/franky). Joint vectors
are length 7; Cartesian poses are [x, y, z, rx, ry, rz] (axis-angle) on the
boundary. The arm only: the gripper is a registered driver
(spark_real.grippers, config `gripper.type`) mounted on it by the factory.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Sequence

import numpy as np

from spark_real.robots.franka.franka_streaming import FrankyStreamingMixin
from spark_real.utils.rotations import (
    axis_angle_to_quat as _axis_angle_to_quat,
    quat_to_axis_angle as _quat_to_axis_angle,
)

logger = logging.getLogger(__name__)

try:
    from franky import (
        Affine,
        CartesianMotion,
        CartesianStopMotion,
        JointMotion,
        RealtimeConfig,
        ReferenceType,
        RelativeDynamicsFactor,
        Robot,
    )

    HAS_FRANKY = True
except ImportError:
    HAS_FRANKY = False

# FR3 ready pose; configs/franka_default.yaml robot.home_config repeats it.
# J5 = -0.15 keeps the wrist out of its spherical singularity at home.
HOME_CONFIG = [0.0, -0.785398, 0.0, -2.356194, -0.15, 1.570796, 0.785398]

JOINT_LIMITS = [
    (-2.7437, 2.7437),
    (-1.7837, 1.7837),
    (-2.9007, 2.9007),
    (-3.0421, -0.1518),
    (-2.8065, 2.8065),
    (0.5445, 4.5169),
    (-3.0159, 3.0159),
]


class FrankaDriver(FrankyStreamingMixin):
    """
    Direct control interface for one Franka FR3 / Panda via franky.

    Public API mirrors UR10eDriver (see robots/driver_protocol.py).
    """

    robot_family = "franka"
    SUPPORTS_URSCRIPT = False
    SUPPORTS_VELOCITY_STREAMING = True
    HOME_CONFIG = list(HOME_CONFIG)
    JOINT_LIMITS = list(JOINT_LIMITS)
    DEFAULT_VELOCITY = 0.25
    DEFAULT_JOINT_VEL = 1.05
    DEFAULT_ACCELERATION = 0.5
    DEFAULT_JOINT_ACC = 1.4

    _MAX_LINEAR_VEL_FR3 = 1.7
    _MAX_JOINT_VEL_FR3 = 2.62

    # Stationarity gate: max |dq| (rad/s) the arm may carry into a new joint
    # motion. Below this, a JointMotion starts without tripping the
    # velocity/acceleration discontinuity reflex.
    STATIONARY_DQ_TOL = 0.01
    STATIONARY_GATE_TIMEOUT = 0.5

    def __init__(
        self,
        robot_ip: str,
        frequency: float = 1000.0,
        collision: Optional[dict] = None,
    ):
        if not HAS_FRANKY:
            raise RuntimeError(
                "franky not installed. pip install franky-control "
                "(libfranka must also be available)"
            )
        self.robot_ip = robot_ip
        self.frequency = frequency
        self._connected = False
        self._robot: Optional["Robot"] = None
        self._max_linear_vel = self._MAX_LINEAR_VEL_FR3
        self._max_joint_vel = self._MAX_JOINT_VEL_FR3

        # Collision thresholds from the family config; without them connect()
        # applies the scalar threshold.
        self._collision_cfg = dict(collision) if collision else {}

        # Realtime joint-servo session state
        self._servo_lock = threading.Lock()
        self._servo_active = False
        self._servo_last_target: Optional[np.ndarray] = None

        # Realtime Cartesian-velocity session state
        self._velocity_lock = threading.Lock()
        self._velocity_active = False
        self._velocity_last_target: Optional[np.ndarray] = None

    @property
    def connected(self) -> bool:
        return self._connected

    # Connection lifecycle

    def connect(self):
        # Open FCI connection to the Franka controller.
        logger.info("Connecting to Franka at %s...", self.robot_ip)
        try:
            self._robot = Robot(self.robot_ip)
        except Exception as e:
            if "realtime" in str(e).lower():
                logger.warning("Non-RT kernel (%s). Using RealtimeConfig.Ignore.", e)
                self._robot = Robot(
                    self.robot_ip, realtime_config=RealtimeConfig.Ignore
                )
            else:
                raise
        try:
            self._max_linear_vel = float(self._robot.translation_velocity_limit.max)
            self._max_joint_vel = float(
                np.max(np.asarray(self._robot.joint_velocity_limit.max))
            )
        except Exception:
            logger.debug("Could not read franky dynamics limits; using defaults.")
        self._connected = True
        self._apply_collision_behavior()
        logger.info("Connected. has_errors=%s", self._robot.has_errors)

    _COLLISION_NOMINAL = (
        "joint_torque_nom_lower", "joint_torque_nom_upper",
        "cartesian_force_nom_lower", "cartesian_force_nom_upper",
    )
    _COLLISION_ACC = (
        "joint_torque_acc_lower", "joint_torque_acc_upper",
        "cartesian_force_acc_lower", "cartesian_force_acc_upper",
    )

    def _apply_collision_behavior(self) -> bool:
        """
        Apply the config `collision_behavior:` thresholds through franky's
        set_collision_behavior overloads: all eight lists (acceleration +
        nominal), the four nominal lists, or a scalar 100 Nm / 100 N.
        """
        cfg = self._collision_cfg
        nom = [cfg.get(k) for k in self._COLLISION_NOMINAL]
        acc = [cfg.get(k) for k in self._COLLISION_ACC]
        try:
            if all(v is not None for v in acc + nom):
                tl, tu, fl, fu = (list(v) for v in nom)
                atl, atu, afl, afu = (list(v) for v in acc)
                self._robot.set_collision_behavior(atl, atu, tl, tu, afl, afu, fl, fu)
            elif all(v is not None for v in nom):
                self._robot.set_collision_behavior(*(list(v) for v in nom))
            else:
                self._robot.set_collision_behavior(100.0, 100.0)
            logger.info("Applied collision thresholds (per-axis=%s)", all(v is not None for v in nom))
            return True
        except Exception as exc:
            logger.warning("set_collision_behavior rejected: %s", exc)
            return False

    def disconnect(self):
        # Tear down the FCI connection.
        with self._servo_lock:
            self._servo_active = False
            self._servo_last_target = None
        with self._velocity_lock:
            self._velocity_active = False
            self._velocity_last_target = None
        if self._robot is not None:
            try:
                self._robot.stop()
            except Exception:
                pass
        self._robot = None
        self._connected = False
        logger.info("Disconnected")

    def _check_connected(self):
        if not self._connected:
            raise RuntimeError("Not connected to robot. Call connect() first.")

    # State queries

    def get_joint_positions(self) -> np.ndarray:
        # Return current joint positions (radians, length 7).
        self._check_connected()
        return np.asarray(self._robot.current_joint_state.position, dtype=float)

    def get_joint_velocities(self) -> np.ndarray:
        # Return current joint velocities (rad/s, length 7).
        self._check_connected()
        return np.asarray(self._robot.current_joint_state.velocity, dtype=float)

    def get_tcp_pose(self) -> np.ndarray:
        # Return current TCP pose [x, y, z, rx, ry, rz] (axis-angle).
        self._check_connected()
        pose = self._robot.current_pose.end_effector_pose
        t = np.asarray(pose.translation, dtype=float)
        q = np.asarray(pose.quaternion, dtype=float)
        rxyz = _quat_to_axis_angle(q)
        return np.concatenate([t, rxyz])

    def get_tcp_force(self) -> np.ndarray:
        # Return external wrench [fx, fy, fz, tx, ty, tz] at the TCP.
        self._check_connected()
        try:
            return np.asarray(self._robot.state.O_F_ext_hat_K, dtype=float)
        except AttributeError:
            return np.zeros(6)

    def get_robot_mode(self) -> int:
        # Return robot mode integer.
        self._check_connected()
        try:
            return int(self._robot.state.robot_mode)
        except Exception:
            return -1

    def is_steady(self) -> bool:
        # Return True when the robot has finished its current motion.
        self._check_connected()
        return bool(self._robot.poll_motion())

    @property
    def has_errors(self) -> bool:
        self._check_connected()
        return bool(self._robot.has_errors)

    def recover_from_errors(self) -> bool:
        self._check_connected()
        return bool(self._robot.recover_from_errors())

    # Motion commands

    def move_to_joint_config(
        self,
        q: Sequence[float],
        velocity: Optional[float] = None,
        acceleration: Optional[float] = None,
        asynchronous: bool = False,
    ):
        # Move to a 7-DOF joint configuration.
        self._check_connected()
        q_arr = list(q)
        if len(q_arr) != 7:
            raise ValueError(f"Franka expects 7 joints, got {len(q_arr)}.")

        self._preempt_velocity_if_active()
        self._preempt_servo_if_active()

        # Auto-recover if in error state
        try:
            if self._robot.has_errors:
                self._robot.recover_from_errors()
        except Exception as exc:
            logger.warning("auto recovery raised: %s", exc)

        # Stationarity gate: only command the JointMotion once residual velocity
        # from the previous motion has bled off, to avoid the velocity/
        # acceleration discontinuity reflex.
        self.gate_joint_stationary()

        try:
            if self._robot.has_errors:
                self._robot.recover_from_errors()
        except Exception:
            pass

        vel_rdf = self._velocity_to_rdf(
            velocity, self._max_joint_vel, self.DEFAULT_JOINT_VEL
        )
        rdf = RelativeDynamicsFactor(vel_rdf, vel_rdf * 0.5, vel_rdf * 0.25)
        motion = JointMotion(q_arr, relative_dynamics_factor=rdf)

        try:
            self._dispatch_motion(motion, asynchronous=asynchronous)
        except Exception as exc:
            msg = str(exc).lower()
            if any(
                k in msg
                for k in (
                    "motion finished commanded",
                    "joint_motion_generator_velocity_discontinuity",
                    "joint_motion_generator_acceleration_discontinuity",
                )
            ):
                logger.info("joint move residual velocity (%s); recovering", exc)
                try:
                    if self._robot.has_errors:
                        self._robot.recover_from_errors()
                except Exception:
                    pass
                self._wait_joints_stationary(max_wait=2.0, vel_tol=2e-3)
                return
            raise

    def gate_joint_stationary(
        self,
        vel_tol: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> float:
        """
        Fast pre-motion gate: ensure the arm is still before a joint move.

        Reads current joint velocities once. If max |dq| is already below
        vel_tol the call returns immediately (no sleep), the common case.
        Otherwise it polls at ~100 Hz until still or timeout elapses.

        Returns the wait duration in seconds (0.0 on the fast path).
        """
        tol = self.STATIONARY_DQ_TOL if vel_tol is None else vel_tol
        t_max = self.STATIONARY_GATE_TIMEOUT if timeout is None else timeout

        def _dq_max() -> Optional[float]:
            try:
                v = np.asarray(self._robot.current_joint_velocities, dtype=float)
            except Exception:
                try:
                    v = np.asarray(
                        self._robot.current_joint_state.velocity, dtype=float
                    )
                except Exception:
                    return None
            return float(np.max(np.abs(v)))

        dqm = _dq_max()
        if dqm is None or dqm < tol:
            return 0.0  # already still (or unreadable): single read, no sleep

        t0 = time.monotonic()
        deadline = t0 + t_max
        while time.monotonic() < deadline:
            time.sleep(0.01)
            dqm = _dq_max()
            if dqm is None or dqm < tol:
                break
        waited = time.monotonic() - t0
        logger.info(
            "joint stationarity gate waited %.0f ms (dq_max=%.4f rad/s, "
            "tol=%.3f) before joint move",
            waited * 1e3,
            dqm if dqm is not None else float("nan"),
            tol,
        )
        return waited

    def _wait_joints_stationary(self, max_wait: float = 1.0, vel_tol: float = 5e-3):
        # Block until joint velocities settle below vel_tol (slow fallback path).
        t_end = time.time() + max_wait
        while time.time() < t_end:
            try:
                v = np.asarray(self._robot.current_joint_state.velocity, dtype=float)
            except Exception:
                return
            if float(np.max(np.abs(v))) < vel_tol:
                return
            time.sleep(0.01)

    def _check_workspace(self, pose: Sequence[float]):
        # Verify a Cartesian target is inside workspace bounds.
        x, y, z = pose[0], pose[1], pose[2]
        if not (-0.9 <= x <= 0.9 and -0.9 <= y <= 0.9 and -0.05 <= z <= 1.2):
            raise ValueError(
                f"Target ({x:.3f}, {y:.3f}, {z:.3f}) outside workspace bounds."
            )

    def move_linear(
        self,
        pose: Sequence[float],
        velocity: Optional[float] = None,
        acceleration: Optional[float] = None,
        asynchronous: bool = False,
    ):
        # Move linearly to a Cartesian pose [x, y, z, rx, ry, rz].
        self._check_connected()
        self._check_workspace(pose)
        self._preempt_velocity_if_active()
        self._preempt_servo_if_active()

        x, y, z, rx, ry, rz = (float(v) for v in pose[:6])
        quat = _axis_angle_to_quat(rx, ry, rz)
        target = Affine([x, y, z], quat.tolist())

        vel_rdf = self._velocity_to_rdf(
            velocity, self._max_linear_vel, self.DEFAULT_VELOCITY
        )
        vel_rdf = min(vel_rdf, 0.35)
        rdf = RelativeDynamicsFactor(vel_rdf, vel_rdf * 0.5, vel_rdf * 0.25)

        try:
            self._robot.move(CartesianStopMotion(), asynchronous=False)
        except Exception:
            pass

        motion = CartesianMotion(target, relative_dynamics_factor=rdf)
        self._dispatch_motion(motion, asynchronous=asynchronous)

        # Wait for joints to physically stop after synchronous move
        if not asynchronous:
            t0 = time.monotonic()
            while time.monotonic() - t0 < 1.0:
                try:
                    dq = np.asarray(
                        self._robot.current_joint_state.velocity, dtype=float
                    )
                except Exception:
                    break
                if float(np.max(np.abs(dq))) < 0.005:
                    break
                time.sleep(0.02)

    def move_linear_relative(
        self,
        delta: Sequence[float],
        velocity: Optional[float] = None,
        acceleration: Optional[float] = None,
    ):
        # Move by [dx, dy, dz, drx, dry, drz] relative to the current TCP.
        self._check_connected()
        self._preempt_velocity_if_active()
        self._preempt_servo_if_active()
        dx, dy, dz, drx, dry, drz = (float(v) for v in delta[:6])
        quat = _axis_angle_to_quat(drx, dry, drz)
        delta_affine = Affine([dx, dy, dz], quat.tolist())
        vel_rdf = self._velocity_to_rdf(
            velocity, self._max_linear_vel, self.DEFAULT_VELOCITY
        )
        rdf = RelativeDynamicsFactor(vel_rdf, vel_rdf * 0.5, vel_rdf * 0.25)
        motion = CartesianMotion(
            delta_affine,
            reference_type=ReferenceType.Relative,
            relative_dynamics_factor=rdf,
        )
        self._dispatch_motion(motion)

    def stop(self):
        # Emergency stop.
        self._check_connected()
        with self._servo_lock:
            self._servo_active = False
            self._servo_last_target = None
        with self._velocity_lock:
            self._velocity_active = False
            self._velocity_last_target = None
        self._robot.stop()

    def go_home(self, velocity: Optional[float] = None):
        self.move_to_joint_config(self.HOME_CONFIG, velocity=velocity)

    def get_observation(self) -> dict:
        obs = {}
        for fn_name, key in (
            ("get_joint_positions", "joint_positions"),
            ("get_tcp_pose", "tcp_pose"),
        ):
            try:
                obs[key] = getattr(self, fn_name)()
            except Exception:
                pass
        return obs

    # Helpers

    def _dispatch_motion(self, motion, asynchronous=False):
        # Send a motion to franky, auto-clearing latched Reflex first.
        if self._robot.has_errors:
            try:
                self._robot.recover_from_errors()
                logger.info("auto-cleared reflex before motion")
            except Exception as exc:
                logger.warning("recover_from_errors failed: %s", exc)
        return self._robot.move(motion, asynchronous=asynchronous)

    def _velocity_to_rdf(
        self, commanded_vel: Optional[float], hard_limit: float, default_vel: float
    ) -> float:
        # Translate an absolute velocity into franky's relative dynamics factor.
        v = float(commanded_vel) if commanded_vel is not None else float(default_vel)
        if hard_limit <= 1e-6:
            return 1.0
        return float(np.clip(v / hard_limit, 1e-3, 1.0))

    # Context manager

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False
