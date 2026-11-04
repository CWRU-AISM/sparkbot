"""
UR10e real-time control via ur_rtde.

State reads and the motion lease live here; connection (ur10e_connection),
motion (ur10e_motion) and the Robotiq 2F-85 (robotiq) are mixins. State
goes through ur_rtde; raw URScript (moves, speedl/speedj for the
CartesianServo, gripper programs) goes through the primary-interface TCP
socket on port 30002.
"""

import logging
import re
import socket
import threading
import time
from typing import Optional, Sequence

import numpy as np

from spark_real.control.command_latch import CommandLatch, CommandSnapshot

logger = logging.getLogger(__name__)

# Commanded-speed parsers, used by _note_motion_script to size the brake.
# A blended program carries one v= per row; the leading row wins.
_MOVE_V_RE = re.compile(r"move[jl]\([^)]*?\bv=([-\d.eE+]+)")
_SPEED_VEC_RE = re.compile(r"speed([jl])\(\[([^\]]*)\]")

try:
    import rtde_control
    import rtde_io
    import rtde_receive

    HAS_RTDE = True
except ImportError:
    HAS_RTDE = False
    # Franka hosts import this module too; UR10eDriver raises at construction if ur-rtde is missing.
    logger.debug("ur-rtde not installed (only the UR10e driver needs it: pip install -e \".[ur10e]\")")

from spark_real.control.robotiq import RobotiqMixin  # noqa: F401
from spark_real.control.ur10e_connection import URConnectionMixin  # noqa: F401
from spark_real.control.ur10e_motion import URMotionMixin  # noqa: F401


class UR10eDriver(URConnectionMixin, URMotionMixin, RobotiqMixin):
    """
    Direct control interface for UR10e robot via RTDE protocol.
    """

    # Family + capability flags consumed by the family-agnostic stack
    # (CartesianServo, executor_ik, executor_grasp). SUPPORTS_URSCRIPT
    # routes CartesianServo through speedl via _send_script; GRIPPER_TYPE
    # selects the Robotiq fast path in executor_grasp.
    robot_family = "ur10e"
    SUPPORTS_URSCRIPT = True
    SUPPORTS_VELOCITY_STREAMING = True
    GRIPPER_TYPE = "robotiq_2f85"

    # UR10e joint limits (radians)
    JOINT_LIMITS = [(-2 * np.pi, 2 * np.pi)] * 6

    # Default motion parameters
    DEFAULT_VELOCITY = 0.25  # m/s for linear moves
    DEFAULT_ACCELERATION = 0.5  # m/s^2
    DEFAULT_JOINT_VEL = 1.05  # rad/s
    DEFAULT_JOINT_ACC = 1.4  # rad/s^2

    # Robotiq 2F-85 gripper parameters
    GRIPPER_OPEN = 0.0
    GRIPPER_CLOSED = 255.0
    GRIPPER_SPEED = 255
    GRIPPER_FORCE = 50

    # Robotiq 2F-85 max jaw opening (meters). Used to map a target grasp
    # width in meters to the 0-100 percent-closed URScript command.
    GRIPPER_MAX_WIDTH_M = 0.085

    # Primary-interface port for raw URScript (speedl/speedj/movej).
    URSCRIPT_PORT = 30002
    # Brake. The stop that works on this rig is a URScript stopj/stopl sent to
    # URSCRIPT_PORT: a program sent there REPLACES the running program, so it
    # brakes a fire-and-forget movej. rtde_c.stopJ cannot (see stop()).
    # The decel is derived from the LAST COMMANDED speed, never hardcoded:
    # stopping distance is v**2/(2a), so holding the overshoot at a budget
    # means a = v**2/(2*budget). A raised velocity cap therefore raises the
    # brake with it instead of silently lengthening the stopping distance.
    BRAKE_OVERSHOOT_BUDGET_RAD = 0.15  # leading joint, rad
    BRAKE_DECEL_MIN = 2.0              # rad/s^2, the historical stopj(2.0)
    BRAKE_DECEL_MAX = 8.0              # above this UR trips C153/C173
    BRAKE_OVERSHOOT_BUDGET_M = 0.05    # TCP, m
    BRAKE_DECEL_LIN_MIN = 1.2          # m/s^2
    BRAKE_DECEL_LIN_MAX = 5.0
    # Bounded wait for the Dashboard escalation. DashboardClient.connect() can
    # block on an unreachable/locked controller, and a stop must not hang.
    DASHBOARD_TIMEOUT_S = 2.0

    # Common configurations
    # Home: TCP xyz=(-0.8144, 0.1089, 0.1304) rot=(-2.2836, -2.1399, 0.0231).
    # A home 170 deg off at wrist 3 rolls the wrist camera a half turn.
    HOME_CONFIG = [3.2551, -1.7610, -1.9082, 5.2296, 1.5712, -2.8697]

    def __init__(
        self,
        robot_ip: str,
        frequency: float = 500.0,
        gripper_script: Optional[str] = None,
    ):
        """
        Args:
            robot_ip: IP address of the UR10e controller
            frequency: RTDE communication frequency (Hz)
            gripper_script: Robotiq URScript export (see _load_gripper_functions)
        """
        if not HAS_RTDE:
            raise RuntimeError("ur-rtde not installed. pip install ur-rtde")

        self.robot_ip = robot_ip
        self.frequency = frequency
        self._rtde_c: Optional[rtde_control.RTDEControlInterface] = None
        self._rtde_r: Optional[rtde_receive.RTDEReceiveInterface] = None
        self._rtde_io: Optional[rtde_io.RTDEIOInterface] = None
        self._urscript_socket: Optional[socket.socket] = None
        self._connected = False
        # Serializes writes on the single URScript socket. _send_script is
        # called concurrently from the executor thread, the primitive-timeout
        # watchdog's abort -> brake, and teleop /api/velocity; interleaved
        # partial writes on one socket corrupt programs.
        self._script_lock = threading.Lock()
        # True after a send failed even through the reconnect attempt.
        # Surfaced so callers/status can tell "command dropped" from "sent".
        self._urscript_down = False
        self._gripper_script_path = gripper_script
        self._gripper_script_header = None
        self._gripper_initialized = False
        # Last commanded twist / gripper target, for the demo recorder's
        # action label. Stamped by send_velocity + the gripper entry points;
        # read (never written) by recording/action_source.py.
        self._command_latch = CommandLatch()
        # Monotonic deadline until which a motion URScript may still be
        # starting or stopping. See _note_motion_script / _motion_in_flight.
        self._motion_lease_until = -1.0
        # Last commanded speed and its space, for sizing the brake. See brake().
        self._last_motion_kind: Optional[str] = None  # "joint" | "linear"
        self._last_joint_vel = 0.0   # rad/s, leading joint
        self._last_linear_vel = 0.0  # m/s, TCP

    @property
    def connected(self) -> bool:
        return self._connected

    # Motion lease (see _publish_gripper_state for why this exists)

    # A speedl is only valid for its own duration; the servo re-sends every
    # ~8 ms, so a short lease self-expires within one tick of the loop ending.
    _SPEEDL_LEASE_S = 0.15
    # movej/movel are fire-and-forget with a ~1 s upload/start lag during which
    # the arm is stationary, so the measured-speed guard cannot see them.
    _MOVEJ_LEASE_S = 1.5

    def _note_motion_script(self, script: str) -> None:
        """Extend the motion lease when ``script`` commands arm motion.

        Also records the commanded speed and its space, which is what brake()
        sizes its deceleration from.
        """
        if "speedl(" in script or "speedj(" in script or "servoj(" in script:
            lease = self._SPEEDL_LEASE_S
        elif "movej(" in script or "movel(" in script:
            lease = self._MOVEJ_LEASE_S
        else:
            return
        self._note_commanded_speed(script)
        self._motion_lease_until = max(
            self._motion_lease_until, time.monotonic() + lease
        )

    def _note_commanded_speed(self, script: str) -> None:
        """Remember how fast, and in which space, the arm was last told to go."""
        for m in _SPEED_VEC_RE.finditer(script):
            try:
                vec = [float(x) for x in m.group(2).split(",")]
            except ValueError:
                continue
            if m.group(1) == "l":
                self._last_motion_kind = "linear"
                self._last_linear_vel = float(np.linalg.norm(vec[:3]))
            else:
                self._last_motion_kind = "joint"
                self._last_joint_vel = max(abs(v) for v in vec) if vec else 0.0
        vels = [float(v) for v in _MOVE_V_RE.findall(script)]
        if vels:
            # movej v is rad/s on the leading joint; movel v is m/s at the TCP.
            if "movel(" in script:
                self._last_motion_kind = "linear"
                self._last_linear_vel = max(vels)
            else:
                self._last_motion_kind = "joint"
                self._last_joint_vel = max(vels)

    def clear_motion_lease(self) -> None:
        """
        Positive confirmation that the last commanded motion has landed.

        The lease is a pessimistic backstop sized for the worst-case upload
        lag, so a move that arrives sooner would otherwise keep suppressing
        gripper publishes and hand the next stationary read a stale register.
        Callers that can prove arrival (executor_motion._wait_for_motion, on a
        joint-target match or Cartesian proximity) retire it early.
        """
        self._motion_lease_until = -1.0

    def _motion_in_flight(self) -> bool:
        """
        True while an arm motion may still be starting, running or stopping.

        Two independent signals, OR-ed, because neither alone is sufficient:
        the measured TCP speed misses the URScript start lag and the final
        millimetres of a servo convergence, while the lease misses a movej
        that outlives its lease.
        """
        return time.monotonic() < self._motion_lease_until or self._tcp_is_moving()

    def _tcp_is_moving(self) -> bool:
        """Measured TCP speed above the standstill threshold."""
        try:
            if self._rtde_r is not None:
                spd = np.asarray(self._rtde_r.getActualTCPSpeed(), dtype=float)
                if spd.size >= 3 and float(np.linalg.norm(spd[:3])) > 0.01:
                    return True
        except Exception:  # noqa: BLE001 - a read failure must not block motion
            pass
        return False

    # Command latch (see control/command_latch.py)

    def latch_command_velocity(
        self, velocity: Sequence[float], source: Optional[str] = None
    ) -> None:
        """
        Stamp a commanded base-frame twist without sending anything.

        Exists for the one caller that formats its own ``speedl`` URScript
        (``CartesianServo`` on the URScript branch) so the latch still has a
        single implementation even though there are two emit paths.
        """
        self._command_latch.latch_velocity(velocity, source)

    def get_last_command(self) -> CommandSnapshot:
        """Most recent commanded velocity / gripper, with monotonic stamps."""
        return self._command_latch.snapshot()


    #State queries

    def get_joint_positions(self) -> np.ndarray:
        """
        Get current joint positions (radians).
        """
        self._check_connected()
        return np.array(self._rtde_r.getActualQ())

    def get_joint_velocities(self) -> np.ndarray:
        """
        Get current joint velocities (rad/s).
        """
        self._check_connected()
        return np.array(self._rtde_r.getActualQd())

    def get_tcp_pose(self) -> np.ndarray:
        """
        Get current TCP pose [x, y, z, rx, ry, rz] (axis-angle).
        """
        self._check_connected()
        return np.array(self._rtde_r.getActualTCPPose())

    def get_tcp_offset(self) -> np.ndarray:
        """
        Pendant-configured TCP offset [x, y, z, rx, ry, rz] (gripper tip
        relative to tool0, tool0 frame). Static, so cached after first read.
        Used by the pyroki IK path (ur10e_ik_pyroki.solve_ik_rtde) to convert
        a gripper-TIP target into the tool0 frame pyroki solves for.
        """
        off = getattr(self, "_tcp_offset_cache", None)
        if off is None:
            self._check_connected()
            off = np.array(self._rtde_c.getTCPOffset(), dtype=float)
            self._tcp_offset_cache = off
        return off

    def get_tcp_force(self) -> np.ndarray:
        """
        Get TCP force/torque [fx, fy, fz, tx, ty, tz].
        """
        self._check_connected()
        return np.array(self._rtde_r.getActualTCPForce())

    def get_robot_mode(self) -> int:
        """
        Get robot mode (7=running, 5=idle, etc.).
        """
        self._check_connected()
        return self._rtde_r.getRobotMode()

    def is_steady(self) -> bool:
        """
        Check if robot has stopped moving.
        """
        self._check_connected()
        return self._rtde_r.isSteady()

    SAFETY_MODE_NAMES = {
        1: "NORMAL", 2: "REDUCED", 3: "PROTECTIVE_STOP", 4: "RECOVERY",
        5: "SAFEGUARD_STOP", 6: "SYSTEM_EMERGENCY_STOP",
        7: "ROBOT_EMERGENCY_STOP", 8: "VIOLATION", 9: "FAULT",
    }


    def get_observation(self) -> dict:
        """
        Proprioceptive observation dict: joint positions/velocities, TCP pose
        (axis-angle 6-vec), and gripper position (0-255, 0=open).
        """
        self._check_connected()
        obs = {
            "joint_positions": self.get_joint_positions(),
            "joint_velocities": self.get_joint_velocities(),
            "tcp_pose": np.array(self._rtde_r.getActualTCPPose()),
            "tcp_velocity": np.array(self._rtde_r.getActualTCPSpeed()),
        }
        # PASSIVE read. get_observation is polled by the demo recorder thread
        # at record_hz WHILE the arm is moving, and a publishing read uploads a
        # URScript program that cancels the in-flight move. The recorder's own
        # docstring already says this channel is diagnostic and that the
        # training label comes from the command latch, so a register that is at
        # worst one gripper actuation stale costs nothing and a cancelled
        # trajectory costs the episode.
        try:
            obs["gripper_position"] = self.get_gripper_position(publish=False)
        except Exception:
            obs["gripper_position"] = 0.0
        return obs


    # Robotiq 2F-85: the rq_* URScript functions come from the Robotiq URCap.
    # RTDE output integer registers used to surface gripper state. This
    # ur_rtde build only exposes getOutputIntRegister for indices 12-19; the
    # general-purpose pair below is read back spin-free (see _publish_gripper_state).
    _POS_REGISTER = 12
    _OBJ_REGISTER = 13
    # Written LAST by _pub_grip as a completion token: when it reads back, the
    # controller has already written 12 and 13, so the publish is done
    # (typically ~10-30 ms).
    _SEQ_REGISTER = 14
    _PUBLISH_POLL_S = 0.005
    # A build that does not surface register 14 waits the full timeout.
    _PUBLISH_TIMEOUT_S = 0.2


    #Context manager

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False
