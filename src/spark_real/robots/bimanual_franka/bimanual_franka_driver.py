"""
Bimanual Franka driver: two per-arm Franka drivers plus external jaws.

* Its own family ("bimanual_franka"), composed from per-arm drivers rather
  than inheriting from FrankaDriver.
* Every command takes ``arm="left"|"right"``; aggregate reads (joint
  positions) concatenate left || right, and commands with no arm refuse to
  run, except stop, homing and the gripper calls, which act on both arms.
* Arm motion backend from `robot.backend`: "franky" (FrankaDriver, no Franka
  Hand client) or "bamboo" (FrankaBambooDriver, one joint-impedance node per
  arm on its own ZMQ port).
* Jaws from the gripper registry (`grippers.type`, spark_real.grippers),
  decoupled from the FCI connections.
* ``for_arm(arm)`` returns a GripperMount: the per-arm driver with that
  arm's jaw behind the single-arm gripper API and the jaw tip as its TCP
  (`grippers.tcp_offset_m` along flange +Z), so a per-arm ScoreExecutor's
  grasp drives the real jaw and its Cartesian targets are jaw-tip targets,
  the frame calibration.bimanual_anchor_tap solves in. get_observation
  reports both the flange (tcp_pose) and the jaw tip (jaw_tip_pose).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from spark_real.grippers import GripperMount

logger = logging.getLogger(__name__)

ARMS: Tuple[str, str] = ("left", "right")
BACKENDS = ("franky", "bamboo")


@dataclass
class BimanualObservation:
    """Both arms and jaws at one timestamp; per-arm dicts match single-arm."""

    ts: float
    left: Dict[str, Any] = field(default_factory=dict)
    right: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Dict[str, Any]:
        if key in ("left", "right", "ts"):
            return getattr(self, key)
        raise KeyError(key)

    def to_dict(self) -> Dict[str, Any]:
        return {"ts": self.ts, "left": self.left, "right": self.right}


class BimanualFrankaDriver:
    """
    Two-arm driver with an ``arm`` kwarg on the UR10eDriver surface.

    Construction opens nothing; call connect().
    """

    SUPPORTS_VELOCITY_STREAMING = True
    SUPPORTS_URSCRIPT = False
    robot_family = "bimanual_franka"

    # Aggregate joint vector layout: left 7 || right 7 = 14
    DOF_PER_ARM = 7
    DOF_TOTAL = 14

    # Per-arm ZMQ control ports for the bamboo backend.
    BAMBOO_PORTS = {"left": 5555, "right": 5556}

    def __init__(
        self,
        left_ip: str,
        right_ip: str,
        dual_gripper,
        backend: str = "franky",
        frequency: float = 1000.0,
        home_left: Optional[Sequence[float]] = None,
        home_right: Optional[Sequence[float]] = None,
        collision: Optional[dict] = None,
        jaw_offset_m: float = 0.0,
    ):
        backend = str(backend or "franky").lower()
        if backend not in BACKENDS:
            raise ValueError(f"robot.backend {backend!r} is not one of {BACKENDS}")
        if not left_ip or not right_ip:
            raise ValueError("robot.left_ip and robot.right_ip must both be set")
        self.left_ip = left_ip
        self.right_ip = right_ip
        self.backend = backend
        self.frequency = frequency
        self.grippers = dual_gripper
        self.GRIPPER_TYPE = dual_gripper.GRIPPER_TYPE
        self.jaw_offset_m = float(jaw_offset_m)
        self._homes = {"left": home_left, "right": home_right}
        self._collision = dict(collision or {})
        self._drivers: Dict[str, Any] = {"left": None, "right": None}
        self._views: Dict[str, GripperMount] = {}
        # Per-arm locks so parallel BT branches do not interleave commands
        # on one arm's control session.
        self._arm_locks = {arm: threading.Lock() for arm in ARMS}
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def _left_driver(self):
        return self._drivers["left"]

    @property
    def _right_driver(self):
        return self._drivers["right"]

    @property
    def left(self) -> GripperMount:
        return self.for_arm("left")

    @property
    def right(self) -> GripperMount:
        return self.for_arm("right")

    def for_arm(self, arm: str) -> GripperMount:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
        if arm not in self._views:
            if self._drivers[arm] is None:
                raise RuntimeError(f"{arm} arm is not connected")
            self._views[arm] = GripperMount(
                self._drivers[arm], self.grippers.for_arm(arm), self.jaw_offset_m, "bimanual_franka"
            )
        return self._views[arm]

    def _make_arm_driver(self, arm: str):
        ip = self.left_ip if arm == "left" else self.right_ip
        if self.backend == "bamboo":
            from spark_real.robots.franka.franka_bamboo_driver import FrankaBambooDriver

            drv = FrankaBambooDriver(ip=ip, port=self.BAMBOO_PORTS[arm])
            # Picks the PyRoki chain when panda_py's analytic IK fails.
            drv._arm_hint = arm
        else:
            from spark_real.robots.franka.franka_driver import FrankaDriver

            drv = FrankaDriver(ip, self.frequency, collision=self._collision)
        if self._homes[arm] is not None:
            drv.HOME_CONFIG = [float(v) for v in self._homes[arm]]
        return drv

    def connect(self) -> None:
        """
        Open both arm sessions in parallel, then the gripper bus. If one arm
        fails the other stays up (bring-up debugging); both failing raises.
        """
        if self._connected:
            return
        errors: Dict[str, Exception] = {}

        def _open(arm: str) -> None:
            try:
                drv = self._make_arm_driver(arm)
                drv.connect()
                self._drivers[arm] = drv
            except Exception as exc:
                errors[arm] = exc

        threads: List[threading.Thread] = [
            threading.Thread(target=_open, args=(arm,), daemon=True, name=f"franka-{arm}-connect")
            for arm in ARMS
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for arm, exc in errors.items():
            logger.error("failed to connect %s arm: %s", arm, exc)
        if len(errors) == len(ARMS):
            raise errors["left"]

        # Gripper bus after the arms, so any USB renumbering from the arm
        # libraries happens first.
        self.grippers.connect()
        try:
            self.grippers.homing()
        except Exception:
            logger.exception("gripper homing failed; continuing")
        self._connected = True

    def disconnect(self) -> None:
        if not self._connected:
            return
        for drv in self._drivers.values():
            if drv is None:
                continue
            try:
                drv.disconnect()
            except Exception:
                logger.exception("driver disconnect failed")
        try:
            self.grippers.disconnect()
        except Exception:
            logger.exception("gripper disconnect failed")
        self._connected = False

    def reconnect_arm(self, arm: str) -> bool:
        """
        Tear down and re-open the FCI session for one arm only.

        Use this after rebooting a robot or re-enabling FCI on its Desk
        without bouncing the whole spark_real server.  Leaves the other
        arm's session, gripper bus, and state publisher untouched.

        Returns True if the arm came back up, False otherwise.  Either
        way, ``self._connected`` stays True so the rest of the server
        keeps running; check ``self.for_arm(arm)._connected`` for the
        per-arm state.
        """
        if arm not in ARMS:
            raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
        drv = self._drivers[arm]
        if drv is None:
            return False
        try:
            drv.disconnect()
        except Exception:
            logger.exception("reconnect_arm(%s): disconnect failed", arm)
        try:
            drv.connect()
        except Exception as exc:
            logger.error("reconnect_arm(%s) failed: %s", arm, exc)
            return False
        logger.info("reconnect_arm(%s): FCI session reopened", arm)
        return True

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

    # arm resolver
    def _resolve(self, arm: Optional[str]):
        if arm is None:
            raise ValueError(
                "bimanual driver requires explicit arm='left' or arm='right' "
                "for per-arm commands"
            )
        return self.for_arm(arm)

    # state readers
    def get_joint_positions(self, arm: Optional[str] = None) -> np.ndarray:
        if arm is not None:
            return self._resolve(arm).get_joint_positions()
        return np.concatenate(
            [
                self._left_driver.get_joint_positions(),
                self._right_driver.get_joint_positions(),
            ]
        )

    def get_joint_velocities(self, arm: Optional[str] = None) -> np.ndarray:
        if arm is not None:
            return self._resolve(arm).get_joint_velocities()
        return np.concatenate(
            [
                self._left_driver.get_joint_velocities(),
                self._right_driver.get_joint_velocities(),
            ]
        )

    def get_tcp_pose(self, arm: Optional[str] = None) -> np.ndarray:
        """
        Return single-arm TCP pose; ``arm`` is REQUIRED.

        Aggregate TCP makes no kinematic sense, so we refuse rather than
        invent a synthetic centroid.
        """
        return self._resolve(arm).get_tcp_pose()

    def get_tcp_force(self, arm: Optional[str] = None) -> np.ndarray:
        return self._resolve(arm).get_tcp_force()

    def get_robot_mode(self, arm: Optional[str] = None):
        if arm is not None:
            return self._resolve(arm).get_robot_mode()
        return {a: self.for_arm(a).get_robot_mode() for a in ARMS}

    def is_steady(self, arm: Optional[str] = None) -> bool:
        if arm is not None:
            return self._resolve(arm).is_steady()
        return all(self.for_arm(a).is_steady() for a in ARMS)

    @property
    def has_errors(self) -> bool:
        return any(bool(getattr(d, "has_errors", False)) for d in self._drivers.values() if d is not None)

    def recover_from_errors(self, arm: Optional[str] = None) -> bool:
        if arm is not None:
            return self._resolve(arm).recover_from_errors()
        ok = True
        for a in ARMS:
            try:
                ok = self.for_arm(a).recover_from_errors() and ok
            except Exception:
                logger.exception("recover_from_errors failed for arm=%s", a)
                ok = False
        return ok

    def get_observation(self, arm: Optional[str] = None) -> Any:
        """Per-arm dict (single-arm schema plus jaw_tip_pose), or both arms."""
        if arm is not None:
            view = self._resolve(arm)
            drv = self._drivers[arm]
            return {
                "joint_positions": drv.get_joint_positions(),
                "tcp_pose": drv.get_tcp_pose(),  # flange
                "jaw_tip_pose": view.get_tcp_pose(),
                "tcp_force": drv.get_tcp_force(),
                "gripper_width": self.grippers.for_arm(arm).get_present_width(),
                "gripper_position": self.grippers.for_arm(arm).get_position_norm(),
            }
        return BimanualObservation(
            ts=time.time(),
            left=self.get_observation("left"),
            right=self.get_observation("right"),
        )

    # motion commands (all require an arm)
    def move_to_joint_config(
        self,
        q,
        arm: Optional[str] = None,
        velocity: Optional[float] = None,
        acceleration: Optional[float] = None,
        asynchronous: bool = False,
    ):
        """
        Move one arm to a 7-vector OR both arms to a 14-vector.

        Bimanual aggregate path runs the two arms in parallel via
        ``asynchronous=True`` on each franky call, then joins.
        """
        q_arr = np.asarray(q, dtype=float).ravel()
        if arm is not None:
            with self._arm_locks[arm]:
                return self._resolve(arm).move_to_joint_config(
                    q_arr,
                    velocity=velocity,
                    acceleration=acceleration,
                    asynchronous=asynchronous,
                )
        if q_arr.size != self.DOF_TOTAL:
            raise ValueError(
                f"aggregate move_to_joint_config expected a {self.DOF_TOTAL}-vector "
                f"(left||right); got {q_arr.size}"
            )
        left_q = q_arr[: self.DOF_PER_ARM]
        right_q = q_arr[self.DOF_PER_ARM :]
        results: Dict[str, Any] = {}

        def _go(name: str, drv, target):
            with self._arm_locks[name]:
                results[name] = drv.move_to_joint_config(
                    target,
                    velocity=velocity,
                    acceleration=acceleration,
                    asynchronous=False,
                )

        threads = [
            threading.Thread(
                target=_go,
                args=("left", self._left_driver, left_q),
                name="left-moveJ",
                daemon=True,
            ),
            threading.Thread(
                target=_go,
                args=("right", self._right_driver, right_q),
                name="right-moveJ",
                daemon=True,
            ),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results

    def move_linear(self, pose, arm: Optional[str] = None, **kwargs):
        with self._arm_locks[arm or "right"]:
            return self._resolve(arm).move_linear(pose, **kwargs)

    def move_linear_relative(self, delta, arm: Optional[str] = None, **kwargs):
        with self._arm_locks[arm or "right"]:
            return self._resolve(arm).move_linear_relative(delta, **kwargs)

    def servo_joint(self, q, arm: Optional[str] = None, **kwargs):
        with self._arm_locks[arm or "right"]:
            return self._resolve(arm).servo_joint(q, **kwargs)

    def servo_stop(self, arm: Optional[str] = None) -> None:
        if arm is not None:
            self._resolve(arm).servo_stop()
            return
        for a in ARMS:
            try:
                self.for_arm(a).servo_stop()
            except Exception:
                logger.exception("servo_stop failed for arm=%s", a)

    def send_velocity(
        self,
        linear,
        angular=None,
        arm: Optional[str] = None,
        acceleration: float = 0.5,
        duration: float = 0.0,
    ):
        with self._arm_locks[arm or "right"]:
            return self._resolve(arm).send_velocity(
                linear, angular=angular, acceleration=acceleration, duration=duration
            )

    def stop_velocity(self, arm: Optional[str] = None) -> None:
        if arm is not None:
            self._resolve(arm).stop_velocity()
            return
        for a in ARMS:
            try:
                self.for_arm(a).stop_velocity()
            except Exception:
                logger.exception("stop_velocity failed for arm=%s", a)

    def stop(self, arm: Optional[str] = None) -> None:
        # Hard stop both arms (default) or one. Always brakes the gripper too.
        if arm is not None:
            self._resolve(arm).stop()
            return
        for a in ARMS:
            try:
                self.for_arm(a).stop()
            except Exception:
                logger.exception("stop failed for arm=%s", a)

    def go_home(
        self, velocity: Optional[float] = None, arm: Optional[str] = None
    ) -> None:
        """
        Home one or both arms. Both-arm path sequences left then right so the
        elbows do not interfere as they swing through the home config.
        """
        if arm is not None:
            self._resolve(arm).go_home(velocity=velocity)
            return
        # Sequential to avoid mid-air collision (the home configs are
        # mirror-symmetric and the elbows pass close to the midline).
        for a in ARMS:
            try:
                self.for_arm(a).go_home(velocity=velocity)
            except Exception:
                logger.exception("go_home failed for arm=%s", a)

    # gripper passthrough
    def activate_gripper(self, arm: Optional[str] = None) -> None:
        if arm is not None:
            self.grippers.for_arm(arm).homing()
            return
        self.grippers.homing()

    def open_gripper(self, arm: Optional[str] = None, speed=None, force=None) -> None:
        if arm is not None:
            self.grippers.for_arm(arm).open(speed=speed)
            return
        for a in ARMS:
            self.grippers.for_arm(a).open(speed=speed)

    def close_gripper(self, arm: Optional[str] = None, speed=None, force=None) -> bool:
        if arm is not None:
            return self.grippers.for_arm(arm).close(force_n=force, speed=speed)
        ok = True
        for a in ARMS:
            ok = self.grippers.for_arm(a).close(force_n=force, speed=speed) and ok
        return ok

    def set_gripper_position(
        self, position: float, arm: Optional[str] = None, speed=None, force=None
    ) -> None:
        # ``position`` in 0..1 (0 = open, 1 = closed).
        for a in (arm,) if arm is not None else ARMS:
            self.for_arm(a).set_gripper_position(position, speed=speed, force=force)

    def grasp_to_width(
        self, width: float, arm: Optional[str] = None, force=None, speed=None
    ) -> bool:
        if arm is None:
            raise ValueError("grasp_to_width requires arm='left' or 'right'")
        return self.grippers.for_arm(arm).grasp_to_width(
            width, force_n=force, speed=speed
        )

    def get_gripper_width(self, arm: Optional[str] = None) -> float:
        if arm is None:
            raise ValueError("get_gripper_width requires arm='left' or 'right'")
        return self.grippers.for_arm(arm).get_present_width()

    def get_gripper_position(self, arm: Optional[str] = None) -> float:
        if arm is None:
            raise ValueError("get_gripper_position requires arm='left' or 'right'")
        return self.grippers.for_arm(arm).get_position_norm()

    def is_object_detected(self, arm: Optional[str] = None) -> bool:
        if arm is None:
            return any(self.grippers.for_arm(a).is_object_detected() for a in ARMS)
        return self.grippers.for_arm(arm).is_object_detected()
