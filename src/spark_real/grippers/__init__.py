"""
Gripper registry: the config `gripper.type` names a driver, on any robot family.

    robotiq_2f85  Robotiq 2F-85 over URScript (the host arm must be a UR controller)
    franka_hand   Franka Hand over franky (gripper.ip, default the host arm's FCI address)
    dynamixel     Dynamixel X-series jaw (gripper.port, gripper.id)
    ssg48         Source Robotics SSG-48 jaw over slcan (gripper.channel)
    package.module:ClassName
                  your own driver: built with the block's other keys as keyword
                  arguments; it must provide the jaw API below

Every non-native driver provides the same jaw API (connect, disconnect,
homing, open, close, move_to_width, grasp_to_width, get_present_width,
get_position_norm on the 0..255 closure scale, is_object_detected) and is
mounted on the arm driver with GripperMount, which presents the single-arm
gripper API the executor calls. The Robotiq is native to the UR10e driver
(its gripper program runs on the UR controller) and needs no mount.

make_gripper raises ValueError when the type has no driver and
RuntimeError when its transport cannot work on this host or arm.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

GRIPPER_TYPES = ("robotiq_2f85", "franka_hand", "dynamixel", "ssg48")


def _robotiq(cfg: Dict[str, Any], host) -> None:
    if not (getattr(host, "SUPPORTS_URSCRIPT", False) and hasattr(host, "close_gripper")):
        raise RuntimeError(
            "gripper.type 'robotiq_2f85' runs as a URScript program on a UR controller; "
            f"{type(host).__name__} has no URScript channel (no Modbus transport in this release)"
        )
    return None  # native to the host driver


def _franka_hand(cfg: Dict[str, Any], host):
    from spark_real.grippers.franka_hand import FrankaHandConfig, FrankaHandGripper

    keys = ("speed", "force", "epsilon_inner", "epsilon_outer")
    kw = {k: float(cfg[k]) for k in keys if cfg.get(k) is not None}
    if cfg.get("open_width") is not None:
        kw["width_open_m"] = float(cfg["open_width"])
    return FrankaHandGripper(FrankaHandConfig(ip=str(cfg.get("ip") or getattr(host, "robot_ip", "") or ""), **kw))


def _dynamixel(cfg: Dict[str, Any], host):
    from spark_real.grippers.dynamixel import DynamixelConfig, DynamixelGripper

    motor_id = cfg.get("id", cfg.get("motor_id"))
    if motor_id is None:
        raise ValueError("gripper.type 'dynamixel' needs the servo id (gripper.id)")
    return DynamixelGripper(
        DynamixelConfig(
            motor_id=int(motor_id),
            port=str(cfg.get("port", "/dev/ttyUSB0")),
            baudrate=int(cfg.get("baudrate", 1_000_000)),
        )
    )


def _ssg48(cfg: Dict[str, Any], host):
    from spark_real.grippers.ssg48 import SSG48Config, SSG48Gripper

    if not cfg.get("channel"):
        raise ValueError("gripper.type 'ssg48' needs the slcan device (gripper.channel)")
    fields = {
        "node_id": (int, 0), "bustype": (str, "slcan"), "bitrate": (int, 1_000_000),
        "default_speed": (int, 60), "default_force_ma": (int, 500), "max_force_ma": (int, 1300),
        "position_deadband": (int, 5), "auto_calibrate": (bool, False),
        "invert_position": (bool, False), "width_open_m": (float, 0.090), "width_closed_m": (float, 0.0),
    }
    kw = {k: cast(cfg.get(k, d)) for k, (cast, d) in fields.items()}
    return SSG48Gripper(SSG48Config(channel=str(cfg["channel"]), **kw))


_BUILDERS = {"robotiq_2f85": _robotiq, "franka_hand": _franka_hand, "dynamixel": _dynamixel, "ssg48": _ssg48}


# Block keys the mount consumes itself; a plugin class gets every other key.
_MOUNT_KEYS = ("type", "tcp_offset_m", "left", "right")


def _plugin(cfg: Dict[str, Any], host):
    import importlib

    module, _, name = str(cfg["type"]).partition(":")
    jaw = getattr(importlib.import_module(module), name)(**{k: v for k, v in cfg.items() if k not in _MOUNT_KEYS})
    if not hasattr(jaw, "GRIPPER_TYPE"):
        jaw.GRIPPER_TYPE = name
    return jaw


def gripper_type(cfg: Optional[Mapping[str, Any]]) -> str:
    kind = str((cfg or {}).get("type") or "")
    if ":" in kind:
        return kind
    kind = kind.lower()
    if kind not in _BUILDERS:
        raise ValueError(
            f"gripper.type {kind or '<missing>'!r} has no driver; registered: {GRIPPER_TYPES}, "
            "or a class as 'package.module:ClassName'"
        )
    return kind


def make_gripper(cfg: Optional[Mapping[str, Any]], host=None):
    """The configured gripper driver (None when it is native to ``host``). Opens nothing."""
    cfg = dict(cfg or {})
    kind = gripper_type(cfg)
    return _plugin(cfg, host) if ":" in kind else _BUILDERS[kind](cfg, host)


def arm_gripper_cfg(cfg: Mapping[str, Any], arm: str) -> Dict[str, Any]:
    """Shared keys of a two-jaw `grippers:` block with the `<arm>:` sub-block over them."""
    shared = {k: v for k, v in cfg.items() if k not in ("left", "right")}
    return {**shared, **dict(cfg.get(arm) or {})}


class GripperMount:
    """
    An arm driver with a registered gripper mounted on it.

    Everything except the gripper API falls through to the arm driver. With
    ``tcp_offset_m`` the TCP is the jaw tip, that far along flange +Z:
    get_tcp_pose / move_linear / get_observation convert between jaw tip and
    flange (rotations in move_linear_relative and send_velocity still pivot
    at the flange).
    """

    def __init__(self, driver, gripper, tcp_offset_m: float = 0.0, robot_family: Optional[str] = None):
        self._robot = driver
        self._jaw = gripper
        self.tcp_offset_m = float(tcp_offset_m)
        self.GRIPPER_TYPE = gripper.GRIPPER_TYPE
        self.robot_family = robot_family or getattr(driver, "robot_family", "")

    def __getattr__(self, name):
        return getattr(self._robot, name)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.disconnect()
        return False

    def connect(self):
        self._robot.connect()
        self._jaw.connect()

    def disconnect(self):
        try:
            self._jaw.disconnect()
        finally:
            self._robot.disconnect()

    def _shift(self, pose, sign: float) -> List[float]:
        pose = [float(v) for v in pose[:6]]
        off = Rotation.from_rotvec(pose[3:6]).apply([0.0, 0.0, sign * self.tcp_offset_m])
        return [pose[0] + off[0], pose[1] + off[1], pose[2] + off[2]] + pose[3:6]

    def get_tcp_pose(self):
        return np.asarray(self._shift(self._robot.get_tcp_pose(), +1.0))

    def move_linear(self, pose, *args, **kwargs):
        return self._robot.move_linear(self._shift(pose, -1.0), *args, **kwargs)

    def get_observation(self) -> Dict[str, Any]:
        obs = dict(self._robot.get_observation())
        obs["tcp_pose"] = self.get_tcp_pose()
        obs["gripper_position"] = self.get_gripper_position()
        return obs

    def _width_band(self) -> Tuple[float, float]:
        cfg = getattr(self._jaw, "config", None)
        return float(getattr(cfg, "width_open_m", 0.090)), float(getattr(cfg, "width_closed_m", 0.0))

    def activate_gripper(self) -> None:
        self._jaw.homing()

    def open_gripper(self, speed=None, force=None) -> None:
        self._jaw.open(speed=speed)

    def close_gripper(self, speed=None, force=None) -> bool:
        return self._jaw.close(force_n=force, speed=speed)

    def set_gripper_position(self, position: float, speed=None, force=None) -> None:
        position = max(0.0, min(1.0, float(position)))
        w_open, w_closed = self._width_band()
        if position >= 0.95:
            self._jaw.close(force_n=force, speed=speed)
        else:
            self._jaw.move_to_width(w_open - position * (w_open - w_closed), speed=speed)

    def grasp_to_width(self, width: float, force=None, speed=None, **_eps) -> bool:
        return self._jaw.grasp_to_width(width, force_n=force, speed=speed)

    def get_gripper_width(self) -> float:
        return self._jaw.get_present_width()

    def get_gripper_position(self) -> float:
        return self._jaw.get_position_norm()

    def is_object_detected(self) -> bool:
        return self._jaw.is_object_detected()


def mount_gripper(driver, cfg: Optional[Mapping[str, Any]], robot_family: Optional[str] = None):
    """``driver`` with the configured gripper: as is for a native gripper, else a GripperMount."""
    jaw = make_gripper(cfg, driver)
    if jaw is None:
        driver.GRIPPER_TYPE = gripper_type(cfg)
        return driver
    return GripperMount(driver, jaw, float((cfg or {}).get("tcp_offset_m", 0.0)), robot_family)


class DualGripper:
    """Two jaws, one per arm, for the bimanual driver's two-arm gripper calls."""

    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.GRIPPER_TYPE = left.GRIPPER_TYPE

    def connect(self) -> None:
        self.left.connect()  # sequential: serial enumeration order stays fixed
        self.right.connect()

    def disconnect(self) -> None:
        for g in (self.left, self.right):
            try:
                g.disconnect()
            except Exception:
                logger.exception("error disconnecting %s gripper", g.GRIPPER_TYPE)

    def homing(self) -> None:
        for g in (self.left, self.right):
            g.homing()

    def for_arm(self, arm: str):
        if arm not in ("left", "right"):
            raise ValueError(f"unknown arm {arm!r}; expected 'left' or 'right'")
        return getattr(self, arm)


def make_dual_gripper(cfg: Optional[Mapping[str, Any]], arm_ips: Optional[Mapping[str, str]] = None) -> DualGripper:
    """One registered jaw per arm from a `grippers:` block (no native grippers here).

    ``arm_ips`` is the default `ip` per arm (a Franka Hand talks to its arm's FCI).
    """
    cfg = dict(cfg or {})
    jaws = []
    for arm in ("left", "right"):
        arm_cfg = arm_gripper_cfg(cfg, arm)
        arm_cfg.setdefault("ip", (arm_ips or {}).get(arm, ""))
        jaw = make_gripper(arm_cfg)
        if jaw is None:
            raise RuntimeError(f"grippers.type {cfg.get('type')!r} cannot be an external jaw")
        jaws.append(jaw)
    return DualGripper(*jaws)
