"""
Franka Hand through franky's Gripper client (`gripper.type: franka_hand`).

Transport: the Franka Control Interface of the arm the Hand is mounted on
(`gripper.ip`, default the host arm's address). Speeds in m/s, forces in N.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from franky import Gripper as _FrankyGripper
except ImportError:
    _FrankyGripper = None


@dataclass
class FrankaHandConfig:
    ip: str
    speed: float = 0.10
    force: float = 70.0
    epsilon_inner: Optional[float] = None  # None: the full jaw width
    epsilon_outer: Optional[float] = None
    width_open_m: float = 0.08  # used only when the Hand does not report max_width
    width_closed_m: float = 0.0


class FrankaHandGripper:
    """The jaw API every registered gripper provides (see grippers/__init__.py)."""

    GRIPPER_TYPE = "franka_hand"

    def __init__(self, config: FrankaHandConfig):
        if _FrankyGripper is None:
            raise RuntimeError(
                "gripper.type 'franka_hand' needs franky (pip install franky-control); "
                "not installed on this host"
            )
        if not config.ip:
            raise ValueError("gripper.type 'franka_hand' needs the arm's FCI address (gripper.ip)")
        self.config = config
        self._hand = None
        self._holding = False

    def connect(self) -> bool:
        self._hand = _FrankyGripper(self.config.ip)
        try:
            self.config.width_open_m = float(self._hand.max_width)
        except Exception:
            pass
        return True

    def disconnect(self) -> None:
        self._hand = None

    def _g(self):
        if self._hand is None:
            raise RuntimeError("Franka Hand not connected; call connect() first")
        return self._hand

    def homing(self) -> None:
        ok = self._g().homing()
        time.sleep(0.5)
        if not ok:
            logger.warning("Franka Hand homing did not report success")

    def open(self, speed: Optional[float] = None) -> None:
        self._holding = False
        self._g().open(float(speed if speed is not None else self.config.speed))

    def move_to_width(self, width_m: float, speed: Optional[float] = None) -> None:
        self._holding = False
        self._g().move_async(float(width_m), float(speed if speed is not None else self.config.speed))

    def grasp_to_width(self, width_m: float, force_n: Optional[float] = None, speed: Optional[float] = None) -> bool:
        hand = self._g()
        w_max = self.config.width_open_m
        eps_in = self.config.epsilon_inner if self.config.epsilon_inner is not None else w_max
        eps_out = self.config.epsilon_outer if self.config.epsilon_outer is not None else w_max
        args = (
            float(width_m),
            float(speed if speed is not None else self.config.speed),
            float(force_n if force_n is not None else self.config.force),
            float(eps_in),
            float(eps_out),
        )
        try:
            hand.stop()
        except Exception:
            pass
        self._holding = True
        for attempt in range(2):
            try:
                return bool(hand.grasp(*args))
            except Exception as exc:
                if attempt:
                    raise
                logger.warning("grasp() raised (%s); stop and retry", exc)
                try:
                    hand.stop()
                except Exception:
                    pass
                time.sleep(0.25)
        return False

    def close(self, force_n: Optional[float] = None, speed: Optional[float] = None) -> bool:
        return self.grasp_to_width(self.config.width_closed_m, force_n=force_n, speed=speed)

    def get_present_width(self) -> float:
        return float(self._g().width)

    def get_position_norm(self) -> float:
        span = max(1e-6, self.config.width_open_m - self.config.width_closed_m)
        frac = 1.0 - (self.get_present_width() - self.config.width_closed_m) / span
        return max(0.0, min(255.0, frac * 255.0))

    def is_object_detected(self) -> bool:
        return self._holding and bool(self._g().is_grasped)
