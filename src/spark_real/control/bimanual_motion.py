"""
Bimanual motion layer: the motion path the bimanual hardware runs used.

Ported from the rig's validated fold and silverware procedures: each target
is solved with PyRoki (BimanualPyrokiPlanner, per-arm Panda/FR3 chains) for
the flange link and sent as a joint move to that arm's driver (bamboo or
franky); two-arm phases solve both targets first and move both arms on
threads; the jaws are the registered grippers of the bimanual driver.

Frames: a target for ``arm`` is in that arm's base frame. Scene points
arrive in the right arm's frame (the external camera's calibration) and
``to_arm`` maps them through T_right_to_left from calibration_bimanual.json.
Targets are jaw-tip points; ``flange_z_offset_m`` (default 0.103) is added
to z for the flange the IK solves, as on the rig.

Tuning comes from the family config `bimanual_motion:` block.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

ARMS = ("left", "right")
R_TOP_DOWN = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)

DEFAULTS = {
    "flange_z_offset_m": 0.103,  # PyRoki flange link target above the jaw tip
    "left_yaw_offset_deg": 90.0,  # left jaw line is rotated 90 deg on the rig
    "left_z_correction_m": 0.0,  # measured left-vs-right height discrepancy
    "move_vel": 0.22,
    "approach_vel": 0.08,
    "arc_vel": 0.16,
    "close_settle_s": 1.2,
    "open_settle_s": 0.6,
    "close_force_ma": 1300,  # SSG-48 full-current close
    "holding_raw_min": 100,  # jaw raw count above this = closed on something
    "empty_raw_min": 250,  # at or above this = closed on air
    "reach_tol_m": 0.030,  # FK error above this = unreachable
}


def _wxyz(R: np.ndarray) -> np.ndarray:
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)


class BimanualMotion:
    def __init__(self, driver, T_right_to_left: Optional[np.ndarray], cfg: Optional[Mapping] = None, planner=None,
                 clearance=None):
        """``clearance(arm, target) -> bool`` is the inter-arm gate (BimanualSafeRobot.is_motion_safe)."""
        self.driver = driver
        self.clearance = clearance
        self.T_right_to_left = None if T_right_to_left is None else np.asarray(T_right_to_left, dtype=float)
        self.cfg = {**DEFAULTS, **dict(cfg or {})}
        self._planner = planner

    # frames

    def to_arm(self, arm: str, p_right: Sequence[float]) -> np.ndarray:
        """A right-arm-frame point in ``arm``'s frame (calibration transform)."""
        p = np.asarray(p_right, dtype=float)[:3]
        if arm == "right":
            return p.copy()
        if self.T_right_to_left is None:
            raise RuntimeError("no bimanual calibration: cannot express the scene in the left arm's frame")
        return (self.T_right_to_left @ np.append(p, 1.0))[:3]

    def orientation(self, arm: str, yaw_rad: Optional[float] = None) -> np.ndarray:
        """Top-down jaw orientation (wxyz) for ``arm``, optionally yawed.

        A detected OBB yaw is 180-degree ambiguous; the branch is fixed per
        arm so the wrist camera faces back toward the robots (right:
        sin(yaw) <= 0, left: sin(yaw) >= 0), as on the rig.
        """
        base = R_TOP_DOWN
        if arm == "left":
            base = Rotation.from_euler("z", self.cfg["left_yaw_offset_deg"], degrees=True).as_matrix() @ R_TOP_DOWN
        if yaw_rad is None:
            return _wxyz(base)
        yaw = float(np.arctan2(np.sin(yaw_rad), np.cos(yaw_rad)))
        if arm == "left" and np.sin(yaw) < 0.0:
            yaw += np.pi
        if arm == "right" and np.sin(yaw) > 0.0:
            yaw -= np.pi
        return _wxyz(Rotation.from_euler("z", yaw).as_matrix() @ base)

    # kinematics

    def planner(self):
        if self._planner is None:
            from spark_real.control.pyroki_planner import BimanualPyrokiPlanner

            self._planner = BimanualPyrokiPlanner(T_right_to_left=self.T_right_to_left)
        return self._planner

    def arm(self, arm: str):
        """The raw per-arm driver (joint moves, joint reads)."""
        return self.driver._drivers[arm]

    def ik(self, arm: str, pos, q_seed=None, yaw_rad: Optional[float] = None) -> Optional[np.ndarray]:
        t = np.asarray(pos, dtype=float)[:3].copy()
        t[2] += self.cfg["flange_z_offset_m"]
        if arm == "left":
            t[2] += self.cfg["left_z_correction_m"]
        if q_seed is None:
            q_seed = np.asarray(self.arm(arm).get_joint_positions(), dtype=float)
        try:
            return np.asarray(
                self.planner().solve(arm=arm, target_position_base=t,
                                     target_wxyz_base=self.orientation(arm, yaw_rad), prev_cfg=q_seed)
            )[:7]
        except Exception as exc:
            logger.warning("bimanual IK %s failed at %s: %s", arm, np.round(t, 3).tolist(), exc)
            return None

    # motion

    def _clear(self, arm: str, pos) -> bool:
        if self.clearance is None or self.clearance(arm, np.asarray(pos, dtype=float)[:3]):
            return True
        logger.warning("%s arm target %s is inside the other arm's keep-out distance", arm,
                       np.round(np.asarray(pos, dtype=float)[:3], 3).tolist())
        return False

    def move(self, arm: str, pos, vel: Optional[float] = None, yaw_rad: Optional[float] = None) -> bool:
        if not self._clear(arm, pos):
            return False
        q = self.ik(arm, pos, yaw_rad=yaw_rad)
        if q is None:
            return False
        self.arm(arm).move_to_joint_config(q.tolist(), velocity=vel or self.cfg["move_vel"])
        return True

    def move_both(self, targets: Dict[str, Sequence[float]], vel: Optional[float] = None,
                  yaws: Optional[Dict[str, float]] = None, check_clearance: bool = True) -> bool:
        """Solve every arm's target first, then move the arms together."""
        yaws = yaws or {}
        if check_clearance and not all(self._clear(a, p) for a, p in targets.items()):
            return False
        sols = {a: self.ik(a, p, yaw_rad=yaws.get(a)) for a, p in targets.items()}
        if any(q is None for q in sols.values()):
            logger.warning("bimanual move: IK failed for %s", [a for a, q in sols.items() if q is None])
            return False
        errors = []

        def _go(a):
            try:
                self.arm(a).move_to_joint_config(sols[a].tolist(), velocity=vel or self.cfg["move_vel"])
            except Exception as exc:  # noqa: BLE001 - report, do not kill the other arm's thread
                errors.append((a, exc))

        threads = [threading.Thread(target=_go, args=(a,), daemon=True) for a in sols]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for a, exc in errors:
            logger.warning("bimanual move: %s arm raised %s", a, exc)
        return not errors

    def arc(self, starts: Dict[str, Sequence[float]], ends: Dict[str, Sequence[float]], lift_z: float,
            peak_m: float, n: int = 8, land_z: Optional[float] = None,
            yaws: Optional[Dict[str, float]] = None) -> bool:
        """Sinusoidal arc at ``lift_z`` from start to end xy for one or both arms.

        The last waypoint lands at ``land_z`` when given. A waypoint whose IK
        fails ends the arc unsuccessfully.
        """
        for i in range(1, n + 1):
            t = i / n
            z = lift_z + peak_m * np.sin(t * np.pi)
            if land_z is not None and i == n:
                z = land_z
            wps = {
                a: [s[0] + t * (ends[a][0] - s[0]), s[1] + t * (ends[a][1] - s[1]), z]
                for a, s in starts.items()
            }
            if not self.move_both(wps, vel=self.cfg["arc_vel"], yaws=yaws):
                return False
            time.sleep(0.02)
        return True

    # jaws

    def jaw(self, arm: str):
        return self.driver.grippers.for_arm(arm)

    def jaw_raw(self, arm: str) -> Optional[float]:
        """Closure on the 0..255 scale; None when the jaw reports nothing fresh."""
        j = self.jaw(arm)
        if hasattr(j, "raw_position_fresh"):
            return j.raw_position_fresh()
        try:
            return float(j.get_position_norm())
        except Exception:
            return None

    def close(self, arms: Sequence[str], force_ma: Optional[int] = None) -> None:
        """Full-current close; one re-send for a jaw still open (CAN bus-off recovery)."""
        force = int(force_ma or self.cfg["close_force_ma"])

        def _send(a):
            j = self.jaw(a)
            if hasattr(j, "send_pack_locked"):
                j.send_pack_locked(255, 255, force, 1, 1, 0, 0)
            else:
                j.close(force_n=force / 80.0)

        for a in arms:
            _send(a)
        time.sleep(self.cfg["close_settle_s"])
        for a in arms:
            raw = self.jaw_raw(a)
            if raw is None or raw < self.cfg["holding_raw_min"]:
                logger.warning("%s jaw raw=%s after close; re-sending", a, raw)
                time.sleep(0.5)
                _send(a)
                time.sleep(self.cfg["close_settle_s"])

    def open(self, arms: Sequence[str]) -> None:
        for a in arms:
            j = self.jaw(a)
            if hasattr(j, "send_pack_locked"):
                j.send_pack_locked(0, 255, 500, 1, 1, 0, 0)
            else:
                j.open()
        time.sleep(self.cfg["open_settle_s"])

    def holding(self, arm: str) -> bool:
        """Post-condition of a grasp: closed on something, not on air."""
        raw = self.jaw_raw(arm)
        return raw is not None and self.cfg["holding_raw_min"] < raw < self.cfg["empty_raw_min"]

    def is_open(self, arm: str) -> bool:
        raw = self.jaw_raw(arm)
        return raw is not None and raw < self.cfg["holding_raw_min"]
