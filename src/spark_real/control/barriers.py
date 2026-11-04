"""Control-barrier constraints for SafeRobot: workspace, reach, singularity, obstacles, force."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class SafetyConfig:
    """
    Tuneable parameters for the CBF safety filter.
    """

    eta_kinematic: float = 0.3
    eta_obstacle: float = 0.5
    eta_force: float = 0.8
    vel_lin_max: float = 0.25
    vel_ang_max: float = 1.0
    # Fallback box used only when SafeRobot is built with config=None. Kept
    # as literals rather than unified with ur10e_default.yaml: this z_min is
    # -0.25 while the live UR10e SafetyConfig uses -0.27, so merging would
    # change a value.
    ws_min: np.ndarray = field(default_factory=lambda: np.array([-1.1, -0.5, -0.25]))
    ws_max: np.ndarray = field(default_factory=lambda: np.array([-0.5, 0.7, 0.50]))
    reach_max: float = 1.15
    eps_singularity: float = 0.20
    r_safe: float = 0.10
    f_threshold: float = 10.0
    f_max: float = 30.0
    force_baseline_samples: int = 20
    stale_timeout: float = 0.5
    depth_hz: float = 30.0
    movel_path_samples: int = 5


@dataclass
class Obstacle:
    """
    Single detected obstacle (sphere approximation).
    """

    position: np.ndarray
    radius: float
    timestamp: float = 0.0


class ObstacleMap:
    """
    Thread-safe obstacle list using atomic ref swap under the GIL.
    """

    def __init__(self, stale_timeout: float = 0.5):
        self._stale_timeout = stale_timeout
        self._obstacles: List[Obstacle] = []

    def update(self, obstacles: List[Obstacle]) -> None:
        self._obstacles = obstacles

    def get(self) -> List[Obstacle]:
        now = time.monotonic()
        return [o for o in self._obstacles if (now - o.timestamp) < self._stale_timeout]

    def clear(self) -> None:
        self._obstacles = []


class BarrierSet:
    """
    Collect all barrier functions and linearise them for the QP.

    Each barrier returns a row (a, b) such that the CBF constraint is
    a @ u >= b where u is the 6-D Cartesian velocity command.
    """

    def __init__(self, cfg: SafetyConfig):
        self.cfg = cfg

    def _reach_barrier(self, p_tcp: np.ndarray) -> Optional[tuple]:
        # h_reach = R_max^2 - ||p||^2
        r2 = float(np.dot(p_tcp, p_tcp))
        R2 = self.cfg.reach_max**2
        h = R2 - r2
        if h > R2:
            return None
        a = np.zeros(6)
        a[:3] = -2.0 * p_tcp
        return a, -self.cfg.eta_kinematic * h

    def _workspace_barriers(self, p_tcp: np.ndarray) -> List[tuple]:
        rows: List[tuple] = []
        eta = self.cfg.eta_kinematic
        for axis in range(3):
            h_lo = p_tcp[axis] - self.cfg.ws_min[axis]
            a_lo = np.zeros(6)
            a_lo[axis] = 1.0
            rows.append((a_lo, -eta * h_lo))
            h_hi = self.cfg.ws_max[axis] - p_tcp[axis]
            a_hi = np.zeros(6)
            a_hi[axis] = -1.0
            rows.append((a_hi, -eta * h_hi))
        return rows

    def _singularity_barrier(self, joints: np.ndarray, dt: float) -> Optional[tuple]:
        """
        UR-elbow specific; disabled when eps_singularity <= 0.
        """
        if self.cfg.eps_singularity <= 0:
            return None
        q2 = float(joints[2]) if len(joints) > 2 else 0.0
        eps2 = self.cfg.eps_singularity**2
        h = (q2**2) * ((q2 - np.pi) ** 2) - eps2
        if h > 10.0 * eps2:
            return None
        dh_dq2 = 2.0 * q2 * ((q2 - np.pi) ** 2) + (q2**2) * 2.0 * (q2 - np.pi)
        scale = max(abs(dh_dq2), 1e-6)
        alpha = max(self.cfg.eta_kinematic * max(h, 0.0) / scale, 0.01)
        rows: List[tuple] = []
        for axis in range(6):
            a_pos = np.zeros(6)
            a_pos[axis] = -1.0
            rows.append((a_pos, -alpha))
            a_neg = np.zeros(6)
            a_neg[axis] = 1.0
            rows.append((a_neg, -alpha))
        return rows

    def _obstacle_barriers(
        self, p_tcp: np.ndarray, obstacles: List[Obstacle]
    ) -> List[tuple]:
        rows: List[tuple] = []
        eta = self.cfg.eta_obstacle
        r_safe = self.cfg.r_safe
        for obs in obstacles:
            diff = p_tcp - obs.position
            dist2 = float(np.dot(diff, diff))
            margin = (r_safe + obs.radius) ** 2
            h = dist2 - margin
            a = np.zeros(6)
            a[:3] = 2.0 * diff
            rows.append((a, -eta * h))
        return rows

    def _force_barrier(
        self, f_tcp: np.ndarray, f_baseline: np.ndarray
    ) -> Optional[tuple]:
        f_net = f_tcp[:3] - f_baseline[:3]
        f_mag2 = float(np.dot(f_net, f_net))
        F2 = self.cfg.f_max**2
        h = F2 - f_mag2
        f_mag = np.sqrt(f_mag2)
        if f_mag < self.cfg.f_threshold:
            return None
        f_hat = f_net / max(f_mag, 1e-8)
        a = np.zeros(6)
        a[:3] = -f_hat
        return a, self.cfg.eta_force * h

    def evaluate(
        self, p_tcp, joints, obstacles, f_tcp, f_baseline, dt=0.1, enable_force=True
    ) -> tuple:
        """
        Return (A_cbf, b_cbf, obstacle_active) for the QP.
        """
        rows: List[tuple] = []
        obstacle_active = False

        rb = self._reach_barrier(p_tcp)
        if rb is not None:
            rows.append(rb)

        rows.extend(self._workspace_barriers(p_tcp))

        sb = self._singularity_barrier(joints, dt)
        if sb is not None:
            if isinstance(sb, list):
                rows.extend(sb)
            else:
                rows.append(sb)

        obs_rows = self._obstacle_barriers(p_tcp, obstacles)
        if obs_rows:
            obstacle_active = True
            rows.extend(obs_rows)

        if enable_force and f_tcp is not None and f_baseline is not None:
            fb = self._force_barrier(f_tcp, f_baseline)
            if fb is not None:
                rows.append(fb)

        if not rows:
            return np.zeros((0, 6)), np.zeros(0), obstacle_active
        A = np.array([r[0] for r in rows])
        b = np.array([r[1] for r in rows])
        return A, b, obstacle_active

    def check_pose_safe(
        self, p_tcp: np.ndarray, joints: Optional[np.ndarray] = None
    ) -> tuple:
        """
        Check whether a static TCP pose violates any kinematic barrier.
        """
        violations: List[str] = []
        r2 = float(np.dot(p_tcp, p_tcp))
        if r2 > self.cfg.reach_max**2:
            violations.append(
                f"reach: ||p||={np.sqrt(r2):.3f}m > {self.cfg.reach_max}m"
            )
        for axis, name in enumerate("xyz"):
            if p_tcp[axis] < self.cfg.ws_min[axis]:
                violations.append(
                    f"workspace: {name}={p_tcp[axis]:.3f} < {self.cfg.ws_min[axis]}"
                )
            if p_tcp[axis] > self.cfg.ws_max[axis]:
                violations.append(
                    f"workspace: {name}={p_tcp[axis]:.3f} > {self.cfg.ws_max[axis]}"
                )
        if joints is not None and len(joints) > 2 and self.cfg.eps_singularity > 0:
            q2 = float(joints[2])
            h = (q2**2) * ((q2 - np.pi) ** 2) - self.cfg.eps_singularity**2
            if h < 0:
                violations.append(f"singularity: h={h:.4f}, q2={q2:.3f}")
        return len(violations) == 0, violations
