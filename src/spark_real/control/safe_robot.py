"""SafeRobot: CBF safety filter wrapper (robot-agnostic).

Drop-in replacement that intercepts velocity and script commands, solves a
Control Barrier Function QP at every tick to enforce kinematic, obstacle,
and contact-force safety constraints, then forwards the (possibly modified)
command to the underlying robot.

Dependencies: pip install osqp scipy numpy
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

import numpy as np

try:
    import osqp
    import scipy.sparse as sp
    from scipy import ndimage as _ndimage

    _OSQP_AVAILABLE = True
except ImportError:
    _OSQP_AVAILABLE = False
    _ndimage = None

from spark_real.control.barriers import (  # noqa: F401
    BarrierSet,
    Obstacle,
    ObstacleMap,
    SafetyConfig,
)

logger = logging.getLogger(__name__)


class SafeRobot:
    """
    Drop-in safety wrapper for any driver implementing the shim interface.

    Intercepts send_velocity (CBF-QP filter), move_linear/move_linear_relative
    (workspace clamp), and _send_script (UR-only URScript rewriting).
    All other methods pass through via __getattr__.
    """

    def __init__(
        self,
        robot,
        enable_obstacles: bool = True,
        enable_force: bool = True,
        config: Optional[SafetyConfig] = None,
    ):
        self._robot = robot
        self._cfg = config or SafetyConfig()
        self._enable_force = enable_force
        self._barriers = BarrierSet(self._cfg)
        self._obstacle_map = ObstacleMap(stale_timeout=self._cfg.stale_timeout)

        self._f_baseline: Optional[np.ndarray] = None
        self._f_baseline_count: int = 0
        self._f_baseline_acc: Optional[np.ndarray] = None
        self.cbf_deviated: bool = False
        self._solver: Optional[Any] = None
        self._solver_n_constraints: int = 0

        lim_lin = self._cfg.vel_lin_max
        lim_ang = self._cfg.vel_ang_max
        self._u_min = np.array([-lim_lin] * 3 + [-lim_ang] * 3)
        self._u_max = np.array([lim_lin] * 3 + [lim_ang] * 3)

        if not _OSQP_AVAILABLE:
            logger.warning("osqp/scipy not installed; CBF-QP disabled")
        logger.info(
            "SafeRobot init (obstacles=%s, force=%s, osqp=%s)",
            enable_obstacles,
            enable_force,
            _OSQP_AVAILABLE,
        )

    # State reading helpers

    def _read_tcp_position(self) -> np.ndarray:
        try:
            obs = self._robot.get_observation()
            pose = obs.get("tcp_pose")
            if pose is not None:
                return np.array(pose[:3], dtype=np.float64)
        except Exception:
            pass
        try:
            p = self._robot.get_tcp_pose()
            if isinstance(p, np.ndarray) and p.shape == (4, 4):
                return p[:3, 3].copy()
            return np.array(p[:3], dtype=np.float64)
        except Exception:
            return np.zeros(3)

    def _read_joints(self) -> np.ndarray:
        try:
            return np.array(
                self._robot.get_observation()["joint_positions"], dtype=np.float64
            )
        except Exception:
            return np.zeros(6)

    def _read_force(self) -> Optional[np.ndarray]:
        if hasattr(self._robot, "get_tcp_force"):
            try:
                return np.array(self._robot.get_tcp_force(), dtype=np.float64)
            except Exception:
                pass
        return None

    def _update_force_baseline(self, f_tcp: np.ndarray) -> None:
        if self._f_baseline is not None:
            return
        if self._f_baseline_acc is None:
            self._f_baseline_acc = np.zeros(6)
        self._f_baseline_acc += f_tcp
        self._f_baseline_count += 1
        if self._f_baseline_count >= self._cfg.force_baseline_samples:
            self._f_baseline = self._f_baseline_acc / self._f_baseline_count
            logger.info("Force baseline calibrated: %s", self._f_baseline)

    # QP solver

    def _build_and_solve_qp(
        self, u_des: np.ndarray, A_cbf: np.ndarray, b_cbf: np.ndarray
    ) -> Optional[np.ndarray]:
        if not _OSQP_AVAILABLE:
            return None
        n = 6
        n_cbf = A_cbf.shape[0] if A_cbf.ndim == 2 else 0
        P = sp.eye(n, format="csc") * 2.0
        q = -2.0 * u_des
        A_top = A_cbf if n_cbf > 0 else np.zeros((0, n))
        A_full = np.vstack([A_top, np.eye(n)])
        A_sp = sp.csc_matrix(A_full)
        l_cbf = b_cbf if n_cbf > 0 else np.array([])
        u_cbf = np.full(n_cbf, np.inf)
        l_full = np.concatenate([l_cbf, self._u_min])
        u_full = np.concatenate([u_cbf, self._u_max])
        total = n_cbf + n

        try:
            if self._solver is not None and self._solver_n_constraints == total:
                self._solver.update(q=q, l=l_full, u=u_full, Ax=A_sp.data)
            else:
                self._solver = osqp.OSQP()
                self._solver.setup(
                    P=P,
                    q=q,
                    A=A_sp,
                    l=l_full,
                    u=u_full,
                    verbose=False,
                    warm_starting=True,
                    max_iter=200,
                    eps_abs=1e-4,
                    eps_rel=1e-4,
                    polish=False,
                    adaptive_rho=True,
                )
                self._solver_n_constraints = total
            result = self._solver.solve()
            if result.info.status in ("solved", "solved_inaccurate"):
                return np.array(result.x, dtype=np.float64)
            logger.warning("QP status: %s", result.info.status)
            return None
        except Exception as exc:
            logger.error("QP solver error: %s", exc)
            self._solver = None
            self._solver_n_constraints = 0
            return None

    # Capability checks

    def _supports_velocity_streaming(self) -> bool:
        flag = getattr(self._robot, "SUPPORTS_VELOCITY_STREAMING", None)
        if flag is not None:
            return bool(flag)
        return callable(getattr(self._robot, "send_velocity", None))

    def _supports_urscript(self) -> bool:
        flag = getattr(self._robot, "SUPPORTS_URSCRIPT", None)
        if flag is not None:
            return bool(flag)
        return callable(getattr(self._robot, "_send_script", None))

    def _forward_velocity(
        self, u: np.ndarray, acceleration: float, time_duration: float
    ) -> None:
        if self._supports_velocity_streaming():
            try:
                self._robot.send_velocity(
                    u.tolist(), acceleration=acceleration, duration=time_duration
                )
            except TypeError:
                self._robot.send_velocity(
                    u.tolist(), acceleration=acceleration, time_duration=time_duration
                )
            return
        raise NotImplementedError(
            f"{type(self._robot).__name__} does not support velocity streaming."
        )

    def _stop_underlying(self) -> None:
        for name in ("stop_motion", "stop", "servo_stop"):
            fn = getattr(self._robot, name, None)
            if callable(fn):
                try:
                    fn()
                    return
                except Exception:
                    continue

    # Velocity entry point

    def send_velocity(
        self, velocity, acceleration: float = 0.5, time_duration: float = 0.1
    ):
        """
        Filter velocity through CBF-QP then forward to robot.
        """
        u_des = np.array(velocity, dtype=np.float64).ravel()[:6]
        if u_des.shape[0] < 6:
            u_des = np.pad(u_des, (0, 6 - u_des.shape[0]))

        if not _OSQP_AVAILABLE:
            self._forward_velocity(
                np.clip(u_des, self._u_min, self._u_max), acceleration, time_duration
            )
            return

        # Pure rotation skips QP
        if np.linalg.norm(u_des[:3]) < 1e-4:
            self._forward_velocity(
                np.clip(u_des, self._u_min, self._u_max), acceleration, time_duration
            )
            return

        p_tcp = self._read_tcp_position()
        joints = self._read_joints()
        obstacles = self._obstacle_map.get()
        f_tcp = self._read_force() if self._enable_force else None
        if f_tcp is not None and self._f_baseline is None:
            self._update_force_baseline(f_tcp)

        A_cbf, b_cbf, obstacle_active = self._barriers.evaluate(
            p_tcp,
            joints,
            obstacles,
            f_tcp,
            self._f_baseline,
            dt=time_duration,
            enable_force=self._enable_force,
        )

        u_safe = self._build_and_solve_qp(u_des, A_cbf, b_cbf)
        if u_safe is None:
            logger.warning("CBF-QP infeasible, emergency stop")
            try:
                self._stop_underlying()
            except Exception:
                pass
            return

        if obstacle_active and float(np.linalg.norm(u_safe - u_des)) > 1e-3:
            self.cbf_deviated = True
            logger.info(
                "CBF obstacle deviation: %.4f m/s",
                float(np.linalg.norm(u_safe - u_des)),
            )
        self._forward_velocity(u_safe, acceleration, time_duration)

    # URScript interception (UR-only)

    _RE_MOVEJ = re.compile(r"movej\s*\(\s*p\[([^\]]+)\]", re.IGNORECASE)
    _RE_MOVEL = re.compile(r"movel\s*\(\s*p\[([^\]]+)\]", re.IGNORECASE)
    _RE_MOVEJ_JOINTS = re.compile(r"movej\s*\(\s*\[([^\]]+)\]", re.IGNORECASE)

    def _send_script(self, script: str) -> bool:
        if not self._supports_urscript():
            raise AttributeError(
                f"{type(self._robot).__name__} does not support URScript."
            )
        # EVERY row is checked, not just the first. A blended program is one
        # script holding N movej rows; .search() validated row 0 and let rows
        # 1..N-1 through unchecked, so with blending on by default most of each
        # move bypassed the singularity/workspace filter entirely.
        #
        # Checked in wire order, and the first rejection refuses the WHOLE
        # script: a blended path is not separable, so partially approving it
        # would run the arm through the rows we rejected.
        for regex, handler in (
            (self._RE_MOVEL, self._check_movel_row),
            (self._RE_MOVEJ, self._check_movej_cartesian_row),
            (self._RE_MOVEJ_JOINTS, self._check_movej_joints_row),
        ):
            for m in regex.finditer(script):
                if not handler(m.group(1)):
                    logger.warning(
                        "[CBF] refusing script: a %s row failed the check "
                        "(%d row(s) inspected)",
                        handler.__name__,
                        len(regex.findall(script)),
                    )
                    return False
        return self._robot._send_script(script)

    def _parse_pose_values(self, csv: str) -> Optional[np.ndarray]:
        try:
            return np.array([float(x.strip()) for x in csv.split(",")])
        except (ValueError, TypeError):
            return None

    # Check-only row predicates used by the multi-row scan in _send_script.
    # They must NOT clamp or rewrite: a blended program's rows are one motion,
    # so silently moving one waypoint changes the path the others were planned
    # against. Unparseable rows pass (the legacy single-row path did the same).

    def _check_pose_row(self, csv: str) -> bool:
        pose = self._parse_pose_values(csv)
        if pose is None or len(pose) < 3:
            return True
        safe, violations = self._barriers.check_pose_safe(pose[:3])
        if not safe:
            logger.warning("[CBF] pose row unsafe: %s", violations)
        return bool(safe)

    def _check_movel_row(self, csv: str) -> bool:
        return self._check_pose_row(csv)

    def _check_movej_cartesian_row(self, csv: str) -> bool:
        return self._check_pose_row(csv)

    def _check_movej_joints_row(self, csv: str) -> bool:
        q = self._parse_pose_values(csv)
        if q is None or len(q) < 6:
            return True
        # A joint row carries no TCP pose, so the workspace box cannot be
        # applied here without forward kinematics we do not have. The elbow
        # singularity barrier IS evaluable from q alone, and it is the only
        # test the previous single-row handler enforced -- so enforcing it on
        # every row is strict parity plus the rows that used to be skipped.
        #
        # Do NOT reintroduce a "no FK available -> return True" branch: that
        # is what silently turned this predicate into a no-op and left
        # movej([q1..q6]) completely unvalidated.
        if self._cfg.eps_singularity <= 0:
            return True
        q2 = float(q[2])
        h = (q2**2) * ((q2 - np.pi) ** 2) - self._cfg.eps_singularity**2
        if h < 0:
            logger.warning(
                "[CBF] movej joint row near singularity: q2=%.3f h=%.4f", q2, h
            )
            return False
        return True

    # Workspace clamping

    def _clamp_to_workspace(self, p: np.ndarray) -> np.ndarray:
        clamped = np.clip(p, self._cfg.ws_min, self._cfg.ws_max)
        r = np.linalg.norm(clamped)
        if r > self._cfg.reach_max:
            clamped = clamped * (self._cfg.reach_max / r) * 0.98
        return clamped

    def _clamp_pose_to_workspace(self, pose) -> np.ndarray:
        p = np.asarray(pose, dtype=np.float64).ravel()
        if p.shape[0] < 6:
            p = np.pad(p, (0, 6 - p.shape[0]))
        else:
            p = p[:6]
        out = p.copy()
        out[:3] = self._clamp_to_workspace(p[:3])
        return out

    # Pose-motion entry points

    def move_linear(self, pose, *args, **kwargs):
        clamped = self._clamp_pose_to_workspace(pose)
        if not np.allclose(clamped[:3], np.asarray(pose, dtype=np.float64).ravel()[:3]):
            logger.info(
                "SafeRobot: clamped move_linear xyz %s -> %s",
                np.asarray(pose).ravel()[:3].tolist(),
                clamped[:3].tolist(),
            )
        return self._robot.move_linear(clamped.tolist(), *args, **kwargs)

    def move_linear_relative(self, delta, *args, **kwargs):
        d = np.asarray(delta, dtype=np.float64).ravel()
        if d.shape[0] < 6:
            d = np.pad(d, (0, 6 - d.shape[0]))
        else:
            d = d[:6]
        p_current = self._read_tcp_position()
        p_target_clamped = self._clamp_to_workspace(p_current + d[:3])
        d_adjusted = d.copy()
        d_adjusted[:3] = p_target_clamped - p_current
        return self._robot.move_linear_relative(d_adjusted.tolist(), *args, **kwargs)

    def move_to_joint_config(self, q, *args, **kwargs):
        return self._robot.move_to_joint_config(q, *args, **kwargs)

    def move_home(self, *args, **kwargs):
        if hasattr(self._robot, "move_home"):
            return self._robot.move_home(*args, **kwargs)
        if hasattr(self._robot, "go_home"):
            return self._robot.go_home(*args, **kwargs)
        raise AttributeError("No move_home or go_home on underlying driver.")

    def open_gripper(self, *args, **kwargs):
        return self._robot.open_gripper(*args, **kwargs)

    def close_gripper(self, *args, **kwargs):
        return self._robot.close_gripper(*args, **kwargs)

    def set_gripper_position(self, position, *args, **kwargs):
        if hasattr(self._robot, "set_gripper_position"):
            return self._robot.set_gripper_position(position, *args, **kwargs)
        return (
            self._robot.close_gripper()
            if position >= 0.5
            else self._robot.open_gripper()
        )

    def get_observation(self, *args, **kwargs):
        if hasattr(self._robot, "get_observation"):
            return self._robot.get_observation(*args, **kwargs)
        obs: Dict[str, Any] = {}
        for shim_name, key in (
            ("get_joint_positions", "joint_positions"),
            ("get_joint_velocities", "joint_velocities"),
            ("get_tcp_pose", "tcp_pose"),
            ("get_tcp_force", "tcp_force"),
            ("get_gripper_position", "gripper_position"),
        ):
            fn = getattr(self._robot, shim_name, None)
            if callable(fn):
                try:
                    obs[key] = fn()
                except Exception:
                    pass
        obs.setdefault("gripper_position", 0.0)
        return obs

    def get_tcp_pose(self, *args, **kwargs):
        return self._robot.get_tcp_pose(*args, **kwargs)

    def stop_motion(self, *args, **kwargs):
        if hasattr(self._robot, "stop_motion"):
            return self._robot.stop_motion(*args, **kwargs)
        if hasattr(self._robot, "stop"):
            return self._robot.stop(*args, **kwargs)
        raise AttributeError("No stop_motion or stop on underlying driver.")

    def __getattr__(self, name: str):
        return getattr(self._robot, name)

    def shutdown(self):
        self._obstacle_map.clear()
        logger.info("SafeRobot shutdown complete")

    def __repr__(self):
        return (
            f"SafeRobot(robot={self._robot.__class__.__name__}, "
            f"osqp={_OSQP_AVAILABLE}, "
            f"force={self._enable_force})"
        )
