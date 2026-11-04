"""
FR3 inverse kinematics using pyroki (JAX-based).

Levenberg-Marquardt over a jaxls factor graph with pose, limit, rest and
manipulability costs, solved for the fr3_hand_tcp frame of the upstream
franka_description FR3 + Franka Hand URDF (robot_descriptions
"fr3_description"). JIT-compiled: first call ~7-10 s, later calls <100 ms.
"""

from __future__ import annotations
import logging
import os
from typing import Optional, Union

# CPU pinned: GPU JAX linalg solves fail here on a mixed nvidia wheel set.
# Clean the nvidia-* wheels before retrying JAX_PLATFORMS=cuda.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import jax
    import jax.numpy as jnp
    import jaxlie
    import jaxls
    import pyroki
    from jaxls import Cost, Var, VarValues
    from jax import Array
    from robot_descriptions.loaders.yourdfpy import load_robot_description
except ImportError:
    jax = jnp = jaxlie = jaxls = pyroki = None
    Cost = Var = VarValues = Array = None
    load_robot_description = None

logger = logging.getLogger(__name__)
_ROBOT_DESC = "fr3_description"
_urdf = None
_robot = _rcoll = _ths = None
_tli = _nact = None
_solvers: dict = {}
_last_q: Optional[np.ndarray] = None
_VEL_DT, _VEL_W = 0.01, 0.1
_SC_M, _SC_W, _WC_M, _WC_W, _TZ = 0.02, 5.0, 0.02, 5.0, -0.025


def _load():
    global _robot, _tli, _nact, _urdf
    if _robot is not None:
        return
    if pyroki is None or load_robot_description is None:
        raise RuntimeError("pyroki/JAX/robot_descriptions not installed; FR3 pyroki IK unavailable")
    _urdf = load_robot_description(_ROBOT_DESC)
    _robot = pyroki.Robot.from_urdf(_urdf)
    names = list(_robot.links.names)
    for c in ("fr3_hand_tcp", "fr3_hand", "panda_hand_tcp"):
        if c in names:
            _tli = names.index(c)
            logger.info("FR3 IK (pyroki): %d links, tcp=%s", len(names), c)
            break
    if _tli is None:
        raise RuntimeError(f"No TCP link in URDF. Available: {names[:20]}...")
    _nact = int(_robot.joints.num_actuated_joints)


def _load_coll():
    global _rcoll, _ths
    if _rcoll is not None:
        return
    _rcoll = pyroki.collision.RobotCollision.from_urdf(_urdf)
    _ths = pyroki.collision.HalfSpace.from_point_and_normal(
        point=jnp.array([0.0, 0.0, _TZ], dtype=jnp.float32),
        normal=jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32),
    )


def _solver(sm: bool, co: bool):
    k = (sm, co)
    if k not in _solvers:
        _load()
        if co:
            _load_coll()
        _solvers[k] = _build(sm, co)
    return _solvers[k]


def _build(sm: bool, co: bool):
    pk = pyroki.costs
    tli = jnp.array(_tli, dtype=jnp.int32)
    if sm:

        @Cost.create_factory
        def _vc(v: VarValues, r, jv: Var[Array], p: Array, dt: float, w) -> Array:
            return (
                jnp.maximum(0.0, jnp.abs((v[jv] - p) / dt) - r.joints.velocity_limits)
                * w
            ).flatten()

    def _solve(tp, tq, sd, pw, ow, rw, lw, mw, *ex):
        ex = iter(ex)
        if sm:
            prev, vw = next(ex), next(ex)
        if co:
            scw, wcw, tg = next(ex), next(ex), next(ex)
        pose = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(tq), tp)
        jv = _robot.joint_var_cls(0)
        fs = [
            pk.pose_cost(_robot, jv, pose, tli, pos_weight=pw, ori_weight=ow),
            pk.limit_cost(_robot, jv, weight=lw),
            pk.rest_cost(jv, rest_pose=sd, weight=rw),
            pk.manipulability_cost(_robot, jv, tli, weight=mw),
        ]
        if sm:
            fs.append(_vc(_robot, jv, prev, _VEL_DT, vw))
        if co:
            fs.append(
                pk.self_collision_cost(
                    _robot, robot_coll=_rcoll, joint_var=jv, margin=_SC_M, weight=scw
                )
            )
            fs.append(
                pk.world_collision_cost(
                    _robot,
                    robot_coll=_rcoll,
                    joint_var=jv,
                    world_geom=tg,
                    margin=_WC_M,
                    weight=wcw,
                )
            )
        prob = jaxls.LeastSquaresProblem(fs, [jv]).analyze()
        sol = prob.solve(
            initial_vals=jaxls.VarValues.make([jv.with_value(sd)]),
            linear_solver="dense_cholesky",
            termination=jaxls.TerminationConfig(max_iterations=150),
            verbose=False,
        )
        qf = sol[jv]
        return qf, jaxlie.SE3(_robot.forward_kinematics(qf)[_tli]).translation()

    return jax.jit(_solve)


def _seed(q7):
    s = np.zeros(_nact, dtype=np.float32)
    n = min(7, _nact)
    s[:n] = q7[:n]
    if abs(s[4]) < 0.30:
        s[4] = -0.50 if s[4] <= 0 else 0.50
    if _nact > 7:
        s[7:] = 0.02
    return s


def _wj7(q, ref):
    q7 = q[6]
    best, bok, bd = q7, abs(q7) <= 2.85, abs(q7 - ref)
    for d in (-np.pi, np.pi):
        c, ok, di = q7 + d, abs(q7 + d) <= 2.85, abs(q7 + d - ref)
        if (ok and not bok) or (ok == bok and di < bd):
            best, bok, bd = c, ok, di
    q[6] = best


def solve_ik(
    target_pos: np.ndarray,
    target_orient: Union[np.ndarray, list],
    q_seed: np.ndarray,
    pos_weight: float = 300.0,
    ori_weight: float = 100.0,
    rest_weight: float = 1.5,
    limit_weight: float = 200.0,
    manip_weight: float = 1.0,
    **_kw,
) -> Optional[np.ndarray]:
    """
    Solve FR3 IK. Returns 7-vec joint positions or None on failure.
    """
    global _last_q
    _load()
    uv = os.environ.get("SPARK_IK_VEL_SMOOTHING", "0") == "1"
    uc = os.environ.get("SPARK_IK_COLLISION", "0") == "1"
    slv = _solver(uv, uc)

    target_pos = np.asarray(target_pos, dtype=float).reshape(3)
    to = np.asarray(target_orient, dtype=float)
    if to.shape == (3,):
        Rt = Rotation.from_rotvec(to).as_matrix()
    elif to.shape == (3, 3):
        Rt = to
    else:
        raise ValueError(f"target_orient must be (3,) or (3,3), got {to.shape}")

    q_seed = np.asarray(q_seed, dtype=float).ravel()
    if q_seed.size < 7:
        raise ValueError(f"q_seed needs >= 7 entries, got {q_seed.size}")
    sd = _seed(q_seed)
    pv = _last_q if _last_q is not None else sd

    def _one(R, rw):
        qx = Rotation.from_matrix(R).as_quat()
        qw = np.array([qx[3], qx[0], qx[1], qx[2]], dtype=np.float32)
        a = (
            jnp.asarray(target_pos, jnp.float32),
            jnp.asarray(qw, jnp.float32),
            jnp.asarray(sd, jnp.float32),
            jnp.float32(pos_weight),
            jnp.float32(ori_weight),
            jnp.float32(rw),
            jnp.float32(limit_weight),
            jnp.float32(manip_weight),
        )
        ex = ()
        if uv:
            ex += (jnp.asarray(pv, jnp.float32), jnp.float32(_VEL_W))
        if uc:
            ex += (jnp.float32(_SC_W), jnp.float32(_WC_W), _ths)
        try:
            qj, ap = slv(*a, *ex)
        except Exception:
            return None, None
        return np.asarray(qj, dtype=float).ravel(), np.asarray(ap, dtype=float)

    # Solve both yaw orientations (gripper symmetry: yaw and yaw+pi).
    Ra = Rt @ Rotation.from_euler("z", np.pi).as_matrix()
    cands = []
    for R, tag in ((Rt, "primary"), (Ra, "yaw+pi")):
        q, ap = _one(R, rest_weight)
        if q is None or q.size < 7:
            continue
        _wj7(q, q_seed[6])
        cands.append(
            (
                q,
                ap,
                np.max(np.abs(q[:7] - q_seed[:7])),
                float(np.linalg.norm(ap - target_pos)),
                tag,
            )
        )

    if not cands:
        logger.warning("[pyroki IK] no candidate solved")
        return None
    cands = [c for c in cands if c[3] <= 0.05]
    if not cands:
        logger.warning("[pyroki IK] all candidates >5cm off")
        return None
    # Prefer J5 away from the wrist singularity; do not add an upper band
    # (it breaks candidate continuity across chained waypoints). The hard
    # clamp below handles the limit case.
    ns = [c for c in cands if abs(c[0][4]) >= 0.25]
    if ns:
        cands = ns
    cands.sort(key=lambda c: c[2])
    qb, _, dqb, eb, tb = cands[0]

    # Heavy-rest retry for large swing or J5 near singularity.
    if dqb > 1.0 or abs(qb[4]) < 0.25:
        logger.info("[pyroki IK] dq=%.2f/J5=%.2f; heavy-rest retry", dqb, qb[4])
        q2, a2 = _one(Rt if tb == "primary" else Ra, rest_weight * 4)
        if q2 is not None and q2.size >= 7:
            _wj7(q2, q_seed[6])
            d2, e2 = np.max(np.abs(q2[:7] - q_seed[:7])), float(
                np.linalg.norm(a2 - target_pos)
            )
            if e2 <= 0.05 and d2 < dqb:
                qb, dqb, eb = q2, d2, e2

    if eb > 0.05:
        return None
    # Hard-clamp to joint limits minus a margin: the soft limit_cost can
    # let a solution land just past a bound, which libfranka rejects with
    # joint_position_limits_violation and a slow fallback cascade.
    lo = np.asarray(_robot.joints.lower_limits, dtype=float)[:7] + 0.02
    hi = np.asarray(_robot.joints.upper_limits, dtype=float)[:7] - 0.02
    qb = np.asarray(qb, dtype=float).copy()
    qb[:7] = np.clip(qb[:7], lo, hi)
    logger.info("[pyroki IK] solved dq=%.2f pos_err=%.4fm", dqb, eb)
    _last_q = np.asarray(qb, dtype=np.float32).copy()
    return qb[:7]


def fk(q: np.ndarray) -> tuple:
    """
    Forward kinematics for TCP. Returns (pos[3], R[3x3]).
    """
    _load()
    qf = np.zeros(_nact, dtype=np.float32)
    q = np.asarray(q, dtype=float).ravel()
    qf[: min(7, _nact)] = q[: min(7, _nact)]
    if _nact > 7:
        qf[7:] = 0.02
    T = jaxlie.SE3(_robot.forward_kinematics(jnp.asarray(qf))[_tli])
    return np.asarray(T.translation(), dtype=float), np.asarray(
        T.rotation().as_matrix(), dtype=float
    )
