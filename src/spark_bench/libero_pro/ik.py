"""
Pure IK helpers for the LIBERO Panda runner.

Each function takes the MuJoCo model/data + joint/site indices explicitly
(no closures over runner state), so they are testable.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import mujoco


__all__ = [
    'get_q',
    'compute_ik',
    'compute_ik_6dof',
    'quat_from_approach',
]


def get_q(model, data, joint_ids: Sequence[int]) -> np.ndarray:
    """
    Read the actuated joint configuration from a MuJoCo data block.
    """
    return np.array([data.qpos[model.jnt_qposadr[j]] for j in joint_ids])


def compute_ik(model, data, joint_ids: Sequence[int], ee_site: int,
                target: np.ndarray, max_iters: int = 600,
                tol: float = 0.003) -> np.ndarray:
    """
    Position-only 3-DOF IK via damped least squares.

    Returns the actuated joint configuration that brings ``ee_site`` close
    to ``target``.  Mutates a scratch ``MjData`` only - caller's ``data``
    is untouched.
    """
    ndof = len(joint_ids)
    q = get_q(model, data, joint_ids)
    d2 = mujoco.MjData(model)
    d2.qpos[:] = data.qpos[:]
    for _ in range(max_iters):
        mujoco.mj_forward(model, d2)
        ee = d2.site_xpos[ee_site].copy()
        err = target - ee
        if np.linalg.norm(err) < tol:
            break
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, d2, jacp, None, ee_site)
        J = np.zeros((3, ndof))
        for i, jid in enumerate(joint_ids):
            J[:, i] = jacp[:, model.jnt_dofadr[jid]]
        JJT = J @ J.T + 0.01 ** 2 * np.eye(3)
        dq = 0.3 * J.T @ np.linalg.solve(JJT, err)
        dq = np.clip(dq, -0.2, 0.2)
        q += dq
        for i, jid in enumerate(joint_ids):
            d2.qpos[model.jnt_qposadr[jid]] = q[i]
    return q


def compute_ik_6dof(model, data, joint_ids: Sequence[int], ee_site: int,
                     target_pos: np.ndarray, target_quat: np.ndarray,
                     max_iters: int = 300) -> np.ndarray:
    """
    6-DOF IK (position + quaternion) via damped least squares.

    ``target_quat`` is in (w, x, y, z) order to match
    ``mujoco.mju_mat2Quat`` output.
    """
    ndof = len(joint_ids)
    q = get_q(model, data, joint_ids)
    d2 = mujoco.MjData(model)
    d2.qpos[:] = data.qpos[:]
    for i, jid in enumerate(joint_ids):
        d2.qpos[model.jnt_qposadr[jid]] = q[i]
    for _ in range(max_iters):
        mujoco.mj_forward(model, d2)
        ee = d2.site_xpos[ee_site].copy()
        pos_err = target_pos - ee
        ee_quat = np.zeros(4)
        mujoco.mju_mat2Quat(ee_quat, d2.site_xmat[ee_site].reshape(9))
        quat_err = np.zeros(4)
        ee_quat_conj = ee_quat.copy()
        ee_quat_conj[1:] *= -1
        mujoco.mju_mulQuat(quat_err, target_quat, ee_quat_conj)
        ori_err = quat_err[1:] * 2.0
        if quat_err[0] < 0:
            ori_err *= -1
        err = np.concatenate([pos_err, ori_err])
        if np.linalg.norm(pos_err) < 0.003 and np.linalg.norm(ori_err) < 0.05:
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, d2, jacp, jacr, ee_site)
        J = np.zeros((6, ndof))
        for i, jid in enumerate(joint_ids):
            dof = model.jnt_dofadr[jid]
            J[:3, i] = jacp[:, dof]
            J[3:, i] = jacr[:, dof]
        JJT = J @ J.T + 0.01 ** 2 * np.eye(6)
        q += 0.3 * J.T @ np.linalg.solve(JJT, err)
        for i, jid in enumerate(joint_ids):
            d2.qpos[model.jnt_qposadr[jid]] = q[i]
    return q


def quat_from_approach(approach_dir_world: np.ndarray,
                       finger_close_world: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Build a gripper orientation quaternion from an approach direction.

    The gripper's Z-axis (palm forward) is aligned with `approach_dir_world`
    (the direction the gripper moves into the object).  Its X-axis (one
    finger closing direction) is aligned with `finger_close_world` if given;
    otherwise a reasonable default is picked perpendicular to approach.

    Args:
        approach_dir_world: (3,) world-frame unit vector - the direction
            the gripper's palm faces (e.g. (0, -1, 0) for a horizontal grab
            facing -Y in world).
        finger_close_world: optional (3,) world vector - the direction one
            finger closes.  Defaults to world +Z (vertical close) for a
            horizontal approach, or world +X for a vertical approach.

    Returns:
        quat in (w, x, y, z) order.
    """
    z = np.asarray(approach_dir_world, dtype=np.float64)
    z = z / (np.linalg.norm(z) + 1e-9)
    if finger_close_world is None:
        # Pick a default perpendicular axis
        if abs(z[2]) < 0.9:
            x = np.array([0.0, 0.0, 1.0])  # world Z (vertical close)
        else:
            x = np.array([1.0, 0.0, 0.0])
    else:
        x = np.asarray(finger_close_world, dtype=np.float64)
    # Orthonormalise X against Z
    x = x - np.dot(x, z) * z
    x = x / (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    # Convert rotation matrix to (w, x, y, z) quaternion
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.array([w, qx, qy, qz])
