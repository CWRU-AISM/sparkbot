"""
Tool-use and pouring skills the planner composes (single arm, any family).

Ported from the motion logic of the FR3 task procedures (sponge wash, mug
pour, sweep into a dustpan); every position comes from the detection map
and every tuning number from the plan, the family `task_motion:` block, or
DEFAULTS in task_motion.py. Each skill returns success only when its
post-condition holds:

    grasp_se3          top-down pick with stepped retries     post: holding
    constrained_scrub  pattern on a surface with a held tool  post: path done, tool held
    wipe               alias of constrained_scrub
    sponge_wash        pick sponge, scrub dish, put sponge back  post: each step's
    pour               tilt a held container over a target, hold, upright  post: upright, held
    sweep              held brush pushes objects into a container  post: path done, tool held,
                       and (with perception) every object within the container radius
    sweep_to_container alias of sweep with the other prompt's parameter names
"""

from __future__ import annotations

import logging
import time

import numpy as np
from scipy.spatial.transform import Rotation

from spark_real.control.grasp_strategy import resolve_grasp_orientation
from spark_real.skills import task_motion as tm
from spark_real.skills.primitives import _result
from spark_real.skills.registry import spark_skill

logger = logging.getLogger(__name__)


def _done(name, t0, ok, msg):
    return _result(name, bool(ok), msg, time.time() - t0)


def _orient(ex):
    try:
        return list(ex.current_orientation(fallback=ex.GRASP_ORIENTATION))
    except Exception:  # noqa: BLE001
        return list(ex.GRASP_ORIENTATION)


@spark_skill(
    "grasp_se3",
    "Pick a detected object: hover, descend, close; step down and retry on an empty close; lift.",
    {"keypoint_label": str, "force": float, "target_width": float, "strategy": str,
     "offset_x": float, "offset_y": float, "offset_z": float},
)
def grasp_se3(ex, params: dict):
    t0 = time.time()
    label = params.get("keypoint_label", "")
    strategy = str(params.get("strategy", "top_down"))
    if strategy not in ("top_down", "auto"):
        logger.warning("grasp_se3 strategy %r: this release picks top-down (as the FR3 procedures did)", strategy)
    p = tm.det_xyz(ex, label) + [float(params.get(k, 0.0)) for k in ("offset_x", "offset_y", "offset_z")]
    # Jaw yaw from the same OBB strategy the core grasp uses (top-down base).
    try:
        orient, _ = resolve_grasp_orientation(params, ex.detection_map.get(label) or {}, ex)
    except Exception as exc:  # noqa: BLE001 - fall back to the plain top-down rotvec
        logger.warning("grasp_se3: yaw resolve failed (%s); top-down", exc)
        orient = list(ex.GRASP_ORIENTATION)
    held = tm.pick(ex, p, orient, params)
    ex._last_pick_label = label
    return _done("grasp_se3", t0, held, f"{label} holding={held} width={tm.jaw_width(ex):.4f}")


@spark_skill(
    "constrained_scrub",
    "Scrub a surface with a held tool: press below the surface top and trace a pattern.",
    {"target_label": str, "n_cycles": int, "scrub_radius_m": float, "pattern": str,
     "press_dz": float, "steps_per_cycle": int, "scrub_vel": float},
)
def constrained_scrub(ex, params: dict, name: str = "constrained_scrub"):
    t0 = time.time()
    if not ex._holding:
        return _done(name, t0, False, "no tool held (grasp the sponge first)")
    surface = tm.det_xyz(ex, params.get("target_label") or params.get("workpiece_label", ""))
    orient = _orient(ex)
    center = surface - [0.0, 0.0, float(tm.cfg(ex, "press_dz", params) or 0.005)]
    ex._move_to(center + [0, 0, float(tm.cfg(ex, "approach_dz", params))], orient)
    steps = int(params.get("steps_per_cycle", 16))
    vel = float(tm.cfg(ex, "scrub_vel", params) or 0.08)
    pts = tm.circle_points(center, float(params.get("scrub_radius_m", 0.03)), steps, str(params.get("pattern", "circle")))
    cycles = int(params.get("n_cycles", 4))
    for _ in range(cycles):
        for p in pts:
            ex._move_to(p, orient, velocity=vel)
    ex._move_to(center + [0, 0, float(tm.cfg(ex, "approach_dz", params))], orient)
    held = tm.holding(ex, params)
    return _done(name, t0, held, f"{cycles} cycles x {steps} points, tool held={held}")


@spark_skill("wipe", "Alias of constrained_scrub (paper grammar name).",
             {"target_label": str, "n_cycles": int, "scrub_radius_m": float, "pattern": str})
def wipe(ex, params: dict):
    return constrained_scrub(ex, params, name="wipe")


@spark_skill(
    "sponge_wash",
    "Pick the sponge, scrub the dish, put the sponge back where it was.",
    {"sponge_label": str, "dish_label": str, "n_cycles": int, "scrub_radius_m": float, "force": float},
)
def sponge_wash(ex, params: dict):
    t0 = time.time()
    sponge = params.get("sponge_label", "sponge")
    origin = tm.det_xyz(ex, sponge)
    pick = grasp_se3(ex, {"keypoint_label": sponge, "force": params.get("force", 20.0),
                          "target_width": params.get("target_width")})
    if not pick.success:
        return _done("sponge_wash", t0, False, f"sponge pick failed: {pick.message}")
    scrub = constrained_scrub(ex, {**params, "target_label": params.get("dish_label", "plate")})
    orient = _orient(ex)
    ex._move_to(origin + [0, 0, float(params.get("place_dz", 0.04))], orient)
    ex.robot.open_gripper()
    ex._holding = False
    ex._move_to(origin + [0, 0, float(tm.cfg(ex, "lift_dz", params))], orient)
    return _done("sponge_wash", t0, scrub.success, f"scrub: {scrub.message}; sponge returned")


def _pour_rotvec(orient, axis_dir_xy, angle) -> list:
    """Tilt about the horizontal axis perpendicular to the pour direction,
    so the rim facing the target tips down (a pitch, whatever the yaw)."""
    d = np.array([axis_dir_xy[0], axis_dir_xy[1], 0.0])
    d /= np.linalg.norm(d) + 1e-9
    axis = np.cross([0.0, 0.0, 1.0], d)
    axis /= np.linalg.norm(axis) + 1e-9
    return (Rotation.from_rotvec(axis * angle) * Rotation.from_rotvec(orient)).as_rotvec().tolist()


@spark_skill(
    "pour",
    "Pour from a held container: move toward the target, tilt by pour_angle_rad, hold, return upright.",
    {"source_label": str, "target_label": str, "pour_angle_rad": float, "max_pour_s": float,
     "reach_m": float, "height_above_m": float},
)
def pour(ex, params: dict):
    t0 = time.time()
    if not ex._holding:
        return _done("pour", t0, False, "no container held (grasp the source first)")
    if params.get("target_fill_fraction") is not None:
        logger.info("pour: fill tracking is not in this release; pouring for max_pour_s")
    target = tm.det_xyz(ex, params.get("target_label", ""))
    here = tm.tcp(ex)
    direction = target[:2] - here[:2]
    if np.linalg.norm(direction) < 1e-6:
        direction = np.array([1.0, 0.0])
    unit = direction / np.linalg.norm(direction)
    reach = float(params.get("reach_m", min(0.08, float(np.linalg.norm(direction)))))
    spot = np.array([here[0] + unit[0] * reach, here[1] + unit[1] * reach,
                     max(here[2], target[2] + float(params.get("height_above_m", 0.20)))])
    upright = _orient(ex)
    ex._move_to(spot, upright)
    ex._move_to(spot, _pour_rotvec(upright, unit, float(params.get("pour_angle_rad", 1.4))),
                velocity=float(tm.cfg(ex, "slow_vel", params)))
    hold = float(params.get("max_pour_s", 3.0))
    deadline = time.time() + hold
    while time.time() < deadline:
        ex._check_abort()
        time.sleep(min(0.1, max(0.0, deadline - time.time())))
    ex._move_to(spot, upright)
    held = tm.holding(ex, params)
    return _done("pour", t0, held, f"poured {hold:.1f}s toward {params.get('target_label')}, held={held}")


def _sweep_line(ex, objects, container, params):
    """Waypoints from behind the objects through them into the container."""
    centroid = np.mean([o[:2] for o in objects], axis=0)
    d = np.array([float(params["dir_x"]), float(params["dir_y"])]) if params.get("dir_x") is not None \
        else container[:2] - centroid
    d = d / (np.linalg.norm(d) + 1e-9)
    back = min(float(np.dot(o[:2] - container[:2], d)) for o in objects)
    start = container[:2] + d * (back - float(params.get("approach_margin_m", 0.06)))
    end = container[:2] + d * float(params.get("overshoot_m", 0.02))
    table_z = float(tm.cfg(ex, "table_z", params) if tm.cfg(ex, "table_z", params) is not None else ex.TABLE_Z_FLOOR)
    z = table_z + float(params.get("sweep_height_m", tm.cfg(ex, "sweep_height_m", params) or 0.04))
    n = max(int(np.linalg.norm(end - start) / float(params.get("waypoint_spacing_m", 0.025))), 2)
    return [np.array([*(start + (end - start) * t), z]) for t in np.linspace(0, 1, n)]


@spark_skill(
    "sweep",
    "With a brush held, push the listed objects along one line into the target container.",
    {"target_label": str, "object_labels": str, "n_passes": int, "sweep_height_m": float,
     "waypoint_spacing_m": float, "sweep_velocity": float, "re_detect": bool, "dir_x": float, "dir_y": float},
)
def sweep(ex, params: dict, name: str = "sweep"):
    t0 = time.time()
    if not ex._holding:
        return _done(name, t0, False, "no brush held (grasp the brush first)")
    objs = tm.labels(params.get("object_labels"))
    if not objs:
        return _done(name, t0, False, "object_labels is empty")
    container = tm.det_xyz(ex, params.get("target_label", ""))
    orient = list(ex.GRASP_ORIENTATION)  # yaw 0: bristles face the sweep direction
    transit = float(params.get("transit_dz", 0.10))
    for _ in range(max(1, int(params.get("n_passes", 1)))):
        wps = _sweep_line(ex, [tm.det_xyz(ex, o) for o in objs], container, params)
        ex._move_to(wps[0] + [0, 0, transit], orient)
        for wp in wps:
            ex._move_to(wp, orient, velocity=float(params.get("sweep_velocity", 0.08)))
        ex._move_to(wps[-1] + [0, 0, transit], orient)
    held = tm.holding(ex, params)
    inside, note = True, "no re-detection"
    pipeline = getattr(ex, "_pipeline", None)
    if params.get("re_detect", True) and pipeline is not None and hasattr(pipeline, "detect"):
        dets = pipeline.merge_detections(pipeline.detect(pipeline.capture(), prompts=objs))
        radius = float(params.get("container_radius_m", 0.08))
        left = [d.label for d in dets if d.position_3d is not None
                and np.linalg.norm(np.asarray(d.position_3d)[:2] - container[:2]) > radius]
        inside, note = not left, f"outside container: {left}"
    return _done(name, t0, held and inside, f"brush held={held}; {note}")


@spark_skill("sweep_to_container", "Alias of sweep: container_label, clutter_label, max_passes, sweep_height.",
             {"container_label": str, "clutter_label": str, "max_passes": int, "sweep_height": float})
def sweep_to_container(ex, params: dict):
    p = dict(params)
    p.setdefault("target_label", p.pop("container_label", ""))
    p.setdefault("object_labels", p.pop("clutter_label", ""))
    if p.get("max_passes") is not None:
        p["n_passes"] = p.pop("max_passes")
    if p.get("sweep_height") is not None:
        p["sweep_height_m"] = p.pop("sweep_height")
    return sweep(ex, p, name="sweep_to_container")
