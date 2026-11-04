"""Agentview + wrist SAM3 detection on LIBERO renders, and the re-detect helpers the executor calls."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import mujoco
import numpy as np
from robosuite.utils.camera_utils import get_real_depth_map

from spark_core import sim_detect
from spark_core.sam3 import SPARKPerception

log = logging.getLogger(__name__)

# Work-surface height for the punch-through correction (the value the protocol runs used).
LIBERO_TABLE_Z = -0.025

_sam3: Optional[SPARKPerception] = None


def get_sam3() -> SPARKPerception:
    """Process-wide SAM3; the first call pays the model load."""
    global _sam3
    if _sam3 is None:
        log.info("loading SAM3")
        _sam3 = SPARKPerception(sam3_threshold=0.03)
    return _sam3


@dataclass
class CameraParams:
    """
    Pose + intrinsic FOV for a MuJoCo camera.
    """
    pos: np.ndarray
    mat: np.ndarray
    fovy: float


@dataclass
class DetectionResult:
    rgb: np.ndarray
    depth: Optional[np.ndarray]
    wrist_rgb: Optional[np.ndarray]
    cam: CameraParams
    det_map: dict[str, Any] = field(default_factory=dict)
    dets: list = field(default_factory=list)


def get_camera_params(env, cam_name: str = 'agentview') -> CameraParams:
    """
    Read pose + FOV for a named MuJoCo camera.
    """
    model = env.sim.model._model
    data = env.sim.data._data
    cam_id = env.sim.model.camera_name2id(cam_name)
    return CameraParams(
        pos=data.cam_xpos[cam_id].copy(),
        mat=data.cam_xmat[cam_id].reshape(3, 3).copy(),
        fovy=float(model.cam_fovy[cam_id]),
    )


def _flip_render(arr: np.ndarray) -> np.ndarray:
    """
    Flip vertically - robosuite renders are upside down vs MuJoCo cam frame.
    """
    return arr[::-1].copy()


def _render_depth(env, raw_depth: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if raw_depth is None:
        return None
    depth = get_real_depth_map(env.sim, _flip_render(raw_depth))
    if depth.ndim == 3:
        depth = depth.squeeze(-1)
    return depth


def _detect_one_cam(sam3, rgb: np.ndarray, depth: Optional[np.ndarray],
                     prompts: list[str], cam: CameraParams,
                     cam_w: int, cam_h: int,
                     multi_instance: set[str],
                     secondary_score: Optional[float] = None):
    """One-camera SAM3 detection, backprojected through the rendered depth when there is one."""
    if depth is not None:
        kw = {} if secondary_score is None else {'secondary_score': secondary_score}
        return sim_detect.detect_rendered(
            sam3, rgb, depth, prompts, cam.pos, cam.mat, cam.fovy,
            cam_w, cam_h, multi_instance_prompts=multi_instance, table_height=LIBERO_TABLE_Z, **kw)
    return sam3.detect(rgb, prompts, cam.pos, cam.mat, cam.fovy)


def _merge_wrist_dets(dets: list, dets_w: list, verbose: bool = False) -> list:
    """
    Use wrist position when its confidence is higher; keep agentview mask.

    SE(3) grasp backprojects with agentview depth/cam params, so a
    wrist-camera mask paired with agentview cam params would produce a
    wrong point cloud.  Wrist-only detections have their mask cleared.

    Every detection that survives keeps ``position_agentview`` - the
    agentview-only backprojected centroid (or None for wrist-only
    detections).  Scene-diff / sticky comparisons against later
    agentview-only re-detections must use THIS position, not the fused
    one: the two cameras disagree by a systematic 1-2 cm on some objects
    (1.7 cm in x on one object), which otherwise reads as a phantom
    'moved' verdict on unperturbed trials.
    """
    wrist_map = {d.label: d for d in dets_w if d.position_3d is not None}
    agent_map = {d.label: d for d in dets if d.position_3d is not None}
    for d in dets:
        if d.position_3d is not None:
            d.position_agentview = d.position_3d.copy()
    for label, wd in wrist_map.items():
        if label in agent_map:
            ad = agent_map[label]
            if verbose and ad.position_3d is not None:
                log.info("wrist fuse %r: agent conf=%.3f wrist conf=%.3f apart=%.1fcm", label, ad.confidence,
                         wd.confidence, 100 * float(np.linalg.norm(np.asarray(wd.position_3d)
                                                                   - np.asarray(ad.position_3d))))
            if wd.confidence > ad.confidence:
                for i, d in enumerate(dets):
                    if d.label == label:
                        dets[i].position_3d = wd.position_3d
                        dets[i].confidence = wd.confidence
                        break
        else:
            wd.mask = None
            wd.position_agentview = None
            dets.append(wd)
    return dets


# Landmarks (fixed singletons in the scene): never multi-instance even
# when the instruction has a compound trigger. "put both X and Y in Z"
# has multiple PICK targets but one shared PLACE landmark.
_LANDMARK_TOKENS = ('basket', 'stove', 'cabinet', 'drawer', 'table',
                     'caddy', 'rack', 'microwave', 'tray', 'shelf',
                     'sink', 'counter', 'wooden cabinet', 'compartment')


def _is_landmark_prompt(p: str) -> bool:
    pl = p.lower()
    return any(tok in pl for tok in _LANDMARK_TOKENS)


def detect_scene(env, sam3, prompts: list[str], obs: dict, cfg,
                  cam: Optional[CameraParams] = None) -> DetectionResult:
    """
    Render + detect from agentview, optionally merge with wrist camera.

    Returns a fully-populated ``DetectionResult`` (rgb, depth, det_map,
    cam params).  When ``cam`` is omitted we resolve agentview ourselves.
    """
    model = env.sim.model._model
    data = env.sim.data._data
    mujoco.mj_forward(model, data)

    rgb = obs.get('agentview_image')
    if rgb is None:
        return DetectionResult(rgb=np.zeros((1, 1, 3), np.uint8),
                                depth=None, wrist_rgb=None,
                                cam=cam or get_camera_params(env))
    rgb = _flip_render(rgb)
    depth = _render_depth(env, obs.get('agentview_depth'))
    wrist_rgb_raw = obs.get('robot0_eye_in_hand_image')
    wrist_rgb_disp = _flip_render(wrist_rgb_raw) if wrist_rgb_raw is not None else None
    if cam is None:
        cam = get_camera_params(env)

    # Multi-instance detection: SAM3 returns ALL masks (not just top-1) for
    # prompts that may have several instances. Handles/knobs are always
    # multi-instance (drawer pulls, stove knobs); any prompt referenced by a
    # compound instruction ("both", "two"/"three") is flagged so the planner
    # can chain a pick-place per instance.
    instr = (getattr(cfg, '_current_instruction', '') or '').lower()
    compound_triggers = ('both', ' two ', ' three ', 'all the ', 'each ')
    has_compound = any(t in instr for t in compound_triggers)
    multi_instance = {p for p in prompts
                       if 'handle' in p or 'knob' in p
                       or (has_compound and not _is_landmark_prompt(p))}
    log.debug("multi-instance has_compound=%s engaged=%s prompts=%s", has_compound, sorted(multi_instance), prompts)
    dets = _detect_one_cam(sam3, rgb, depth, prompts, cam,
                            cfg.cam_width, cfg.cam_height, multi_instance)

    # Optional wrist camera fusion.
    wrist_depth_raw = obs.get('robot0_eye_in_hand_depth')
    if wrist_rgb_disp is not None and wrist_depth_raw is not None:
        wrist_depth = _render_depth(env, wrist_depth_raw)
        if wrist_depth is not None:
            mujoco.mj_forward(model, data)
            cam_w_id = env.sim.model.camera_name2id('robot0_eye_in_hand')
            if cam_w_id >= 0:
                wrist_cam = CameraParams(
                    pos=data.cam_xpos[cam_w_id].copy(),
                    mat=data.cam_xmat[cam_w_id].reshape(3, 3).copy(),
                    fovy=float(model.cam_fovy[cam_w_id]))
                dets_w = _detect_one_cam(
                    sam3, wrist_rgb_disp, wrist_depth, prompts, wrist_cam,
                    cfg.cam_width, cfg.cam_height, multi_instance)
                dets = _merge_wrist_dets(dets, dets_w,
                                          verbose=bool(getattr(cfg, 'verbose',
                                                                False)))

    det_map = _build_det_map(dets, cfg)
    return DetectionResult(rgb=rgb, depth=depth, wrist_rgb=wrist_rgb_disp,
                             cam=cam, det_map=det_map, dets=dets)


def _build_det_map(dets: list, cfg) -> dict:
    """Index detections by label, suffixing duplicates (e.g. ``bowl_2``)."""
    det_map: dict = {}
    label_counts: dict[str, int] = {}
    for d in dets:
        if d.position_3d is None:
            continue
        base = d.label
        if base in det_map:
            label_counts[base] = label_counts.get(base, 1) + 1
            key = f"{base}_{label_counts[base]}"
        else:
            key = base
        det_map[key] = d
        if cfg.verbose:
            p = d.position_3d
            log.info("%s conf=%.3f pos=(%.3f,%.3f,%.3f)", key, d.confidence, p[0], p[1], p[2])
    _drawer_part_of_cabinet(det_map)
    return det_map


def _drawer_part_of_cabinet(det_map: dict) -> None:
    """A drawer is part of a cabinet: a 'drawer' mask more than 15 cm from every cabinet and drawer-handle
    detection landed on another object (the goal-suite wine rack, 30 cm off; 'put the X on the top of the
    drawer' then placed onto the rack, real-PRO goal 2/4 position 28 %/26 %). Point it at the nearest
    cabinet. In place."""
    xy = lambda d: np.asarray(d.position_3d, float)[:2]
    cab = [d for k, d in det_map.items() if 'cabinet' in k.lower()]
    if not cab:
        return
    anchors = cab + [d for k, d in det_map.items() if 'drawer' in k.lower() and 'handle' in k.lower()]
    for k in list(det_map):
        kl = k.lower()
        if 'drawer' not in kl or 'handle' in kl or 'cabinet' in kl:
            continue
        p = xy(det_map[k])
        if all(np.linalg.norm(p - xy(a)) > 0.15 for a in anchors):
            det_map[k] = min(cab, key=lambda d: np.linalg.norm(p - xy(d)))


def redetect_wrist(env, sam3, prompts: list[str], cfg,
                    all_instances: bool = False) -> Optional[DetectionResult]:
    """
    Re-render + re-detect from the eye-in-hand camera only.

    The agentview loses a target the moment the gripper hovers over it;
    the wrist camera sees exactly that region.  Same contract as
    redetect_agentview: None on render failure or no detections.
    """
    model = env.sim.model._model
    data = env.sim.data._data
    obs2, _, _, _ = env.step(np.zeros(7))
    rgb2 = obs2.get('robot0_eye_in_hand_image')
    depth_raw = obs2.get('robot0_eye_in_hand_depth')
    if rgb2 is None or depth_raw is None:
        return None
    rgb2 = _flip_render(rgb2)
    depth2 = _render_depth(env, depth_raw)
    if depth2 is None:
        return None
    mujoco.mj_forward(model, data)
    cam_id = env.sim.model.camera_name2id('robot0_eye_in_hand')
    if cam_id < 0:
        return None
    cam = CameraParams(pos=data.cam_xpos[cam_id].copy(),
                       mat=data.cam_xmat[cam_id].reshape(3, 3).copy(),
                       fovy=float(model.cam_fovy[cam_id]))
    multi = set(prompts) if all_instances else set()
    # wrist-verify asks for generic phrases and lets the VLM choose among the instances (its own filter
    # keeps conf >= 0.2): the 0.50 secondary floor left two candidates and dropped the ketchup that SAM3
    # scores level with the bbq sauce (real-PRO object 4 swap, 14 %)
    kw = ({'secondary_score': 0.2}
          if all_instances and getattr(cfg, 'wrist_verify', False) else {})
    dets = sim_detect.detect_rendered(
        sam3, rgb2, depth2, prompts, cam.pos, cam.mat, cam.fovy,
        cfg.cam_width, cfg.cam_height,
        multi_instance_prompts=multi, table_height=LIBERO_TABLE_Z, **kw)
    if not dets:
        return None
    det_map = {d.label: d for d in dets if d.position_3d is not None}
    return DetectionResult(rgb=rgb2, depth=depth2, wrist_rgb=None,
                             cam=cam, det_map=det_map, dets=dets)


def redetect_agentview(env, sam3, prompts: list[str], cfg,
                        timing: Optional[dict] = None,
                        all_instances: bool = False,
                        render_only: bool = False,
                        secondary_score: Optional[float] = None,
                        ) -> Optional[DetectionResult]:
    """
    Re-render + re-detect from agentview only (recovery-loop helper).

    No wrist fusion (the recovery scenario doesn't need it).  Returns
    ``None`` if rendering fails or there are no detections.

    ``timing`` (when supplied) receives ``t_capture`` - the wall-clock
    time right after the RGB/depth frame is in hand and before the SAM3
    call - so event-capture callers can split capture latency from
    verdict latency.

    ``all_instances=True`` flags every prompt multi-instance so SAM3
    returns EVERY mask above threshold, not just top-1.  The
    ``DetectionResult.dets`` list then carries all instances (several
    entries may share a label) for position-sticky association; the
    ``det_map`` keeps its legacy one-entry-per-label shape.
    """
    import time as _time
    model = env.sim.model._model
    data = env.sim.data._data
    if render_only:
        # Render the current state without stepping: a zero-action step freezes an OSC arm for
        # one control step per detection.
        rgb2, raw2 = env.sim.render(width=cfg.cam_width, height=cfg.cam_height,
                                    camera_name='agentview', depth=True)
    else:
        obs2, _, _, _ = env.step(np.zeros(7))
        rgb2, raw2 = obs2.get('agentview_image'), obs2.get('agentview_depth')
    if rgb2 is None:
        return None
    rgb2 = _flip_render(rgb2)
    depth2 = _render_depth(env, raw2)
    if depth2 is None:
        return None
    if timing is not None:
        timing['t_capture'] = _time.time()
    mujoco.mj_forward(model, data)
    cam = get_camera_params(env)
    multi = set(prompts) if all_instances else set()
    # With relation-bind every instance is a candidate (the tracker seeds on the relation-bound
    # one); the 0.50 secondary floor dropped most of a row of look-alike items
    kw = ({'secondary_score': 0.15}
          if all_instances and getattr(cfg, 'relation_bind', False) else {})
    if secondary_score is not None:
        kw = {'secondary_score': secondary_score}
    dets = sim_detect.detect_rendered(
        sam3, rgb2, depth2, prompts, cam.pos, cam.mat, cam.fovy,
        cfg.cam_width, cfg.cam_height,
        multi_instance_prompts=multi, table_height=LIBERO_TABLE_Z, **kw)
    if not dets:
        return None
    det_map = {d.label: d for d in dets if d.position_3d is not None}
    return DetectionResult(rgb=rgb2, depth=depth2, wrist_rgb=None,
                             cam=cam, det_map=det_map, dets=dets)
