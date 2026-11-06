"""POST /api/detect_approve: detect for operator approval, with merged overlay and depth diagnostics."""

import asyncio
import copy
from collections import defaultdict

import cv2
import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from spark_real.perception.prompt_registry import PromptCountMismatch
from spark_real.routes import state, streaming
from spark_real.routes.models import (
    DetectApproveRequest,
)
from spark_real.routes.visualization import (
    depth_to_colormap,
    draw_bbox_overlay,
    draw_grasp_preview,
    encode_image_b64,
    serialize_detections,
)

router = APIRouter()


@router.post("/api/detect_approve")
async def detect_for_approval(req: DetectApproveRequest):
    """
    Detect objects and return annotated image for user approval.
    """
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=400, content={"error": "Pipeline not initialized"}
        )
    loop = asyncio.get_event_loop()

    def _detect_sync():
        with pipeline.activity("capturing"):
            captures = pipeline.capture()
        if not captures:
            return None

        # IR swap for low-light
        if req.use_ir:
            for cam_name in ("sideview", "birdview"):
                cd = captures.get(cam_name)
                if cd is None or cd.get("rgb") is None:
                    continue
                target_hw = cd["rgb"].shape[:2]
                try:
                    ir_rgb, _ = streaming.capture_single_camera(
                        cam_name,
                        need_depth=False,
                        ir_as_rgb=True,
                    )
                    if ir_rgb is None:
                        continue
                    if ir_rgb.shape[:2] != target_hw:
                        ir_rgb = cv2.resize(
                            ir_rgb,
                            (target_hw[1], target_hw[0]),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    cd["rgb"] = ir_rgb
                except Exception:
                    pass

        scene_img = None
        scene_cam = None
        for cam_name in ["birdview", "sideview"]:
            cam_data = captures.get(cam_name)
            if cam_data and cam_data.get("rgb") is not None:
                scene_img = cam_data["rgb"]
                scene_cam = cam_name
                break

        # Registry first, LLM second. A registered task has its prompts and
        # its instance ordering pinned in version control, so this both skips
        # an LLM call and removes the run-to-run prompt drift that
        # generate_prompts introduces at temperature 0.3, which makes a
        # cached BT unsafe.
        spec = pipeline.task_spec(req.instruction)
        prompts = req.prompts
        if prompts is None and spec is not None:
            prompts = spec.prompts
        elif prompts is None and pipeline._planner is not None and scene_img is not None:
            try:
                with pipeline.activity("planning"):
                    prompts = pipeline._planner.generate_prompts(
                        req.instruction,
                        scene_img,
                    )
            except Exception:
                prompts = pipeline._extract_prompts(req.instruction)
        elif prompts is None:
            prompts = pipeline._extract_prompts(req.instruction)

        with pipeline.activity("detecting"):
            # Registered task, prompts not overridden by the operator -> the
            # frozen contract, which means the shared detect_for_task ladder
            # (relax -> alt_prompts -> LLM proposal). Anything else keeps the
            # legacy pass: an operator's own prompt list is an experiment, not
            # a contract to abort on.
            if spec is not None and req.prompts is None:
                merged, all_dets, _res, spec, prompts = pipeline.detect_for_task(
                    captures, req.instruction, spec=spec
                )
            else:
                all_dets = pipeline.detect(captures, prompts, multi_instance=True)
                merged = pipeline.merge_detections(all_dets, spec=spec)

        # Per-camera overlays
        per_cam_overlays = {}
        cam_names_found = []
        for cam_name, cam_data in captures.items():
            rgb_c = cam_data.get("rgb")
            if rgb_c is None:
                continue
            cam_d = [d for d in all_dets if getattr(d, "camera", None) == cam_name]
            if not cam_d:
                continue
            cam_names_found.append(cam_name)
            cam_d_numbered = [copy.copy(d) for d in cam_d]
            lbl_grp = defaultdict(list)
            for d in cam_d_numbered:
                lbl_grp[d.label].append(d)
            for lbl, ds in lbl_grp.items():
                if len(ds) > 1:
                    for idx, d in enumerate(ds):
                        d.label = f"{lbl} {idx + 1}"
            per_cam_overlays[cam_name] = {
                "mask": encode_image_b64(
                    pipeline._annotate_image(rgb_c, cam_d_numbered)
                ),
                "bbox": encode_image_b64(draw_bbox_overlay(rgb_c, cam_d_numbered)),
                "grasp": encode_image_b64(draw_grasp_preview(rgb_c, cam_d_numbered)),
                "count": len(cam_d_numbered),
                "detections": serialize_detections(cam_d_numbered),
            }

        # Scene camera raw detections
        cam_dets = [d for d in all_dets if getattr(d, "camera", None) == scene_cam]
        raw_dets = cam_dets if cam_dets else merged
        raw_dets_numbered = [copy.copy(d) for d in raw_dets]
        raw_grp = defaultdict(list)
        for d in raw_dets_numbered:
            raw_grp[d.label].append(d)
        for lbl, ds in raw_grp.items():
            if len(ds) > 1:
                for idx, d in enumerate(ds):
                    d.label = f"{lbl} {idx + 1}"

        mask_img = pipeline._annotate_image(scene_img, raw_dets_numbered)
        bbox_img = draw_bbox_overlay(scene_img, raw_dets_numbered)
        grasp_img = draw_grasp_preview(scene_img, raw_dets_numbered)

        # Merged overlays with cross-camera projection
        scene_cal = (
            captures.get(scene_cam, {}).get("calibration") if scene_cam else None
        )
        merged_for_overlay = _build_merged_overlay(
            merged,
            scene_cam,
            scene_cal,
            scene_img,
        )
        mask_img_merged = pipeline._annotate_image(
            scene_img,
            merged_for_overlay,
        )
        bbox_img_merged = draw_bbox_overlay(scene_img, merged_for_overlay)
        grasp_img_merged = draw_grasp_preview(scene_img, merged_for_overlay)

        # Depth diagnostics
        depth_diag = _build_depth_diag(captures, all_dets)

        return {
            "prompts": prompts,
            "detections": serialize_detections(merged),
            "raw_detections": serialize_detections(raw_dets_numbered),
            "raw_count": len(raw_dets_numbered),
            "merged_count": len(merged),
            "mask_image": encode_image_b64(mask_img),
            "bbox_image": encode_image_b64(bbox_img),
            "grasp_image": encode_image_b64(grasp_img),
            "mask_merged": encode_image_b64(mask_img_merged),
            "bbox_merged": encode_image_b64(bbox_img_merged),
            "grasp_merged": encode_image_b64(grasp_img_merged),
            "per_cam": per_cam_overlays,
            "cam_names": cam_names_found,
            "captures": captures,
            "all_detections": all_dets,
            "merged": merged,
            "depth_diag": depth_diag,
        }

    try:
        result = await loop.run_in_executor(None, _detect_sync)
    except PromptCountMismatch as exc:
        # Same contract as /api/plan: the task's own on_mismatch policy said
        # abort, so the operator gets the counts, not a green approval screen
        # with the object silently absent.
        return JSONResponse(
            status_code=422,
            content={
                "error": str(exc),
                "instruction": req.instruction,
                "reason": "prompt_count_mismatch",
            },
        )
    if result is None:
        return JSONResponse(status_code=500, content={"error": "No captures"})

    state.pending_detections = result["merged"]
    state.pending_captures = result["captures"]
    state.pending_instruction = req.instruction
    state.pending_prompts = result["prompts"]

    return {
        k: v
        for k, v in result.items()
        if k not in ("captures", "all_detections", "merged")
    }


def _build_merged_overlay(merged, scene_cam, scene_cal, scene_img):
    """
    Build merged overlay list, projecting cross-camera detections.
    """
    merged_for_overlay = []
    for d in merged:
        if getattr(d, "camera", None) == scene_cam:
            merged_for_overlay.append(d)
        elif d.position_3d is not None and scene_cal is not None:
            try:
                pos = np.array(d.position_3d[:3])
                R = scene_cal.extrinsic[:3, :3]
                t = scene_cal.extrinsic[:3, 3]
                p_cam = R.T @ (pos - t)
                if p_cam[2] > 0.01:
                    u = scene_cal.fx * p_cam[0] / p_cam[2] + scene_cal.cx
                    v = scene_cal.fy * p_cam[1] / p_cam[2] + scene_cal.cy
                    h_img, w_img = scene_img.shape[:2]
                    if 0 <= u < w_img and 0 <= v < h_img:
                        proj = copy.copy(d)
                        proj.centroid_2d = (float(u), float(v))
                        proj.mask = None
                        merged_for_overlay.append(proj)
            except Exception:
                pass
        else:
            merged_for_overlay.append(d)
    return merged_for_overlay


def _build_depth_diag(captures, all_dets):
    """
    Build per-camera depth diagnostics for the detect_approve response.
    """
    depth_diag = {}
    for cam_name, cam_data in captures.items():
        depth_arr = cam_data.get("depth")
        cal = cam_data.get("calibration")
        if depth_arr is None or cal is None:
            continue
        depth_m = depth_arr.astype(np.float32)
        if depth_m.max() > 100.0:
            depth_m = depth_m / 1000.0
        H, W = depth_m.shape
        R_cw = cal.extrinsic[:3, :3]
        t_cw = cal.extrinsic[:3, 3]

        # Table samples at canonical positions
        table_samples = []
        for uv_label, uu, vv in [
            ("center", W // 2, H // 2),
            ("upper-left", W // 4, H // 4),
            ("lower-right", 3 * W // 4, 3 * H // 4),
        ]:
            if not (0 <= vv < H and 0 <= uu < W):
                continue
            dz = float(depth_m[vv, uu])
            if not (0.05 < dz < 5.0):
                continue
            xc = (uu - cal.cx) * dz / cal.fx
            yc = (vv - cal.cy) * dz / cal.fy
            pw = R_cw @ np.array([xc, yc, dz]) + t_cw
            table_samples.append(
                {
                    "where": uv_label,
                    "u": uu,
                    "v": vv,
                    "depth_m": dz,
                    "world_z_m": float(pw[2]),
                }
            )

        per_det = []
        cam_d = [d for d in all_dets if getattr(d, "camera", None) == cam_name]
        for d in cam_d:
            m = d.mask
            if m is None or m.shape != depth_m.shape:
                continue
            valid = m & (depth_m > 0.05) & (depth_m < 5.0)
            n = int(valid.sum())
            if n < 20:
                continue
            ys, xs = np.where(valid)
            zs = depth_m[ys, xs]
            xc = (xs - cal.cx) * zs / cal.fx
            yc = (ys - cal.cy) * zs / cal.fy
            pts_cam = np.stack([xc, yc, zs], axis=1)
            pts_world = pts_cam @ cal.extrinsic[:3, :3].T + cal.extrinsic[:3, 3]
            wz = pts_world[:, 2]
            pcts = np.percentile(wz, [5, 25, 50, 75, 95]).tolist()
            per_det.append(
                {
                    "label": d.label,
                    "n_pixels": n,
                    "world_z_p25_m": pcts[1],
                    "world_z_p50_m": pcts[2],
                    "world_z_range_m": float(wz.max() - wz.min()),
                }
            )

        depth_diag[cam_name] = {
            "depth_colormap": encode_image_b64(depth_to_colormap(depth_arr)),
            "table_samples": table_samples,
            "detections": per_det,
        }
    return depth_diag
