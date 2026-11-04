# Detection routes: detect, detect_click, detect_box, detect_approve.

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from spark_real.perception.prompt_registry import PromptCountMismatch
from spark_real.routes import state
from spark_real.routes.detect_approve import router as _approve_router
from spark_real.routes.detect_prompts import (  # noqa: F401
    _backproject_prompt_mask,
    _cal_for_camera,
    _run_box_prompt,
    _run_point_prompt,
    _save_anchor_if_requested,
    _snapshot_frame,
)
from spark_real.routes.models import (
    BoxDetectRequest,
    ClickDetectRequest,
    DetectRequest,
)
from spark_real.routes.visualization import (
    create_detection_overlay,
    create_tiled_detection_overlay,
    serialize_detections,
)

logger = logging.getLogger("spark_server")
router = APIRouter()
router.include_router(_approve_router)


def _merge_into_pending(dets, *, source):
    """Add hand-made detections to the pending set instead of REPLACING it.

    A re-annotation of the same label supersedes the old one; everything else
    accumulates.
    """
    prior = list(state.pending_detections or [])
    fresh_labels = {str(d.label).lower() for d in dets}
    kept = [d for d in prior if str(d.label).lower() not in fresh_labels]
    merged = kept + list(dets)
    state.pending_detections = merged
    if kept:
        logger.info(
            "%s: kept %d earlier detection(s) [%s] alongside the new one(s) [%s]",
            source, len(kept), ", ".join(str(d.label) for d in kept),
            ", ".join(str(d.label) for d in dets),
        )
    return merged


@router.post("/api/detect")
def detect_objects(req: DetectRequest):
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=400, content={"error": "Pipeline not initialized"}
        )
    with pipeline.activity("capturing"):
        captures = pipeline.capture()
    if not captures:
        return JSONResponse(status_code=500, content={"error": "No captures"})

    # A registered task with no operator override runs its frozen contract via
    # detect_for_task, the one implementation of the count gate and its
    # escalation ladder (relax the secondary-mask score, then the registered
    # alt_prompts, then an LLM prompt proposal).
    #
    # Operator-supplied prompts stay on the legacy path deliberately. Typing
    # prompts into the box is an experiment, not the contract; holding it to
    # the task's declared counts would abort a probe of a scene the operator
    # already knows is wrong.
    spec = pipeline.task_spec(req.instruction) if not req.prompts else None
    prompts = req.prompts or (spec.prompts if spec is not None else [])
    # A registered task declares which of its prompts can legitimately return
    # more than one mask. Without that, multi_instance defaults False and the
    # second of two identical blocks is never even segmented.
    multi = req.multi_instance or (spec is not None and bool(spec.multi_instance_prompts))

    with pipeline.activity("detecting"):
        if spec is not None:
            try:
                merged, all_detections, _res, spec, prompts = pipeline.detect_for_task(
                    captures, req.instruction, spec=spec
                )
            except PromptCountMismatch as exc:
                # The task's own policy is `abort`: a 200 with the object
                # missing sends the arm into a spiral search at z=0.35.
                return JSONResponse(
                    status_code=422,
                    content={
                        "error": str(exc),
                        "instruction": req.instruction,
                        "reason": "prompt_count_mismatch",
                    },
                )
        else:
            all_detections = pipeline.detect(
                captures,
                prompts,
                multi_instance=multi,
            )
            merged = pipeline.merge_detections(all_detections, spec=spec)
        annotated, tiled_bboxes = create_tiled_detection_overlay(
            captures,
            all_detections,
        )
    serialized = serialize_detections(merged)
    for i, s in enumerate(serialized):
        s["tiled_bbox"] = tiled_bboxes.get(s["label"])
    serialized_all = serialize_detections(all_detections)
    # Text prompts accumulate on the same terms as a click or a box: a text
    # detect for "spoon" must not silently erase a boxed "tan button" the
    # operator drew a minute earlier. Same-label results supersede, everything
    # else survives, and the approval list's discard control is there for
    # anything that should not have.
    # Do NOT rebind `merged`: the response must report what THIS call found.
    # Accumulation belongs to the pending set alone -- returning the union
    # would make a free-form detect claim credit for every earlier annotation.
    _merge_into_pending(merged, source="detect(text)")
    state.pending_captures = captures
    return {
        "detections": serialized,
        "all_detections": serialized_all,
        "annotated_image": annotated,
        "count": len(merged),
    }


@router.get("/api/detections")
async def list_detections():
    """The pending set, so the UI can show and prune what will actually run."""
    dets = state.pending_detections or []
    return {
        "detections": [
            {
                "index": i,
                "label": str(d.label),
                "confidence": float(d.confidence or 0.0),
                "camera": getattr(d, "camera", None),
                "position_3d": (
                    [float(v) for v in d.position_3d]
                    if d.position_3d is not None else None
                ),
            }
            for i, d in enumerate(dets)
        ]
    }


@router.post("/api/detections/drop")
async def drop_detections(req: dict):
    """Remove detections from the pending set by index or by label."""
    dets = list(state.pending_detections or [])
    if not dets:
        return {"ok": False, "error": "no pending detections", "remaining": 0}

    idxs = req.get("indices")
    labels = req.get("labels")
    if idxs is None and labels is None:
        return {"ok": False, "error": "pass indices or labels", "remaining": len(dets)}

    drop = set()
    for i in (idxs or []):
        try:
            i = int(i)
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(dets):
            drop.add(i)
    want = {str(x).lower() for x in (labels or [])}
    for i, d in enumerate(dets):
        if str(d.label).lower() in want:
            drop.add(i)

    kept = [d for i, d in enumerate(dets) if i not in drop]
    state.pending_detections = kept
    logger.info(
        "detections/drop: removed %d of %d (%s); %d remain",
        len(drop), len(dets),
        ", ".join(str(dets[i].label) for i in sorted(drop)) or "-",
        len(kept),
    )
    return {"ok": True, "dropped": len(drop), "remaining": len(kept)}


@router.post("/api/detect_click")
def detect_click(req: ClickDetectRequest):
    """
    Run SAM3 with a box prompt centred on the click point.
    """
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=400, content={"error": "Pipeline not initialized"}
        )

    # ONE snapshot of the whole triple, so a concurrent stream publish cannot
    # pair one camera's RGB with another's depth and a third's extrinsics.
    rgb, depth, cam_name = _snapshot_frame(pipeline)
    if rgb is None:
        return JSONResponse(status_code=500, content={"error": "No captures"})

    h_img, w_img = rgb.shape[:2]
    scale_x = w_img / req.img_width
    scale_y = h_img / req.img_height
    cx = req.x * scale_x
    cy = req.y * scale_y

    with pipeline.activity("detecting"):
        # Try true point prompt first (SAM2 instance predictor path)
        dets = _run_point_prompt(
            pipeline, rgb, depth, cx, cy, req.label, cam_name=cam_name
        )
        # Fall back to box prompt if point prompt returned nothing
        if not dets:
            box_half = 40 * max(scale_x, scale_y)
            x1 = max(0, cx - box_half)
            y1 = max(0, cy - box_half)
            x2 = min(w_img, cx + box_half)
            y2 = min(h_img, cy + box_half)
            dets = _run_box_prompt(
                pipeline, rgb, depth, x1, y1, x2, y2, req.label, cam_name=cam_name
            )
    dets = _merge_into_pending(dets, source="detect_click")
    annotated = create_detection_overlay(rgb, dets)
    # A click has no box, so anchor a small pad around it -- enough for SAM3 to
    # box-prompt with later, small enough not to swallow a neighbour.
    _pad_x = 0.03
    _pad_y = 0.03
    _nx, _ny = req.x / req.img_width, req.y / req.img_height
    saved = _save_anchor_if_requested(
        pipeline,
        req.save_for_task,
        req.label,
        cam_name,
        (max(0.0, _nx - _pad_x), max(0.0, _ny - _pad_y),
         min(1.0, _nx + _pad_x), min(1.0, _ny + _pad_y)),
    )
    return {
        "detections": serialize_detections(dets),
        "annotated_image": annotated,
        "count": len(dets),
        "click": {"x": cx, "y": cy},
        "anchor_saved": saved,
    }


@router.post("/api/detect_box")
def detect_box(req: BoxDetectRequest):
    """
    Run SAM3 with a user-drawn bounding box prompt.
    """
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=400, content={"error": "Pipeline not initialized"}
        )
    rgb, depth, cam_name = _snapshot_frame(pipeline)
    if rgb is None:
        return JSONResponse(status_code=500, content={"error": "No captures"})

    h_img, w_img = rgb.shape[:2]
    sx, sy = w_img / req.img_width, h_img / req.img_height
    x1, y1 = req.x1 * sx, req.y1 * sy
    x2, y2 = req.x2 * sx, req.y2 * sy

    with pipeline.activity("detecting"):
        dets = _run_box_prompt(
            pipeline, rgb, depth, x1, y1, x2, y2, req.label, cam_name=cam_name
        )
    dets = _merge_into_pending(dets, source="detect_box")
    annotated = create_detection_overlay(rgb, dets)
    saved = _save_anchor_if_requested(
        pipeline,
        req.save_for_task,
        req.label,
        cam_name,
        (req.x1 / req.img_width, req.y1 / req.img_height,
         req.x2 / req.img_width, req.y2 / req.img_height),
    )
    return {
        "detections": serialize_detections(dets),
        "annotated_image": annotated,
        "count": len(dets),
        "anchor_saved": saved,
    }


