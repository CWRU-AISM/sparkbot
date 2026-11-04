"""Cross-camera merge of detections and slot enrichment."""

import copy
import logging
import os
import re
from collections import defaultdict
from typing import List

import numpy as np

from spark_core import dedup
from spark_core.sam3 import ObjectDetection
from spark_real.perception.prompt_registry import (
    TaskSpec,
    base_label,
    resolve_labels,
)
from spark_real.perception.slot_detector import detect_slots
from spark_real.perception_settings import MAX_ON_TABLE_HEIGHT_M

logger = logging.getLogger(__name__)


class MergeMixin:
    """Cross-camera merge of detections and slot enrichment."""

    def merge_detections(
        self,
        detections: List[ObjectDetection],
        spec: "TaskSpec" = None,
    ) -> List[ObjectDetection]:
        """
        Merge detections across cameras. Birdview is the primary authority.

        1. Start with birdview (top-down, sees everything) as the base set
        2. For each sideview detection, find closest same-label birdview match
        3. Matched: keep birdview detection (correct pixel coords for overlay)
        4. Unmatched: add only if birdview has no instance of that label
        5. Number multi-instance, normalize Z
        6. Uses copy.copy so original detections aren't mutated

        `spec` is an optional :class:`TaskSpec` from the prompt registry. When
        given it replaces confidence-rank instance numbering with the task's
        declared geometric order and honours its per-instance-Z setting, so a
        cached behaviour tree's "<label> 1" refers to the same physical object
        on every run. Omitting it preserves the legacy behaviour exactly.
        """
        by_camera = defaultdict(list)
        for det in detections:
            by_camera[getattr(det, "camera", "unknown")].append(det)

        # Primary camera: birdview > sideview (birdview sees entire table).
        # SPARK_PRIMARY_CAM overrides the preference order at runtime (e.g.
        # =sideview to trust sideview's hand-eye during a calibration test);
        # unset restores the birdview-first default. Purely a debug toggle.
        _pref = os.environ.get("SPARK_PRIMARY_CAM", "").strip().lower()
        cam_pref = [_pref, "birdview", "sideview"] if _pref else ["birdview", "sideview"]
        primary_cam = None
        for cam in cam_pref:
            if cam in by_camera and by_camera[cam]:
                primary_cam = cam
                break
        if primary_cam is None and by_camera:
            primary_cam = next(iter(by_camera))
        if not primary_cam:
            return []

        # Base = copies of primary camera detections.
        # position_agentview keeps the primary camera's own backprojection
        # even after the sideview Z-override / wrist fusion rewrites
        # position_3d. Any later same-camera displacement measurement (sticky
        # gate, redetect teleport gate, scene diff) must compare against this,
        # not the fused position: the cameras disagree by a systematic 1-2 cm,
        # which otherwise reads as a phantom 'moved'. position_3d is
        # re-materialized as a fresh array because copy.copy is shallow.
        merged = []
        for d in by_camera[primary_cam]:
            if d.position_3d is None:
                continue
            c = copy.copy(d)
            c.position_3d = np.array(d.position_3d, dtype=float)
            c.position_agentview = np.array(d.position_3d, dtype=float)
            merged.append(c)
        logger.info(
            "Merge: primary=%s (%d dets), secondary cameras: %s",
            primary_cam,
            len(merged),
            [f"{c}({len(ds)})" for c, ds in by_camera.items() if c != primary_cam],
        )

        # Add sideview-only detections that birdview missed entirely.
        primary_labels = {d.label for d in merged}
        for cam, dets_c in by_camera.items():
            if cam == primary_cam:
                continue
            for d in dets_c:
                if d.label not in primary_labels and d.position_3d is not None:
                    c = copy.copy(d)
                    c.position_3d = np.array(d.position_3d, dtype=float)
                    # Secondary-only detection: its "same-source" anchor is
                    # its own camera's backprojection at merge time.
                    c.position_agentview = np.array(d.position_3d, dtype=float)
                    merged.append(c)
                    primary_labels.add(d.label)
                    logger.info(
                        "Merge: added %s-only detection '%s' " "(conf=%.2f, z=%.1fmm)",
                        cam,
                        d.label,
                        d.confidence,
                        d.position_3d[2] * 1000 if d.position_3d is not None else 0,
                    )

        # Birdview owns POSITION: it has the most accurate hand-eye.
        #
        # OBB enrichment from secondary cameras: birdview's mask of a
        # flat container (tray, plate) viewed straight-down can be roughly
        # square (low AR) or noisy because the container's vertical sides
        # project to a thin rim and SAM3's mask edges wobble at low
        # contrast. Sideview sees the footprint at an angle and gets a
        # cleaner OBB. For PLACEMENT we want yaw alignment, so keep
        # birdview position but enrich orientation_angle/aspect_ratio/
        # obb_minor_m from whichever camera has the strongest AR for the
        # same label.
        # Minimum SAM3 confidence for a secondary-camera detection to be
        # trusted for OBB enrichment; below this is treated as a noise mask.
        OBB_ENRICH_MIN_CONF = 0.30

        # Cameras eligible to PROVIDE OBB info. Wrist is excluded: its
        # hand-eye is the noisiest so its world-frame orientation can't be
        # trusted even at high SAM3 confidence. OBB enrichment is restricted
        # to a single camera (birdview) because instance numbers are
        # per-camera, so the same label string can refer to a different
        # object across cameras and cross-wire the grasp. Single-camera OBB
        # keeps the instance IDs self-consistent.
        OBB_ENRICH_TRUSTED_CAMS = {"birdview"}

        # Max world-XY distance (meters) between a merged instance and a
        # trusted-camera detection for that detection's OBB to enrich the
        # instance. Prevents applying one birdview OBB to a different
        # same-label instance across the table (two spoons at different
        # angles each get their own nearest birdview OBB). Env-overridable.
        OBB_ENRICH_MAX_XY_M = float(
            os.environ.get("OBB_ENRICH_MAX_XY_M", "0.12")  # 12 cm default gate
        )

        # Per-instance OBB enrichment via greedy nearest-neighbor matching:
        # for each trusted donor camera, match each merged instance to the
        # nearest same-label donor detection by world XY (closest pair first,
        # each donor consumed by at most one instance). Instances with no
        # donor inside the gate keep their own OBB; the AR-improvement gate
        # below means a donor only wins when its OBB is stronger.
        for cam, dets_c in by_camera.items():
            if cam == primary_cam or cam not in OBB_ENRICH_TRUSTED_CAMS:
                continue
            donors = [
                x
                for x in dets_c
                if x.position_3d is not None and float(x.confidence or 0.0) >= OBB_ENRICH_MIN_CONF
            ]
            if not donors:
                continue
            # Candidate (distance, merged_idx, donor_idx) pairs within gate,
            # same label only.
            pairs = []
            for mi, d in enumerate(merged):
                if d.position_3d is None:
                    continue
                m_xy = np.array(d.position_3d[:2], dtype=float)
                for di, cand in enumerate(donors):
                    if cand.label != d.label:
                        continue
                    dist = float(np.linalg.norm(m_xy - np.array(cand.position_3d[:2], dtype=float)))
                    if dist <= OBB_ENRICH_MAX_XY_M:
                        pairs.append((dist, mi, di))
            # Greedy nearest-first assignment; each merged instance and each
            # donor used at most once.
            pairs.sort(key=lambda p: p[0])
            used_m = set()
            used_d = set()
            for dist, mi, di in pairs:
                if mi in used_m or di in used_d:
                    continue
                used_m.add(mi)
                used_d.add(di)
                d = merged[mi]
                cand = donors[di]
                cand_ar = float(cand.aspect_ratio or 1.0)
                # Only enrich when the donor's OBB is stronger (higher AR)
                # than what the instance already has.
                if cand_ar <= float(d.aspect_ratio or 1.0):
                    continue
                logger.info(
                    "  OBB-enrich '%s': orient %.3f->%.3f rad, AR %.2f->%.2f "
                    "(from %s @ %.1fcm; position stays %s)",
                    d.label,
                    float(d.orientation_angle or 0.0),
                    float(cand.orientation_angle or 0.0),
                    float(d.aspect_ratio or 1.0),
                    cand_ar,
                    cam,
                    dist * 100.0,
                    d.camera,
                )
                d.orientation_angle = cand.orientation_angle
                d.aspect_ratio = cand_ar
                # The fusion gate's axis_trust scored the DONOR's mask axis, so
                # it has to travel with the axis it scored. Without this the
                # instance wears one camera's angle and another camera's trust,
                # and a wrist can be aimed off an axis nothing vouched for.
                # Donor unmeasured -> clear it: absent means unknown, and the
                # base's own trust no longer describes the axis being carried.
                d.axis_trust = getattr(cand, "axis_trust", None)
                # world-frame major axis follows the donor's OBB when present.
                if getattr(cand, "world_major_axis_rad", None) is not None:
                    d.world_major_axis_rad = cand.world_major_axis_rad
                # Keep the primary's width if the enriching camera has none:
                # sideview often reports obb_minor_m=0.
                cand_obb_minor = float(cand.obb_minor_m or 0.0)
                if cand_obb_minor > 0:
                    d.obb_minor_m = cand_obb_minor

        # Z override from secondary camera: when the two cameras disagree on
        # height, the primary's depth is wrong (Kinect IR averages table +
        # object pixels for small/shiny/translucent surfaces from above,
        # producing Z anywhere from table level to mid-object) and sideview
        # resolves height much better. No "near table" gate; compare the two
        # cameras directly. Bidirectional: birdview over-reads as easily as it
        # under-reads (a tray measured at z=+0.071 against a table at ~-0.27,
        # 34 cm in the air).
        Z_ELEVATION_THRESH = 0.05  # cameras must disagree by >=50mm
        for d in merged:
            if d.position_3d is None:
                continue
            pri_z = float(d.position_3d[2])
            for cam, dets_c in by_camera.items():
                if cam == primary_cam:
                    continue
                # A detection implausibly far above the table cannot be right
                # for an object resting on it, so widen the search: accept a
                # base-label match at any confidence (sideview has called a
                # tray 'purple tray grip' at 0.12, which an exact-label 0.20
                # gate rejected).
                implausible = pri_z > self.table_z_floor() + MAX_ON_TABLE_HEIGHT_M
                same = [
                    x
                    for x in dets_c
                    if x.label == d.label
                    and x.position_3d is not None
                    and float(x.confidence or 0) >= 0.20
                ]
                if not same and implausible:
                    base = str(d.label).split()[0].lower()
                    same = [
                        x
                        for x in dets_c
                        if x.position_3d is not None
                        and str(x.label).lower().startswith(base)
                    ]
                    if same:
                        logger.warning(
                            "  '%s' at z=%.0fmm is >%.0fmm above the table; "
                            "accepting a loose '%s' match from %s to correct it",
                            d.label, pri_z * 1000,
                            MAX_ON_TABLE_HEIGHT_M * 1000, base, cam,
                        )
                if not same:
                    continue
                sec = max(same, key=lambda x: float(x.confidence or 0))
                sec_z = float(sec.position_3d[2])
                if abs(sec_z - pri_z) > Z_ELEVATION_THRESH:
                    logger.info(
                        "  Z-override '%s': %s z=%.0fmm -> %s z=%.0fmm "
                        "(%+.0fmm, sideview wins)",
                        d.label,
                        primary_cam,
                        pri_z * 1000,
                        cam,
                        sec_z * 1000,
                        (sec_z - pri_z) * 1000,
                    )
                    d.position_3d[2] = sec_z

        # Per-camera detection counts (kept for diagnostics).
        for cam in list(by_camera.keys()):
            if cam != primary_cam:
                logger.info(
                    "  Secondary camera %s contributed %d detection(s) "
                    "(used for OBB enrichment + Z override; position XY from %s)",
                    cam,
                    len(by_camera[cam]),
                    primary_cam,
                )

        # Filter low confidence. The floor is low because cloth sub-parts
        # (shirt collar, hem) ground at very low SAM3 confidence on plain
        # fabric while still landing in the correct region of the body
        # mask. The 3D backprojection + downstream verify catch any
        # mislocalised low-conf detections, so the low floor buys
        # part-grounding on fabric without producing false placements.
        MIN_MERGED_CONFIDENCE = 0.05
        before_filter = len(merged)
        merged = [d for d in merged if d.confidence >= MIN_MERGED_CONFIDENCE]
        if len(merged) < before_filter:
            logger.info(
                "Dropped %d merged detections below %.0f%% confidence",
                before_filter - len(merged),
                MIN_MERGED_CONFIDENCE * 100,
            )

        # Cross-label spatial dedup: SAM3 often detects the same object for
        # multiple prompts (e.g., "knife handle" mask on a spoon). If two
        # detections from different labels are the same physical object,
        # keep the higher-confidence one.
        #
        # "Same physical object" is decided by perception.dedup, not by XY
        # proximity alone: an object resting in a container is centimetres from
        # the container's centroid, and deleting it here makes
        # inside(obj, container) unverifiable. See perception/dedup.py.
        CROSS_LABEL_DEDUP_M = 0.05  # 5cm; same physical object
        aware = dedup.containment_aware()
        before_dedup = len(merged)
        deduped = []
        for d in merged:
            if d.position_3d is None:
                deduped.append(d)
                continue
            duplicate = False
            for i, existing in enumerate(deduped):
                if existing.position_3d is None:
                    continue
                # Part-of guard: never dedup a part against its parent
                # ('mug handle' vs 'mug'). The parent mask includes the
                # part so their centroids collide, which would otherwise
                # delete the very label the planner needs to grasp.
                if d.label in existing.label or existing.label in d.label:
                    continue
                if dedup.same_object_3d(d, existing, CROSS_LABEL_DEDUP_M, aware=aware):
                    if d.confidence > existing.confidence:
                        logger.info(
                            "  Cross-label dedup: %s (%.0f%%) replaces %s (%.0f%%) at same pos",
                            d.label,
                            d.confidence * 100,
                            existing.label,
                            existing.confidence * 100,
                        )
                        deduped[i] = d
                    else:
                        logger.info(
                            "  Cross-label dedup: dropping %s (%.0f%%), kept %s (%.0f%%)",
                            d.label,
                            d.confidence * 100,
                            existing.label,
                            existing.confidence * 100,
                        )
                    duplicate = True
                    break
            if not duplicate:
                deduped.append(d)
        merged = deduped
        if len(merged) < before_dedup:
            logger.info("Cross-label dedup removed %d duplicate(s)", before_dedup - len(merged))

        # Number multi-instance labels and normalize Z.
        # With a task spec the numbering follows the registry's declared
        # geometric order (a total order, so "<label> 1" is the same physical
        # object on every run). Without one it falls back to confidence rank,
        # which is arbitrary for identical objects, so a cached BT must always
        # run under a spec.
        if spec is not None:
            resolution = resolve_labels(spec, merged)
            # Published for detect_for_task, which needs the count gate but
            # cannot get it from the return value without breaking every other
            # caller's signature. Re-resolving instead would be wrong once Z
            # flattening below has run, since order_by may reference world_z.
            self._last_resolution = resolution
            merged = resolution.detections + resolution.extras
            if not resolution.ok:
                logger.warning("merge_detections: %s", resolution.describe())
        else:
            label_counts = defaultdict(list)
            for d in merged:
                label_counts[d.label].append(d)

            for label, dets in label_counts.items():
                if len(dets) > 1:
                    dets.sort(key=lambda d: d.confidence, reverse=True)
                    for i, d in enumerate(dets):
                        d.label = f"{label} {i + 1}"

        # Z flattening across same-label instances: two views of one flat
        # object can disagree on height, but flattening is fatal for a stack.
        # per_instance_z opts a task out; the default applies to callers with
        # no spec.
        if spec is not None and spec.per_instance_z:
            logger.debug("merge_detections: per-instance Z kept for task %r", spec.task)
        else:
            flatten_groups = defaultdict(list)
            for d in merged:
                flatten_groups[base_label(d.label)].append(d)
            for label, dets in flatten_groups.items():
                if len(dets) < 2:
                    continue
                zs = [d.position_3d[2] for d in dets if d.position_3d is not None]
                if not zs:
                    continue
                median_z = float(np.median(zs))
                for d in dets:
                    if d.position_3d is not None:
                        d.position_3d[2] = median_z

        for d in merged:
            pos_str = (
                f"({d.position_3d[0]:.3f},{d.position_3d[1]:.3f},{d.position_3d[2]:.3f})"
                if d.position_3d is not None
                else "N/A"
            )
            logger.info(
                "  Merged: %s -> cam=%s conf=%.2f pos=%s",
                d.label,
                getattr(d, "camera", "?"),
                d.confidence,
                pos_str,
            )
        return merged

    def _enrich_with_slots(self, merged, captures):
        """
        For container-like merged detections, compute slot poses +
        world-frame major-axis orientation. Mutates merged in place.

        Slot detection requires the source camera's depth + intrinsics
        + hand-eye extrinsic. Birdview is preferred (top-down view +
        best calibration); if a container was merged from another
        camera, we'll still try slot detection from THAT camera's
        depth, but the result is most reliable on birdview.
        """
        for d in merged:
            base_label = re.sub(r"\s+\d+$", "", str(d.label or "")).strip().lower()
            if base_label not in self.SLOT_CONTAINER_LABELS:
                continue
            cam_name = getattr(d, "camera", None)
            cap = captures.get(cam_name) if cam_name else None
            if cap is None:
                continue
            mask = d.mask
            depth = cap.get("depth")
            cal = cap.get("calibration")
            if mask is None or depth is None or cal is None:
                continue
            T_cam_to_base = getattr(cal, "extrinsic", None)
            try:
                K = cal.intrinsic_matrix
            except Exception:
                continue
            try:
                # depth here is cap["depth"], already corrected by capture()'s
                # _scaled (z*scale+offset), so pass depth_scale=1.0 to avoid
                # double-applying the per-camera depth correction.
                slots = detect_slots(
                    rgb=cap.get("rgb"),
                    depth_m=depth,
                    tray_mask=mask,
                    K=K,
                    T_cam_to_base=T_cam_to_base,
                    depth_scale=1.0,
                )
            except Exception as exc:
                logger.warning("detect_slots(%s) raised: %s", d.label, exc)
                continue
            d.slots = slots
            # World-frame major-axis orientation from the existing
            # world-XY PCA OBB code already populated d.orientation_angle.
            # For container labels, orientation_angle from
            # _detect_with_rendered_depth is in WORLD frame (radians
            # from world +X) when the world-XY PCA path fires, which
            # it does for top-down containers on the table.
            d.world_major_axis_rad = float(d.orientation_angle or 0.0)
            n_slots = len(slots) if slots else 0
            modes = sorted({s.get("mode", "?") for s in (slots or [])})
            logger.info(
                "  Slots(%s): %d slot(s) via %s, world_axis=%.3f rad",
                d.label,
                n_slots,
                "+".join(modes) if modes else "none",
                d.world_major_axis_rad or 0.0,
            )
