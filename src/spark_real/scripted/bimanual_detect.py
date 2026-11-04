"""
Standalone ZED + SAM3 detection for the scripted bimanual demos.

Captures one ZED frame, runs SAM3 on text prompts, and writes every
detection's position in BOTH arm base frames (from
<calibrations>/calibration_bimanual.json) to
<SPARK_OUTPUT_DIR>/detections/detections.json. Positions are jaw-tip
targets: the calibration was solved with the jaw tip on the board corners.
Stop the server first: the ZED is exclusive.

Usage:
    python -m spark_real.scripted.bimanual_detect fork knife spoon tray --grab --annotate
"""

import argparse
import json
import sys
import time

import cv2
import numpy as np

from spark_real.config import calibrations_dir, output_root

CAL_PATH = calibrations_dir() / "calibration_bimanual.json"


def det_path():
    return output_root() / "detections" / "detections.json"


def load_calibration():
    """(T_zed_to_right_base, T_zed_to_left_base)."""
    cal = json.loads(CAL_PATH.read_text())
    T_r = np.array(cal["arms"]["right"]["T_zed_to_base"]).reshape(4, 4)
    T_l = np.array(cal["arms"]["left"]["T_zed_to_base"]).reshape(4, 4)
    return T_r, T_l


def right_to_left(pos_r, T_r, T_l):
    pt_zed = (np.linalg.inv(T_r) @ np.append(pos_r, 1.0))[:3]
    return (T_l @ np.append(pt_zed, 1.0))[:3]


def capture_zed():
    """One HD2K frame (utensil handles are ~25 px wide at 720p)."""
    from spark_real.perception.zed import ZEDMiniCamera

    cam = ZEDMiniCamera(width=2208, height=1242, depth_mode="NEURAL")
    cam.open()  # raises when the ZED is absent: never detect on fake frames
    time.sleep(0.5)
    rgb, depth = cam.read(depth=True)
    config = cam.config
    cam.close()
    if rgb is None:
        sys.exit("ZED capture failed")
    K = np.array([[config.fx, 0, config.cx], [0, config.fy, config.cy], [0, 0, 1]])
    return rgb, depth, K


def run_sam3(rgb, depth, prompts, K, T_zed_to_base, threshold=0.5):
    from spark_core.sam3 import SPARKPerception

    perc = SPARKPerception(sam3_threshold=threshold)
    h, w = rgb.shape[:2]
    fovy = float(2 * np.degrees(np.arctan(h / (2 * K[1, 1]))))
    return perc._detect_with_rendered_depth(
        rgb=rgb, depth=depth, prompts=prompts,
        cam_pos=T_zed_to_base[:3, 3], cam_mat=T_zed_to_base[:3, :3], cam_fovy=fovy, w=w, h=h,
        intrinsic_matrix=K, table_height=-0.08,
    )


def _backproject_pixel(u, v, depth, K, T_zed_to_base):
    """One pixel to the base frame; a 10x10 median fills a depth hole."""
    h, w = depth.shape[:2]
    ui, vi = min(max(int(round(u)), 0), w - 1), min(max(int(round(v)), 0), h - 1)
    d = float(depth[vi, ui])
    if d < 0.01:
        patch = depth[max(0, vi - 5):vi + 5, max(0, ui - 5):ui + 5]
        valid = patch[(patch > 0.01) & (patch < 10)]
        if len(valid) == 0:
            return None
        d = float(np.median(valid))
    p_cam = np.array([(u - K[0, 2]) * d / K[0, 0], (v - K[1, 2]) * d / K[1, 1], d])
    return T_zed_to_base[:3, :3] @ p_cam + T_zed_to_base[:3, 3]


def compute_grab_point(mask, depth, K, T_zed_to_base, inset_m=0.015):
    """Mask centroid, or the deepest interior pixel when the centroid is near an edge."""
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    d = float(depth[int(cy), int(cx)])
    if d < 0.01:
        valid = depth[mask > 0]
        valid = valid[(valid > 0.01) & (valid < 10)]
        d = float(np.median(valid)) if len(valid) else 0.5
    if dist[int(cy), int(cx)] <= inset_m * K[0, 0] / d:
        i = int(np.argmax(dist[mask > 0]))
        cx, cy = float(xs[i]), float(ys[i])
    return _backproject_pixel(cx, cy, depth, K, T_zed_to_base)


def compute_obb_angle(mask, depth, K, T_zed_to_base):
    """Mask long-axis angle in the base XY plane (rad); 0.0 on failure."""
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return 0.0
    coords = np.column_stack([xs, ys]).astype(float)
    mean = coords.mean(axis=0)
    axis = np.linalg.eigh(np.cov((coords - mean).T))[1][:, -1] * 40.0
    p1 = _backproject_pixel(*(mean + axis), depth, K, T_zed_to_base)
    p2 = _backproject_pixel(*(mean - axis), depth, K, T_zed_to_base)
    if p1 is None or p2 is None:
        return 0.0
    return float(np.arctan2(p1[1] - p2[1], p1[0] - p2[0]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("prompts", nargs="+")
    p.add_argument("--grab", action="store_true", help="grab points and long-axis angles")
    p.add_argument("--annotate", action="store_true", help="save an annotated capture")
    p.add_argument("--inset", type=float, default=0.015, help="grab inset from the mask edge (m)")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args(argv)

    out = det_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    T_r, T_l = load_calibration()
    rgb, depth, K = capture_zed()
    detections = run_sam3(rgb, depth, args.prompts, K, T_r, args.threshold)

    results = []
    for det in detections:
        if det.position_3d is None:
            continue
        pos_r = np.asarray(det.position_3d, dtype=float)
        entry = {
            "label": det.label,
            "confidence": float(det.confidence),
            "right_frame": pos_r.tolist(),
            "left_frame": right_to_left(pos_r, T_r, T_l).tolist(),
        }
        if args.grab and det.mask is not None:
            gp = compute_grab_point(det.mask, depth, K, T_r, args.inset)
            if gp is not None:
                entry["grab_right"] = gp.tolist()
                entry["grab_left"] = right_to_left(gp, T_r, T_l).tolist()
            entry["obb_angle_right"] = compute_obb_angle(det.mask, depth, K, T_r)
            entry["obb_angle_left"] = compute_obb_angle(det.mask, depth, K, T_l)
        results.append(entry)
        print(f"{det.label:<16} {det.confidence:5.2f}  right={np.round(pos_r, 4).tolist()}")

    if args.annotate:
        img = rgb.copy()
        for det in detections:
            if det.mask is not None:
                img[det.mask > 0] = (0.5 * img[det.mask > 0] + (0, 127, 0)).astype(np.uint8)
            cx, cy = (int(v) for v in det.centroid_2d)
            cv2.putText(img, det.label, (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.imwrite(str(out.parent / "annotated.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {len(results)} detections to {out}")


if __name__ == "__main__":
    main()
