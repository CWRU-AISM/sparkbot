"""Geometry finds a tool's two ends; the planner says which one is the working end.

A mask gives a precise line but cannot tell a spoon's bowl from its handle in general, and a model
reads that from an image far better than it reads an angle. So each elongated detection gets its
ends marked A and B on the image the planner already receives, and the answer comes back with the
plan as `head_end`. Nothing here is specific to one tool.
"""

import json
import logging
import re

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

MIN_ASPECT = 2.5
TILE_PX = 300
NOTE = (
    "\n\nSome tools in the images have their two ends marked '<label>:A' and '<label>:B'. Add a "
    "top-level YAML key `head_end` that maps each of those labels to the letter of its working "
    "end (a spoon's bowl, a fork's tines, a knife's blade, a screwdriver's tip), or null if you "
    "cannot tell or the tool has no working end."
)
ASK = (
    "These tools have their two ends marked '<label>:A' and '<label>:B': {labels}. For each one, "
    "which letter is its working end (a spoon's bowl, a fork's tines, a knife's blade, a "
    "screwdriver's tip)? Use null if you cannot tell or the tool has no working end. Reply with "
    'JSON only, for example {{"tool 1": "A", "tool 2": null}}.'
)


def measure(detections, captures) -> dict:
    """{label: {"camera", "A": (px, xy), "B": (px, xy)}} for elongated masks. A has the larger x."""
    out = {}
    for det in detections or []:
        mask, pos = getattr(det, "mask", None), getattr(det, "position_3d", None)
        cap = (captures or {}).get(getattr(det, "camera", None)) or {}
        cal = cap.get("calibration")
        if mask is None or pos is None or cal is None:
            continue
        ys, xs = np.nonzero(np.asarray(mask) > 0)
        if xs.size < 50:
            continue
        T = np.asarray(cal.extrinsic, dtype=float)
        rays = T[:3, :3] @ np.linalg.inv(np.asarray(cal.intrinsic_matrix, dtype=float)) @ np.stack(
            [xs, ys, np.ones_like(xs)]
        )
        if np.any(np.abs(rays[2]) < 1e-6):
            continue
        xy = T[:2, 3:] + rays[:2] * (float(pos[2]) - T[2, 3]) / rays[2]
        mid = xy.mean(axis=1, keepdims=True)
        vals, vecs = np.linalg.eigh((xy - mid) @ (xy - mid).T)
        axis = vecs[:, 1] * (1.0 if vecs[0, 1] >= 0 else -1.0)
        proj = axis @ (xy - mid)
        lo, hi = np.percentile(proj, [1.0, 99.0])
        if hi - lo < MIN_ASPECT * 4.0 * np.sqrt(max(vals[0], 0.0) / xs.size):
            continue  # 4 sigma approximates the width of a solid bar
        end = lambda sel, at: (  # noqa: E731
            (int(np.median(xs[sel])), int(np.median(ys[sel]))),
            (mid[:, 0] + axis * at).tolist(),
        )
        out[str(det.label)] = {"camera": det.camera, "A": end(proj >= hi, hi), "B": end(proj <= lo, lo)}
    return out


def draw(scene_image, ends: dict, scene_camera=None):
    """(views with the ends marked plus a zoomed tile of each tool, the labels actually marked).

    Marks sit just past each end: on the tool they hide the very thing the model must read.
    """
    views = (
        [(scene_camera, scene_image)]
        if not isinstance(scene_image, (list, tuple))
        else list(scene_image)
    )
    marked, out, tiles = set(), [], []
    for cam, img in views:
        vis = None if img is None else np.ascontiguousarray(np.array(img))
        for label, e in ends.items():
            if vis is None or e["camera"] != cam:
                continue
            a, b = (np.asarray(e[k][0], dtype=float) for k in ("A", "B"))
            out_dir = (a - b) / max(float(np.linalg.norm(a - b)), 1e-6)
            clean = vis.copy()
            for name, at, sign in (("A", a, 1.0), ("B", b, -1.0)):
                px = tuple(int(v) for v in at + sign * out_dir * 16)
                cv2.circle(vis, px, 5, (255, 255, 0), 2)
                text = f"{label}:{name}"
                cv2.putText(vis, text, (px[0] + 8, px[1] - 8), 0, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis, text, (px[0] + 8, px[1] - 8), 0, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
            tiles.append(_tile(clean, vis, a, b, label))
            marked.add(label)
        out.append((cam, vis if vis is not None else img))
    if tiles:
        out.append(("zoomed tools", np.hstack(tiles)))
    return out, marked


def _tile(clean, marked, a, b, label):
    """A fixed-size close-up: the tool as filmed beside it with its marks."""
    h, w = clean.shape[:2]
    half = int(max(np.abs(a - b).max() / 2 + 40, 60))
    cx, cy = ((a + b) / 2).astype(int)
    x0, x1, y0, y1 = max(cx - half, 0), min(cx + half, w), max(cy - half, 0), min(cy + half, h)
    pair = [cv2.resize(m[y0:y1, x0:x1], (TILE_PX, TILE_PX), interpolation=cv2.INTER_CUBIC) for m in (clean, marked)]
    tile = np.hstack(pair)
    cv2.putText(tile, label, (6, 18), 0, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    return tile


def headings(ends: dict, head_end) -> dict:
    """{label: unit world-XY vector from tail to working end}, for the labels the planner decided."""
    out = {}
    for label, letter in (head_end if isinstance(head_end, dict) else {}).items():
        letter = str(letter).strip().upper()
        if label in ends and letter in ("A", "B"):
            a, b = (np.asarray(ends[label][k][1], dtype=float) for k in ("A", "B"))
            vec = (a - b) if letter == "A" else (b - a)
            out[label] = vec / max(float(np.linalg.norm(vec)), 1e-9)
    logger.info("[tool-ends] planner head_end=%s; decided %d of %d", head_end, len(out), len(ends))
    return out


def ask(planner, scene_image, labels) -> dict:
    """A cached plan means no planning call was made this run, so ask just this."""
    contents = [ASK.format(labels=", ".join(sorted(labels)))]
    contents += [
        img if not isinstance(img, np.ndarray) else Image.fromarray(img)
        for _, img in scene_image
        if img is not None
    ]
    reply = planner.generate(contents, tag="tool_ends", json_only=True)
    return json.loads(re.search(r"\{.*\}", reply, re.S).group())
