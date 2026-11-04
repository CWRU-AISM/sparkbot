"""
Role-keyed camera registry.

Single-arm rigs: the pipeline opens the Kinects and the wrist RealSense
itself (sync order, retries, stagger) and wraps the open devices with
``registry_from_open_devices``. The bimanual rig opens every camera from
its config roster with ``build_camera_registry``.

``spec(profile)`` returns the declared ``cameras:`` roster without touching
hardware and rejects an unknown camera type.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from spark_real.calibration import CameraCalibration

logger = logging.getLogger(__name__)

# Camera driver registry: config `cameras[].type` -> the module its driver
# needs. Any family may name any type. "azure_kinect" and "usb_camera"/"uvc"
# are accepted spellings of "kinect" and "webcam".
_SDK = {"kinect": "pyk4a", "realsense": "pyrealsense2", "zed_mini": "pyzed.sl", "webcam": "cv2"}
CAMERA_TYPES = tuple(_SDK)
_TYPE_ALIASES = {"azure_kinect": "kinect", "usb_camera": "webcam", "uvc": "webcam"}
# (role, type) pairs the single-arm pipeline opens itself (Kinect sync order,
# retries, RealSense recovery); every other roster entry opens generically.
PIPELINE_OPENED = {("sideview", "kinect"), ("birdview", "kinect"), ("wrist", "realsense")}


def _is_plugin(kind: str) -> bool:
    # A driver outside this package: "package.module:ClassName" (see _open_plugin).
    return ":" in kind


def camera_type(entry: Dict[str, Any]) -> str:
    """The canonical type of a `cameras:` entry; ValueError if no driver is registered."""
    raw = str(entry.get("type") or "")
    if _is_plugin(raw):
        return raw
    raw = raw.lower()
    kind = _TYPE_ALIASES.get(raw, raw)
    if kind not in CAMERA_TYPES:
        raise ValueError(
            f"camera role {entry.get('role')!r}: type {raw or '<missing>'!r} has no driver; "
            f"registered: {CAMERA_TYPES}, or a class as 'package.module:ClassName'"
        )
    return kind


def require_driver(kind: str, role: str = "") -> None:
    """RuntimeError when the type's SDK (its transport) is not installed here."""
    import importlib

    module = kind.split(":")[0] if _is_plugin(kind) else _SDK[kind]
    try:
        importlib.import_module(module)
    except ImportError as exc:
        raise RuntimeError(
            f"camera role {role!r}: type {kind!r} needs {module} ({exc}); "
            "install its SDK or remove the entry from `cameras:`"
        ) from None


KINECT_ROLES = ("sideview", "birdview")  # the single-arm pipeline's two Kinect slots


def assign_kinect_roles(
    serials: Dict[int, Optional[str]],
    roster: List[Dict[str, Any]],
    master_serial: str = "",
    use_hw_sync: bool = False,
) -> List[tuple]:
    """
    (device_id, role, sync_mode) for each connected Kinect, in open order.

    A Kinect whose serial is in the `cameras:` roster takes that entry's role;
    the others fill the remaining Kinect roles in roster order, so a rig with
    one Kinect, or Kinects whose serials are not set, needs no serials.
    With hardware sync the master is ``master_serial`` when set, else the
    roster entry with `sync: master`, else sideview; subordinates open first.
    """
    entries = [e for e in roster if camera_type(e) == "kinect" and e.get("role") in KINECT_ROLES
               and e.get("enabled", True) is not False]
    roles = [e["role"] for e in entries] or list(KINECT_ROLES)
    roles += [r for r in KINECT_ROLES if r not in roles]
    by_serial = {str(e["serial"]): e["role"] for e in entries if e.get("serial")}
    if master_serial:
        by_serial = {sn: r for sn, r in by_serial.items() if r != "sideview"}
        by_serial[str(master_serial)] = "sideview"
    assigned = {d: by_serial[str(sn)] for d, sn in serials.items() if sn is not None and str(sn) in by_serial}
    free = [r for r in roles if r not in assigned.values()]
    for d in sorted(serials):
        if d not in assigned and free:
            assigned[d] = free.pop(0)
            logger.warning("Kinect %d (SN=%s) has no roster serial; using role %r", d, serials[d], assigned[d])
    master_role = "sideview"
    if not master_serial:
        master_role = next((e["role"] for e in entries if str(e.get("sync", "")).lower() == "master"), "sideview")
    out = []
    for d in sorted(assigned):
        role = assigned[d]
        sync = ("MASTER" if role == master_role else "SUBORDINATE") if use_hw_sync else "STANDALONE"
        out.append((d, role, sync))
    out.sort(key=lambda t: t[2] == "MASTER")  # subordinates listen before the master triggers
    return out


@dataclass
class CameraEntry:
    """One open camera with its calibration."""

    role: str  # e.g. 'sideview', 'birdview', 'wrist', 'external', 'wrist_left'
    kind: str  # one of CAMERA_TYPES
    device: Any
    calibration: CameraCalibration
    arm: Optional[str] = None
    tool_offset_xyz: Optional[List[float]] = None  # TCP->camera (wrist cams)
    serial: str = ""


@dataclass
class CameraRegistry:
    """Role-keyed cameras; ``roles`` order is fixed for a stable UI layout."""

    cameras: Dict[str, CameraEntry] = field(default_factory=dict)
    roles: List[str] = field(default_factory=list)
    primary_role: str = ""
    wrist_for_arm: Dict[str, str] = field(default_factory=dict)  # arm -> role

    def __getitem__(self, role: str) -> CameraEntry:
        return self.cameras[role]

    def get(self, role: str) -> Optional[CameraEntry]:
        return self.cameras.get(role)

    def __contains__(self, role: str) -> bool:
        return role in self.cameras

    def __iter__(self):
        for r in self.roles:
            yield r, self.cameras[r]


def spec(profile) -> Dict[str, Dict[str, Any]]:
    """Role -> copy of the YAML camera entry. Entries without a role are skipped."""
    out: Dict[str, Dict[str, Any]] = {}
    for entry in profile.cameras():
        role = entry.get("role")
        if not role:
            logger.warning("camera spec missing 'role'; skipping: %s", entry)
            continue
        camera_type(entry)
        out[role] = dict(entry)
    return out


def registry_from_open_devices(
    entries: List[CameraEntry], primary_role: str = ""
) -> CameraRegistry:
    """Wrap already-open devices; the primary role is listed first."""
    registry = CameraRegistry(primary_role=primary_role)
    for entry in entries:
        registry.cameras[entry.role] = entry
        registry.roles.append(entry.role)
    if primary_role in registry.cameras:
        registry.roles.remove(primary_role)
        registry.roles.insert(0, primary_role)
    return registry


# Roster keys the registry consumes itself; a plugin class gets every other key.
_REGISTRY_KEYS = ("role", "type", "arm", "tool_offset_xyz", "enabled", "sync")


def _open_plugin(kind: str, spec: Dict[str, Any]):
    """
    A camera class named in config as "package.module:ClassName", built with the
    entry's other keys as keyword arguments. It must provide open(), close(),
    read() -> (rgb HxWx3 uint8, depth HxW in meters or None) and a `config` with
    width, height, fx, fy, cx, cy (camera_base.CameraConfig).
    """
    import importlib

    module, _, name = kind.partition(":")
    cls = getattr(importlib.import_module(module), name)
    return cls(**{k: v for k, v in spec.items() if k not in _REGISTRY_KEYS})


def _open_device(kind: str, spec: Dict[str, Any]):
    """Construct and open one camera from its config entry."""
    from spark_real.perception.camera_base import CameraConfig

    width, height = int(spec.get("width", 640)), int(spec.get("height", 480))
    if _is_plugin(kind):
        device = _open_plugin(kind, spec)
    elif kind == "zed_mini":
        from spark_real.perception.zed import ZEDMiniCamera

        device = ZEDMiniCamera(
            serial=spec.get("serial") or None,
            width=width,
            height=height,
            fps=int(spec.get("fps", 30)),
            depth_mode=spec.get("depth_mode", "NEURAL"),
            flip=spec.get("flip", "auto"),
        )
    elif kind == "realsense":
        from spark_real.perception.realsense import RealSenseCamera

        device = RealSenseCamera(serial=spec.get("serial") or None, config=CameraConfig(width=width, height=height))
    elif kind == "kinect":
        from spark_real.perception.kinect import AzureKinectCamera

        device = AzureKinectCamera(
            color_resolution=spec.get("resolution", "720P"),
            depth_mode=spec.get("depth_mode", "WFOV_2X2BINNED"),
            camera_fps=int(spec.get("fps", 15)),
        )
    else:
        from spark_real.perception.camera_base import USBCamera

        # `device`: an index or a /dev/v4l/by-path/... string (identical
        # webcams often share one USB serial, so by-id collides).
        dev_id = spec.get("device", spec.get("device_id"))
        if dev_id is None:
            raise ValueError("webcam `device` is not set")
        device = USBCamera(device_id=dev_id, config=CameraConfig(width=width, height=height))
    device.open()
    return device


def build_camera_registry(
    cam_specs: List[Dict[str, Any]], primary_role: str = "", registry: Optional[CameraRegistry] = None
) -> CameraRegistry:
    """
    Open every camera in a `cameras:` roster, keyed by role (added to
    ``registry`` when given).

    Before anything opens, a type with no driver raises ValueError and a
    type whose SDK is missing raises RuntimeError. A camera that then fails
    to open (unplugged, busy) is logged and left out, so nothing downstream
    reads a dead or simulated device. The primary role is listed first, then
    the left and right wrist cameras.
    """
    kinds = [(spec, camera_type(spec)) for spec in cam_specs if spec.get("role")]
    for spec, kind in kinds:
        require_driver(kind, spec["role"])
    if registry is None:
        registry = CameraRegistry(primary_role=primary_role or (kinds[0][0]["role"] if kinds else ""))
    for spec, kind in kinds:
        role = spec["role"]
        try:
            device = _open_device(kind, spec)
        except Exception as exc:
            logger.error("camera %r (%s) did not open: %s", role, kind, exc)
            continue
        c = device.config
        entry = CameraEntry(
            role=role,
            kind=kind,
            device=device,
            # Extrinsic stays identity until a calibration file sets it.
            calibration=CameraCalibration(
                name=role, width=int(c.width), height=int(c.height),
                fx=float(c.fx), fy=float(c.fy), cx=float(c.cx), cy=float(c.cy),
            ),
            arm=spec.get("arm"),
            tool_offset_xyz=spec.get("tool_offset_xyz"),
            serial=str(spec.get("serial") or ""),
        )
        registry.cameras[role] = entry
        registry.roles.append(role)
        if entry.arm in ("left", "right"):
            registry.wrist_for_arm[entry.arm] = role
    preferred = [registry.primary_role, registry.wrist_for_arm.get("left"), registry.wrist_for_arm.get("right")]
    head = [r for r in preferred if r and r in registry.cameras]
    registry.roles = head + [r for r in registry.roles if r not in head]
    logger.info("camera registry: roles=%s primary=%s", registry.roles, registry.primary_role)
    return registry
