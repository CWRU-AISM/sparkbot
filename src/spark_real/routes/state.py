"""
Shared server state accessible by all route modules.
"""

import asyncio
import threading
from typing import Optional

pipeline = None
pipeline_lock = asyncio.Lock()

# The SparkConfig the server was LAUNCHED with (set by server.main before
# uvicorn starts). /api/initialize builds from this so a --no-init boot on
# rebuilds the launched configuration, not a default one.
launch_config = None

# BT library runtime overrides for the LLM-vs-cache resolver in
# pipeline.plan() / pipeline.run_task(). Both default to None so the
# legacy SPARK_DISABLE_LLM env var still governs at boot.
#
# use_cached_bt: None = fall back to env var, True/False = force.
#   Toggled by POST /api/bt/toggle_cache from the frontend panel.
# pinned_bt_hash: single-shot pointer to a specific cached BT. plan()
#   consumes and clears it so the next call returns to normal resolution.
use_cached_bt: Optional[bool] = None
pinned_bt_hash: Optional[str] = None

# RF-DETR second-opinion runtime override, same shape as use_cached_bt:
# None = fall back to config (YAML, then SPARK_PROPOSER_MODE), otherwise one of
# pipeline_perception.PROPOSER_MODES ("auto" / "always" / "off"). Written only
# by POST /api/perception/proposer_mode, which also drops the pipeline's cached
# fusion gate -- setting this alone is inert until the gate is rebuilt.
proposer_mode: Optional[str] = None

active_camera = "sideview"

# The most recently published frame, as ONE consistent (rgb, depth, camera)
# triple. Always publish with set_last_frame() and consume with
# get_last_frame(); never assign or read the three names directly.
#
# Three producers write it concurrently (capture_frame_sync on the
# _STREAM_POOL, the /ws/camera loop, /api/capture). Read as three separate
# globals, a tiled publish interleaved with a single-camera publish hands
# /api/detect_click one camera's RGB, another camera's depth and a third
# camera's extrinsics; the Kinects share a resolution, so nothing raises and
# the arm moves to a coordinate that corresponds to nothing in the scene.
last_rgb = None
last_depth = None
last_cam_name = None
_last_frame_lock = threading.Lock()


def set_last_frame(rgb, depth, cam_name):
    """Publish an rgb/depth/camera triple as one indivisible update."""
    global last_rgb, last_depth, last_cam_name
    with _last_frame_lock:
        last_rgb = rgb
        last_depth = depth
        last_cam_name = cam_name


def get_last_frame():
    """
    Return the last published ``(rgb, depth, cam_name)``, guaranteed to be
    from a single camera and a single capture.

    Consumers must take all three from ONE call. Re-reading
    ``state.last_depth`` afterwards reintroduces exactly the tear this
    exists to remove.
    """
    with _last_frame_lock:
        return last_rgb, last_depth, last_cam_name

# Progress log: pipeline steps pushed here, frontend polls /api/progress.
#
# Entries are {"seq": int, "ts": float, "msg": str}. `seq` is a monotonic
# counter that NEVER resets while the buffer trims, and it is what the
# frontend uses to ask for "everything after N" (GET /api/progress?since=N).
#
# A client-side list index would silently point at the wrong element once
# the buffer trims from the front.
#
# progress_seq is guarded by progress_lock, like the buffer itself.
progress_log: list = []
progress_seq: int = 0
progress_lock = threading.Lock()

# Buffer cap. Generous because entries are small and `since` means a poll
# only ever transfers what is new; the cap only bounds how far a client can
# fall behind before it misses lines.
PROGRESS_MAX = 500

# Pending detections waiting for user approval
pending_detections = None
pending_captures = None
pending_instruction = None
pending_prompts = None

# DA3 estimator (lazy, /api/da3_estimate)
da3_estimator = None

# Calibration points
calibration_points = []

# Camera stream state
stream_mode = "rgb"
stream_camera = "all"

# Tile layout for click-to-camera mapping
tile_layout = []

# EMA-smoothed velocity for teleop
smoothed_vel = None
# EMA smoothing factor for teleop velocity. 0.4 is the default; higher
# values reduce smoothing and increase jitter.
VEL_ALPHA = 0.4

# Maximum downward (-Z, world-frame) velocity for /api/velocity commands.
# Caps both teleop and any other client that streams Cartesian velocity to
# the robot, so server-side safety doesn't depend on the frontend obeying
# its own gain map. Only descent gets clamped; None = no cap.
MAX_DOWN_VEL = 0.03

# Execution serialization. Acquired NON-BLOCKING at the top of
# /api/execute, /api/execute_approved and /api/run_bt; a second request
# while one run is in flight gets 409 instead of a second ScoreExecutor
# traversal over the same driver (interleaved URScript programs replace
# each other on the controller; the second run's completion would clear
# executor_running while the first is still driving the arm).
execute_lock = threading.Lock()

# Executor mutex. True while /api/execute_approved / /api/execute is
# actively driving the arm. While True, teleop endpoints
# (/api/velocity, /api/move_relative) become no-ops so they don't
# interleave Cartesian-velocity commands with the executor's
# Cartesian-pose / joint motions.
executor_running = False

# Calibration mutex. True while /api/calibrate/move_to_pose is
# driving the CartesianServo. Same rationale as executor_running:
# any concurrent /api/velocity call (from teleop with stick drift
# past the deadzone) would fight the servo, leaving the robot
# averaged near zero motion, hitting a servo timeout.
cal_active = False

# Current episode recorder (EpisodeRecorder instance). Holds the
# cache for the most recent run (success or fail). Cleared on the
# next execute_approved call. The user reviews via the result panel
# and saves with /api/episode/save (renames cache -> episodes_kept/)
# or discards with /api/episode/discard.
current_episode = None
