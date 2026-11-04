# Shared pipeline dataclasses, split out so the mixin modules can import
# them without a cycle back through pipeline.py.

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from spark_real.config import output_root


@dataclass
class PipelineConfig:
    """
    Configuration for the real robot pipeline.
    """

    # robot_ip is set by server.py from --ip / configs/<family>_default.yaml.
    # The empty default is a sentinel meaning "no robot configured"; code
    # paths that try to connect should treat "" as a "skip connect" signal.
    robot_ip: str = ""

    # Robot identity surfaced to the frontend so the UI can swap banner +
    # control panel per family. server.py overrides via --robot.
    robot_family: str = "ur10e"
    robot_model: str = "UR10e"  # human-readable model string

    use_kinect: bool = True
    use_realsense: Optional[bool] = None
    """
    None = auto (init iff a RealSense is enumerated); True/False forces.
    """
    # 720P halves the color stream's USB isochronous reservation vs 1080P;
    # intrinsics follow the resolution, so hand-eye stays valid.
    kinect_resolution: str = "720P"
    kinect_master_serial: str = ""  # names the sideview (master) Kinect
    kinect_use_hw_sync: bool = False  # Set True only if sync cable connected
    # Depth modes (Azure Kinect DK):
    #   NFOV_UNBINNED:  75x65 FOV,  640x576, 30 fps, range 0.5-3.86 m
    #   NFOV_2X2BINNED: 75x65 FOV,  320x288, 30 fps, range 0.5-5.46 m
    #   WFOV_UNBINNED:  120x120,    1024x1024, 15 fps, range 0.25-2.21 m
    #   WFOV_2X2BINNED: 120x120,    512x512,  30 fps, range 0.25-2.88 m
    # WFOV_2X2BINNED is the right default for tabletop manipulation:
    # full 30 fps for the VLA loop, wide enough that one Kinect sees the
    # whole cell, and close-range support down to 25 cm. Override in the
    # per-family YAML's `kinect_depth_mode` field.
    kinect_depth_mode: str = "WFOV_2X2BINNED"
    # 15 fps halves USB and depth-engine load vs 30 with two Kinects on one
    # host controller.
    kinect_fps: int = 15
    realsense_serial: str = ""
    # Stream the wrist RealSense color-only (no depth) to halve SuperSpeed USB
    # bandwidth and avoid the xHCI event-ring desync that drops the stream.
    # Depth is unused while wrist_refine is off; set False to restore it.
    realsense_color_only: bool = True

    sam3_threshold: float = 0.03
    use_hardware_depth: bool = True

    gemini_model: str = "gemini-3.5-flash"
    planner_temperature: float = 0.3

    velocity: float = 0.25
    safe_height: float = 0.35
    # Recovery search floor (robot frame, open-gripper TCP at the work
    # surface). None means: use the executor's config-fed TABLE_Z_FLOOR
    # for the active family. Set per machine in the family YAML
    # (table_height) only when the rig has a raised surface like the
    # original UR10e cell's foam mat (-0.28).
    table_height: float | None = None

    output_dir: str = field(default_factory=lambda: str(output_root() / "real_runs"))
    save_captures: bool = True

    # Behavior-tree cache. Once a task has one verified run in the library,
    # every later run of it is served from disk with no LLM call. Set via the
    # `bt:` block in configs/<family>_default.yaml.
    bt_library_dir: Optional[str] = None
    """Library root. None => <output_dir>/../bt_library (NOT cwd-relative)."""
    bt_seed_dir: Optional[str] = None
    """Seed YAML root. None => the packaged configs/bt_seeds."""
    bt_frozen: bool = False
    """
    Freeze the BT library: serve NOTHING as few-shot context and record
    nothing back.

    The no-adaptation control: the paper's Problem Setup defines the
    no-adaptation protocol as seeing no trial of a task before that task is
    scored, so with bt_frozen set, trial N plans with exactly the context
    trial 1 had.

    Env override: SPARK_BT_FROZEN=1.
    """

    bt_seed_on_start: bool = True
    """
    Install the packaged seed trees into the library at startup.

    Turn this OFF when collecting a fresh demonstration corpus: a seeded tree
    is cache-eligible immediately, so the task would be served from disk
    instead of being planned, and the run you meant to record fresh never
    calls the planner at all.
    """
    bt_plan_mode: str = "auto"
    # The recovery ladder's terminal rung: after every LLM-free rung fails
    # WITH evidence, consult the planner once with the failure context.
    # Off: surface the failure and stop.
    llm_last_resort: bool = True
    """auto (cache first, LLM on miss) | cache_only (miss raises) | llm."""
    bt_min_similarity: float = 0.75
    """Jaccard floor for fuzzy retrieval. Below it is a miss, not a hit."""
    bt_auto_promote_after: int = 3
    """Successes (with zero failures) before an entry auto-promotes."""

    # Frozen per-task SAM3 prompt contracts (prompts, expected object counts,
    # deterministic instance ordering). None => the packaged configs/tasks.
    # Also overridable by $SPARK_TASK_PROMPTS.
    task_prompts_dir: Optional[str] = None
    """Directory of task prompt YAMLs; see perception/prompt_registry.py."""

    # Experimental: strict placement verification.
    # When True, replaces the binary mask-intersection check with a
    # containment ratio (>= 50% of the picked object's mask must lie
    # inside the place target's mask), allows the post-task retry to
    # re-detect the placed object's current location, and caps retries
    # at 1. Catches "tumbled out next to container" failures that the
    # legacy verify reports as success.
    # Off by default, flip via --strict-verify on the server CLI.
    strict_placement_verify: bool = False

    save_to_library: bool = True
    """
    If True, every successful task appends its (instruction, score) to bt_library/.
    """

    # Closed-loop task execution (generic, all families).
    # After a behavior tree runs, the pipeline re-perceives the scene and
    # checks whether the target objects actually reached their destination
    # (in a receptacle, swept into a dustpan, lifted off the source). Any
    # target that did NOT make it, plus any object the scene was perturbed
    # with mid-task (swapped utensil, added item, moved receptacle), drives
    # another plan-and-execute pass. Already-handled objects stay closed via
    # the destination filter and the executor's placed-label set, so the
    # re-plan never re-visits a node already taken care of.
    # max_task_passes == 1 is single shot.
    max_task_passes: int = 1
    """Outer closed-loop cap. 1 = single shot (legacy). >1 re-perceives and
    re-plans over what remains until the goal holds or the budget is spent."""
    closed_loop_settle_s: float = 0.8
    """Seconds to let the scene settle (arm home, objects still) before the
    between-pass re-perception so SAM3 sees the workspace, not motion."""
    closed_loop_clear_radius_m: float = 0.12
    """A target whose centroid is within this distance of a destination
    centroid is treated as handled (inside the receptacle / swept in). Used
    as a floor; the destination's own OBB minor radius widens it per object."""

    use_grasp_calibration: bool = True
    """
    If True, ScoreExecutor uses per-object learned grasp depth offsets.
    """

    # Wrist-camera translation from the TCP (identity rotation). None: use
    # <calibrations>/handeye_wrist.json, else a placeholder offset.
    wrist_tool_offset_xyz: Optional[List[float]] = None


@dataclass
class TaskResult:
    """
    Result from a single task execution.
    """

    instruction: str
    timestamp: str
    detections: List[Dict]
    plan: Dict
    execution_results: List[Dict]
    success: bool
    duration: float
    captures: Dict[str, str] = field(default_factory=dict)

    # Plan provenance. Declared (not just set as ad-hoc attributes) because
    # the recorder copies them verbatim into metadata["spark"], so an episode
    # on disk says exactly which tree drove it and where that tree came from.
    plan_source: Optional[str] = None
    """pin | exact | alias | jaccard | seed | llm -- see BTLibrary.resolve."""
    bt_hash: Optional[str] = None
    """Content hash of the executed behavior tree, or None if unlibraried."""
    label_resolutions: Dict[str, str] = field(default_factory=dict)
    """Cached-BT label -> the differently-named detection it re-bound to."""
    # Task-success verdict (control/success_verifier.VerifyOutcome.to_dict()).
    # ``success`` above is exactly ``verify_status == "pass"``; this carries the
    # evidence -- predicates, per-camera votes, gates, depth source.
    verify: Optional[Dict] = None
    """Full VerifyOutcome dict, or None when no verdict was produced."""
    verify_status: str = "unverified"
    """pass | fail | unverified. `unverified` must not train and must not
    reinforce a cached tree; it is not a quieter kind of failure."""
