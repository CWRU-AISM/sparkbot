"""FairConfig: the LIBERO-PRO runner's CLI (tyro) and the fields the executor and LIBERO-Dyn read."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class FairConfig:
    suite: str = "object"
    """LIBERO suite: object, goal, spatial."""
    perturbation: Literal['position', 'task', 'language', 'object'] = "position"
    """LIBERO-PRO perturbation."""
    num_trials: int = 5
    """Trials per task; trial i uses init state (i + trial_offset)."""
    cam_width: int = 640
    cam_height: int = 480
    no_gemini: bool = False
    """No LLM: scripted plan from the BDDL pick/place hints (smoke and fixed-plan runs)."""
    verbose: bool = False
    output_dir: str = "results/libero_pro"
    """Result JSON goes to <output_dir>/<suite>.json."""
    only_task_id: int = -1
    """If >= 0, run only this task."""
    trial_offset: int = 0
    """Added to the trial index for the env seed and the init state."""
    freshness_timeout_s: Optional[float] = 0.5
    """Re-run SAM3 between primitives when the cached detections are older than this. None disables."""

    # perception
    use_gemini_prompts: bool = False
    """The planner VLM proposes the SAM3 prompts from the instruction and the scene image."""
    no_bddl_hints: bool = False
    """Pass only the instruction to the planner, not the BDDL pick/place names."""
    adaptive_prompts: bool = False
    """The VLM proposes adaptive_k phrasings per object; SAM3 keeps one per object."""
    adaptive_k: int = 3
    language_only: bool = False
    """Nothing from the BDDL goal or the curated phrase table reaches perception: prompts come from the
    VLM and the instruction (forces use_gemini_prompts)."""
    wrist_verify: bool = False
    """Re-ground a weakly detected pick target from the home-pose wrist image (one VLM call)."""
    relation_bind: bool = False
    """With twin pick targets, take the instance that satisfies the instruction's spatial relation, from
    3D positions only."""

    # verify-and-adapt arm of the fixed-plan displacement result
    pre_close_retarget_min_cm: float = 4.0
    """Apparent target moves below this at the pre-close capture are treated as occlusion bias."""
    event_captures: bool = False
    """Capture and re-detect at pre-close and post-release."""
    scene_diff: bool = False
    """Diff detections against the plan-time binding before each keypoint primitive."""
    retarget_max_cm: float = 10.0
    """Targets that moved less than this are retargeted in place; larger moves go to recovery."""
    scene_diff_moved_cm: float = 1.5
    scene_diff_self_radius_cm: float = 12.0
    """Changes within this radius of the gripper are attributed to the robot."""
    layered_recovery: bool = False
    """Repair at the lowest responsible layer (perception re-bind, local retry, replan)."""
    max_local_retries: int = 1
