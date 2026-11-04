"""
Shared types for the ScoreExecutor family.
"""

from dataclasses import dataclass


# Leaf primitive names the executor dispatches directly (not via the skill
# registry). The planner validates emitted leaf types against this same set.
# The `action` meta-type and structural node types (sequence, selector,
# parallel, sync_barrier) are added locally at each site.
BUILTIN_PRIMITIVES = frozenset(
    {
        "move_to_keypoint",
        "grasp",
        "release",
        "move_relative",
        "wait",
    }
)


class AbortRequested(Exception):
    """Raised when the operator requests an emergency stop mid-motion.

    Propagated out of every motion-issuing helper so that no new motion
    command can overwrite the stop that was just sent to the controller.
    """

    pass


@dataclass
class ExecutionResult:
    action_type: str
    success: bool
    message: str = ""
    duration: float = 0.0
