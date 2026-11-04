import logging

from spark_real.perception_capture import CaptureMixin  # noqa: F401
from spark_real.perception_fusion import FusionMixin  # noqa: F401
from spark_real.perception_merge import MergeMixin  # noqa: F401
from spark_real.perception_settings import (  # noqa: F401
    _NULL_BACKENDS,
    _PROPOSER_OFF_LABEL,
    MAX_ON_TABLE_HEIGHT_M,
    PROPOSER_MODE_ALWAYS,
    PROPOSER_MODE_AUTO,
    PROPOSER_MODE_OFF,
    PROPOSER_MODES,
    normalize_proposer_mode,
)
from spark_real.perception_task import TaskPromptMixin  # noqa: F401

logger = logging.getLogger(__name__)


class PerceptionMixin(FusionMixin, CaptureMixin, MergeMixin, TaskPromptMixin):

    # Count-gate outcome of the most recent spec-driven merge_detections.
    # Class-level default because this is a mixin with no __init__ of its own.
    _last_resolution = None
    _last_captures = None
    _last_captures_t = None
    _prompt_registry = None

    # Detector fusion / ASPIRE quality gate (config-gated, default OFF)
    # `None` = not resolved yet, `False` = resolved and OFF, otherwise the gate.
    _fusion_gate = None
    _fusion_gate_error = None

    FUSION_ENV = "SPARK_DETECTION_FUSION"
    PROPOSER_MODE_ENV = "SPARK_PROPOSER_MODE"


    MERGE_PROXIMITY_M = 0.15  # 15cm, loose enough for cross-camera calibration drift


    # Container labels whose detections should receive slot poses +
    # world-frame major-axis orientation. Slot detection runs in birdview
    # using surface-normal clustering and falls back to stripe-along-axis
    # when no compartments are separable. Matched case-insensitively
    # against the detection's base label (digit suffix from multi-
    # instance numbering stripped).
    SLOT_CONTAINER_LABELS = frozenset(
        {
            "tray",
            "grey tray",
            "silverware tray",
            "plate",
            "dish",
            "bowl",
            "dustpan",
            "pan",
            "drawer",
            "container",
            "compartment",
            "box",
        }
    )


    # Camera preference for the frame the LLM is shown. Birdview first for the
    # same reason merge_detections trusts it first: it is the view the merged
    # detection's XY comes from, so it is the view whose vocabulary has to work.
    _PROPOSAL_CAM_ORDER = ("birdview", "sideview", "wrist")


