"""SPARK real-robot pipeline for the UR10e + Robotiq 2F-85 (optional extra)."""

import os

# A real table and hand-eye are never perfectly level, see spark_core.mask_obb._grow_top_surface.
os.environ.setdefault("SPARK_OBB_TILT_FIT", "1")
