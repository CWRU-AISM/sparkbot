"""
Bimanual Franka rig (Panda left + FR3 right) as one driver family: two
per-arm drivers (franky or bamboo) plus one registered jaw per arm
(spark_real.grippers).
"""

from .bimanual_franka_driver import ARMS, BimanualFrankaDriver, BimanualObservation

__all__ = ["ARMS", "BimanualFrankaDriver", "BimanualObservation"]
