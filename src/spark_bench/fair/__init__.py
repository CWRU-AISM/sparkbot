"""LIBERO-PRO runner internals: config, env loading, planning, execution, timing.

Process-wide defaults, set before mujoco / torch load: headless EGL rendering, and no mid-episode reads of
the simulator's success predicate (SPARK_NO_ORACLE). The protocol scores each trial with one predicate
read after the last action. SPARK_NO_ORACLE=0 restores the mid-episode reads; only the archived
fixed-plan displacement runs used them.
"""
import os

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('SPARK_NO_ORACLE', '1')
