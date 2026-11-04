"""Components of the LIBERO-PRO runner (``spark_bench.run_libero_pro``).

* ``executor.py``, ``binding.py``, ``grasp_close.py`` - ``LiberoExecutor``: state, binding, grasp close.
* ``primitives/`` - per-primitive handlers.
* ``motion.py``, ``ik.py`` - OSC moves, gripper helpers, IK helpers.
* ``perception/`` - prompt selection, SAM3 detection, relation binding, wrist verify.
* ``planning/`` - LLM planner and the scripted (LLM-free) plan.
* ``bddl.py`` - BDDL parsing to SAM3 prompts.
"""
