"""spark_core is hardware-free: importing every module pulls in no spark_real, camera or robot code."""
import pkgutil
import subprocess
import sys

import spark_core

HARDWARE = ("spark_real", "pyk4a", "pyrealsense2", "rtde_control", "rtde_receive", "depth_anything_3")


def test_core_imports_no_hardware():
    mods = sorted(m.name for m in pkgutil.iter_modules(spark_core.__path__, "spark_core."))
    assert {"spark_core.planner", "spark_core.sam3", "spark_core.sim_detect"} <= set(mods)
    code = ("import importlib, sys\n"
            f"for m in {mods!r}: importlib.import_module(m)\n"
            f"print(sorted(m for m in sys.modules if m.split('.')[0] in {HARDWARE!r}))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout
