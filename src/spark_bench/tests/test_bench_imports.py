"""The LIBERO-PRO runner pulls in no spark_real, camera or robot code."""
import subprocess
import sys

import pytest

HARDWARE = ("spark_real", "pyk4a", "pyrealsense2", "rtde_control", "rtde_receive", "depth_anything_3")


def test_runner_imports_no_hardware():
    pytest.importorskip('libero.libero')
    code = ("import importlib, sys\n"
            "for m in ('spark_bench.run_libero_pro', 'spark_bench.summarize'): importlib.import_module(m)\n"
            f"print(sorted(m for m in sys.modules if m.split('.')[0] in {HARDWARE!r}))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stdout
