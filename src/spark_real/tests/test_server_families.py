"""
The server boots the franka and bimanual pipelines without a robot and
answers /api/status with 200. No device is opened: Kinects, RealSense and
the camera roster are off, and SAM3 loading is stubbed (the GPU is not
needed to check the wiring).

Usage:
    PYTHONPATH=src python -m pytest src/spark_real/tests/test_server_families.py -q
"""

from __future__ import annotations

import pytest

from spark_real.config import SparkConfig, load_profile


@pytest.mark.parametrize("family,expected", [("franka", "franka"), ("bimanual", "bimanual_franka")])
def test_status_is_200_without_devices(monkeypatch, tmp_path, family, expected):
    monkeypatch.setenv("SPARK_OUTPUT_DIR", str(tmp_path))
    from fastapi.testclient import TestClient

    import spark_core.sam3 as sam3
    from spark_real import server
    from spark_real.pipeline import PipelineConfig, SPARKRealPipeline
    from spark_real.routes import state

    monkeypatch.setattr(sam3.SPARKPerception, "load_models", lambda self, load_da3=True: None)
    cfg = SparkConfig(family=family, no_robot=True, no_kinect=True)
    profile = load_profile(cfg)
    profile.raw.update(use_realsense=False, use_kinect=False, cameras=[])
    pipeline = SPARKRealPipeline(PipelineConfig(**profile.to_pipeline_kwargs()), profile=profile)
    pipeline.initialize()
    monkeypatch.setattr(state, "pipeline", pipeline)
    monkeypatch.setattr(state, "launch_config", cfg)

    client = TestClient(server.app)
    status = client.get("/api/status")
    assert status.status_code == 200
    body = status.json()
    assert body["robot_family"] == expected
    assert body["kinect_connected"] is False and body["realsense_connected"] is False
    assert client.get("/api/robot").json()["family"] == expected
    assert client.get("/api/bimanual/state").status_code == 400  # no robot connected
