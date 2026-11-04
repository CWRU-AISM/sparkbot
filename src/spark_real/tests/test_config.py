"""load_profile -> PipelineConfig per family, with overlays and flags."""

import pytest

from spark_real import config as config_mod
from spark_real.config import SparkConfig, load_profile
from spark_real.pipeline_types import PipelineConfig


def test_default_profile_builds_a_pipeline_config():
    profile = load_profile(SparkConfig(family="ur10e"))
    kwargs = profile.to_pipeline_kwargs()
    cfg = PipelineConfig(**kwargs)
    assert cfg.robot_family == "ur10e"
    assert cfg.kinect_resolution == "720P"
    assert cfg.kinect_fps == 15
    assert kwargs["robot_ip"] == "127.0.0.1"  # placeholder until an overlay sets it


def test_flags_override_the_yaml():
    profile = load_profile(SparkConfig(family="ur10e", ip="192.0.2.7", strict_verify=True))
    kwargs = profile.to_pipeline_kwargs()
    assert kwargs["robot_ip"] == "192.0.2.7"
    assert kwargs["strict_placement_verify"] is True
    assert load_profile(SparkConfig(family="ur10e", no_robot=True)).to_pipeline_kwargs()[
        "robot_ip"
    ] == ""


def test_machine_overlay_deep_merges(tmp_path, monkeypatch):
    (tmp_path / "machines").mkdir()
    (tmp_path / "ur10e_default.yaml").write_text(
        "robot: {ip: null, frequency: 500}\nkinect_fps: 15\n"
    )
    (tmp_path / "machines" / "rig.yaml").write_text(
        "robot: {ip: 192.0.2.10}\nkinect_master_serial: 'abc'\n"
    )
    monkeypatch.setattr(config_mod, "CONFIGS_DIR", tmp_path)
    profile = load_profile(SparkConfig(family="ur10e", machine="rig"))
    assert profile.raw["robot"] == {"ip": "192.0.2.10", "frequency": 500}
    kwargs = profile.to_pipeline_kwargs()
    assert kwargs["robot_ip"] == "192.0.2.10"
    assert kwargs["kinect_master_serial"] == "abc"


def test_output_root_follows_the_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARK_OUTPUT_DIR", str(tmp_path / "out"))
    assert config_mod.output_root() == (tmp_path / "out").resolve()
    assert config_mod.calibrations_dir() == (tmp_path / "out" / "calibrations").resolve()
    assert PipelineConfig().output_dir == str((tmp_path / "out" / "real_runs").resolve())


@pytest.mark.parametrize(
    "arg,family",
    [("ur10e", "ur10e"), ("franka", "franka"), ("bimanual", "bimanual_franka"),
     ("bimanual_franka", "bimanual_franka")],
)
def test_robot_flag_selects_the_family(arg, family):
    import tyro

    assert tyro.cli(SparkConfig, args=["--robot", arg]).family == family


def test_unknown_family_is_rejected_at_the_cli():
    import tyro

    with pytest.raises(SystemExit):
        tyro.cli(SparkConfig, args=["--family", "g1"])
