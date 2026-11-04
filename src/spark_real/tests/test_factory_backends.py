"""
make_robot_driver builds, or raises RuntimeError for a missing backend, for
every family. Never TypeError: a kwarg the driver does
not take, or a value landing in the wrong positional slot, is a factory bug
(SMOKE-01/02). The gripper is any registered type; the config always names
it. TEST-NET IPs only; nothing here connects.

Usage:
    PYTHONPATH=src python -m pytest src/spark_real/tests/test_factory_backends.py -q
"""

from __future__ import annotations

import pytest

from spark_real.robots.driver_protocol import RobotDriver
from spark_real.robots.factory import KNOWN_FAMILIES, make_robot_driver

IP = "192.0.2.1"

DYN = {"type": "dynamixel", "left": {"id": 1}, "right": {"id": 2}}
CASES = [
    ("ur10e", None, {"gripper": {"type": "robotiq_2f85"}}),
    ("franka", None, {"gripper": {"type": "franka_hand"}}),
    ("bimanual_franka", "bamboo", {"robot": {"left_ip": "192.0.2.2", "backend": "bamboo"}, "gripper": DYN}),
]


def _build(monkeypatch, family, backend, **kwargs):
    try:
        return make_robot_driver(family, IP, **kwargs)
    except RuntimeError as exc:
        # The driver's own "backend not installed" guard; the only allowed
        # failure. TypeError and everything else propagate and fail the test.
        assert "install" in str(exc).lower(), exc
        return None


@pytest.mark.parametrize("family,backend,kwargs", CASES)
def test_builds_or_raises_runtime_error(monkeypatch, family, backend, kwargs):
    driver = _build(monkeypatch, family, backend, **kwargs)
    if driver is None:
        pytest.skip(f"{family}/{backend or '-'} backend not installed here")
    assert isinstance(driver, RobotDriver)


@pytest.mark.parametrize("family,backend,kwargs", CASES)
def test_frequency_kwarg_is_accepted(monkeypatch, family, backend, kwargs):
    # run.py passes frequency positionally; it must not shift into another slot.
    _build(monkeypatch, family, backend, frequency=1000.0, **kwargs)


def test_known_families_are_all_covered():
    assert set(KNOWN_FAMILIES) == {c[0] for c in CASES}


def test_unknown_family_is_value_error():
    with pytest.raises(ValueError):
        make_robot_driver("kuka", IP, gripper={"type": "robotiq_2f85"})


def test_any_family_mounts_any_registered_jaw():
    ur = make_robot_driver("ur10e", IP, gripper={"type": "ssg48", "channel": "/dev/x"})
    assert ur.GRIPPER_TYPE == "ssg48" and type(ur._robot).__name__ == "UR10eDriver"


@pytest.mark.parametrize(
    "family,kwargs",
    [
        ("ur10e", {"gripper": {}}),
        ("ur10e", {"gripper": {"type": "parallel_jaw"}}),
        ("franka", {"gripper": {"type": "franka_hand"}, "robot": {"backend": "bamboo"}}),
        ("bimanual_franka", {"robot": {"left_ip": "192.0.2.2"}, "gripper": {}}),
        ("bimanual_franka", {"robot": {"left_ip": "192.0.2.2", "backend": "torque"}, "gripper": DYN}),
    ],
)
def test_unregistered_gripper_or_unsupported_backend_is_value_error(family, kwargs):
    with pytest.raises(ValueError):
        make_robot_driver(family, IP, **kwargs)


def test_bimanual_needs_both_arm_ips():
    with pytest.raises(ValueError):
        make_robot_driver("bimanual_franka", "", gripper=DYN)


def test_robotiq_without_a_urscript_transport_is_runtime_error(monkeypatch):
    import spark_real.robots.franka.franka_driver as fd

    monkeypatch.setattr(fd, "HAS_FRANKY", True)
    with pytest.raises(RuntimeError, match="URScript"):
        make_robot_driver("franka", IP, gripper={"type": "robotiq_2f85"})
    with pytest.raises(RuntimeError, match="URScript"):
        make_robot_driver("bimanual_franka", IP, robot={"left_ip": "192.0.2.2"}, gripper={"type": "robotiq_2f85"})
