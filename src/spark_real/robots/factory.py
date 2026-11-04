"""Robot driver factory: one family name plus its config blocks -> a driver.

Families: ur10e (ur_rtde), franka (one FR3/Panda through franky) and
bimanual_franka (Panda + FR3, franky or bamboo arms). The gripper is not
tied to the family: `gripper.type` (`grippers.type` for bimanual) names a
driver from spark_real.grippers, mounted on the arm here. Backend imports
are deferred to their branch, so a host only needs its own install extra.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from spark_real.grippers import make_dual_gripper, mount_gripper
from spark_real.robots.driver_protocol import missing_members

logger = logging.getLogger(__name__)

KNOWN_FAMILIES = ("ur10e", "franka", "bimanual_franka")

# Config `robot.backend` each family accepts; the first is the default.
SUPPORTED_BACKENDS = {
    "ur10e": ("ur_rtde",),
    "franka": ("franky",),
    "bimanual_franka": ("franky", "bamboo"),
}


def check_family(family: str) -> str:
    fam = (family or "").strip().lower()
    if fam not in KNOWN_FAMILIES:
        raise ValueError(f"Unknown robot family {family!r}. Expected one of: {', '.join(KNOWN_FAMILIES)}.")
    return fam


def _backend(fam: str, robot: Mapping[str, Any]) -> str:
    allowed = SUPPORTED_BACKENDS[fam]
    value = str(robot.get("backend") or allowed[0]).lower()
    if value not in allowed:
        raise ValueError(f"robot.backend {value!r} is not supported for {fam!r}; expected one of {allowed}")
    return value


def make_robot_driver(
    family: str,
    robot_ip: str,
    frequency: float | None = None,
    gripper: Optional[Mapping[str, Any]] = None,
    robot: Optional[Mapping[str, Any]] = None,
    collision: Optional[Mapping[str, Any]] = None,
):
    """Build the arm driver for ``family``, mount the configured gripper, and
    check the result implements RobotDriver.

    ``gripper``, ``robot`` and ``collision`` are the family config's
    `gripper:` (`grippers:` for bimanual), `robot:` and `collision_behavior:`
    blocks. Raises ValueError for an unknown family, backend or gripper type,
    RuntimeError when the gripper's transport cannot work, and TypeError
    when the driver lacks a RobotDriver member.
    """
    fam = check_family(family)
    gripper = dict(gripper or {})
    robot = dict(robot or {})
    backend = _backend(fam, robot)

    if fam == "ur10e":
        from spark_real.control.ur10e_driver import UR10eDriver

        freq = 500.0 if frequency is None else float(frequency)
        arm = UR10eDriver(robot_ip, freq, gripper_script=gripper.get("script"))
        driver = mount_gripper(arm, gripper)
    elif fam == "franka":
        from spark_real.robots.franka.franka_driver import FrankaDriver

        freq = 1000.0 if frequency is None else float(frequency)
        arm = FrankaDriver(robot_ip, freq, collision=collision)
        if robot.get("home_config") is not None:
            arm.HOME_CONFIG = [float(v) for v in robot["home_config"]]
        driver = mount_gripper(arm, gripper)
    else:
        from spark_real.robots.bimanual_franka import BimanualFrankaDriver

        freq = 1000.0 if frequency is None else float(frequency)
        # --ip, when given, names the right arm; robot.left_ip/right_ip otherwise.
        ips = {"left": str(robot.get("left_ip", "")), "right": str(robot_ip or robot.get("right_ip", ""))}
        driver = BimanualFrankaDriver(
            left_ip=ips["left"],
            right_ip=ips["right"],
            dual_gripper=make_dual_gripper(gripper, ips),
            backend=backend,
            frequency=freq,
            home_left=robot.get("home_config_left"),
            home_right=robot.get("home_config_right"),
            collision=collision,
            jaw_offset_m=float(gripper.get("tcp_offset_m", 0.0)),
        )
    logger.info("make_robot_driver: %s @ %s (%s, gripper %s)", fam, robot_ip, backend, driver.GRIPPER_TYPE)
    missing = missing_members(driver)
    if missing:
        raise TypeError(f"{type(driver).__name__} does not implement RobotDriver; missing: {missing}")
    return driver
