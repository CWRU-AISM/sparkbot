"""
Bimanual rig routes under /api/bimanual. Every handler answers 400 unless
the running pipeline is the bimanual_franka family.

GET  /state                 both arms: joints, flange pose, jaw-tip pose, jaw width
POST /{arm}/gripper         {"action": "open"|"close", "force", "speed"}
POST /{arm}/home            home one arm
POST /home_both             home left, then right
GET  /inter_arm_distance    jaw-tip to jaw-tip distance (m)
"""

import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from spark_real.routes import state

logger = logging.getLogger("spark_server")
router = APIRouter(prefix="/api/bimanual")
_ARMS = ("left", "right")


class ArmGripperRequest(BaseModel):
    action: str  # "open" | "close"
    force: Optional[float] = None
    speed: Optional[float] = None


def _driver(arm: Optional[str] = None):
    """(bimanual driver, None) or (None, error response)."""
    pipeline = state.pipeline
    if pipeline is None or getattr(pipeline, "_robot", None) is None:
        return None, JSONResponse(status_code=400, content={"error": "robot not connected"})
    if (getattr(pipeline.config, "robot_family", "") or "").lower() != "bimanual_franka":
        return None, JSONResponse(status_code=400, content={"error": "active robot is not bimanual_franka"})
    if arm is not None and arm not in _ARMS:
        return None, JSONResponse(status_code=400, content={"error": "arm must be left|right"})
    return pipeline._robot, None


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _call(what: str, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return {"status": "ok"}
    except Exception as e:
        logger.exception("bimanual %s failed", what)
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/state")
def bimanual_state():
    drv, err = _driver()
    if err is not None:
        return err
    try:
        obs = drv.get_observation()
        return _jsonable(obs.to_dict() if hasattr(obs, "to_dict") else obs)
    except Exception as e:
        logger.exception("bimanual state failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/{arm}/gripper")
def arm_gripper(arm: str, body: ArmGripperRequest):
    drv, err = _driver(arm)
    if err is not None:
        return err
    if body.action not in ("open", "close"):
        return JSONResponse(status_code=400, content={"error": "action must be open|close"})
    fn = drv.open_gripper if body.action == "open" else drv.close_gripper
    return _call("gripper", fn, arm=arm, speed=body.speed, force=body.force)


@router.post("/{arm}/home")
def arm_home(arm: str):
    drv, err = _driver(arm)
    if err is not None:
        return err
    return _call("home", drv.go_home, arm=arm)


@router.post("/home_both")
def home_both():
    drv, err = _driver()
    if err is not None:
        return err
    return _call("home_both", drv.go_home)


@router.get("/inter_arm_distance")
def inter_arm_distance():
    drv, err = _driver()
    if err is not None:
        return err
    return {"distance_m": float(drv.get_inter_arm_distance())}
