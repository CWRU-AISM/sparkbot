# spark_real

The SPARK pipeline served as a FastAPI web UI on a UR10e, a Franka FR3 and a
bimanual Franka. No simulated numbers in the
paper depend on this module.

Install and launch per robot: [docs/real_world.md](../../docs/real_world.md).

Grippers and cameras are not tied to a family. `gripper.type`
(`grippers.type` for bimanual) names a driver from `spark_real/grippers`
(robotiq_2f85, franka_hand, dynamixel, ssg48). `cameras[].type` names a
camera driver (kinect, realsense, zed_mini, webcam). An unregistered type,
or one whose SDK or transport is missing, raises at startup. The bimanual
rig's typed primitives are in `skills/bimanual_primitives.py`. The
Kinect and RealSense SDKs, the ZED SDK and SAM3 are installed separately
(see docs/setup.md). Bring-up per platform: docs/real_world.md. `spark_real/scripted/` holds demos that are not SPARK. They are fixed
whole-task procedures without the planner, for example the bimanual
silverware sort.

## Data flow

```
RGB + depth (Azure Kinect x2, wrist RealSense)
    |  SAM3 text prompts, 3D back-projection        perception_capture.py, spark_core.sam3
    |  cross-camera merge, fusion gate, task contract perception_merge.py, perception_fusion.py,
    v                                                perception_task.py
detection map (label -> position_3d, OBB, ...)
    |  BT cache first, else one planner call         pipeline_plan.py, bt_library.py, spark_core.planner
    v
behavior tree (YAML)
    |  ScoreExecutor walks the tree                  control/score_executor.py and mixins
    v
UR10eDriver (RTDE + URScript socket) behind SafeRobot (CBF filter)
    |  verify the task, recover, replay on evidence  control/success_verifier.py, executor_verify.py
```

## Launch

```bash
cp src/spark_real/configs/machines/example.yaml src/spark_real/configs/machines/<host>.yaml
# edit robot.ip, the Kinect serials and gripper.script in <host>.yaml
export SPARK_OUTPUT_DIR=$PWD/output          # runs, calibrations, BT library
PYTHONPATH=src python -m spark_real.server --robot ur10e --machine <host> --port 8888
```

`--no-robot` runs perception and planning without connecting the arm,
`--no-kinect` skips the Kinects, `--no-init` serves the UI without building
the pipeline. Open `http://localhost:8888`.

## Configuration

- `configs/ur10e_default.yaml`: control box, grasp tuning, verification,
  timeouts, camera roster, BT cache and perception settings.
- `configs/machines/<host>.yaml` (not tracked): robot IP, Kinect and
  RealSense serials, gripper script path. Deep-merged over the default.
- `configs/tasks/*.yaml`: frozen per-task SAM3 prompt contracts.
- `configs/bt_seeds/*.yaml`: seed behavior trees for the demo tasks.

Environment:

| Variable | Effect |
|---|---|
| `SPARK_OUTPUT_DIR` | output root (default `./output`); calibrations live in `<root>/calibrations` |
| `SPARK_GRIPPER_SCRIPT` | Robotiq URScript export, when `gripper.script` is unset |
| `SPARK_LLM_BACKEND` | `openai` (default: any OpenAI-compatible endpoint), `anthropic` or `gemini` |
| `SPARK_LLM_MODEL` | model name; default Qwen3.8-27B on `openai`, claude-opus-5-5 on `anthropic`, gemini-3.5-flash on `gemini` |
| `SPARK_VERIFY_LLM` | `0` stops the planner LLM from checking runs that geometry could not verify |
| `OPENAI_BASE_URL` | endpoint, e.g. `http://127.0.0.1:8000/v1` from `scripts/serve_planner.sh` |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` | planner credentials (`anthropic` and `gemini` also need their uv extra) |
| `ANTHROPIC_WORKSPACE_ID` | only for an Anthropic key not scoped to a workspace |

## Robotiq gripper script

The driver sends the Robotiq URCap function definitions with every gripper
command. They are not redistributed here. Export them from the pendant: save
any program on a controller with the Robotiq Grippers URCap installed, copy the
generated `.script` file off the controller, and point `gripper.script` (or
`SPARK_GRIPPER_SCRIPT`) at it. Without it the arm still moves; gripper
commands fail.

## Calibration

`calibration/handeye.py` solves each camera's extrinsic from AprilTag views
and writes `<SPARK_OUTPUT_DIR>/calibrations/handeye_<camera>.json`; the
pipeline loads them at start. `POST /api/calibrate_table` records the table
plane used as the descent floor.

## Tests

Hardware-free tests run without a robot or cameras:

```bash
PYTHONPATH=src python -m pytest src/spark_real/tests
```
