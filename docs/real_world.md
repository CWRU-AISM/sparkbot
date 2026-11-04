# Real world deployment

`src/spark_real` serves the SPARK pipeline as a web UI (FastAPI, port 8888) on three robots. Each
robot has its own extra:

| `--robot` | Hardware | Install |
|---|---|---|
| `ur10e` | UR10e, Robotiq 2F-85, two Azure Kinects, wrist RealSense | `uv sync --extra bench --extra dev --extra ur10e` |
| `franka` | Franka FR3 and Franka Hand (franky), same cameras | `uv sync --extra bench --extra dev --extra franka` |
| `bimanual` | Panda and FR3, SSG-48 or Dynamixel jaws, ZED Mini and two wrist webcams | `uv sync --extra bench --extra dev --extra bimanual` |

List every extra you use in one `uv sync`: uv removes the packages of extras left out.
The Azure Kinect SDK (`libk4a1.4-dev`, needed by `pyk4a`), the ZED SDK and the libfranka version
your robot runs are installed outside uv. Rig values (IPs, camera serials, calibration) go in
`src/spark_real/configs/machines/<host>.yaml` and the `*_example.yaml` files show the keys.

```bash
source .venv/bin/activate
python -m spark_real.server --robot franka --machine <host> --no-robot   # dry run, no hardware
python -m spark_real.server --robot franka --machine <host>              # then open localhost:8888
```

## Other cameras and grippers

`cameras[].type` and `gripper.type` take a built-in driver or your own class as
`package.module:ClassName`. The entry's other keys are passed to the class as keyword arguments.

```yaml
cameras:
  - role: "front"
    type: "my_rig.cameras:OrbbecCamera"   # open(), close(), read() -> (rgb, depth in m), .config intrinsics
    serial: "AY1234"
gripper:
  type: "my_rig.grippers:RobotiqUSB"      # connect(), open(), close(), move_to_width(), ...
  port: "/dev/ttyUSB0"
```

| | Built in | Your class must provide |
|---|---|---|
| Camera | `kinect`, `realsense`, `zed_mini`, `webcam` | `open()`, `close()`, `read()` returning (rgb HxWx3 uint8, depth HxW in meters or None), `config` with width, height, fx, fy, cx, cy (`perception/camera_base.CameraConfig`) |
| Gripper | `robotiq_2f85`, `franka_hand`, `dynamixel`, `ssg48` | the jaw API in `src/spark_real/grippers/__init__.py`: connect, disconnect, homing, open, close, move_to_width, grasp_to_width, get_present_width, get_position_norm, is_object_detected |

Kinects take their role from the serial in their `cameras:` entry. With one Kinect, or without
serials, they fill the Kinect roles in the order listed.

More on the UR10e pipeline, its configuration and calibration: [src/spark_real/README.md](../src/spark_real/README.md).

# Bring-up checklist: Franka FR3 and bimanual Franka

The UR10e, FR3 and bimanual Franka modules share one pipeline
(`python -m spark_real.server --robot <family>`). The unit tests cover
config parsing, driver construction, gripper dispatch and the executor
with mocked hardware. Only the steps below confirm the platforms on a
real robot.

Common to both:

- `export SPARK_OUTPUT_DIR=<dir>`. Calibrations go to
  `$SPARK_OUTPUT_DIR/calibrations`, run records to `$SPARK_OUTPUT_DIR/real_runs`.
- Copy the machine overlay template and fill in the rig values:
  `src/spark_real/configs/machines/franka_example.yaml` or
  `bimanual_example.yaml` to `configs/machines/<host>.yaml` (gitignored).
  Launch with `--machine <host>`.
- In Desk: unlock the joints and activate FCI on each arm.
- Grippers and cameras are config-selected drivers (see "Other cameras and
  grippers"). An unknown type, or a missing SDK or transport, stops the
  server at startup with the reason.
- Server smoke without the robot first:
  `python -m spark_real.server --robot <family> --machine <host> --no-robot`
  and `curl localhost:8888/api/status`.

### Franka FR3 (single arm, franky)

Install: `uv sync --extra bench --extra dev --extra franka`. franky-control must match the
robot's libfranka version. If `connect()` reports a version mismatch,
build franky against the system libfranka.

1. Overlay: `robot.ip`, the two Kinect serials, `kinect_master_serial`
   (sideview), and `kinect_use_hw_sync: true` only if the sync cable is
   connected.
2. Start `python -m spark_real.server --robot franka --machine <host>`.
   The log must show `franka connected at <ip> (gripper franka_hand,
   SafeRobot off)`. A `gripper.type` error names the missing driver or
   transport (for example, `robotiq_2f85` needs a UR controller).
3. Gripper: in the UI, Open and then Close. The Hand must home on the
   first command and then move.
4. Home: press Home (`/api/home`). The arm goes to `robot.home_config`
   (J5 = -0.15). Teleop at low speed in X, Y and Z. A velocity reflex at
   home means that J5 is too near 0.
5. Calibration: the reference rig's Kinect extrinsics are in
   `src/spark_real/calibration/fr3_reference/handeye_{sideview,birdview}.json`
   (8-9 mm mean error at the time). Copy them to
   `$SPARK_OUTPUT_DIR/calibrations/` only if the cameras did not move.
   Otherwise run `python -m spark_real.calibration.handeye --camera
   birdview` (and `sideview`). Check: detect a block and touch it with
   the closed Hand. The error must be 1 cm or less.
6. Wrist camera: `wrist_tool_offset_xyz` is a placeholder until
   `handeye_wrist.json` exists. `control.wrist_refine` stays false.
7. Table floor: jog the closed Hand to the table and read the TCP z. Set
   `table_z_floor` (default -0.025) to that value plus a few mm.
8. Motion path: `move_to_keypoint` solves FR3 IK with pyroki (first call
   ~10 s JIT) and moves with a franky JointMotion. Run one
   "pick up the <object>" at the default executor speed cap
   (`control.max_velocity: 0.10`). Look for `[movej_via_ik:franka]` in the log.
9. Timeouts: measure the wall-clock time of grasp, release and
   move_to_keypoint, then reduce the values in the `timeouts:` block.
   They are copied from the UR10e.
10. Utensils in tray, 3 trials, to compare with the paper's FR3 row.

### Bimanual Franka (Panda left + FR3 right)

Install: `uv sync --extra bench --extra dev --extra bimanual`, the ZED SDK with pyzed (from
Stereolabs), and a build of the bamboo controller
(`bamboo_control_node`). Set `SPARK_BAMBOO_BINARY` to its path, and set
`SPARK_BAMBOO_LD_PATH` if the binary needs extra library directories. One
node runs per arm on ZMQ ports 5555 (left) and 5556 (right). The driver
starts and stops the nodes. Bamboo logs go to `$SPARK_OUTPUT_DIR/logs/`.

1. Overlay: `robot.left_ip` / `right_ip`, `grippers.type` (`ssg48`
   or `dynamixel`), the jaw buses (`grippers.left.channel` /
   `right.channel` for SSG-48; `port`, `left_id`, `right_id` for
   Dynamixel), and the two webcam `/dev/v4l/by-path/...` devices.
2. Start `python -m spark_real.server --robot bimanual --machine <host>`.
   The log must show `Bimanual Franka connected (backend=bamboo,
   grippers=ssg48, ...)`. It must also list the camera registry roles
   `external, wrist_left, wrist_right`. A camera that fails to open is
   logged and left out. It is never replaced with a simulated camera.
3. Jaw mapping: `POST /api/bimanual/left/gripper {"action":"open"}` must
   move the LEFT jaw. If the right jaw moves, swap the two channels.
4. Home: `POST /api/bimanual/home_both` (left, then right) goes to
   `home_config_left/right`.
5. Geometry to measure (the family defaults are placeholders):
   `robot.base_left/base_right` (inter-arm distance gate),
   `workspace_left/right`, and `grippers.tcp_offset_m` (flange to closed
   jaw tip, 0.1275 m on the reference SSG-48 jaws).
6. Calibration (stop the server for step a, because the ZED is exclusive):
   a. `python -m spark_real.calibration.bimanual_charuco_stereo` with the
      ChArUco board flat in view.
   b. Restart the server. Run
      `python -m spark_real.calibration.bimanual_anchor_tap --grip-offset <tcp_offset_m>`.
      Drive each closed jaw tip onto the 5 corners with the arm pilot
      buttons, as the script prompts. This writes
      `$SPARK_OUTPUT_DIR/calibrations/calibration_bimanual.json`.
   c. Restart the server. The log must show `bimanual calibration loaded`.
      Without the file, the bimanual executor refuses every score.
7. Confirm the frame convention before any grasp (audit item 19). Send a
   `move_to_keypoint_arm` score with `offset_z: 0.10` on each arm. The
   jaw tip must stop 10 cm above the detected object (the executor TCP is
   the jaw tip, flange + `tcp_offset_m` along flange +Z). The left arm
   receives the scene through `T_right_to_left`, so check the left arm
   separately.
8. Typed primitives through the planner path (`/api/execute`), each
   checked by its post-condition, in this order:
   `pick_with_arm` on one arm, then the paper's two-arm silverware BT (a
   `parallel` of two `pick_with_arm` branches, `move_to_keypoint_arm`,
   `place_with_arm`). Then the fold chain `bimanual_grasp_points`,
   `bimanual_lift_together`, `bimanual_arc_drape`. Then `handoff`, and
   `bimanual_lift` plus `bimanual_place` on a tray. These run on the motion
   path of the rig's fold and sort procedures (`bimanual_motion:` block:
   PyRoki flange IK, joint moves, full-current SSG-48 close). Re-check
   `flange_z_offset_m` and `left_z_correction_m` against the measured jaw
   offset before the first grasp. A jaw that closes fully (closure >= 250)
   fails the grasp post-condition. No hardware record exists for
   `handoff`, `bimanual_lift` or `bimanual_place`. Run them first with
   empty jaws, the E-stop in hand, and the meeting point well above the
   table.
9. Demo, not SPARK: the fixed silverware-sort procedure behind the
   paper's bimanual silverware result (SSG-48 only, no planner):
   `python -m spark_real.scripted.sort_silverware --machine <host> --dry-run`
   (IK only), then run it without `--dry-run`. The script calls
   `spark_real.scripted.bimanual_detect`, which needs the calibration from
   step 6. Stop the server first, because the ZED is exclusive. The
   T-shirt fold in the paper was also a fixed procedure; it is not shipped.
   The fold chain in step 8 is its planner-composable form.

### Paper tasks: the skills each plan uses

Every task runs through the full pipeline: perception, a planner call that
writes a behavior tree, and the executor running registered skills. Each
skill checks its post-condition and hands failures to the recovery layer.
Positions come from the detection map. Tuning comes from the plan, then the
family `task_motion:` / `bimanual_motion:` block. The plans below are the
shapes the frozen prompts ask for. `test_paper_tasks_fr3.py` and
`test_paper_tasks_bimanual.py` run each of them end to end on mocked drivers.

| Task | Skills, in plan order | Post-conditions |
|---|---|---|
| FR3 sponge wash | `grasp_se3`(sponge), `move_relative`, `constrained_scrub`(plate), `move_to_keypoint`(sponge), `release` (or one `sponge_wash`) | sponge held after the pick; the pattern completed with the sponge still held |
| FR3 mug pour | `grasp_se3`(mug), `pour`(target: bowl), `move_relative` | mug held; back upright after `max_pour_s` with the mug still held |
| FR3 sweep to dustpan | `grasp_se3`(brush), `sweep`(object_labels, target_label) | brush held; the path ran from behind the objects through the dustpan; with perception, every object ends within the dustpan radius |
| FR3 T-shirt fold | `cloth_fold` (+Y sleeve, -Y sleeve past the center line, re-perceive, hem to collar); each step is a `cloth_fold_single` | each pinch held before its swing; each jaw open after it |
| FR3 utensils in tray | `grasp_se3`(utensil), `move_relative`, `place_in_slot` or `move_to_keypoint`(tray), `release` | utensil held; placement verification |
| Bimanual silverware sort | `parallel`{`pick_with_arm`(left), `move_relative`}, {`pick_with_arm`(right), `move_relative`}; `parallel`{`move_to_keypoint_arm`, `place_with_arm`} per arm | each jaw closed on its utensil (not on air); each jaw open after the place |
| Bimanual T-shirt fold | `bimanual_shirt_fold` = `bimanual_grasp_points`(sleeves), `bimanual_lift_together`, `bimanual_arc_drape`(center line), re-perceive, the same three on the hem corners to the collar | both jaws holding after each pinch and lift; both open after each drape |

An untagged leaf inside a `parallel` branch (the paper plans leave
`move_relative` untagged) runs on the arm of the branch's last tagged leaf.
Not in this release: `pour_to_level` (fill tracking; `pour` holds for
`max_pour_s` instead), side or angled `grasp_se3` strategies (the pick is
top-down, as in the FR3 runs), `bimanual_pants_fold` and
`bimanual_handover_sponge`.
