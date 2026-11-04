"""Robotiq 2F-85 over URScript: activation, open/close, width, and register state."""

import logging
import os
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class RobotiqMixin:
    """Robotiq 2F-85 over URScript: activation, open/close, width, and register state."""

    def _load_gripper_functions(self):
        """
        Load the Robotiq gripper URScript function defs.

        The script is the Robotiq URCap installation export (a program saved
        on the pendant with the gripper URCap installed); it is not shipped.
        First existing of: the gripper_script argument (config
        gripper.script), $SPARK_GRIPPER_SCRIPT, robots/ur10e/grippy.script.
        """
        search_paths = [
            Path(p).expanduser()
            for p in (self._gripper_script_path, os.environ.get("SPARK_GRIPPER_SCRIPT"))
            if p
        ]
        search_paths.append(Path(__file__).parent.parent / "robots/ur10e/grippy.script")
        for script_path in search_paths:
            if not script_path.exists():
                continue
            try:
                full_script = script_path.read_text(encoding="utf-8")
                lines = full_script.split("\n")
                gripper_start = None
                gripper_end = None
                for i, line in enumerate(lines):
                    if "#   Type: Gripper" in line:
                        for j in range(i, max(0, i - 10), -1):
                            if (
                                "# begin: URCap Installation Node" in lines[j]
                                and "#   Source: Robotiq_Grippers" in lines[j + 1]
                            ):
                                gripper_start = j
                                break
                    if (
                        gripper_start
                        and "# end: URCap Installation Node" in line
                        and i > gripper_start + 10
                    ):
                        gripper_end = i + 1
                        break
                if gripper_start and gripper_end:
                    self._gripper_script_header = "\n".join(
                        lines[gripper_start:gripper_end]
                    )
                    logger.info("Loaded Robotiq gripper functions")
                    return
            except Exception as e:
                logger.info("Gripper functions not loaded from %s: %s", script_path, e)
        logger.warning(
            "Robotiq gripper script not found (set gripper.script or "
            "SPARK_GRIPPER_SCRIPT); gripper commands will not work"
        )

    def activate_gripper(self):
        """
        Activate gripper (required once after power-on).
        """
        self._check_connected()
        # Sent over the primary-interface URScript socket (_send_script),
        # the same transport speedl uses, rather than RTDEControl's
        # sendCustomScriptFunction: this ur_rtde build only accepts the
        # 2-arg (name, script) form AND auto-wraps+indents the body into a
        # def, which double-wraps our already-self-invoking scripts. The
        # header is embedded so rq_* are defined even when the running
        # program has not loaded the Robotiq URCap; the script self-invokes.
        if self._gripper_script_header:
            # reset=True forces deactivate -> re-activate, which re-runs the
            # Robotiq auto-calibration (physically cycles the gripper to relearn
            # its full open/close stroke). Plain rq_activate_and_wait only
            # activates if needed and keeps the stale calibration that was
            # capping the close short of the mechanical stop.
            script = (
                "def rq_activate():\n"
                f"{self._gripper_script_header}\n"
                "  rq_activate_all_grippers(True)\n"
                "end\n"
                "rq_activate()\n"
            )
        else:
            script = "def rq_activate():\n  rq_activate_and_wait()\nend\nrq_activate()\n"
        if not self._send_script(script):
            raise RuntimeError("gripper activation dropped: URScript channel down")
        time.sleep(5.0)  # reset+re-activate cycles the gripper (~3-5s)
        logger.info("Gripper reset + activated (re-calibrated)")

    def open_gripper(self, speed: int = None, force: int = None):
        """
        Open the Robotiq 2F-85 gripper.
        """
        self._send_gripper_command(0.0, speed, force)

    def close_gripper(self, speed: int = None, force: int = None):
        """
        Fully close the Robotiq 2F-85: drive to the MECHANICAL stop (rPR=255).

        rq_move_and_wait_norm(100) stops short of the mechanical stop (rPR~226,
        a ~29/255 gap), so the FULL close uses rq_close_and_wait() to drive the
        raw position register to the hard stop. Intermediate widths keep the
        norm-scaled path (set_gripper_position), where the mapping is correct.
        """
        # Latched here as well as in _send_gripper_command: the script-header
        # branch below bypasses that method entirely.
        self._command_latch.latch_gripper(1.0)
        self._check_connected()
        spd = int(np.clip(speed if speed is not None else 100, 0, 100))
        frc = int(np.clip(force if force is not None else 100, 0, 100))
        if self._gripper_script_header:
            script = (
                "def gripper_close():\n"
                f"{self._gripper_script_header}\n"
                f'    rq_set_force_norm({frc}, "1")\n'
                f'    rq_set_speed_norm({spd}, "1")\n'
                '    rq_close_and_wait("1")\n'
                "end\n"
                "gripper_close()\n"
            )
            if not self._send_script(script):
                raise RuntimeError(
                    "gripper close dropped: URScript channel down"
                )
            time.sleep(0.5)
        else:
            self._send_gripper_command(1.0, speed, force)

    def set_gripper_position(
        self, position: float, speed: int = None, force: int = None
    ):
        """
        Set gripper to specific position.

        Args:
            position: 0.0 (fully open) to 1.0 (fully closed)
            speed: 0-100 (default: 100)
            force: 0-100 (default: 50)
        """
        self._send_gripper_command(position, speed, force)

    def _send_gripper_command(
        self, position: float, speed: int = None, force: int = None
    ):
        """
        Send command to Robotiq gripper via URScript.

        Args:
            position: 0.0 (open) to 1.0 (closed)
            speed: 0-100 percent (default: 100)
            force: 0-100 percent (default: 50)
        """
        # Single funnel for open_gripper / set_gripper_position / the
        # no-header close: the commanded target is the recorder's label.
        self._command_latch.latch_gripper(position)
        self._check_connected()
        speed = speed if speed is not None else 100
        force = force if force is not None else 50
        # rq_*_norm take percent (0-100). configs/ur10e_default.yaml carries
        # gripper.speed/force = 255 (the raw Robotiq register scale); those
        # CLIP to 100 here, which is full speed/force, so the legacy YAML
        # values still mean "max" rather than overflowing the norm command.
        pos_pct = int(np.clip(position * 100, 0, 100))
        spd = int(np.clip(speed, 0, 100))
        frc = int(np.clip(force, 0, 100))

        if self._gripper_script_header:
            script = f"""def gripper_move():
{self._gripper_script_header}

    rq_set_force_norm({frc}, "1")
    rq_set_speed_norm({spd}, "1")
    rq_move_and_wait_norm({pos_pct}, "1")
end
gripper_move()
"""
        else:
            # Fallback without script header (may not work on all setups)
            pos_raw = int(np.clip(position * 255, 0, 255))
            spd_raw = int(np.clip(speed * 2.55, 0, 255))
            frc_raw = int(np.clip(force * 2.55, 0, 255))
            script = (
                f"def gripper_cmd():\n"
                f"  rq_set_speed({spd_raw})\n"
                f"  rq_set_force({frc_raw})\n"
                f"  rq_move_and_wait({pos_raw})\n"
                f"end\n"
            )
        # Self-contained, self-invoking script -> primary-interface URScript
        # socket (see activate_gripper for why not sendCustomScriptFunction).
        if not self._send_script(script):
            raise RuntimeError("gripper command dropped: URScript channel down")
        time.sleep(0.5)

    def grasp_to_width(
        self,
        width: float,
        force: float = None,
        speed: float = None,
        epsilon_inner: float = None,
        epsilon_outer: float = None,
    ) -> bool:
        """
        Close the Robotiq to a target jaw width in METERS.

        The ScoreExecutor (_gripper_squeeze) targets a mask-derived grasp
        width through this call. The executor passes
        ``speed`` as a 0-1 fraction; both that and a raw 0-100 percent are
        accepted. The 2F-85 has no width API, so width is mapped to a
        percent-closed command via GRIPPER_MAX_WIDTH_M. epsilon_* are
        accepted for signature parity and ignored (the Robotiq has no grasp
        tolerance window).
        """
        w = float(np.clip(width, 0.0, self.GRIPPER_MAX_WIDTH_M))
        # 0 m -> fully closed (1.0); max width -> fully open (0.0).
        position = 1.0 - (w / self.GRIPPER_MAX_WIDTH_M)
        spd = speed
        if spd is not None and spd <= 1.0:
            spd = spd * 100.0  # fractional 0-1 -> percent
        frc = force
        if frc is not None:
            frc = min(float(frc), 100.0)
        try:
            self._send_gripper_command(position, speed=spd, force=frc)
            return True
        except Exception as e:
            logger.warning("grasp_to_width(%.3f) failed: %s", width, e)
            return False

    def get_gripper_width(self) -> float:
        """
        Current jaw opening in meters (0=closed, GRIPPER_MAX_WIDTH_M=open).
        """
        try:
            pos255 = self.get_gripper_position()  # 0=open, 255=closed
        except Exception:
            return 0.0
        frac_closed = float(np.clip(pos255 / 255.0, 0.0, 1.0))
        return (1.0 - frac_closed) * self.GRIPPER_MAX_WIDTH_M

    def _publish_gripper_state(self, force: bool = False):
        """Push gripper position + object-detect into RTDE output registers.

        The Robotiq state (rq_current_pos_norm / rq_is_object_detected) is only
        reachable from URScript, but a FIRE-AND-FORGET _send_script that WRITES
        an output register does NOT need the rtde_c control script and never
        spins (unlike a sendCustomScriptFunction round-trip, which hangs once
        _send_script has stopped the control script). The values are then read
        back with rtde_r.getOutputIntRegister, a pure receive read, also
        spin-free. The embedded URCap header defines the rq_* functions.
        """
        # Do not publish gripper state while the ARM is moving: _send_script
        # replaces the running URScript program, so the _pub_grip write mid-move
        # cancels the in-flight movel/movej/speedl. Return immediately and let
        # the read fall back to the last-published register values (the gripper
        # isn't actuating during a move, so they're current). The grasp squeeze
        # moves the gripper with the arm STATIONARY, so its verify reads still
        # refresh.
        #
        # A measured TCP speed > 0.01 m/s guard alone has two holes: a
        # Cartesian servo converging on its target drops BELOW 0.01 m/s while
        # still emitting speedl every 8 ms, and a fire-and-forget movej is
        # stationary for its whole ~1 s upload lag. A publish landing in either
        # hole cancels the motion, which then re-accelerates on the next tick.
        # _motion_in_flight closes both with a command-side lease.
        #
        # ``force`` is for a caller that KNOWS the arm is parked and needs a
        # fresh sample to decide something (a drop check at a transport
        # waypoint). It skips the lease (only a pessimistic upper bound on an
        # unconfirmed move) but still honours measured motion, so it can never
        # cancel a move that is genuinely running.
        if force:
            if self._tcp_is_moving():
                return
        elif self._motion_in_flight():
            return

        header = self._gripper_script_header or ""
        # write_output_integer_register requires an INTEGER: rq_is_object_detected
        # returns a Bool and rq_current_pos_norm a float (0-100), so floor the
        # position and map the bool to 0/1 via if/else. A raw bool write throws
        # "Must be an integer, not 'Bool'" and leaves register 13 unwritten.
        # Sequence token cycles 1..255 (never 0, so a cold register reads as
        # "not yet acknowledged" rather than matching by accident).
        self._publish_seq = getattr(self, "_publish_seq", 0) % 255 + 1
        seq = self._publish_seq
        script = (
            "def _pub_grip():\n"
            f"{header}\n"
            f'  write_output_integer_register({self._POS_REGISTER}, floor(rq_current_pos_norm("1")))\n'
            "  obj_i = 0\n"
            '  if (rq_is_object_detected("1")):\n'
            "    obj_i = 1\n"
            "  end\n"
            f"  write_output_integer_register({self._OBJ_REGISTER}, obj_i)\n"
            f"  write_output_integer_register({self._SEQ_REGISTER}, {seq})\n"
            "end\n"
            "_pub_grip()\n"
        )
        if not self._send_script(script):
            return
        self._await_publish(seq)

    def _await_publish(self, seq: int) -> bool:
        """
        Block until the controller acknowledges publish ``seq``.

        The token is written last, so seeing it means registers 12 and 13
        already hold this publish's values. Returns False on timeout, in which
        case the caller reads whatever the registers currently hold.
        """
        deadline = time.monotonic() + self._PUBLISH_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                if self._rtde_r.getOutputIntRegister(self._SEQ_REGISTER) == seq:
                    return True
            except Exception:  # noqa: BLE001 - unsupported register: just wait
                time.sleep(max(0.0, deadline - time.monotonic()))
                return False
            time.sleep(self._PUBLISH_POLL_S)
        return False

    def get_gripper_position(self, publish: bool = True) -> float:
        """Current gripper position, 0 (open) to 255 (closed). Spin-free.

        rq_current_pos_norm is 0-100; scaled to 0-255 to match the Robotiq
        convention the verify thresholds use.

        ``publish=False`` reads the last-published register without uploading
        a URScript program. Use it from any loop that runs CONCURRENTLY with
        motion (the demo recorder), where the refresh is worth less than the
        motion it would cancel.
        """
        self._check_connected()
        if publish:
            self._publish_gripper_state()
        return self._rtde_r.getOutputIntRegister(self._POS_REGISTER) * 2.55

    def is_object_detected(self, publish: bool = True) -> bool:
        """Robotiq gOBJ flag: True when the jaws stopped on an object (gOBJ
        1/2) rather than reaching the commanded position empty (gOBJ 3). The
        reliable grasp signal; detects a lateral grip TCP force can't. Read
        spin-free via the output register (see _publish_gripper_state)."""
        self._check_connected()
        if publish:
            self._publish_gripper_state()
        return self._rtde_r.getOutputIntRegister(self._OBJ_REGISTER) == 1
