"""UR10e connection: RTDE, Dashboard preflight, URScript socket, health and reconnect."""

import logging
import socket
import threading
import time

import numpy as np

try:
    import rtde_control
    import rtde_io
    import rtde_receive

    HAS_RTDE = True
except ImportError:
    HAS_RTDE = False

logger = logging.getLogger(__name__)


class URConnectionMixin:
    """UR10e connection: RTDE, Dashboard preflight, URScript socket, health and reconnect."""

    def connect(self):
        """
        Establish RTDE connection to robot.

        Runs a best-effort Dashboard preflight (port 29999) first so a fresh
        connect recovers WITHOUT a trip to the teach pendant: it closes
        blocking popups, unlocks a protective stop, and powers on / releases
        brakes if the arm is idle. The one thing it cannot override is Local
        mode (a physical safety choice); it warns clearly in that case.
        The RTDEControl construction is then retried, because the usual cause
        of "Failed to start control script" is a half-open control script
        left by a previous client's unclean exit, which a dashboard stop +
        retry clears.
        """
        logger.info("Connecting to %s...", self.robot_ip)
        self._dashboard_preflight()
        self._rtde_c = self._connect_control_with_retry()
        # getTCPOffset() fails once a _send_script has replaced the control
        # script (first move), so cache the static pendant TCP offset now. The
        # pyroki IK path (ur10e_ik_pyroki.solve_ik_rtde) needs it to map a
        # gripper-TIP target into the tool0 frame; without it the UR10e falls
        # back to the legacy Cartesian path.
        try:
            self._tcp_offset_cache = np.array(
                self._rtde_c.getTCPOffset(), dtype=float
            )
            logger.info("Cached pendant TCP offset: %s", self._tcp_offset_cache.round(4))
        except Exception as e:
            logger.warning(
                "getTCPOffset() at connect failed (%s); pyroki IK will use the "
                "legacy fallback until this is available.", e,
            )
        self._rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
        # RTDEIOInterface is best-effort: it reserves RTDE input registers and
        # raises "input registers already in use" when a fieldbus (EtherNet/IP,
        # PROFINET, MODBUS) is enabled. It is currently UNUSED (the Robotiq
        # gripper is driven via URScript, not RTDE IO), so a failure here must
        # not abort the connection. Construct opportunistically, None on failure.
        try:
            self._rtde_io = rtde_io.RTDEIOInterface(self.robot_ip)
        except RuntimeError as e:
            self._rtde_io = None
            logger.warning(
                "RTDEIOInterface unavailable (%s); continuing without RTDE IO "
                "(unused; gripper goes through URScript).", e,
            )
        self._open_urscript_socket()
        self._connected = True
        self._load_gripper_functions()
        # The Robotiq 2F-85 deactivates on every controller power-cycle and then
        # silently ignores move commands until re-activated, so activate on
        # connect (the URCap auto-activation this URScript path bypasses).
        # rq_activate_and_wait self-guards, so it is a no-op when already active.
        try:
            self.activate_gripper()
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Gripper activation failed on connect (%s); gripper moves will "
                "be ignored until it is activated.", e,
            )
        logger.info("Connected. Mode: %s", self.get_robot_mode())

    def _dashboard(self):
        """Open a connected DashboardClient, or None if unavailable."""
        try:
            import dashboard_client
        except ImportError:
            return None
        try:
            d = dashboard_client.DashboardClient(self.robot_ip)
            d.connect()
            return d
        except Exception as e:  # noqa: BLE001
            logger.warning("Dashboard (29999) connect failed: %s", e)
            return None

    def _dashboard_preflight(self):
        """Clear states that block RTDE control-script upload. No-op when the
        dashboard is unreachable. Requires Remote Control mode (one-time
        pendant toggle); everything else is recovered programmatically."""
        d = self._dashboard()
        if d is None:
            return
        try:
            if not d.isInRemoteControl():
                logger.warning(
                    "UR is in LOCAL mode; RTDE cannot start the control "
                    "script. Flip the pendant top-right menu to Remote "
                    "Control (one-time); after that, power/brake/protective "
                    "stop are all recovered from here without the pendant."
                )
                return
            for fn in ("closeSafetyPopup", "closePopup"):
                try:
                    getattr(d, fn)()
                except Exception:  # noqa: BLE001
                    pass
            safety = d.safetystatus()
            mode = d.robotmode()
            if "PROTECTIVE_STOP" in safety:
                logger.info("Protective stop -> unlocking via dashboard")
                try:
                    d.unlockProtectiveStop()
                    # UR enforces a ~5s settle after unlock before motion (or a
                    # control-script upload) is accepted. Waiting less is the
                    # usual cause of RTDEControl's "failed to start control
                    # script" right after a protective stop.
                    time.sleep(5.0)
                except Exception as e:  # noqa: BLE001
                    logger.warning("unlockProtectiveStop failed: %s", e)
            if "FAULT" in safety or "VIOLATION" in safety:
                logger.info("Safety fault -> restartSafety via dashboard")
                try:
                    d.restartSafety()
                    time.sleep(2.0)
                except Exception:  # noqa: BLE001
                    pass
            if any(s in mode for s in ("POWER_OFF", "IDLE", "BOOTING")):
                logger.info("Arm not running -> powerOn + brakeRelease")
                try:
                    d.powerOn()
                    time.sleep(0.5)
                    d.brakeRelease()
                    time.sleep(3.0)
                except Exception as e:  # noqa: BLE001
                    logger.warning("powerOn/brakeRelease failed: %s", e)
        finally:
            try:
                d.disconnect()
            except Exception:  # noqa: BLE001
                pass

    def _connect_control_with_retry(self, attempts: int = 3):
        """RTDEControlInterface with retry. A stale control script from a
        prior unclean exit is the common cause of the 5 s start timeout;
        a dashboard stop between tries clears it."""
        last = None
        for i in range(attempts):
            try:
                return rtde_control.RTDEControlInterface(
                    self.robot_ip, self.frequency
                )
            except RuntimeError as e:
                last = e
                logger.warning(
                    "RTDEControl start failed (attempt %d/%d): %s",
                    i + 1, attempts, e,
                )
                d = self._dashboard()
                if d is not None:
                    for fn in ("stop", "closeSafetyPopup", "closePopup"):
                        try:
                            getattr(d, fn)()
                        except Exception:  # noqa: BLE001
                            pass
                    try:
                        d.disconnect()
                    except Exception:  # noqa: BLE001
                        pass
                time.sleep(1.5)
        raise last

    def unlock_protective_stop(self) -> bool:
        """Dashboard-unlock a protective stop mid-session (no pendant).
        Returns True if the unlock command was issued. Note the UR enforces
        a ~5 s settle after unlock before motion resumes."""
        d = self._dashboard()
        if d is None:
            return False
        try:
            d.unlockProtectiveStop()
            for fn in ("closeSafetyPopup", "closePopup"):
                try:
                    getattr(d, fn)()
                except Exception:  # noqa: BLE001
                    pass
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("unlock_protective_stop failed: %s", e)
            return False
        finally:
            try:
                d.disconnect()
            except Exception:  # noqa: BLE001
                pass

    def _open_urscript_socket(self):
        """
        Open the primary-interface TCP socket for raw URScript.

        speedl/speedj can't go through ur_rtde's blocking moveL/servoJ, so
        CartesianServo velocity streaming pushes raw URScript to port
        30002. A failure
        here degrades to RTDE-only motion (move_linear/servo_joint) rather
        than crashing the bring-up; SUPPORTS_URSCRIPT stays True so the
        servo path is still attempted and reconnects lazily in _send_script.
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((self.robot_ip, self.URSCRIPT_PORT))
            self._urscript_socket = sock
            logger.info("URScript socket up (port %d)", self.URSCRIPT_PORT)
        except OSError as e:
            self._urscript_socket = None
            logger.warning(
                "URScript socket (port %d) failed: %s; speedl/speedj will "
                "retry-connect on first use",
                self.URSCRIPT_PORT,
                e,
            )

    def _drop_urscript_socket(self):
        """
        Close and forget the URScript socket, holding ``_script_lock``.

        NEVER close this socket without the lock. _send_script resolves
        self._urscript_socket and then calls sendall() on it, both inside
        _script_lock; a closer that skips the lock can free the descriptor
        between those two steps:

            executor thread                 /api/abort (event loop)
            ---------------                 ------------------------
            with _script_lock:
              sock = self._urscript_socket
                                            sock.close()          # fd N freed
                                            self._urscript_socket = None
                                            (another thread open()s -> gets N)
              sock.sendall(script)          # writes URScript into fd N,
                                            # which is now someone else's
                                            # socket / log file / device node

        The fd is freed the moment close() returns and the kernel reissues
        the lowest free number, so this is not merely "the send fails": it
        is a cross-descriptor write of a multi-KB URScript program into an
        unrelated open file. /api/stop and /api/abort are async handlers on
        the event loop and run concurrently with the executor thread, so
        both closers below were reachable during any movel/movej.

        Must not be called from inside _send_script, which already holds
        the lock (it is a plain Lock, not an RLock).
        """
        with self._script_lock:
            sock = self._urscript_socket
            self._urscript_socket = None
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass

    def disconnect(self):
        """
        Safely disconnect from robot.

        Brakes FIRST. Closing the URScript socket does not stop a program that
        is already running on the controller, so a teardown mid-move (server
        Ctrl-C -> pipeline.shutdown -> disconnect) used to leave the arm
        running to its target with nobody left to stop it -- the operator's
        "Ctrl-C made the arm just go high".
        """
        if self._connected:
            self.brake()
        # Clear _connected FIRST. A demo-recorder or streaming thread polling
        # get_joint_positions()/get_observation() during teardown must get the
        # "Not connected to robot" RuntimeError, not reach rtde_r.getActualQ()
        # on an interface rtde_r.disconnect() is tearing down: ur_rtde reads
        # its state buffer through a pointer disconnect() drops, and that read
        # can segfault the process rather than raise.
        self._connected = False
        if self._rtde_c:
            self._rtde_c.stopScript()
            self._rtde_c.disconnect()
        if self._rtde_r:
            self._rtde_r.disconnect()
        # Under _script_lock: a concurrent _send_script may have already
        # resolved this socket and be about to sendall() on it.
        self._drop_urscript_socket()
        logger.info("Disconnected")

    def _check_connected(self):
        if not self._connected:
            raise RuntimeError("Not connected to robot. Call connect() first.")

    def health_check(self) -> bool:
        """
        Probe the ACTUAL RTDE link, not the _connected flag.

        Checks rtde_r.isConnected() plus a getActualQ read; on failure clears
        _connected so _check_connected reflects reality. reconnect() (or
        /api/connect_robot) is the recovery.
        """
        if not self._connected:
            return False
        try:
            rr = self._rtde_r
            if rr is None:
                raise RuntimeError("no RTDE receive interface")
            is_conn = getattr(rr, "isConnected", None)
            if callable(is_conn) and not is_conn():
                raise RuntimeError("RTDE receive link reports disconnected")
            q = rr.getActualQ()
            if q is None or len(q) < 6:
                raise RuntimeError("joint state unavailable")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Driver health check FAILED: %s (marking disconnected)", exc)
            self._connected = False
            return False

    def reconnect(self) -> bool:
        """
        Tear down and re-run the full connect sequence (dashboard preflight,
        RTDE control retry, URScript socket, gripper). Recovery path after
        health_check() fails, so /api/connect_robot is not the only way back.
        """
        logger.info("Driver reconnect requested")
        try:
            self.disconnect()
        except Exception as exc:  # noqa: BLE001
            logger.warning("reconnect: disconnect raised: %s", exc)
        self._rtde_c = None
        self._rtde_r = None
        self._urscript_socket = None
        self._connected = False
        self.connect()
        return self._connected

    def _send_dashboard_command(self, command: str, timeout_s: float = None) -> bool:
        """Run one Dashboard (29999) command under a bounded wait.

        DashboardClient.connect() blocks on an unreachable or Local-mode
        controller, and an escalation that hangs is not an escalation, so the
        call runs on a daemon worker that is joined only for timeout_s.
        """
        timeout_s = self.DASHBOARD_TIMEOUT_S if timeout_s is None else timeout_s
        result = {"ok": False}

        def _run():
            d = self._dashboard()
            if d is None:
                return
            try:
                fn = getattr(d, command, None)
                if not callable(fn):
                    logger.warning("Dashboard has no command %r", command)
                    return
                fn()
                result["ok"] = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("Dashboard %s failed: %s", command, exc)
            finally:
                try:
                    d.disconnect()
                except Exception:  # noqa: BLE001
                    pass

        worker = threading.Thread(
            target=_run, name="ur-dashboard-%s" % command, daemon=True
        )
        worker.start()
        worker.join(timeout_s)
        if worker.is_alive():
            logger.warning(
                "Dashboard %s did not answer in %.1fs", command, timeout_s
            )
            return False
        return bool(result["ok"])
