"""UR10e motion: joint and linear moves, URScript send, velocity, brake and stop."""

import logging
import os
import time
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class URMotionMixin:
    """UR10e motion: joint and linear moves, URScript send, velocity, brake and stop."""

    def move_to_joint_config(
        self,
        q: list,
        velocity: float = None,
        acceleration: float = None,
        asynchronous: bool = False,
    ):
        """
        Move to joint configuration.

        Args:
            q: Target joint positions [j0..j5] in radians
            velocity: Joint velocity (rad/s)
            acceleration: Joint acceleration (rad/s^2)
            asynchronous: If True, return immediately
        """
        self._check_connected()
        vel = velocity or self.DEFAULT_JOINT_VEL
        acc = acceleration or self.DEFAULT_JOINT_ACC
        self._ensure_control_script("moveJ")
        self._rtde_c.moveJ(list(q), vel, acc, asynchronous)

    def move_to_joint_config_urscript(
        self, q: list, velocity: float = None, acceleration: float = None
    ):
        """Joint move via raw URScript on the primary interface (port 30002),
        the SAME channel as speedl/speedj.

        Use this after velocity streaming: a speedl program stops the control
        script that RTDEControlInterface uploaded, so a subsequent rtde_c.moveJ
        silently no-ops. This routes movej through the URScript socket instead,
        which is conflict-free with speedl. Fire-and-forget: returns
        immediately, it does NOT block until the move completes.
        """
        vel = velocity or self.DEFAULT_JOINT_VEL
        acc = acceleration or self.DEFAULT_JOINT_ACC
        q_str = ", ".join(f"{float(x):.6f}" for x in q)
        if not self._send_script(f"movej([{q_str}], a={acc}, v={vel})"):
            # A dropped motion command must FAIL the primitive, not silently
            # pass while the caller sleeps and proceeds as if the arm moved.
            raise RuntimeError(
                "movej dropped: URScript channel down (see health_check/reconnect)"
            )

    def go_home_urscript(self, velocity: float = None):
        """Home via the URScript channel (see move_to_joint_config_urscript).
        Safe to call from velocity-streaming teleop where go_home() no-ops."""
        self.move_to_joint_config_urscript(
            self.HOME_CONFIG, velocity=velocity or 0.6, acceleration=1.0
        )

    def _check_workspace(self, pose: list):
        """
        Driver-level sanity envelope for the UR10e cell: a coarse backstop
        against wild targets only, since the ScoreExecutor already clips every
        move to the precise control.workspace_min/max ([-1.1,-0.5] x, [-0.5,0.7]
        y, [-0.27,0.50] z). The UR10e grasps at NEGATIVE z (table ~ -0.25 in
        base frame) and reaches X to -1.1, so the envelope spans the real
        reachable range (with margin).
        """
        x, y, z = pose[0], pose[1], pose[2]
        if not (-1.2 <= x <= 0.9 and -0.9 <= y <= 0.9 and -0.35 <= z <= 1.2):
            raise ValueError(
                f"Target ({x:.3f}, {y:.3f}, {z:.3f}) outside workspace bounds. "
                f"x:[-1.2,0.9] y:[-0.9,0.9] z:[-0.35,1.2]"
            )

    def move_linear(
        self,
        pose: list,
        velocity: float = None,
        acceleration: float = None,
        asynchronous: bool = False,
    ):
        """
        Linear motion to Cartesian pose.

        Args:
            pose: Target [x, y, z, rx, ry, rz] in meters and axis-angle
            velocity: Linear velocity (m/s)
            acceleration: Linear acceleration (m/s^2)
            asynchronous: If True, return immediately
        """
        self._check_connected()
        self._check_workspace(pose)
        vel = velocity or self.DEFAULT_VELOCITY
        acc = acceleration or self.DEFAULT_ACCELERATION
        self._ensure_control_script("moveL")
        self._rtde_c.moveL(list(pose), vel, acc, asynchronous)

    def _check_safety_and_recover(self, what: str):
        """Make protective stops VISIBLE and self-clearing.

        Measured 2026-08-21 12:39: a wrist yaw at low height collided, the
        controller went into PROTECTIVE_STOP, and the server's log showed
        nothing but moved=False timeouts -- zero occurrences of the word
        'protective' across the whole session, while the operator cleared the
        pendant three times.

        On PROTECTIVE_STOP: wait the 5 s the UR firmware requires, unlock via
        the dashboard, re-upload the control script -- and then RAISE anyway.
        A cleared protective stop means the pending motion's predecessor
        COLLIDED; silently continuing with the same target would re-run the
        collision (which is exactly what the operator watched happen when
        they cleared it by hand). Raising drops this motion into the
        executor's failure path, whose recovery ladder begins with a lift --
        the correct next move after any collision. Emergency stops are never
        auto-cleared: a physical stop is an operator decision.
        """
        try:
            mode = int(self._rtde_r.getSafetyMode())
        except Exception:
            return  # can't read; let the motion surface its own error
        if mode in (1, 2):
            return
        name = self.SAFETY_MODE_NAMES.get(mode, str(mode))
        if mode in (6, 7):
            raise RuntimeError(
                f"robot is in {name}; refusing {what} -- emergency stops "
                "must be cleared by the operator"
            )
        logger.warning("robot is in %s before %s; attempting recovery", name, what)
        if mode == 3:
            time.sleep(5.0)  # UR firmware refuses unlock before 5 s
            d = self._dashboard()
            if d is not None:
                try:
                    d.unlockProtectiveStop()
                    logger.warning("protective stop UNLOCKED via dashboard")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("unlockProtectiveStop failed: %s", exc)
                finally:
                    try:
                        d.disconnect()
                    except Exception:  # noqa: BLE001
                        pass
            try:
                self._rtde_c.reuploadScript()
                time.sleep(0.2)
            except Exception as exc:  # noqa: BLE001
                logger.warning("script re-upload after unlock failed: %s", exc)
            # CONTINUE, do not raise. The colliding motion is not the one
            # waiting behind this check -- it already died with moved=False
            # and routed to the failure path before the arm ever got here;
            # the pending command is recovery's own next move (typically a
            # lift). This mirrors the teleop /api/recover behavior the
            # operator relies on: unlock and keep driving. Only a stop that
            # WON'T clear escalates.
            try:
                if int(self._rtde_r.getSafetyMode()) in (1, 2):
                    # Stamp the recovery so the executor can reconcile its
                    # world model at the next action boundary: a protective
                    # stop means a COLLISION happened, and the collision may
                    # have changed things the plan believes (the object may
                    # have been knocked from the jaws, the target nudged).
                    self.last_protective_recovery_ts = time.time()
                    logger.warning(
                        "protective stop cleared; continuing with %s", what
                    )
                    return
            except Exception:  # noqa: BLE001
                pass
        raise RuntimeError(
            f"robot is in {name} and did not recover before {what}; "
            "operator attention needed"
        )

    def _ensure_control_script(self, what: str):
        """Re-upload the RTDE control script if it has died.

        After a servoj session ends (servo_stop / speed_stop), ur_rtde's
        control script on the controller can stop and stay stopped; every
        subsequent moveL is then a silent no-op that returns immediately.
        """
        self._check_safety_and_recover(what)
        try:
            if self._rtde_c.isProgramRunning():
                return
        except Exception:
            return  # can't ask; let the motion call surface its own error
        logger.warning(
            "RTDE control script not running before %s; re-uploading", what
        )
        try:
            ok = self._rtde_c.reuploadScript()
            time.sleep(0.1)  # give the controller a beat to start it
            if not ok or not self._rtde_c.isProgramRunning():
                raise RuntimeError("reuploadScript did not take")
        except Exception as exc:
            # Reconnect is the heavier fallback, the same path a fresh connect
            # uses, so a controller that can run at all will accept it.
            logger.warning(
                "reuploadScript failed (%s); reconnecting RTDE control", exc
            )
            self._rtde_c.reconnect()
            if not self._rtde_c.isProgramRunning():
                raise RuntimeError(
                    f"RTDE control script would not restart before {what}; "
                    "refusing to silently drop the motion"
                )

    def move_linear_relative(
        self, delta: list, velocity: float = None, acceleration: float = None
    ):
        """
        Move relative to current TCP pose.

        Args:
            delta: [dx, dy, dz, drx, dry, drz] relative offset
        """
        current = self.get_tcp_pose()
        target = current + np.array(delta)
        self.move_linear(target.tolist(), velocity, acceleration)

    def _send_script(self, script: str) -> bool:
        """
        Send a raw URScript line to the primary interface (port 30002).

        Auto-reconnects once on a broken pipe. Returns True on success,
        False if the socket can't be (re)established.
        """
        if not self._connected:
            return False
        if not script.endswith("\n"):
            script += "\n"
        # rtde_c keeps a control-script keepalive running that re-stomps a
        # one-shot movej/movel sent over _send_script before the arm moves (arm
        # silently never moves; /api/home no-ops cold). A gripper _send_script
        # stops the keepalive, which is why a move works right after a grasp but
        # not from a fresh state. All motion here goes through _send_script (see
        # executor_ik._movej_to_pose: rtde_c.moveL no-ops), so release the
        # keepalive right before a one-shot move. speedl/speedj are EXCLUDED: the
        # 30 Hz servo re-sends faster than the stomp and must not eat a
        # stopScript() per tick. SPARK_UR_RELEASE_KEEPALIVE=0 restores legacy.
        if (
            self._rtde_c is not None
            and ("movej(" in script or "movel(" in script)
            and os.environ.get("SPARK_UR_RELEASE_KEEPALIVE", "1") == "1"
        ):
            try:
                self._rtde_c.stopScript()
            except Exception as e:  # noqa: BLE001
                logger.debug("rtde_c.stopScript() before move failed: %s", e)
        # One writer at a time on the socket: concurrent sends (executor +
        # watchdog brake + teleop) interleaving on one TCP stream corrupt
        # programs mid-line.
        with self._script_lock:
            for attempt in range(2):
                try:
                    if self._urscript_socket is None:
                        self._open_urscript_socket()
                        if self._urscript_socket is None:
                            self._urscript_down = True
                            logger.error(
                                "URScript DOWN: could not open socket; "
                                "command dropped: %s", script.split("\n", 1)[0]
                            )
                            return False
                    # sendall, not send: gripper scripts embed the multi-KB
                    # URCap header, and an unchecked partial send truncates a
                    # program mid-line (the next send then concatenates
                    # garbage onto it).
                    self._urscript_socket.sendall(script.encode())
                    # Stamp AFTER a successful send: a script that never left
                    # the socket must not suppress the next gripper publish.
                    self._note_motion_script(script)
                    self._urscript_down = False
                    return True
                except (BrokenPipeError, ConnectionResetError, OSError) as e:
                    if attempt == 0:
                        logger.warning("URScript socket lost, reconnecting (%s)", e)
                        try:
                            if self._urscript_socket is not None:
                                self._urscript_socket.close()
                        except OSError:
                            pass
                        self._urscript_socket = None
                    else:
                        self._urscript_down = True
                        logger.error(
                            "URScript DOWN: send failed after reconnect (%s); "
                            "command dropped: %s", e, script.split("\n", 1)[0]
                        )
                        return False
            return False

    def send_velocity(
        self,
        velocity: Sequence[float],
        acceleration: float = 0.5,
        duration: float = 0.1,
    ):
        """
        Stream a base-frame Cartesian velocity via URScript speedl.

        Args:
            velocity: [vx, vy, vz, wrx, wry, wrz] in m/s and rad/s
            acceleration: linear acceleration limit (m/s^2)
            duration: how long the command stays active (seconds)

        CartesianServo calls this with keyword ``acceleration``/``duration``
        on the non-URScript fallback branch; here the URScript branch is
        used directly, but we keep the same signature so the servo's
        TypeError fallback never has to fire on this driver.
        """
        v = [float(x) for x in velocity]
        # Latch BEFORE emitting: the recorder's staleness gate is keyed on the
        # commit time of this value.
        self._command_latch.latch_velocity(v)
        vel_str = ", ".join(f"{x:.5f}" for x in v)
        self._send_script(f"speedl([{vel_str}], a={acceleration}, t={duration})")

    def stop_velocity(self):
        """
        Decelerate any active speedl/speedj to zero. True == on the wire.

        Named for the velocity-streaming path, but several abort handlers reach
        for it generically (/api/abort, /api/recover), so it brakes in whatever
        space the arm was last commanded in: stopl after a speedl, stopj after
        a movej, where a Cartesian decel constraint would be the wrong one.
        """
        return self.brake(kind=self._last_motion_kind or "linear")

    def brake_decel(self, kind: Optional[str] = None) -> float:
        """Deceleration for a stop, sized from the last commanded speed.

        overshoot = v**2 / (2a), so a = v**2 / (2 * budget). Returned in
        rad/s^2 for a joint stop and m/s^2 for a linear one. Clamped: below
        BRAKE_DECEL_MIN a stop is not a stop, above BRAKE_DECEL_MAX the UR
        answers with a C153/C173 protective stop instead of braking.
        """
        kind = kind or self._last_motion_kind or "joint"
        if kind == "linear":
            v, budget = self._last_linear_vel, self.BRAKE_OVERSHOOT_BUDGET_M
            lo, hi = self.BRAKE_DECEL_LIN_MIN, self.BRAKE_DECEL_LIN_MAX
        else:
            v, budget = self._last_joint_vel, self.BRAKE_OVERSHOOT_BUDGET_RAD
            lo, hi = self.BRAKE_DECEL_MIN, self.BRAKE_DECEL_MAX
        return float(min(max(v * v / (2.0 * budget), lo), hi))

    def brake(self, decel: Optional[float] = None, kind: Optional[str] = None) -> bool:
        """Decelerate the arm NOW over the URScript socket. True == on the wire.

        This is the only mechanism that brakes this rig, and the abort path for
        blended paths already proved it on the wire: a program sent to
        URSCRIPT_PORT REPLACES the running program, so a stopj both brakes the
        arm and kills whatever movej/blended path was running. Fire-and-forget
        like every other motion here -- it returns as soon as the bytes are
        sent, the deceleration happens on the controller.

        Safe to call when disconnected or mid-abort: _send_script returns False
        and the caller escalates (see emergency_stop).
        """
        kind = kind or self._last_motion_kind or "joint"
        a = self.brake_decel(kind) if decel is None else float(decel)
        cmd = "stopl" if kind == "linear" else "stopj"
        ok = False
        try:
            ok = bool(self._send_script(f"{cmd}({a:.3f})"))
        except Exception as exc:  # noqa: BLE001 - a stop never raises
            logger.warning("brake: %s send failed: %s", cmd, exc)
        if ok:
            # The commanded motion is over: retire the lease so the next
            # gripper publish is not suppressed by a move that no longer runs.
            # The commanded-speed memory is deliberately NOT cleared: /api/stop
            # brakes twice (executor abort, then the driver), and a second
            # stopj REPLACES the first one's program; computed from a zeroed
            # velocity it would be the BRAKE_DECEL_MIN floor, i.e. a redundant
            # stop that decelerates more gently than the one already running.
            self.clear_motion_lease()
        else:
            logger.error("brake: %s(%.2f) DID NOT REACH THE ARM", cmd, a)
        return ok

    def emergency_stop(self, decel: Optional[float] = None) -> dict:
        """Brake the arm, escalating until something reaches it.

        1. URScript stopj/stopl on the primary socket.
        2. Same, on a freshly reopened socket (a dead socket is the one
           failure mode step 1 has).
        3. Dashboard ``stop`` on 29999, bounded by DASHBOARD_TIMEOUT_S. This
           ends the running program too, just slower to issue and vulnerable
           to another dashboard client holding the single slot.

        Returns {"braked": bool, "steps": [...]}: ``braked`` is True only if a
        stop actually left this process. Deliberately NOT raising: every caller
        is already on an abort path.
        """
        steps = []
        if self.brake(decel):
            return {"braked": True, "steps": ["urscript"]}
        steps.append("urscript:failed")
        # A stale socket is invisible until a send fails; drop it and retry.
        # Under _script_lock (see _drop_urscript_socket): /api/stop and
        # /api/abort reach here on the event loop while the executor thread
        # may be mid-_send_script on this very descriptor.
        self._drop_urscript_socket()
        if self.brake(decel):
            steps.append("urscript-reconnect")
            return {"braked": True, "steps": steps}
        steps.append("urscript-reconnect:failed")
        if self._send_dashboard_command("stop"):
            steps.append("dashboard")
            return {"braked": True, "steps": steps}
        steps.append("dashboard:failed")
        logger.critical(
            "EMERGENCY STOP DID NOT REACH THE ARM (%s). The arm may still be "
            "moving; use the physical E-stop.",
            ", ".join(steps),
        )
        return {"braked": False, "steps": steps}

    def servo_joint(
        self,
        q: list,
        velocity: float = 0.5,
        acceleration: float = 0.5,
        dt: float = 0.002,
        lookahead_time: float = 0.1,
        gain: int = 300,
    ):
        """
        Real-time joint servo.

        Args:
            q: Target joint positions
            velocity: Not used directly but defines profile
            acceleration: Not used directly but defines profile
            dt: Time step (1/frequency)
            lookahead_time: Smoothing parameter (0.03-0.2)
            gain: Proportional gain (100-2000)
        """
        self._check_connected()
        self._rtde_c.servoJ(list(q), velocity, acceleration, dt, lookahead_time, gain)

    def servo_stop(self):
        """
        Stop servo mode.
        """
        if self._rtde_c is not None:
            try:
                self._rtde_c.servoStop()
            except Exception as exc:  # noqa: BLE001
                logger.debug("rtde_c.servoStop failed: %s", exc)
        # servoStop is an RTDE register command read by the control script, so
        # it is a no-op once _send_script has stopped that script. Brake for
        # real on the way out, and report whether that reached the arm; a
        # caller polling stop methods must not read None as success.
        return self.brake()

    def stop(self):
        """
        Emergency stop: decelerate to zero velocity.

        This used to be ``rtde_c.stopJ(2.0)``, which CANNOT stop this rig.
        stopJ writes an RTDE input register that ur_rtde's control script
        polls, and every motion here goes through _send_script, which
        deliberately stops that control script (see _send_script). No control
        script, nobody reads the register: a silent no-op that satisfied every
        best-effort stop loop in the codebase while the arm ran on to its
        target. The brake is URScript on the primary socket; the arm actually
        decelerates. Kept as ``stop`` because SafeRobot.stop_motion and several
        route handlers look this name up by getattr.
        """
        return self.brake()

    def go_home(self, velocity: float = None):
        """
        Move to home configuration.
        """
        self.move_to_joint_config(self.HOME_CONFIG, velocity=velocity)
