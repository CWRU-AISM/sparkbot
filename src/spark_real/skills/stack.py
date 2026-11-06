"""stack skill: place a held object on another with a force-guided descent."""

import logging
import time

import numpy as np

from spark_real.control.grasp_strategy import resolve_grasp_orientation
from spark_real.skills.primitives import _result
from spark_real.skills.registry import spark_skill

logger = logging.getLogger(__name__)


@spark_skill(
    name="stack",
    description=(
        "Place a held object on top of another using force-guided descent. "
        "Descends slowly until contact is detected via F/T sensor, then releases. "
        "Much more reliable than guessing offset_z for stacking."
    ),
    params={
        "target_label": str,
        "force_threshold": float,
        "max_descent": float,
        "clearance": float,
    },
)
def stack(executor, params: dict):
    """
    Force-guided descent onto target; release on contact.
    """
    t0 = time.time()
    target_label = params.get("target_label", "")
    # Config-driven tuning (control/executor_core.py class defaults, overridden
    # by the `stack:` YAML block); explicit BT params still win per-call.
    force_thresh = float(params.get("force_threshold", executor.STACK_CONTACT_FORCE_N))
    max_descent = float(params.get("max_descent", executor.STACK_MAX_DESCENT_M))
    clearance = float(params.get("clearance", executor.STACK_CLEARANCE_M))
    hard_stop_n = float(params.get("hard_stop_delta_n", executor.STACK_HARD_STOP_DELTA_N))
    desc_vel = max(0.01, executor.velocity * float(executor.STACK_DESCENT_VEL_FRAC))

    if not executor._holding:
        return _result(
            "stack", False, "Not holding anything to stack", time.time() - t0
        )

    det = executor.detection_map.get(target_label)
    if det is None:
        return _result(
            "stack", False, f"Target '{target_label}' not found", time.time() - t0
        )

    target_pos = np.array(det["position_3d"])
    # Placement orientation: orient the held object to the TARGET's OBB (as if we
    # were grasping the target) so it lands ALIGNED with it, and hold that yaw
    # through transport instead of reverting the wrist to the base orientation.
    # When the pick and place objects share an angle (e.g. two same-size blocks)
    # the wrist barely moves from pick to place.
    place_orient = executor.GRASP_ORIENTATION
    try:
        # Same resolver as the grasp/place paths: no bare AR comparison.
        _orient, _strategy = resolve_grasp_orientation(
            params, det, executor, context="stack"
        )
        if _strategy == "obb":
            place_orient = _orient
    except Exception as _e:  # noqa: BLE001
        logger.debug("Stack: oriented placement unavailable (%s)", _e)
    above = target_pos.copy()
    above[2] += clearance
    current = executor._get_current_position()
    safe_z = max(current[2], above[2] + 0.05)
    lift = current.copy()
    lift[2] = safe_z
    executor._move_to(lift, place_orient)
    if hasattr(executor, "_verify_holding_during_transport"):
        if not executor._verify_holding_during_transport():
            executor._holding = False
            return _result(
                "stack", False, "Object lost during transport", time.time() - t0
            )
    above_high = above.copy()
    above_high[2] = safe_z
    executor._move_to(above_high, place_orient)
    executor._move_to(above, place_orient)

    # GUARD: confirm the arm is actually ABOVE the target before descending.
    # The positioning moves can early-return short of the target (movej's
    # _wait_for_motion treats the slow initial creep as "stationary"). If we
    # then enter the descent loop, each "2mm step" becomes a full lateral move
    # to target_xy at the crawl descent velocity -> a 20-30cm diagonal drag
    # that times out (observed: arm at Y=0.23, target Y=-0.047, dist=0.285m).
    arrived = False
    for _ in range(3):
        executor._check_abort()
        cur = executor._get_current_position()
        if float(np.linalg.norm(cur[:2] - above[:2])) < 0.01:  # within 1cm XY
            arrived = True
            break
        executor._move_to(above, place_orient)  # full speed, retry
    if not arrived:
        cur = executor._get_current_position()
        xy_err = float(np.linalg.norm(cur[:2] - above[:2]))
        # Bail only on a REAL drag risk (>4cm). A couple-cm residual (the pyroki
        # IK sometimes leaves ~2cm on far/yawed place targets) is fine to place;
        # a slightly offset stack beats aborting (a 2cm gate tripped on a 22mm
        # residual). CRITICAL: do NOT clear _holding here; the block is still
        # physically in the jaws, and marking not-holding lets the post-task
        # cleanup open the gripper and DROP it from height. Keep holding so a
        # failure leaves the block safely gripped, not dropped.
        if xy_err > 0.04:
            return _result(
                "stack",
                False,
                f"Could not position above '{target_label}' before descent "
                f"(xy_err={xy_err * 1000:.0f}mm)",
                time.time() - t0,
            )
        logger.info(
            "Stack: %.0fmm off above '%s', within place tolerance, descending",
            xy_err * 1000, target_label,
        )

    baseline_fz = 0.0
    try:
        if hasattr(executor.robot, "get_tcp_force"):
            ft = executor.robot.get_tcp_force()
            baseline_fz = abs(ft[2]) if len(ft) > 2 else 0.0
    except Exception:
        pass
    contact_z = None
    floor_z = executor.TABLE_Z_FLOOR + 0.002

    def _fz_delta():
        try:
            if hasattr(executor.robot, "get_tcp_force"):
                ft = executor.robot.get_tcp_force()
                fz = abs(ft[2]) if len(ft) > 2 else 0.0
                return fz, fz - baseline_fz
        except Exception:
            pass
        return baseline_fz, 0.0

    # Hard stop / contact both test the baseline-SUBTRACTED delta, NOT absolute
    # fz. getActualTCPForce carries a large constant gripper/payload offset
    # (~69N); an absolute threshold tripped on step 1 and DROPPED the
    # block 12cm up. The delta cancels the offset so only a real collision spikes.
    # Continuous force-guided descent for ANY arm with velocity control (UR
    # speedl, FR3 via franky send_velocity). The stepped path below is only a
    # last-resort fallback for a driver that exposes no send_velocity at all.
    use_continuous = hasattr(executor.robot, "send_velocity")
    if use_continuous:
        # Continuous constant-velocity descent: smooth (no discrete-movel
        # crawl/under-shoot), and a steady velocity keeps the wrist force flat so
        # a real block-on-block contact spikes cleanly (discrete 2mm stepping
        # moved ~30mm in 60s and timed out).
        logger.info(
            "Stack: force-guided descent toward '%s' from z=%.3f "
            "(v=%.3f m/s, contact>%.1fN, cap=%.0fmm)",
            target_label, above[2], desc_vel, force_thresh, max_descent * 1000,
        )
        # Two-phase descent: fast until we near the target top, then CRAWL.
        # Rigid block-on-block contact builds force in <1mm, so at full descent
        # speed the ~10Hz force loop only samples every ~2.5mm and overshoots
        # the threshold hard (e.g. 4N -> 19N in one step). Crawling once within
        # slow_margin of the target's detected top keeps that final overshoot
        # small so the press is gentle, without slowing the whole descent.
        slow_margin = float(params.get(
            "slow_margin", getattr(executor, "STACK_SLOW_MARGIN_M", 0.04)))
        slow_vel = max(0.006, desc_vel * float(
            getattr(executor, "STACK_SLOW_VEL_FRAC", 0.35)))
        slow_z = target_pos[2] + slow_margin
        t_start = time.time()
        _step = 0
        _crawling = False
        while not executor._abort:
            cur = executor._get_current_position()
            if cur[2] <= floor_z or (above[2] - cur[2]) >= max_descent:
                contact_z = cur[2]
                logger.info("Stack: descent cap/floor reached at z=%.3f", cur[2])
                break
            if time.time() - t_start > 25.0:
                contact_z = cur[2]
                logger.warning("Stack: descent time cap at z=%.3f", cur[2])
                break
            fz, delta = _fz_delta()
            _step += 1
            # Continuous force readout: log the whole descent profile (~10 Hz)
            # so the fz/delta curve up to contact is inspectable in the log.
            if _step % 3 == 0:
                logger.info(
                    "Stack descent: z=%.4f  fz=%.2f  delta=%.2fN  (contact>%.1f)",
                    cur[2], fz, delta, force_thresh,
                )
            # Contact = a big CHANGE in the wrist z-force, EITHER SIGN. On a
            # block-on-block landing the held object's weight is partly borne by
            # the lower block, so abs(ft_z) DROPS (delta goes strongly negative);
            # a straight push-into would raise it. A positive-only test never
            # detects the landing (the arm rams down and holds ~42N until the
            # 25s time cap), so use |delta|.
            if abs(delta) > hard_stop_n:
                logger.warning("Stack: HARD STOP |delta|=%.1fN at z=%.3f", delta, cur[2])
                contact_z = cur[2]
                break
            if abs(delta) > force_thresh:
                contact_z = cur[2]
                logger.info("Stack: contact at z=%.3f (delta=%.1fN)", cur[2], delta)
                break
            # Crawl for the final approach near the target top (gentle landing).
            v = desc_vel
            if cur[2] <= slow_z:
                v = slow_vel
                if not _crawling:
                    _crawling = True
                    logger.info(
                        "Stack: entering slow zone at z=%.3f (v=%.3f m/s)",
                        cur[2], slow_vel,
                    )
            # Positional args: SafeRobot names the 3rd param `time_duration`,
            # the raw driver names it `duration`; positional works for both.
            # Short duration (0.06s) so a contact-break overshoots ~1.5mm, not ~3mm.
            executor.robot.send_velocity([0.0, 0.0, -v, 0.0, 0.0, 0.0], 0.5, 0.06)
            time.sleep(0.03)
        # stop residual motion
        try:
            executor.robot.send_velocity([0.0] * 6, 1.0, 0.05)
        except Exception:
            pass
        for _stop in ("stop", "stop_motion"):
            if hasattr(executor.robot, _stop):
                try:
                    getattr(executor.robot, _stop)()
                except Exception:
                    pass
                break
    else:
        # Stepped _move_to fallback (drivers without send_velocity).
        step_size = float(executor.STACK_STEP_M)
        num_steps = max(int(max_descent / step_size), 1)
        descent_pos = above.copy()
        logger.info("Stack: stepped descent toward '%s' from z=%.3f", target_label, above[2])
        for i in range(num_steps):
            if executor._abort:
                logger.warning("Stack: aborted at step %d/%d", i, num_steps)
                break
            descent_pos[2] -= step_size
            descent_pos[:2] = above[:2]
            executor._move_to(descent_pos, place_orient, velocity=desc_vel)
            time.sleep(0.05)
            fz, force_delta = _fz_delta()
            if force_delta > hard_stop_n:
                logger.warning("Stack: HARD STOP delta=%.1fN at step %d", force_delta, i + 1)
                descent_pos[2] += step_size * 3
                executor._move_to(descent_pos, place_orient, velocity=desc_vel)
                contact_z = descent_pos[2]
                break
            if force_delta > force_thresh:
                contact_z = descent_pos[2]
                logger.info("Stack: contact at z=%.3f (delta=%.1fN) step %d/%d",
                            contact_z, force_delta, i + 1, num_steps)
                break

    if executor._abort:
        executor._holding = False
        return _result("stack", False, "Aborted during descent", time.time() - t0)

    if contact_z is None:
        contact_z = executor._get_current_position()[2]
        logger.warning(
            "Stack: no contact detected after %.0fmm descent, " "releasing at z=%.3f",
            max_descent * 1000,
            contact_z,
        )

    if hasattr(executor.robot, "_send_gripper_command"):
        executor.robot._send_gripper_command(0.0, speed=30, force=20)
    else:
        executor.robot.open_gripper()
    time.sleep(0.8)
    executor._holding = False
    retract = executor._get_current_position()
    retract[2] += 0.06
    executor._move_to(
        retract, executor.current_orientation(), velocity=executor.velocity * 0.3
    )

    descent_mm = (above[2] - contact_z) * 1000
    return _result(
        "stack",
        True,
        f"Stacked on '{target_label}' at z={contact_z:.3f} "
        f"(descended {descent_mm:.0f}mm, contact={'yes' if contact_z is not None else 'no'})",
        time.time() - t0,
    )
