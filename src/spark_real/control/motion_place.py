"""Transport to a place target: release height, standoff, descent and arrival."""

import logging
import os

import numpy as np

from spark_real.control import success_verifier
from spark_real.control.grasp_strategy import (
    compose_yaw,
    resolve_grasp_orientation,
)
from spark_real.control.release_height import resolve_release_z
from spark_real.control.success_predicates import container_obb_xy, obb_contains
from spark_real.perception.mask_features import resolve_slot_direction

logger = logging.getLogger(__name__)


class PlaceMixin:
    """Transport to a place target: release height, standoff, descent and arrival."""

    def _place_release_z(self, plan_z: float, container, params: dict) -> float:
        """Resolve the Z a place transport flies to; see control/release_height.

        Binds executor state (held detection, table plane, grasp Z offset) to
        the pure resolver and logs the verdict. Every branch returns a number;
        an unmeasurable container returns ``plan_z`` unchanged.
        """
        held_label = getattr(self, "_active_grasp_label", "") or ""
        held = self.detection_map.get(held_label) if held_label else None
        strict = bool(params.get("strict_offset_z", False))
        # Effective grasp-z offset for held_drop. The static constant assumes
        # the TCP closed at perceived_top + GRIPPER_OPEN_Z_OFFSET; the real
        # grasp descends further (SPARK_GRASP_DEPTH_M, contact guard, learned
        # per-label offsets, recovery Z bias), so gripping N cm lower means the
        # object hangs N cm LESS below the TCP. Behind place.release_giveback
        # (default off); plan-strict and plan-no-geometry bypass the drop
        # model either way.
        gz = float(self.GRIPPER_OPEN_Z_OFFSET)
        if self._release_giveback_enabled():
            actual = getattr(self, "_actual_grasp_tcp_z", None)
            ptz = getattr(self, "_grasp_perception_target_z", None)
            if actual is not None and ptz is not None:
                gz = float(actual) - float(ptz)
                logger.info(
                    "Release give-back: effective grasp_z_offset=%.3f "
                    "(actual close TCP %.3f - perceived top %.3f; constant "
                    "was %.3f)",
                    gz,
                    float(actual),
                    float(ptz),
                    float(self.GRIPPER_OPEN_Z_OFFSET),
                )
        try:
            out = resolve_release_z(
                plan_z=plan_z,
                container=container,
                held=held,
                table_z=float(self.TABLE_Z_FLOOR),
                strict=strict,
                grasp_z_offset=gz,
            )
        except Exception as exc:  # noqa: BLE001 - a height quirk must not stop a place
            logger.warning(
                "Release height: could not resolve (%s); keeping the plan's " "z=%.3f",
                exc,
                plan_z,
            )
            return float(plan_z)
        if abs(out.z - plan_z) < 1e-6:
            logger.info("Release height: %s (unchanged)", out.describe())
        else:
            logger.info(
                "Release height: plan z=%.3f -> %.3f (%+.1f cm) %s",
                plan_z,
                out.z,
                (out.z - plan_z) * 100.0,
                out.describe(),
            )
        self._last_release_height = out
        return float(out.z)

    def _release_giveback_enabled(self) -> bool:
        """place.release_giveback in the family YAML (default OFF = the static
        GRIPPER_OPEN_Z_OFFSET). SPARK_RELEASE_GIVEBACK env overrides."""
        env = os.environ.get("SPARK_RELEASE_GIVEBACK")
        if env is not None:
            return env.strip() not in ("", "0", "false", "no")
        try:
            profile = getattr(self._pipeline, "profile", None)
            raw = getattr(profile, "raw", None) or {}
            return bool((raw.get("place") or {}).get("release_giveback", False))
        except Exception:  # noqa: BLE001
            return False

    def _place_arrival(self, label: str, detection) -> tuple:
        """Did the arm end up OVER container ``label``? -> (ok, detail).

        ``ok`` is True whenever the question cannot be answered: a container
        with no slots and no measured width has an UNKNOWN extent
        (success_predicates._container_obb returns None). Missing perception
        never blocks a release.

        The test is the container's own OBB in XY, widened by
        PLACE_ARRIVAL_XY_MARGIN_M, evaluated against the CURRENT TCP, not
        against the pose the servo was commanding.
        """
        obb = container_obb_xy(detection) if detection is not None else None
        if obb is None:
            return True, ""
        tcp = self._get_current_position()
        if obb_contains(tcp[:2], obb, self.PLACE_ARRIVAL_XY_MARGIN_M):
            return True, ""
        centre, half_a, half_b, theta = obb
        d = np.asarray(tcp[:2], dtype=float) - np.asarray(centre, dtype=float)
        ca, sa = np.cos(theta), np.sin(theta)
        along = abs(d[0] * ca + d[1] * sa)
        across = abs(-d[0] * sa + d[1] * ca)
        detail = (
            f"TCP ({tcp[0]:.3f},{tcp[1]:.3f}) is outside '{label or '?'}': "
            f"{along * 100:.1f}/{across * 100:.1f} cm from centre vs "
            f"half=({half_a * 100:.1f},{half_b * 100:.1f})cm "
            f"+{self.PLACE_ARRIVAL_XY_MARGIN_M * 100:.0f}cm margin"
        )
        return False, detail

    def _note_place_arrival(self, label: str, arrived: bool, detail: str) -> None:
        """Record that a place transport ran, and how it went.

        Read by ReleaseMixin._transport_delivered, which uses the LABEL to know
        which container to judge against and recomputes the verdict from the
        live TCP; ``arrived``/``detail`` feed the failed-action message and
        the log. The release ignores a record whose label no longer matches
        ``_last_place_label``.
        """
        self._place_arrival_record = {
            "label": label,
            "arrived": bool(arrived),
            "detail": detail,
        }

    def _transport_to(
        self,
        target: np.ndarray,
        target_detection: dict = None,
        target_label: str = "",
        params: dict = None,
        lead=None,
    ):
        """
        Lift, horizontal move, servo descend with orientation matching.

        `params` are the plan node's params; only grasp_strategy /
        grasp_yaw_deg are read.

        `lead` is an optional list of ``(position, orientation, velocity,
        label)`` transit waypoints to fly BEFORE the lift, as part of the same
        blended phrase (the legato look-ahead folds a preceding
        ``move_relative`` into the transport). Ignored when blending is off.

        Returns True if the transport reached its place pose. False when:

          * the grip check found a DROP (``_holding`` is already cleared), or
          * the arm did not ARRIVE inside the target container's planar
            footprint (``_place_arrival``). The object is still held; the
            caller reports the action failed so the tree can recover.
        """
        self._check_abort()
        self._last_place_label = target_label
        self._place_arrival_record = None

        orient = self.GRASP_ORIENTATION
        # Does `orient` end up carrying a slot/fit/OBB alignment? Read by the
        # deep-dip side choice below: an aligned yaw may only be traded for
        # its grasp-symmetric 180 twin, a free yaw may be picked outright.
        yaw_aligned = False
        # Slotted-container safety net for a generic place (move_to_keypoint
        # + release). For a slotted tray the raw container OBB major axis is
        # PERPENDICULAR to the slots (world-PCA axis-swap on a near-square
        # tray), so derive the slot direction the SAME way place_in_slot does
        # and align to that instead of the raw OBB.
        slots = target_detection.get("slots") if target_detection else None
        if slots:
            slot_dir, slot_src = resolve_slot_direction(target_detection, target_label)
            yaw = (slot_dir + np.pi / 2.0) % np.pi - np.pi / 2.0
            yaw = self._nearest_symmetric_yaw(yaw)
            # Slot direction is a validated source and bypasses the strategy
            # gates, but the cable limit still applies to the composed rotation.
            slot_orient = compose_yaw(self, yaw, context="place-slots")
            if slot_orient is not None:
                # Same directed head/tail decision as the obb place path and
                # place_in_slot; axis alignment alone (mod pi) is a coin flip
                # on which end of the slot the tool's tip lands in.
                orient = self._orient_head_to_head(slot_orient, target_detection)
                yaw_aligned = True
            logger.info(
                "[place_orient] slotted container '%s': aligning to slots "
                "(dir=%.1f deg src=%s) -> yaw=%.1f deg, NOT raw OBB",
                target_label or "?",
                np.rad2deg(slot_dir),
                slot_src,
                np.rad2deg(yaw),
            )
        elif target_detection:
            target_orient = target_detection.get("orientation_angle")

            # SHAPE FIT FIRST; it decides on its own whether orientation
            # matters. A slot or tool cut-out is shaped like the thing that
            # goes in it, so one rotation fits with a large margin; for a bowl
            # the margin collapses and the fit says "do not rotate".
            fitted = self._fit_place_yaw(target_detection)
            if fitted is not None:
                orient = fitted
                yaw_aligned = True
                logger.info("[place_orient] using the shape fit for the place")
                target_orient = None  # the fit is the only source for this target

            # The axis path gives the right LINE; the fit is also used for the
            # half it answers without any image-to-world frame conversion:
            # whether that line should be flipped 180 deg. See
            # _orient_head_to_head, which compares both in one image frame.

            if target_orient is not None and abs(float(target_orient)) > 1e-3:
                place_orient, place_strategy = resolve_grasp_orientation(
                    params or {}, target_detection, self, context="place"
                )
                # Adopt ANY resolved yaw ("obb", "plane", "target_axis",
                # "caller_yaw").
                if place_strategy != "topdown" and (
                    place_orient is not None
                ):
                    orient = place_orient
                    place_orient = self._orient_head_to_head(
                        place_orient, target_detection
                    )
                    orient = place_orient
                    yaw_aligned = True
                    logger.info(
                        "[place_orient] adopting '%s' yaw for the place",
                        place_strategy,
                    )
            else:
                logger.info(
                    "[place_orient] skipping yaw-align for '%s' " "(orient<=1mrad or missing)",
                    target_label or "?",
                )

        # DEEP-DIP SIDE CHOICE, decided BEFORE the transport flies and folded
        # into the approach orientation. When the upcoming release asks for a
        # dip past the positive pitch budget (wrist-camera clearance, 40 deg),
        # rotate the approach yaw so the dip lands on the NEGATIVE side (90 deg
        # budget). See ReleaseMixin._dip_side_orient for the rules.
        orient = self._dip_side_orient(orient, yaw_aligned=yaw_aligned)

        place_target = target.copy()
        logger.info("Place '%s': target_z=%.3f", target_label, place_target[2])

        current_pose = self._get_current_position()
        clearance_from = np.asarray(lead[-1][0], dtype=float) if lead else current_pose
        safe_z = self._safe_clearance_z(clearance_from, place_target)
        lift = clearance_from.copy()
        lift[2] = safe_z
        above_target = place_target.copy()
        above_target[2] = safe_z
        # Hand the servo only the last centimetres of the descent. The servo
        # is a PD loop whose gain falls with the error (kp_pos=2.0,
        # pos_threshold=3 mm), so it crawls near convergence (3.2 s for the
        # full descent). A movej covers the free-space part; the servo keeps
        # the part that can touch something, with its compliance and its
        # unreachable-target stall exit intact.
        standoff = place_target.copy()
        standoff[2] = min(
            float(place_target[2]) + self._place_standoff_m(place_target), safe_z
        )

        if self._blend_enabled():
            phrase = list(lead or [])
            # Lift at the orientation the arm is ALREADY holding, not the base
            # constant: snapping to GRASP_ORIENTATION would rotate the wrist
            # to zero on the way up and out to the place yaw on the way
            # across. Holding the carry angle leaves a SINGLE rotation during
            # the horizontal transit.
            carry_orient = getattr(self, "_active_grasp_orient", None)
            if not carry_orient:
                try:
                    carry_orient = list(
                        np.asarray(self.robot.get_tcp_pose(), dtype=float)[3:6]
                    )
                except Exception:  # noqa: BLE001
                    carry_orient = list(self.GRASP_ORIENTATION)
            phrase.append((lift, list(carry_orient), None, "lift"))
            phrase.append((above_target, list(orient), None, "above_target"))
            if standoff[2] < above_target[2] - 1e-4:
                # Terminal row of the SAME program: no extra upload; the r=0
                # stop is the one the exact joint-arrival test applies to.
                phrase.append((standoff, list(orient), None, "place_standoff"))
            if self._blend_transit(phrase, final_radius_m=0.0, phrase="transport"):
                # ONE grip check, at the parked end of the phrase. A
                # mid-transport check sends a gripper URScript, which REPLACES
                # the running program and would cancel the blend; a passive
                # register read cannot see a drop since the last publish.
                if not self._transport_grip_ok("above_target"):
                    self._holding = False
                    return False
                return self._place_descent(place_target, orient, target_label, target_detection)
            for pos_i, ori_i, _v, _lbl in lead or []:
                self._move_to(pos_i, ori_i)

        self._move_to(lift, self.GRASP_ORIENTATION)

        # Per-waypoint drop detection at EACH stationary transport arrival
        # (lift, then above-target). On a confirmed drop, clear _holding and
        # return; the caller fails the action and recovery re-grasps.
        if not self._transport_grip_ok("lift"):
            self._holding = False
            return False

        self._move_to(above_target, orient)
        if not self._transport_grip_ok("above_target"):
            self._holding = False
            return False

        # Same standoff as the blended phrase, at the cost of one extra
        # program upload.
        if standoff[2] < above_target[2] - 1e-4:
            self._move_to(standoff, orient)
        return self._place_descent(place_target, orient, target_label, target_detection)

    def _place_standoff_m(self, place_target) -> float:
        """How much of the descent the slow servo must own, in metres.

        A held object first meets the container rim when its BOTTOM reaches
        rim height, i.e. at TCP z = rim_z + held_drop. For anything taller
        than the container is deep that point is ABOVE the release height,
        and contact would land in the fast blended move (no compliance, no
        stall exit). Example: rim -0.220, held_drop 10.3 cm, release z -0.139:
        contact begins at -0.117, but a fixed 2 cm standoff takes over at
        -0.119.

        So: start the slow phase strictly ABOVE the first possible contact,
        with a margin for the held-extent estimate. Short objects keep the
        fixed 2 cm because their computed value is smaller.
        """
        base = float(self.PLACE_SERVO_STANDOFF_M)
        geom = getattr(self, "_last_release_height", None)
        rim = getattr(geom, "rim_z", None)
        drop = getattr(geom, "held_drop_m", None)
        if rim is None or drop is None:
            return base
        try:
            contact_z = float(rim) + float(drop)
            need = (contact_z - float(place_target[2])) + self.PLACE_CONTACT_MARGIN_M
        except (TypeError, ValueError):
            return base
        if not np.isfinite(need) or need <= base:
            return base
        need = min(need, self.PLACE_SERVO_STANDOFF_MAX_M)
        logger.info(
            "Place standoff: %.1f cm (rim=%.3f + held_drop=%.3f -> first "
            "contact at z=%.3f, %.1f cm above the release height); the slow "
            "servo owns the whole contact zone instead of the blend",
            need * 100.0,
            float(rim),
            float(drop),
            contact_z,
            (contact_z - float(place_target[2])) * 100.0,
        )
        return need

    def _place_descent(self, place_target, orient, target_label: str, target_detection) -> bool:
        """Final servo of a transport, plus the arrival question it answers.

        A stall alone does NOT fail the transport: a 4.9 mm stall in the
        middle of a bowl has delivered the object. What fails it is ending up
        outside the container's own footprint.
        """
        # ORIENTATION IS SETTLED AT HEIGHT, NEVER IN THE CONTACT ZONE. A held
        # tool extends far below the TCP, so a wrist rotation near the rim
        # sweeps a volume the planner never cleared (a held screwdriver swept
        # into the tool bed and the controller went into PROTECTIVE_STOP). If
        # orientation error remains here, rise to a clearance height, fix the
        # wrist THERE, and come back down; the contact-zone servo then owns
        # position only.
        try:
            cur_o = self.current_orientation()
            if cur_o is not None and orient is not None:
                from scipy.spatial.transform import Rotation as _R

                ori_err = float(
                    (_R.from_rotvec(list(orient)).inv()
                     * _R.from_rotvec(list(cur_o))).magnitude()
                )
                if ori_err > self.PLACE_ORI_FIX_GATE_RAD:
                    fix = np.array(place_target, dtype=float).copy()
                    fix[2] = float(place_target[2]) + self.ROTATE_CLEARANCE_M
                    logger.warning(
                        "[place] %.1f deg of orientation left at the "
                        "standoff; correcting at z=%.3f (+%.0f cm clearance) "
                        "before the contact-zone descent",
                        np.rad2deg(ori_err), fix[2],
                        self.ROTATE_CLEARANCE_M * 100,
                    )
                    self._move_to(fix.tolist(), list(orient))
        except Exception as exc:  # noqa: BLE001 - the descent must still run
            logger.warning("[place] orientation pre-check failed: %s", exc)
        # 0.25, not 0.5: this servo owns the rim-contact zone (see
        # _place_standoff_m), so its speed sets the impulse delivered to the
        # container on contact. Keeps a light bowl from being knocked over.
        landed = self._servo_to(place_target, orient, velocity=self.velocity * 0.25)
        arrived, detail = self._place_arrival(target_label, target_detection)
        self._note_place_arrival(target_label, arrived, detail)
        if arrived:
            if not landed:
                logger.info(
                    "Place '%s': servo stalled but the TCP is over the "
                    "container; releasing here is still correct",
                    target_label or "?",
                )
            return True
        logger.warning(
            "Place '%s' DID NOT ARRIVE (%s): %s",
            target_label or "?",
            "servo stalled" if not landed else "servo converged",
            detail,
        )
        return False

    def _transport_grip_ok(self, waypoint: str) -> bool:
        """Maintain + verify the grip at a STATIONARY transport waypoint.

        Robotiq path: VERIFY FIRST via the gObj-primary _grip_intact() (one
        forced publish + two register reads, ~30-200 ms, no jaw motion). Only
        if that does not confirm, re-squeeze to the fully-closed POSITION at
        GRASP_CLOSE_FORCE to take up slow slip, then re-verify. A False after
        that is a real drop.

        Non-robotiq path: delegate to the force-based
        _verify_holding_during_transport().
        """
        if self._gripper_type() != "robotiq_2f85":
            return self._verify_holding_during_transport()
        if self._grip_intact():
            return True
        close_force = int(os.environ.get("SPARK_GRASP_CLOSE_FORCE", self.GRASP_CLOSE_FORCE))
        logger.info(
            "Transport '%s': grip unconfirmed, re-squeezing to take up slack",
            waypoint,
        )
        self._robotiq_resqueeze(speed=50, force=close_force, settle=self.GRASP_VERIFY_SETTLE_S)
        intact = self._grip_intact()
        if not intact:
            logger.warning("Transport: object DROPPED at waypoint '%s'", waypoint)
            # G2: the run can no longer pass without a later successful re-grasp.
            success_verifier.note_transport_drop(self, waypoint)
        return intact
