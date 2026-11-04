"""Release dip side: rotate the approach yaw so a tilted release lands on the allowed side."""

import logging

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


class ReleaseDipMixin:
    """Release dip side: rotate the approach yaw so a tilted release lands on the allowed side."""

    def _dip_side_orient(self, orient, yaw_aligned: bool = False):
        """Fold the deep-dip side choice into the place-approach orientation.

        Called by MotionMixin._transport_to with the approach orientation it
        is about to command. When the upcoming release (executor_core's
        _pending_release_tilt look-ahead) asks for a dip past the positive
        budget (MAX_POS_PITCH_RAD, the wrist-camera clearance) and `orient`
        would put the dip on that side (tool_y_world[0] > 0 at the release's
        own pitch_sign test), rotate the approach yaw so the dip lands on the
        negative side and gets the 90 deg budget instead. The yaw rides the
        transport's horizontal transit, so it costs no extra wrist travel at
        the release.

        Two cases:
          * `yaw_aligned` False (bin/bowl: no slot, no shape fit, no OBB
            adopted): the yaw is free, so pick the smallest yaw that puts
            tool Y DIP_SIDE_MARGIN_RAD past the boundary, cable limit
            enforced by compose_yaw.
          * `yaw_aligned` True: the yaw carries a slot/fit/OBB alignment and
            only its 180-twin preserves it (a two-jaw grasp is symmetric mod
            180). Try _flip_yaw_180; when the twin exceeds the wrist-3 cable
            limit, keep the alignment and let the release clamp to 40 deg. A
            shallow dip beats a crooked seat.

        Returns `orient` (possibly replaced). Never raises: any read failure
        keeps the alignment untouched and the release clamps.
        """
        pend = getattr(self, "_pending_release_tilt", None)
        if not pend:
            return orient
        try:
            tilt = abs(float(pend.get("tilt_angle", 0.0) or 0.0))
        except (TypeError, ValueError, AttributeError):
            return orient
        if tilt <= self.MAX_POS_PITCH_RAD + 1e-9:
            return orient
        try:
            tool_y = Rotation.from_rotvec(
                np.asarray(orient, dtype=float)
            ).apply([0.0, 1.0, 0.0])
        except Exception as exc:  # noqa: BLE001 - never stop a transport
            logger.warning("[dip-side] could not read orient (%s); unchanged", exc)
            return orient
        if tool_y[0] <= 0.0:
            logger.info(
                "[dip-side] release wants %.0f deg; approach already on the "
                "negative-pitch side (tool_y_x=%.2f), budget %.0f deg",
                np.rad2deg(tilt), tool_y[0], np.rad2deg(self.MAX_NEG_PITCH_RAD),
            )
            return orient

        if yaw_aligned:
            flipped = self._flip_yaw_180(orient)
            if flipped is not None:
                logger.info(
                    "[dip-side] release wants %.0f deg > %.0f positive budget; "
                    "flipping the ALIGNED place yaw 180 (grasp-symmetric) so "
                    "the dip gets the %.0f deg negative budget",
                    np.rad2deg(tilt), np.rad2deg(self.MAX_POS_PITCH_RAD),
                    np.rad2deg(self.MAX_NEG_PITCH_RAD),
                )
                return flipped
            logger.warning(
                "[dip-side] release wants %.0f deg but the aligned yaw's 180 "
                "twin exceeds the wrist-3 cable limit; keeping the alignment, "
                "the release will clamp to %.0f deg",
                np.rad2deg(tilt), np.rad2deg(self.MAX_POS_PITCH_RAD),
            )
            return orient

        # Free yaw: aim tool Y's world heading past the +-90 deg boundary by
        # the margin, with the least wrist travel from base. The heading of
        # tool Y after a world-Z yaw psi is (delta + psi), where delta is its
        # heading at the base orientation, and pitch_sign goes negative when
        # cos(delta + psi) < 0.
        from spark_real.control.grasp_strategy import base_orientation, compose_yaw

        try:
            ty_base = Rotation.from_rotvec(base_orientation(self)).apply(
                [0.0, 1.0, 0.0]
            )
            delta = float(np.arctan2(ty_base[1], ty_base[0]))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[dip-side] base orientation unreadable (%s)", exc)
            return orient
        edge = np.pi / 2.0 + self.DIP_SIDE_MARGIN_RAD
        candidates = sorted(
            (
                float((s * edge - delta + np.pi) % (2 * np.pi) - np.pi)
                for s in (1.0, -1.0)
            ),
            key=abs,
        )
        for psi in candidates:
            picked = compose_yaw(self, psi, context="place-dip")
            if picked is None:
                continue  # over the cable limit; try the other side
            logger.info(
                "[dip-side] release wants %.0f deg > %.0f positive budget; "
                "free yaw -> %.1f deg so the dip gets the %.0f deg negative "
                "budget (folded into the approach, no extra motion)",
                np.rad2deg(tilt), np.rad2deg(self.MAX_POS_PITCH_RAD),
                np.rad2deg(psi), np.rad2deg(self.MAX_NEG_PITCH_RAD),
            )
            return picked
        logger.warning(
            "[dip-side] release wants %.0f deg but no dip-side yaw fits the "
            "wrist-3 cable limit; the release will clamp to %.0f deg",
            np.rad2deg(tilt), np.rad2deg(self.MAX_POS_PITCH_RAD),
        )
        return orient
