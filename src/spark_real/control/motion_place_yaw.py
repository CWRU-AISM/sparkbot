"""Place-time wrist yaw: fit to the target and head-to-head orientation."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class PlaceYawMixin:
    """Place-time wrist yaw: fit to the target and head-to-head orientation."""

    def _fit_place_yaw(self, target_detection):
        """Place yaw from SHAPE AGREEMENT, or None when shape cannot decide.

        Registers the held object's mask against the receptacle's over every
        rotation and keeps the best. Returns a composed orientation when the
        winner beats every orientation >=90 deg away by `JIGSAW_MIN_MARGIN`,
        and None otherwise (a round container).

        The angle is applied as a DELTA on the orientation the arm is already
        carrying, so it never depends on an image-to-world sign convention.
        """
        try:
            held = getattr(self, "_held_mask", None)
            tgt = (
                target_detection.get("_mask")
                if isinstance(target_detection, dict)
                else None
            )
            if held is None or tgt is None:
                return None
            from spark_real.perception.mask_features import best_fit_rotation

            ang, iou, margin = best_fit_rotation(held, tgt)
            logger.info(
                "[fit] rotation %.1f deg  IoU %.3f  margin %.3f "
                "(need margin>=%.3f, IoU>=%.3f)",
                np.rad2deg(ang), iou, margin,
                self.JIGSAW_MIN_MARGIN, self.JIGSAW_MIN_IOU,
            )
            if margin < self.JIGSAW_MIN_MARGIN or iou < self.JIGSAW_MIN_IOU:
                logger.info(
                    "[fit] shape does not decide the orientation here "
                    "(margin %.3f); leaving the yaw to the axis path",
                    margin,
                )
                return None

            from spark_real.control.grasp_strategy import (
                base_orientation,
                compose_yaw,
                measured_yaw_offset,
            )

            cur = measured_yaw_offset(
                self._active_grasp_orient or base_orientation(self),
                base_orientation(self),
            )
            # want = cur + ang: the fit reports the signed rotation that
            # carries the held shape onto the target, ADDED to the carry yaw,
            # no extra 90 and no negation (verified: cur -5.7, ang 273 = -87
            # -> want -92.7 -> tool_X -8.6 against a measured -9). `cur - ang`
            # and `cur - ang - 90` were each wrong by a clean 90 or 180.
            want = cur + float(ang)
            want = (want + np.pi) % (2 * np.pi) - np.pi
            if abs(want) > self.MAX_YAW_OFFSET + 1e-9:
                alt = want + (np.pi if want < 0 else -np.pi)
                if abs(alt) <= self.MAX_YAW_OFFSET + 1e-9:
                    # Substitute only when the shapes cannot tell the two ends
                    # apart. A DECISIVE fit means the 180-equivalent is the
                    # orientation the fit rejected: refusing to orient beats
                    # orienting backwards (grasp_strategy.place_aware_yaw
                    # prevents most of these upstream).
                    if margin >= 2.0 * self.JIGSAW_MIN_MARGIN:
                        logger.warning(
                            "[fit] wanted %.1f deg (past the wrist limit) and "
                            "the fit is DECISIVE (margin %.3f), so the "
                            "180-equivalent %.1f deg would seat the object "
                            "end-for-end; refusing to orient rather than "
                            "placing it backwards",
                            np.rad2deg(want), margin, np.rad2deg(alt),
                        )
                        return None
                    logger.warning(
                        "[fit] wanted %.1f deg (past the wrist limit); using "
                        "the 180-equivalent %.1f deg -- margin %.3f says the "
                        "ends are interchangeable here",
                        np.rad2deg(want), np.rad2deg(alt), margin,
                    )
                    want = alt
                else:
                    logger.warning(
                        "[fit] wanted %.1f deg, beyond the wrist limit and no "
                        "reachable equivalent; leaving the yaw to the axis path",
                        np.rad2deg(want),
                    )
                    return None
            return compose_yaw(self, want, context="place")
        except Exception as exc:  # noqa: BLE001 - never let this stop a place
            logger.warning("[fit] failed (%s); falling back to the axis path", exc)
            return None

    def _orient_head_to_head(self, place_orient, target_detection):
        """Flip the place yaw by 180 deg when the object would go in backwards.

        The yaw resolved so far aligns the two AXES; a PCA axis is a line,
        identical under a 180 deg flip, and the wrong half is a tool seated
        end-for-end.

        Both masks break the tie by being fatter at one end
        (mask_geometry.heavy_end_sign). The comparison stays in ONE image
        frame, so no image-to-world sign convention can invert it.

        Requires BOTH heavy ends to be known. Either coming back 0.0 means the
        shape is too symmetric to call, and the yaw passes through untouched.
        """
        # FIRST: jigsaw. Which rotation makes the two SHAPES coincide answers
        # direction and angle together. Only fall through to the heavy-end
        # heuristic when the shapes cannot decide it.
        try:
            mask_t0 = (
                target_detection.get("_mask")
                if isinstance(target_detection, dict)
                else None
            )
            mask_h0 = getattr(self, "_held_mask", None)
            if mask_t0 is not None and mask_h0 is not None:
                from spark_real.perception.mask_features import best_fit_rotation

                _ang, _iou, _margin = best_fit_rotation(mask_h0, mask_t0)
                logger.info(
                    "[jigsaw] best-fit rotation %.1f deg, IoU %.3f, margin %.3f "
                    "(need margin >= %.3f)",
                    np.rad2deg(_ang), _iou, _margin, self.JIGSAW_MIN_MARGIN,
                )
                if _margin >= self.JIGSAW_MIN_MARGIN and _iou >= self.JIGSAW_MIN_IOU:
                    ang_h = getattr(self, "_held_axis_img", None)
                    mt, _, _, _ = __import__(
                        "spark_core.mask_obb",
                        fromlist=["_pca_obb"],
                    )._pca_obb(np.asarray(mask_t0))
                    if ang_h is not None:
                        # Does the shape fit agree with the axis alignment, or
                        # with its 180 flip? Same image frame throughout.
                        def _w(a):
                            return (a + np.pi) % (2 * np.pi) - np.pi

                        axis_rot = _w(mt - ang_h)
                        # Reduce into [-pi/2, pi/2] BEFORE comparing. A PCA
                        # eigenvector's sign is arbitrary, so mt - ang_h is
                        # only defined mod pi (36/50 vs 50/50 correct flips on
                        # synthetic trials). Same reduction as the heavy-end
                        # rung below.
                        if axis_rot > np.pi / 2:
                            axis_rot -= np.pi
                        elif axis_rot < -np.pi / 2:
                            axis_rot += np.pi
                        if abs(_w(_ang - axis_rot)) > np.pi / 2:
                            flipped = self._flip_yaw_180(place_orient)
                            if flipped is not None:
                                logger.info(
                                    "[jigsaw] shape fit disagrees with the axis "
                                    "alignment by ~180; flipping the place yaw"
                                )
                                return flipped
                            logger.warning(
                                "[jigsaw] shape fit wants a 180 flip but it "
                                "exceeds the wrist limit; seating end-for-end"
                            )
                        else:
                            logger.info("[jigsaw] shape fit agrees; no flip")
                        return place_orient
        except Exception as exc:  # noqa: BLE001
            logger.warning("[jigsaw] failed (%s); falling back to heavy-end", exc)

        try:
            sgn_h = getattr(self, "_held_axis_sign", None)
            ang_h = getattr(self, "_held_axis_img", None)
            mask_t = (
                target_detection.get("_mask")
                if isinstance(target_detection, dict)
                else None
            )
            if sgn_h in (None, 0.0) or ang_h is None or mask_t is None:
                logger.info(
                    "[head-to-head] not decidable (held sign=%s, target mask=%s); "
                    "leaving the yaw as resolved",
                    sgn_h,
                    "yes" if mask_t is not None else "no",
                )
                return place_orient

            from spark_core.mask_obb import _pca_obb
            from spark_real.perception.mask_features import heavy_end_sign

            ang_t, ar_t, _, _ = _pca_obb(np.asarray(mask_t))
            sgn_t = heavy_end_sign(np.asarray(mask_t), ang_t)
            if sgn_t == 0.0:
                logger.info(
                    "[head-to-head] target heavy end UNKNOWN (ar %.2f); leaving "
                    "the yaw as resolved",
                    ar_t,
                )
                return place_orient

            def _wrap(a):
                return (a + np.pi) % (2 * np.pi) - np.pi

            dir_h = ang_h + (0.0 if sgn_h > 0 else np.pi)
            dir_t = ang_t + (0.0 if sgn_t > 0 else np.pi)
            # The rotation the pipeline effectively applied, reduced mod 180,
            # which is all an axis alignment can express.
            rot = _wrap(ang_t - ang_h)
            if rot > np.pi / 2:
                rot -= np.pi
            elif rot < -np.pi / 2:
                rot += np.pi
            residual = _wrap(dir_t - (dir_h + rot))
            if abs(residual) <= np.pi / 2:
                logger.info(
                    "[head-to-head] heads agree (residual %.0f deg); no flip",
                    np.rad2deg(residual),
                )
                return place_orient

            flipped = self._flip_yaw_180(place_orient)
            if flipped is None:
                logger.warning(
                    "[head-to-head] object would seat BACKWARDS (residual "
                    "%.0f deg) but the 180 flip exceeds the wrist limit; "
                    "placing anyway, end-for-end",
                    np.rad2deg(residual),
                )
                return place_orient
            logger.info(
                "[head-to-head] object would seat BACKWARDS (residual %.0f deg); "
                "flipping the place yaw 180",
                np.rad2deg(residual),
            )
            return flipped
        except Exception as exc:  # noqa: BLE001 - never let this stop a place
            logger.warning("[head-to-head] failed (%s); yaw unchanged", exc)
            return place_orient

    def _flip_yaw_180(self, orient):
        """`orient` rotated 180 deg about world Z, or None if out of limit."""
        from spark_real.control.grasp_strategy import (
            base_orientation,
            compose_yaw,
            measured_yaw_offset,
        )

        cur = measured_yaw_offset(orient, base_orientation(self))
        for cand in (cur + np.pi, cur - np.pi):
            if abs(cand) <= self.MAX_YAW_OFFSET + 1e-9:
                return compose_yaw(self, cand, context="place")
        return None
