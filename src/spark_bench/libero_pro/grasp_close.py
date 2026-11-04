"""Grasp close and the informative-moment captures for ``LiberoExecutor``.

Gripper telemetry is the first, instant vote on a grasp (sim analog of the Robotiq gObj flag); the
schedule-triggered captures (cfg.event_captures) are the camera vote: before closure, after a telemetry
EMPTY_CLOSE, and after release. ``GraspCloseMixin`` needs ``BindingMixin`` and the executor state.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from spark_bench.libero_pro.motion import step_env
from spark_bench.libero_pro.perception import redetect_agentview
from spark_bench.libero_pro.primitives import keypoint as _keypoint
from spark_bench.libero_pro.primitives._common import fuzzy_get_det
from spark_bench.libero_pro.telemetry import (
    FAST_CLOSE, GraspClassification, aperture_settled, classify_grasp_outcome, find_finger_qpos_addrs, read_aperture)
from spark_core.binding import STATUS_MOVED, fuzzy_key


class GraspCloseMixin:

    def close_gripper_with_telemetry(self, steps: int = 300) -> list[float]:
        """Close the gripper while sampling the finger aperture every step; returns the trace."""
        if self._finger_addrs is None:
            self._finger_addrs = find_finger_qpos_addrs(self.model)
        trace: list[float] = []
        for _ in range(steps):
            a = np.zeros(7)
            a[6] = 1.0
            if step_env(self.env, a):
                break
            w = read_aperture(self.data, self._finger_addrs)
            if w is not None:
                trace.append(w)
                if FAST_CLOSE and aperture_settled(trace):
                    break
        return trace

    def record_grasp_outcome(self, cls: GraspClassification, *, is_retry: bool = False) -> None:
        rec = cls.to_meta()
        rec['t'] = time.time()
        rec['label'] = self.last_pick_label
        rec['is_retry'] = bool(is_retry)
        self.trial_meta.setdefault('grasp_outcomes', []).append(rec)
        if self.cfg.verbose:
            print(f"[GraspTelemetry] {cls.outcome.value} (final={cls.final_aperture:.4f}m "
                  f"plateau={cls.plateau_aperture:.4f}m n={cls.n_samples}{' retry' if is_retry else ''})")

    def event_capture(self, event: str) -> Optional[dict]:
        """
        Capture + SAM3 verdict at a primitive-declared informative moment (cfg.event_captures).

        Merges the fresh detections like the freshness gate and records {event, t_event, t_capture,
        t_verdict} in trial_meta['capture_events']. Returns the record (the caller may attach a
        verdict), or None when disabled or unavailable.
        """
        if not getattr(self.cfg, 'event_captures', False):
            return None
        if self.sam3 is None or not self.prompts_for_refresh:
            return None
        t_event = time.time()
        timing: dict = {}
        try:
            fresh = redetect_agentview(self.env, self.sam3, self.prompts_for_refresh, self.cfg,
                                       timing=timing, all_instances=self._sticky_enabled())
        except Exception as e:
            if self.cfg.verbose:
                print(f"[EventCapture:{event}] redetect failed: {e}")
            return None
        t_verdict = time.time()
        n_labels, n_updated = self._merge_fresh_detections(fresh)
        self._last_perception_t = t_verdict
        rec = {
            'event': event,
            't_event': t_event,
            't_capture': timing.get('t_capture', t_event),
            't_verdict': t_verdict,
            'latency_capture_s': round(timing.get('t_capture', t_event) - t_event, 4),
            'latency_verdict_s': round(t_verdict - t_event, 4),
            'n_labels': n_labels,
            'n_updated': n_updated,
        }
        if self._sticky_flags:
            rec['sticky'] = dict(self._sticky_flags)
        self.trial_meta.setdefault('capture_events', []).append(rec)
        if self.cfg.verbose:
            print(f"[EventCapture:{event}] verdict in {rec['latency_verdict_s']*1000:.0f}ms "
                  f"({rec['n_labels']} labels, {n_updated} updated)")
        return rec

    def pre_close_capture(self) -> None:
        """
        Capture right before closure, then (cfg.scene_diff) re-check the pick target: a small move
        re-approaches the fresh detection, a large move or a missing target sets self.abort so the
        grasp never closes on air.
        """
        rec = self.event_capture('pre_close')
        if rec is None or not getattr(self.cfg, 'scene_diff', False):
            return
        label = self.last_pick_label
        if not label:
            return
        sd = self._run_scene_diff(label, context='pre_close')
        if not self._handle_target_diff(sd, context='pre_close'):
            return
        if sd.target_status == STATUS_MOVED:
            # Retargeted: the arm hovers at the stale grasp pose; re-approach before the close.
            _keypoint.handle(self, {'keypoint_label': label})
            evs = self.trial_meta.get('retarget_events') or []
            if evs:
                evs[-1]['t_resume'] = time.time()

    def post_release_capture(self) -> bool:
        """
        Capture right after release; verdict = did the released object land near the place target.
        Returns True when the capture ran (the caller then skips the blunt perception invalidate).

        The verdict uses only detections from this capture, and decides identity by positive
        evidence: a fresh instance within the sticky gate of the landing spot is the placed object
        (report its xy error); else one within the gate of the pre-place origin means it never left
        (``ok: False``, ``still_at_origin``); anything else abstains.
        """
        rec = self.event_capture('post_release')
        if rec is None:
            return False
        label = self.last_pick_label
        verdict: Optional[dict] = None
        if label:
            pos = None
            at_origin = False
            if self._sticky_enabled():
                key = fuzzy_key(self.det_map, label) or label
                cands = self._fresh_groups.get(key) or []
                gate = self._sticky_gate_m()

                def _nearest(anchor):
                    if anchor is None or not cands:
                        return None, None
                    a = np.asarray(anchor, dtype=float)[:2]
                    best = min(cands, key=lambda c: float(np.linalg.norm(
                        np.asarray(c.position_3d, float)[:2] - a)))
                    return best, float(np.linalg.norm(np.asarray(best.position_3d, float)[:2] - a))

                # Binding: the placed label's motion is self-caused, so track the nearest fresh
                # instance to the release point whatever its distance.
                rebind_det, _ = _nearest(self.last_place_target)
                cand, dist = _nearest(self.last_place_target)
                if cand is not None and dist <= gate:
                    pos = np.asarray(cand.position_3d, dtype=float)
                else:
                    # Still at the pre-place origin? The hold suppressed association, so the standing
                    # binding is where it was picked up.
                    cand, dist = _nearest(self.plan_binding.get(key))
                    if cand is not None and dist <= gate:
                        pos = np.asarray(cand.position_3d, dtype=float)
                        at_origin = True
                if rebind_det is not None:
                    p = np.asarray(rebind_det.position_3d, dtype=float)
                    self.plan_binding[key] = p.copy()
                    self.agent_ref[key] = p.copy()
                    self.det_map[key] = rebind_det
            else:
                det = fuzzy_get_det(self.det_map, label)
                pos = getattr(det, 'position_3d', None) if det is not None else None
                if pos is not None:
                    key = fuzzy_key(self.det_map, label) or label
                    self.plan_binding[key] = np.asarray(pos, dtype=float).copy()
            if pos is not None:
                verdict = {'placed_label': label}
                if at_origin:
                    verdict['reason'] = 'still_at_origin'
                if self.last_place_target is not None:
                    err = float(np.linalg.norm(np.asarray(pos, float)[:2]
                                               - np.asarray(self.last_place_target, float)[:2]))
                    verdict['xy_error_m'] = round(err, 4)
                    verdict['ok'] = bool(err < 0.10)
            else:
                verdict = {'placed_label': label, 'ok': None, 'abstain': True,
                           'reason': 'not_freshly_detected'}
        rec['verdict'] = verdict
        return True

    def post_grasp_second_vote(self) -> None:
        """
        Camera second vote behind a telemetry EMPTY_CLOSE (only with cfg.event_captures): if the pick
        target is still detected away from the hand the grasp missed, and one local retry runs (reopen,
        re-approach the fresh detection, re-close), its telemetry outcome recorded alongside.
        """
        label = self.last_pick_label
        if not label:
            return
        # Drop the holding flag before the capture so the sticky merge does not suppress the pick
        # label as held (telemetry already said the hand is empty).
        was_holding = self.holding
        self.holding = False
        rec = self.event_capture('post_grasp_verify')
        if rec is None:
            self.holding = was_holding
            return
        fresh_det = None
        if self._sticky_enabled():
            key = fuzzy_key(self.det_map, label) or label
            fresh_det = self._fresh_dets.get(key) or self._fresh_out_of_gate.get(key)
            pos = np.asarray(fresh_det.position_3d, dtype=float) if fresh_det is not None else None
        else:
            det = fuzzy_get_det(self.det_map, label)
            pos = getattr(det, 'position_3d', None) if det is not None else None
        missed = False
        if pos is not None:
            dist_to_hand = float(np.linalg.norm(np.asarray(pos, float) - self._ee()))
            missed = dist_to_hand > 0.10
            rec['verdict'] = {'label': label, 'dist_to_hand_m': round(dist_to_hand, 4),
                              'confirms_empty': missed}
        else:
            rec['verdict'] = {'label': label, 'confirms_empty': False, 'reason': 'not_detected'}
        if not missed:
            self.holding = was_holding
            return
        if self.cfg.verbose:
            print(f"[GraspTelemetry] camera confirms empty close on {label!r}, local regrasp retry")
        if self._sticky_enabled() and fresh_det is not None:
            # The stale bind missed and the camera sees the object elsewhere: adopt the fresh
            # observation outright before re-approaching.
            key = fuzzy_key(self.det_map, label) or label
            p = np.asarray(fresh_det.position_3d, dtype=float).copy()
            self.plan_binding[key] = p
            self.agent_ref[key] = p.copy()
            self.det_map[key] = fresh_det
        self._gripper(True, steps=60)
        self.holding = False
        _keypoint.handle(self, {'keypoint_label': label})
        trace = self.close_gripper_with_telemetry(steps=300)
        cls2 = classify_grasp_outcome(trace)
        self.holding = True
        self.record_grasp_outcome(cls2, is_retry=True)
