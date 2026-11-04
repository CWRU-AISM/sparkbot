"""Label binding for ``LiberoExecutor``: fresh-detection merge, position-sticky association and the
pre-primitive scene diff with in-flight retarget (cfg.scene_diff).

``BindingMixin`` expects the executor attributes set in ``LiberoExecutor.__init__`` (det_map,
plan_binding, agent_ref, the ``_fresh_*`` stores, holding, last_pick_label, trial_meta, cfg).
"""
from __future__ import annotations

import dataclasses
import time
from typing import Optional

import numpy as np

from spark_bench.libero_pro.motion import move_to
from spark_core.binding import (
    STATUS_MISSING,
    STATUS_MOVED,
    STATUS_SELF,
    STICKY_AMBIGUOUS,
    STICKY_HELD,
    STICKY_OCCLUDED,
    STICKY_OUT_OF_GATE,
    _labels_match,
    compute_scene_diff,
    dedup_candidate_indices,
    fuzzy_key,
    is_self_occluded,
    sticky_associate,
)

_LABEL_KEYS = ('keypoint_label', 'target_label', 'label', 'pick_label', 'place_label')


def action_uses_keypoint(act: dict) -> bool:
    params = act.get('params', {}) or {}
    return any(params.get(k) for k in _LABEL_KEYS)


def action_target_label(act: dict) -> str:
    params = act.get('params', {}) or {}
    for k in _LABEL_KEYS:
        v = params.get(k)
        if v:
            return str(v)
    return ''


class BindingMixin:

    def _sticky_enabled(self) -> bool:
        return bool(getattr(self.cfg, 'scene_diff', False))

    def _sticky_gate_m(self) -> float:
        return max(float(getattr(self.cfg, 'retarget_max_cm', 10.0)), 12.0) / 100.0

    def _merge_fresh_detections(self, fresh) -> tuple[int, int]:
        """
        Merge a fresh DetectionResult into the executor's binding state.

        Scene diff off: freshest-per-label wins, only keys the fresh frame produced are overwritten.

        Scene diff on: bind-preserving. The fresh frame never rewrites ``det_map``; it fills the
        per-capture stores (``_fresh_dets`` / ``_fresh_out_of_gate`` / ``_fresh_groups``). The diff
        measures displacement same-source (fresh agentview centroid vs the ``agent_ref`` centroid the
        label was bound at), and the retarget / self-caused / recovery paths move the authoritative
        binding by that delta, never to the fresh absolute position (the two cameras disagree by a
        systematic 1-2 cm, which would read as phantom 'moved' verdicts).

        Per bound label: a held label is skipped (the camera cannot relocate the object in the
        gripper); the nearest in-gate instance is recorded; ambiguous / occluded-out-of-gate holds; an
        out-of-gate instance is recorded separately so the diff routes to recovery without poisoning
        det_map.

        Returns ``(n_labels_seen, n_labels_updated)``.
        """
        self._sticky_flags = {}
        self._fresh_dets = {}
        self._fresh_out_of_gate = {}
        self._fresh_groups = {}
        if fresh is None:
            return 0, 0
        if not self._sticky_enabled() or not getattr(fresh, 'dets', None):
            n_updated = 0
            det_map = fresh.det_map or {}
            for k, v in det_map.items():
                if k in self.det_map:
                    n_updated += 1
                self.det_map[k] = v
            return len(det_map), n_updated

        rename = getattr(self.cfg, '_mp_phrase_to_concept', None) or {}
        groups: dict[str, list] = {}
        for d in fresh.dets:
            if getattr(d, 'position_3d', None) is None:
                continue
            groups.setdefault(rename.get(d.label, d.label), []).append(d)

        n_updated = 0
        for label, cands in groups.items():
            # Collapse SAM3 duplicate masks first; otherwise nearest-to-previous association favours the
            # duplicate that moved least and under-measures real motion.
            if len(cands) > 1:
                keep = dedup_candidate_indices(
                    [c.position_3d for c in cands],
                    [float(getattr(c, 'confidence', 0.0)) for c in cands])
                cands = [cands[i] for i in keep]
            key = label if label in self.det_map else (fuzzy_key(self.det_map, label) or label)
            self._fresh_groups[key] = list(cands)
            prev = self.det_map.get(key)
            if prev is None:
                # Never seen before: adopt confidence top-1 (agentview-only, so the ref is the position).
                best = max(cands, key=lambda c: float(getattr(c, 'confidence', 0.0)))
                self.det_map[key] = best
                self.agent_ref[key] = np.asarray(best.position_3d, dtype=float).copy()
                if self.plan_binding.get(key) is None:
                    self.plan_binding[key] = np.asarray(best.position_3d, dtype=float).copy()
                continue
            if self.holding and self.last_pick_label and _labels_match(key, self.last_pick_label):
                # In the gripper: any mask is the object-in-hand or a phantom.
                self._sticky_flags[key] = STICKY_HELD
                continue
            ref = self.agent_ref.get(key)
            if ref is None:
                p = getattr(prev, 'position_3d', None)
                ref = np.asarray(p, dtype=float).copy() if p is not None else self.plan_binding.get(key)
            if ref is None:
                continue
            res = sticky_associate(ref, [c.position_3d for c in cands],
                                   gate_m=self._sticky_gate_m(), ambiguity_sep_m=0.03)
            self._sticky_flags[key] = res.status
            if res.status == STICKY_OUT_OF_GATE:
                if is_self_occluded(ref, self._ee()):
                    # The arm hovers on top of this object: the far re-association is a lookalike. Hold
                    # the binding; grasp telemetry disambiguates milliseconds later.
                    self._sticky_flags[key] = STICKY_OCCLUDED
                    if self.cfg.verbose:
                        print(f"[Sticky] {key!r}: out-of-gate re-association while the EE hovers over "
                              f"it, holding binding (self-occlusion)")
                else:
                    # Record the far candidate for the diff (moved beyond retarget -> recovery), but do
                    # NOT rebind det_map to a possible lookalike.
                    self._fresh_out_of_gate[key] = cands[res.chosen_index]
                continue
            if res.status == STICKY_AMBIGUOUS or res.chosen_index is None:
                if self.cfg.verbose and res.status == STICKY_AMBIGUOUS:
                    print(f"[Sticky] {key!r}: {len(cands)} instances in-gate and inseparable; keeping "
                          f"previous binding")
                continue
            fresh_det = cands[res.chosen_index]
            self._fresh_dets[key] = fresh_det
            n_updated += 1
            if self.cfg.verbose:
                newp = np.asarray(fresh_det.position_3d, float).reshape(-1)
                pv = np.asarray(ref, float).reshape(-1)[:3]
                d_cm = float(np.linalg.norm(newp[:3] - pv)) * 100.0
                if d_cm > 0.5:
                    print(f"[StickyObs] {key!r} ref=({pv[0]:.4f},{pv[1]:.4f},{pv[2]:.4f}) "
                          f"fresh=({newp[0]:.4f},{newp[1]:.4f},{newp[2]:.4f}) delta={d_cm:.1f}cm "
                          f"conf={float(getattr(fresh_det, 'confidence', 0.0)):.3f}")
                if len(cands) > 1:
                    print(f"[Sticky] {key!r}: associated instance {res.chosen_index}/{len(cands)} at "
                          f"{(res.chosen_dist_m or 0)*100:.1f}cm ({res.status})")
        return len(groups), n_updated

    def _shift_binding(self, key: str, fresh_pos, fresh_det=None) -> None:
        """
        Move the binding of ``key`` by the same-source delta of a fresh agentview observation.

        ``plan_binding`` (and the det_map position) shifts by ``fresh_pos - agent_ref[key]`` so a
        wrist-fused bind keeps its cross-camera correction; ``agent_ref`` re-anchors at the fresh
        centroid. Without a ref the fresh position is adopted outright.
        """
        fresh_pos = np.asarray(fresh_pos, dtype=float).reshape(-1)[:3].copy()
        ref = self.agent_ref.get(key)
        bound = self.plan_binding.get(key)
        if ref is not None and bound is not None:
            new_fused = np.asarray(bound, dtype=float) + (fresh_pos - ref)
        else:
            new_fused = fresh_pos.copy()
        self.plan_binding[key] = new_fused
        self.agent_ref[key] = fresh_pos.copy()
        prev = self.det_map.get(key)
        base = fresh_det if fresh_det is not None else prev
        if base is not None:
            try:
                self.det_map[key] = dataclasses.replace(
                    base, position_3d=new_fused.copy(), position_agentview=fresh_pos.copy())
            except Exception:
                self.det_map[key] = base

    def _current_positions(self) -> dict:
        """
        Same-source 'current' map for the scene diff: fresh agentview observations from the last merge,
        else the agentview reference. No evidence means hold (delta 0), never 'missing'.
        """
        out: dict = {}
        for k in self.det_map:
            if k in self._fresh_dets:
                out[k] = np.asarray(self._fresh_dets[k].position_3d, dtype=float)
            elif k in self._fresh_out_of_gate:
                out[k] = np.asarray(self._fresh_out_of_gate[k].position_3d, dtype=float)
            else:
                ref = self.agent_ref.get(k)
                if ref is None:
                    d = self.det_map.get(k)
                    p = getattr(d, 'position_3d', None) if d is not None else None
                    ref = np.asarray(p, dtype=float) if p is not None else None
                out[k] = ref
        return out

    def _reference_positions(self) -> dict:
        """Same-source 'bound' map: agentview refs, else the fused binding (wrist-only labels)."""
        out: dict = {}
        for k in self.det_map:
            ref = self.agent_ref.get(k)
            if ref is None:
                ref = self.plan_binding.get(k)
            out[k] = np.asarray(ref, dtype=float) if ref is not None else None
        return out

    def _run_scene_diff(self, target_label: str, *, context: str, action_idx: Optional[int] = None):
        """
        Compute and record one same-source scene diff. Labels the diff attributes to the robot itself
        (with fresh evidence) have their binding delta-shifted. Returns the SceneDiff.
        """
        cfg = self.cfg
        sd = compute_scene_diff(
            self._reference_positions(), self._current_positions(),
            target_label=target_label or None,
            held_label=self.last_pick_label if self.holding else None,
            gripper_pos=self._ee(),
            moved_threshold_m=float(getattr(cfg, 'scene_diff_moved_cm', 1.5)) / 100.0,
            self_radius_m=float(getattr(cfg, 'scene_diff_self_radius_cm', 12.0)) / 100.0)
        rec = sd.to_meta()
        rec['context'] = context
        if action_idx is not None:
            rec['action_idx'] = action_idx
        if self._sticky_flags:
            rec['sticky'] = dict(self._sticky_flags)
        self.trial_meta.setdefault('scene_diffs', []).append(rec)
        for lbl, ld in sd.labels.items():
            if ld.status == STATUS_SELF and ld.current_pos is not None and lbl in self._fresh_dets:
                self._shift_binding(lbl, ld.current_pos)
        return sd

    def _handle_target_diff(self, sd, *, context: str) -> bool:
        """
        Retarget-or-abort for a diff whose target moved or vanished. True to continue (ok or
        retargeted), False when the executor falls through to recovery (self.abort set).
        """
        cfg = self.cfg
        if sd.target_status == STATUS_MOVED:
            delta_cm = sd.target_delta_m * 100.0
            # Pre-close occlusion floor: with the gripper over the target, the agentview centroid shifts
            # 2.3 to 3.8 cm on unperturbed trials (phantom retargets at pre_close took the adaptive arm's
            # 0 cm controls to 9/45 against the baseline's 27/45). Below the floor trust the standing
            # bind; a true move is caught by the telemetry EMPTY_CLOSE vote and its regrasp.
            if context == 'pre_close' and delta_cm < float(getattr(cfg, 'pre_close_retarget_min_cm', 4.0)):
                if cfg.verbose:
                    print(f"[Retarget] suppressed at pre_close: {sd.target_key} delta {delta_cm:.1f}cm is "
                          f"below the occlusion floor")
                return True
            if delta_cm < float(getattr(cfg, 'retarget_max_cm', 10.0)):
                # In-flight retarget by the same-source delta.
                cur = sd.labels[sd.target_key].current_pos
                if cur is not None:
                    self._shift_binding(sd.target_key, cur)
                self.trial_meta.setdefault('retarget_events', []).append({
                    'label': sd.target_key, 'delta_cm': round(delta_cm, 2),
                    'context': context, 't_flag': sd.t, 't_resume': time.time()})
                if cfg.verbose:
                    print(f"[Retarget] {sd.target_key} moved {delta_cm:.1f}cm (< {cfg.retarget_max_cm}cm); "
                          f"re-aimed at fresh detection ({context})")
                return True
            self.abort = {'reason': 'target_moved_beyond_retarget', 'label': sd.target_key,
                          'delta_cm': round(delta_cm, 2), 'context': context, 't_flag': sd.t,
                          't': time.time()}
        elif sd.target_status == STATUS_MISSING:
            self.abort = {'reason': 'target_missing', 'label': sd.target_label, 'context': context,
                          't_flag': sd.t, 't': time.time()}
        if self.abort is not None:
            self.trial_meta['scene_abort'] = self.abort
            return False
        return True

    def _pre_primitive_gate(self, act_idx: int, act: dict) -> bool:
        """cfg.scene_diff gate run before every keypoint-referencing leaf."""
        if not getattr(self.cfg, 'scene_diff', False) or not action_uses_keypoint(act):
            return True
        sd = self._run_scene_diff(action_target_label(act), context='pre_primitive', action_idx=act_idx)
        return self._handle_target_diff(sd, context='pre_primitive')

    def _move_to_monitored(self, label: str, target, gripper_open, steps: int = 200,
                           chunk: int = 100) -> float:
        """
        Chunked move that re-checks the scene between chunks (cfg.scene_diff; otherwise a plain move).

        Bounds mid-motion intervention-detection latency to about one chunk plus one verdict instead of
        the whole primitive. A small target move shifts the remaining waypoint by the binding delta; a
        large move or a vanished target sets self.abort and the motion stops where it is.
        """
        if not self._sticky_enabled() or chunk >= steps:
            return self._move_to(target, gripper_open, steps=steps)
        target = np.asarray(target, dtype=float).copy()
        sd0 = self._run_scene_diff(label, context='mid_motion')
        key = getattr(sd0, 'target_key', None) if sd0 is not None else None
        bind0 = (np.asarray(self.plan_binding.get(key), float).copy()
                 if key is not None and key in self.plan_binding else None)
        residual = 0.0
        done = 0
        while done < steps:
            n = min(chunk, steps - done)
            residual = move_to(self.env, self.model, self.data, self.joint_ids, self.ee_site,
                               self.arm_actuator_ids, self.gripper_actuator_ids,
                               target, gripper_open, steps=n)
            done += n
            if done >= steps:
                break
            self._maybe_refresh_perception({'type': 'move_to_keypoint', 'params': {'keypoint_label': label}})
            sd = self._run_scene_diff(label, context='mid_motion')
            if sd is None:
                continue
            if not self._handle_target_diff(sd, context='mid_motion'):
                return residual
            if key is not None and bind0 is not None and key in self.plan_binding:
                delta = np.asarray(self.plan_binding[key], float) - bind0
                if float(np.linalg.norm(delta)) > 0.005:
                    target = target + delta
                    bind0 = np.asarray(self.plan_binding[key], float).copy()
        return residual
