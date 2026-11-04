"""Scripted mid-episode displacement of the pick target (the fixed-plan displacement experiment)."""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from spark_bench.dyn.schedule import Phase


class PerturbationInjector:
    """Wraps ``env.step`` and fires a one-shot displacement at the scheduled phase.

    MID_APPROACH fires once the EE has covered ``progress_frac`` of its start distance to
    ``pick_target_pos``. POST_GRASP_PLAN fires on the first step that commands a close (action[6] > 0.5),
    before that step runs. ``fire_on_first_step`` fires on the first step (POST_PLAN slide). With
    ``slide_steps`` > 1, ``displace_fn`` runs once per step for that many steps (a continuous slide).
    """

    def __init__(self, env, phase: Phase, *,
                 displace_fn: Callable[[], Optional[str]],
                 pick_target_pos: Optional[np.ndarray] = None,
                 get_ee_pos: Optional[Callable[[], Optional[np.ndarray]]] = None,
                 progress_frac: float = 0.5,
                 meta: Optional[dict] = None,
                 slide_steps: int = 0,
                 fire_on_first_step: bool = False):
        self.env = env
        self.phase = Phase(phase)
        self.displace_fn = displace_fn
        self.pick_target_pos = (None if pick_target_pos is None else
                                np.asarray(pick_target_pos, float)[:3])
        self.get_ee_pos = get_ee_pos
        self.progress_frac = float(progress_frac)
        self.meta = meta if meta is not None else {}
        self.fired = False
        self.slide_steps = int(slide_steps)
        self.fire_on_first_step = bool(fire_on_first_step)
        self._slide_left = 0
        self._start_dist: Optional[float] = None
        self._orig_step = None

    def install(self) -> None:
        if self._orig_step is not None:
            return
        self._orig_step = self.env.step

        def _patched(action):
            if not self.fired and self._should_fire(action):
                self.fire()
            elif self._slide_left > 0:
                self._slide_left -= 1
                try:
                    self.displace_fn()
                except Exception as e:
                    self.meta.setdefault('perturb_error', str(e))
                if self._slide_left == 0:
                    self.meta['t_perturb_end'] = time.time()
            return self._orig_step(action)

        self.env.step = _patched

    def uninstall(self) -> None:
        if self._orig_step is not None:
            try:
                self.env.step = self._orig_step
            except Exception:
                pass
            self._orig_step = None

    def _should_fire(self, action) -> bool:
        if self.fire_on_first_step:
            return True
        if self.phase == Phase.POST_GRASP_PLAN:
            try:
                return float(np.asarray(action).reshape(-1)[6]) > 0.5
            except Exception:
                return False
        if self.phase == Phase.MID_APPROACH:
            if self.pick_target_pos is None or self.get_ee_pos is None:
                return False
            ee = self.get_ee_pos()
            if ee is None:
                return False
            dist = float(np.linalg.norm(np.asarray(ee, float)[:3] - self.pick_target_pos))
            if self._start_dist is None:
                self._start_dist = max(dist, 1e-6)
                return False
            return 1.0 - dist / self._start_dist >= self.progress_frac
        return False

    def fire(self) -> None:
        """Start the displacement: one write, or the first of ``slide_steps`` increments."""
        if self.fired:
            return
        self.fired = True
        t = time.time()
        body = None
        try:
            body = self.displace_fn()
        except Exception as e:  # an injection failure never breaks the trial
            self.meta['perturb_error'] = str(e)
        if self.slide_steps > 1:
            self._slide_left = self.slide_steps - 1
            self.meta['slide_steps'] = self.slide_steps
        self.meta['t_perturb'] = t
        self.meta['perturb_phase'] = self.phase.value
        if body is not None:
            self.meta['perturbed_body'] = body
