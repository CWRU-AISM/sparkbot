"""Shared helpers for the motion mixins."""

import numpy as np


class _StillnessTracker:
    """Motion / stillness detector on a FIXED sample cadence.

    The poll rate and the motion test are independent: displacement is
    evaluated over its own ``sample_s`` window. ``speed_eps`` is 3 mm/s, i.e.
    0.3 mm of travel per 100 ms sample; over a 25 ms window it would be
    0.075 mm, inside TCP measurement noise, and a spuriously latched
    ``ever_moved`` lets the stillness exit fire early.
    """

    __slots__ = ("_sample_s", "_eps", "_need", "_prev", "_prev_t", "ever_moved", "still")

    def __init__(self, tcp, now, sample_s=0.1, speed_eps=0.003, still_samples=3):
        self._sample_s = float(sample_s)
        self._eps = float(speed_eps)
        self._need = int(still_samples)
        self._prev = np.asarray(tcp, dtype=float).copy()
        self._prev_t = float(now)
        self.ever_moved = False
        self.still = 0

    def update(self, tcp, now) -> None:
        """Fold one poll in; only whole sample windows change the verdict."""
        dt = float(now) - self._prev_t
        if dt < self._sample_s:
            return
        step_speed = float(np.linalg.norm(np.asarray(tcp, dtype=float) - self._prev)) / dt
        if step_speed > self._eps:
            self.ever_moved = True
            self.still = 0
        else:
            self.still += 1
        self._prev = np.asarray(tcp, dtype=float).copy()
        self._prev_t = float(now)

    def stopped(self) -> bool:
        """Enough consecutive still windows to call the arm stopped."""
        return self.still >= self._need
