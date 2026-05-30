"""Lightweight pose post-processing for monocular VO trajectories."""

from __future__ import annotations

import numpy as np


class PoseSmoother:
    """Reject outlier motion steps and smooth the integrated trajectory."""

    def __init__(
        self,
        *,
        alpha: float = 0.35,
        max_step: float = 0.35,
        enabled: bool = True,
    ):
        self.alpha = float(np.clip(alpha, 0.05, 1.0))
        self.max_step = max_step
        self.enabled = enabled
        self._position: np.ndarray | None = None
        self._last_raw: np.ndarray | None = None

    def reset(self) -> None:
        self._position = None
        self._last_raw = None

    def update(self, raw_position: np.ndarray) -> np.ndarray:
        raw = raw_position.reshape(3, 1).astype(np.float64)

        if self._position is None:
            self._position = raw.copy()
            self._last_raw = raw.copy()
            return self._position.copy()

        # VO returns cumulative position; gate on per-frame increment, not total offset.
        increment = raw - self._last_raw
        step = float(np.linalg.norm(increment))

        if self.enabled and step > self.max_step:
            starting_from_origin = float(np.linalg.norm(self._last_raw)) < 1e-9
            if not (starting_from_origin and float(np.linalg.norm(raw)) > 1e-9):
                return self._position.copy()

        self._last_raw = raw.copy()

        if not self.enabled:
            self._position = raw.copy()
            return self._position.copy()

        self._position = self.alpha * raw + (1.0 - self.alpha) * self._position
        return self._position.copy()
