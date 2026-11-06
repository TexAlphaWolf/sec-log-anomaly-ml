"""The baseline: how far is the most unusual column from normal?

No learning beyond the scaler. Each column is already expressed in robust standard deviations;
the score is the mean of the log of the three largest. Included to answer an honest question:
how much of the work do the models do that this one line of arithmetic does not?
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import Batch, Detector, Floats

TOP = 3


class RobustZ(Detector):
    name = "zscore"

    def fit(self, batch: Batch) -> None:
        """Nothing to learn: the scaler already holds the medians and spreads."""

    def score(self, batch: Batch) -> Floats:
        k = min(TOP, batch.z.shape[1])
        top = np.log1p(np.sort(np.abs(batch.z), axis=1)[:, -k:])
        return np.asarray(top.mean(axis=1))

    def params(self) -> dict[str, Any]:
        return {}

    def arrays(self) -> dict[str, NDArray[Any]]:
        return {}

    @classmethod
    def from_state(cls, params: dict[str, Any], arrays: dict[str, NDArray[Any]]) -> RobustZ:
        return cls()
