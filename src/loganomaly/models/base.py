"""What every detector has in common."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

Floats = NDArray[np.float64]
CLIP = 1e4  # only guards against absurd values; the models compress large ones themselves


@dataclass
class Batch:
    """The windows a detector sees: scaled features, and each window's event kinds in order."""

    z: Floats  # (windows, features), robust-scaled
    tokens: list[list[int]]  # per window, kind indices in time order

    def __len__(self) -> int:
        return len(self.z)


class Scaler:
    """Robust scaling: subtract the median, divide by 1.4826 * MAD.

    Most columns are counts that are zero almost everywhere, where MAD is 0. Those fall back
    to the standard deviation, and to 1 for a column that never changes. A value far from
    anything seen in training therefore gets a large score instead of a division by zero."""

    def __init__(self, center: Floats, scale: Floats) -> None:
        self.center = center
        self.scale = scale

    @classmethod
    def fit(cls, x: Floats) -> Scaler:
        center = np.median(x, axis=0)
        mad = np.median(np.abs(x - center), axis=0) * 1.4826
        std = x.std(axis=0)
        scale = np.where(mad > 1e-9, mad, np.where(std > 1e-9, std, 1.0))
        return cls(center, scale)

    def transform(self, x: Floats) -> Floats:
        return np.clip((x - self.center) / self.scale, -CLIP, CLIP)


class Detector(ABC):
    """Scores windows: higher means more unusual."""

    name: str

    @abstractmethod
    def fit(self, batch: Batch) -> None: ...

    @abstractmethod
    def score(self, batch: Batch) -> Floats: ...

    @abstractmethod
    def params(self) -> dict[str, Any]:
        """Settings and small values, saved as JSON."""

    @abstractmethod
    def arrays(self) -> dict[str, NDArray[Any]]:
        """Weights, saved in the model's .npz file (never pickled)."""

    @classmethod
    @abstractmethod
    def from_state(cls, params: dict[str, Any], arrays: dict[str, NDArray[Any]]) -> Detector: ...
