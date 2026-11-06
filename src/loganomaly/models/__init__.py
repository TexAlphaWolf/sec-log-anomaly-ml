"""The four detectors, from lightest to heaviest."""

from __future__ import annotations

from .autoencoder import Autoencoder
from .base import Batch, Detector, Scaler
from .iforest import IsolationForest
from .sequence import NextEvent
from .zscore import RobustZ

DETECTORS: dict[str, type[Detector]] = {
    "zscore": RobustZ,
    "iforest": IsolationForest,
    "autoencoder": Autoencoder,
    "sequence": NextEvent,
}

__all__ = [
    "DETECTORS",
    "Autoencoder",
    "Batch",
    "Detector",
    "IsolationForest",
    "NextEvent",
    "RobustZ",
    "Scaler",
]
