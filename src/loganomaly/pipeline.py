"""Training and detection, end to end: events in, scores and flags out."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .events import Event
from .features import DEFAULT_WINDOW, Tracker, Window, featurize, tokens
from .models import (
    DETECTORS,
    Autoencoder,
    Batch,
    Detector,
    IsolationForest,
    NextEvent,
    RobustZ,
    Scaler,
)
from .store import Bundle

Floats = NDArray[np.float64]
ALL_MODELS = ("zscore", "iforest", "autoencoder", "sequence")
LEARNED = ("iforest", "autoencoder", "sequence")  # the detectors that take part in the vote
DAY = 86400.0
MIN_TRAIN_WINDOWS = 50


@dataclass
class TrainConfig:
    window: int = DEFAULT_WINDOW
    quantile: float = 0.995  # the share of training windows that stays under the threshold
    seed: int = 0
    models: tuple[str, ...] = ALL_MODELS
    warmup: float | None = None  # seconds ignored at the start; None picks one automatically


@dataclass
class Analysis:
    """Detection results: one row per window, one column of scores per detector."""

    windows: list[Window]
    raw: Floats
    batch: Batch
    scores: dict[str, Floats]
    thresholds: dict[str, float]
    flags: dict[str, NDArray[np.bool_]] = field(default_factory=dict)
    ratios: dict[str, Floats] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, values in self.scores.items():
            self.ratios[name] = values / self.thresholds[name]
            self.flags[name] = values > self.thresholds[name]
        learned = [n for n in LEARNED if n in self.scores]
        if len(learned) >= 2:
            votes = sum(self.flags[n].astype(int) for n in learned)
            need = 2
            self.flags["vote"] = np.asarray(votes >= need)
            self.ratios["vote"] = np.mean([self.ratios[n] for n in learned], axis=0)
            self.scores["vote"] = self.ratios["vote"]
            self.thresholds["vote"] = 1.0


def _make(name: str, seed: int) -> Detector:
    if name == "iforest":
        return IsolationForest(seed=seed)
    if name == "autoencoder":
        return Autoencoder(seed=seed)
    if name == "sequence":
        return NextEvent(seed=seed)
    return RobustZ()


def train(events: list[Event], config: TrainConfig | None = None) -> Bundle:
    """Fit every requested detector on `events`, which should be (mostly) normal activity."""
    config = config or TrainConfig()
    unknown = set(config.models) - set(DETECTORS)
    if unknown:
        raise ValueError(f"unknown model(s): {', '.join(sorted(unknown))}")
    if not events:
        raise ValueError("no events to train on")
    started = time.perf_counter()
    tracker = Tracker()
    windows, raw = featurize(events, config.window, tracker)
    span = events[-1].ts - events[0].ts
    warmup = config.warmup if config.warmup is not None else min(DAY, 0.1 * span)
    keep = np.array([w.start >= events[0].ts + warmup for w in windows])
    if keep.sum() < MIN_TRAIN_WINDOWS:
        keep[:] = True  # too little data to spare a warm-up period
    if keep.sum() < 10:
        raise ValueError(
            f"only {int(keep.sum())} windows of activity: need more logs to learn from"
        )
    scaler = Scaler.fit(raw[keep])
    train_windows = [w for w, k in zip(windows, keep, strict=True) if k]
    batch = Batch(scaler.transform(raw[keep]), [tokens(w) for w in train_windows])

    detectors = {}
    thresholds = {}
    timings = {}
    for name in config.models:
        t0 = time.perf_counter()
        detector = _make(name, config.seed)
        detector.fit(batch)
        scores = detector.score(batch)
        thresholds[name] = float(np.quantile(scores, config.quantile))
        detectors[name] = detector
        timings[name] = round(time.perf_counter() - t0, 3)
    return Bundle(
        window=config.window,
        quantile=config.quantile,
        scaler=scaler,
        tracker=tracker,
        detectors=detectors,
        thresholds=thresholds,
        info={
            "events": len(events),
            "windows": int(keep.sum()),
            "warmup_seconds": warmup,
            "first_event": events[0].ts,
            "last_event": events[-1].ts,
            "seed": config.seed,
            # The range 99 % of normal windows stay inside, per feature: what "unusual" means.
            "ranges": np.quantile(raw[keep], [0.005, 0.995], axis=0).T.tolist(),
            "train_seconds": timings,
            "total_seconds": round(time.perf_counter() - started, 3),
        },
    )


def analyse(bundle: Bundle, events: list[Event]) -> Analysis:
    """Score `events` with a trained model. The bundle itself is left untouched."""
    windows, raw = featurize(events, bundle.window, bundle.tracker.copy())
    batch = Batch(bundle.scaler.transform(raw), [tokens(w) for w in windows])
    scores = {name: det.score(batch) for name, det in bundle.detectors.items()}
    return Analysis(windows, raw, batch, scores, dict(bundle.thresholds))
