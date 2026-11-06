"""Scoring detectors against a dataset whose attacks are known.

Train on the first days, test on the rest. A test window is *positive* when it holds activity of
an attack campaign, *negative* otherwise. Low-and-slow campaigns (one try every few minutes) are
outside what a five-minute window can see by design: their windows are left out of the window
metrics and reported on their own as campaign recall.

Reported per detector:
* ROC AUC and average precision: how well the scores rank attacks above normal windows, with
  no threshold involved
* precision, recall and F1 at the detector's own threshold (the 99.5th percentile of its
  training scores): what you would get in production
* false alerts per day: negative windows flagged, divided by the days tested. This is the
  number a person on call lives with
* campaigns detected: an attack counts as found when any of its windows is flagged
"""

from __future__ import annotations

import io
import json
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .events import Event
from .features import Window
from .pipeline import Analysis, TrainConfig, analyse, train
from .sources import read_all
from .synth import Attack, DatasetSpec, generate, write

Floats = NDArray[np.float64]
DAY = 86400.0
FORMATS = (("auth", "auth.log"), ("journald", "journal.jsonl"), ("web", "access.log"),
           ("windows", "windows-security.jsonl"))  # fmt: skip


def load_events(root: Path) -> tuple[list[Event], dict[str, float]]:
    """Parse the log files of a generated dataset, and read its manifest."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    inputs = [(fmt, str(root / name)) for fmt, name in FORMATS if (root / name).is_file()]
    events, _ = read_all(inputs, year=int(manifest["year"]))
    return events, manifest


def load_attacks(root: Path) -> list[Attack]:
    lines = (root / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    return [Attack(**json.loads(line)) for line in lines if line.strip()]


def label_windows(
    windows: list[Window], attacks: list[Attack], size: int
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], list[int | None]]:
    """For every window: is it an attack, is it a low-and-slow one, and which campaign."""
    by_entity: dict[str, list[Attack]] = defaultdict(list)
    for attack in attacks:
        for entity in attack.entities:
            by_entity[entity].append(attack)
    positive = np.zeros(len(windows), dtype=bool)
    hard = np.zeros(len(windows), dtype=bool)
    owner: list[int | None] = [None] * len(windows)
    for i, window in enumerate(windows):
        for attack in by_entity.get(window.entity, ()):
            if window.start < attack.end and window.start + size > attack.start:
                positive[i] = True
                owner[i] = attack.id
                hard[i] = attack.hard
                break
    return positive, hard, owner


# ------------------------------------------------------------------------------- metrics


def roc_auc(y: NDArray[np.bool_], score: Floats) -> float:
    """Probability that a random attack scores higher than a random normal window."""
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_scores = score[order]
    start = 0
    while start < len(score):  # tied scores share their average rank
        end = start
        while end + 1 < len(score) and sorted_scores[end + 1] == sorted_scores[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2 + 1
        start = end + 1
    return float((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def average_precision(y: NDArray[np.bool_], score: Floats) -> float:
    """Area under the precision-recall curve, as the mean precision at each hit."""
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    hits = y[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision[hits].sum() / positives)


@dataclass
class DetectorResult:
    name: str
    auc: float
    ap: float
    precision: float
    recall: float
    f1: float
    false_alerts_per_day: float
    campaigns_found: int
    campaigns_total: int
    per_type: dict[str, tuple[int, int]]
    train_seconds: float
    score_seconds: float
    model_kb: float


@dataclass
class Evaluation:
    results: list[DetectorResult]
    test_windows: int
    positive_windows: int
    slow_windows: int
    test_days: float
    events: int
    extra: dict[str, float] = field(default_factory=dict)

    def get(self, name: str) -> DetectorResult:
        return next(r for r in self.results if r.name == name)


def _result(
    name: str,
    analysis: Analysis,
    y: NDArray[np.bool_],
    hard: NDArray[np.bool_],
    owner: list[int | None],
    attacks: list[Attack],
    days: float,
    train_s: float,
    score_s: float,
    model_kb: float,
) -> DetectorResult:
    scores, flags = analysis.scores[name], analysis.flags[name]
    use = ~hard
    ys, ss, fs = y[use], scores[use], flags[use]
    tp = int((ys & fs).sum())
    fp = int((~ys & fs).sum())
    fn = int((ys & ~fs).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    found: dict[int, bool] = {a.id: False for a in attacks}
    for i, attack_id in enumerate(owner):
        if attack_id is not None and flags[i]:
            found[attack_id] = True
    per_type: dict[str, tuple[int, int]] = {}
    for attack in attacks:
        done, total = per_type.get(attack.type, (0, 0))
        per_type[attack.type] = (done + found[attack.id], total + 1)
    return DetectorResult(
        name=name,
        auc=roc_auc(ys, ss),
        ap=average_precision(ys, ss),
        precision=precision,
        recall=recall,
        f1=f1,
        false_alerts_per_day=fp / days,
        campaigns_found=sum(found.values()),
        campaigns_total=len(attacks),
        per_type=per_type,
        train_seconds=train_s,
        score_seconds=score_s,
        model_kb=model_kb,
    )


def evaluate(root: str | Path, config: TrainConfig | None = None) -> Evaluation:
    """Train on a dataset directory's training days and score its test days."""
    config = config or TrainConfig()
    path = Path(root)
    events, manifest = load_events(path)
    attacks = [a for a in load_attacks(path) if a.split == "test"]
    cut = manifest["train_end"]
    train_events = [e for e in events if e.ts < cut]
    test_events = [e for e in events if e.ts >= cut]

    bundle = train(train_events, config)
    started = time.perf_counter()
    analysis = analyse(bundle, test_events)
    score_total = time.perf_counter() - started
    # What each detector costs on its own: scoring time and the size of its saved arrays.
    score_time: dict[str, float] = {}
    size_kb: dict[str, float] = {}
    for name, detector in bundle.detectors.items():
        t0 = time.perf_counter()
        detector.score(analysis.batch)
        score_time[name] = time.perf_counter() - t0
        buffer = io.BytesIO()
        arrays: dict[str, Any] = dict(detector.arrays())
        np.savez_compressed(buffer, **arrays)
        size_kb[name] = buffer.getbuffer().nbytes / 1024

    y, hard, owner = label_windows(analysis.windows, attacks, bundle.window)
    days = (manifest["end"] - cut) / DAY
    names = [*config.models, *(["vote"] if "vote" in analysis.flags else [])]
    seconds = bundle.info["train_seconds"]
    results = [
        _result(
            name,
            analysis,
            y,
            hard,
            owner,
            attacks,
            days,
            seconds.get(name, sum(seconds.values())),
            score_time.get(name, 0.0),
            size_kb.get(name, 0.0),
        )
        for name in names
    ]
    return Evaluation(
        results=results,
        test_windows=len(analysis.windows) - int(hard.sum()),
        positive_windows=int((y & ~hard).sum()),
        slow_windows=int(hard.sum()),
        test_days=days,
        events=len(events),
        extra={"featurise_seconds": max(score_total - sum(score_time.values()), 0.0)},
    )


def evaluate_spec(spec: DatasetSpec, config: TrainConfig | None = None) -> Evaluation:
    """Generate a dataset in a temporary directory and evaluate on it."""
    with tempfile.TemporaryDirectory() as scratch:
        write(generate(spec), spec, scratch)
        return evaluate(scratch, config)
