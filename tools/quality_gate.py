"""Fail when the detectors get clearly worse on the demo data.

The floors sit well below what the detectors reach today, so ordinary noise does not trip
them, but a broken feature or a bad refactor does. Run it after any change to features or
models: `python tools/quality_gate.py`.
"""

from __future__ import annotations

import sys

from loganomaly.evaluate import evaluate_spec
from loganomaly.pipeline import TrainConfig
from loganomaly.synth import DatasetSpec

FLOORS = {"autoencoder": {"auc": 0.95, "recall": 0.60}, "vote": {"auc": 0.95, "precision": 0.55}}
MAX_ALERTS_PER_DAY = {"vote": 8.0}


def main() -> int:
    ev = evaluate_spec(DatasetSpec(days=14, train_days=7, seed=1), TrainConfig(seed=1))
    failures = []
    for name, floors in FLOORS.items():
        result = ev.get(name)
        for metric, floor in floors.items():
            value = getattr(result, metric)
            print(f"{name:12s} {metric:10s} {value:6.3f}  (floor {floor})")
            if value < floor:
                failures.append(f"{name} {metric} {value:.3f} < {floor}")
    for name, ceiling in MAX_ALERTS_PER_DAY.items():
        value = ev.get(name).false_alerts_per_day
        print(f"{name:12s} alerts/day {value:6.1f}  (ceiling {ceiling})")
        if value > ceiling:
            failures.append(f"{name} false alerts/day {value:.1f} > {ceiling}")
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
