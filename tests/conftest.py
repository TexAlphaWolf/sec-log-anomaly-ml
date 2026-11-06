"""Shared fixtures: one small synthetic dataset, generated once per test session."""

from __future__ import annotations

from pathlib import Path

import pytest

from loganomaly.evaluate import load_events
from loganomaly.events import Event
from loganomaly.pipeline import TrainConfig, train
from loganomaly.store import Bundle
from loganomaly.synth import DatasetSpec, generate, write

SPEC = DatasetSpec(days=5, train_days=3, seed=3, scale=0.4)


@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("dataset")
    write(generate(SPEC), SPEC, out)
    return out


@pytest.fixture(scope="session")
def events(dataset_dir: Path) -> list[Event]:
    return load_events(dataset_dir)[0]


@pytest.fixture(scope="session")
def bundle(events: list[Event]) -> Bundle:
    cut = min(e.ts for e in events) + 3 * 86400
    return train([e for e in events if e.ts < cut], TrainConfig(seed=1))
