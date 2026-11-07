from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from loganomaly.evaluate import (
    average_precision,
    evaluate_spec,
    label_windows,
    load_attacks,
    roc_auc,
)
from loganomaly.events import Event
from loganomaly.features import make_windows
from loganomaly.pipeline import TrainConfig, analyse, train
from loganomaly.store import Bundle, ModelError, load, save
from loganomaly.synth import DatasetSpec, generate

from .conftest import SPEC


def test_synth_is_deterministic(tmp_path: Path) -> None:
    spec = DatasetSpec(days=2, train_days=1, seed=9, scale=0.2)
    a, b = generate(spec), generate(spec)
    assert a.auth == b.auth
    assert a.web == b.web
    assert [x.type for x in a.attacks] == [x.type for x in b.attacks]
    other = generate(DatasetSpec(days=2, train_days=1, seed=10, scale=0.2))
    assert other.web != a.web


def test_dataset_files_and_labels(dataset_dir: Path) -> None:
    names = {p.name for p in dataset_dir.iterdir()}
    assert {
        "auth.log",
        "journal.jsonl",
        "access.log",
        "windows-security.jsonl",
        "labels.jsonl",
        "manifest.json",
    } <= names
    attacks = load_attacks(dataset_dir)
    assert attacks
    assert {a.split for a in attacks} == {"train", "test"} or {a.split for a in attacks} == {"test"}
    assert all(a.end >= a.start for a in attacks)
    assert len({a.type for a in attacks}) >= 6


def test_dirty_training_hides_attacks_in_train_days() -> None:
    clean = generate(DatasetSpec(days=4, train_days=2, seed=2, scale=0.2))
    dirty = generate(DatasetSpec(days=4, train_days=2, seed=2, scale=0.2, dirty=0.5))
    assert sum(a.split == "train" for a in dirty.attacks) > sum(
        a.split == "train" for a in clean.attacks
    )


def test_train_and_analyse(bundle: Bundle, events: list[Event]) -> None:
    assert set(bundle.detectors) == {"zscore", "iforest", "autoencoder", "sequence"}
    assert all(t > 0 for t in bundle.thresholds.values())
    assert "ranges" in bundle.info
    cut = min(e.ts for e in events) + 3 * 86400
    analysis = analyse(bundle, [e for e in events if e.ts >= cut])
    assert "vote" in analysis.flags
    n = len(analysis.windows)
    assert all(len(v) == n for v in analysis.scores.values())
    assert analysis.flags["vote"].sum() < n * 0.2


def test_analyse_does_not_change_the_model(bundle: Bundle, events: list[Event]) -> None:
    before = json.dumps(bundle.tracker.to_json(), sort_keys=True)
    analyse(bundle, events[:500])
    assert json.dumps(bundle.tracker.to_json(), sort_keys=True) == before


def test_detects_the_attacks(bundle: Bundle, events: list[Event], dataset_dir: Path) -> None:
    cut = min(e.ts for e in events) + 3 * 86400
    test = [e for e in events if e.ts >= cut]
    analysis = analyse(bundle, test)
    attacks = [a for a in load_attacks(dataset_dir) if a.split == "test"]
    y, hard, _ = label_windows(analysis.windows, attacks, bundle.window)
    use = ~hard
    assert roc_auc(y[use], analysis.scores["autoencoder"][use]) > 0.85


def test_too_little_training_data() -> None:
    with pytest.raises(ValueError, match="no events"):
        train([])
    with pytest.raises(ValueError, match="unknown model"):
        train([Event(1.0, "web", "web_ok", "h", "1.2.3.4", path="/")], TrainConfig(models=("x",)))


def test_save_load_roundtrip(bundle: Bundle, events: list[Event], tmp_path: Path) -> None:
    save(bundle, tmp_path / "m")
    loaded = load(tmp_path / "m")
    sample = events[:3000]
    a, b = analyse(bundle, sample), analyse(loaded, sample)
    for name in a.scores:
        assert np.allclose(a.scores[name], b.scores[name])
    assert loaded.thresholds == pytest.approx(bundle.thresholds)


def test_load_rejects_missing_tampered_and_foreign_models(bundle: Bundle, tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="not a model directory"):
        load(tmp_path)
    good = tmp_path / "good"
    save(bundle, good)

    tampered = tmp_path / "tampered"
    shutil.copytree(good, tampered)
    with (tampered / "weights.npz").open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(ModelError, match="does not match"):
        load(tampered)

    broken = tmp_path / "broken"
    shutil.copytree(good, broken)
    (broken / "model.json").write_text("{not json")
    with pytest.raises(ModelError, match="not valid JSON"):
        load(broken)

    for key, value, message in (("schema", 99, "schema"), ("features", ["a"], "other features")):
        other = tmp_path / key
        shutil.copytree(good, other)
        doc = json.loads((other / "model.json").read_text())
        doc[key] = value
        (other / "model.json").write_text(json.dumps(doc))
        with pytest.raises(ModelError, match=message):
            load(other)


def test_weights_are_never_pickled(bundle: Bundle, tmp_path: Path) -> None:
    save(bundle, tmp_path / "m")
    with np.load(tmp_path / "m" / "weights.npz", allow_pickle=False) as data:
        assert all(data[k].dtype != object for k in data.files)


def test_metrics_on_known_cases() -> None:
    y = np.array([0, 0, 1, 1], dtype=bool)
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert average_precision(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    # without any attack the metrics are undefined, and say so instead of inventing a number
    none = np.zeros(3, dtype=bool)
    assert np.isnan(average_precision(none, np.array([1.0, 2.0, 3.0])))
    assert np.isnan(roc_auc(none, np.array([1.0, 2.0, 3.0])))
    assert np.isnan(roc_auc(~none, np.array([1.0, 2.0, 3.0])))


def test_label_windows_matches_entity_and_time(events: list[Event], dataset_dir: Path) -> None:
    attacks = load_attacks(dataset_dir)
    windows = make_windows(events, 300)
    y, hard, owner = label_windows(windows, attacks, 300)
    assert y.sum() > 0
    assert (hard <= y).all()
    assert all((o is not None) == bool(flag) for o, flag in zip(owner, y, strict=True))


def test_evaluate_spec_end_to_end() -> None:
    ev = evaluate_spec(SPEC, TrainConfig(seed=1))
    assert {r.name for r in ev.results} >= {"zscore", "iforest", "autoencoder", "sequence", "vote"}
    for r in ev.results:
        assert 0 <= r.auc <= 1
        assert 0 <= r.precision <= 1
        assert r.campaigns_found <= r.campaigns_total
        assert r.model_kb >= 0
    assert ev.get("vote").name == "vote"
    assert ev.test_days > 0


def test_vote_ratio_is_above_one_exactly_when_flagged(bundle: Bundle, events: list[Event]) -> None:
    analysis = analyse(bundle, events[:20000])
    flagged = analysis.flags["vote"]
    assert flagged.any()
    assert (analysis.ratios["vote"][flagged] > 1).all()
    assert (analysis.ratios["vote"][~flagged] <= 1).all()


def test_python_dash_m_entry_point() -> None:
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "loganomaly", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "loganomaly" in out.stdout
