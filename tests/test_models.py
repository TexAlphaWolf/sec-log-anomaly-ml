from __future__ import annotations

import numpy as np
import pytest

from loganomaly.models import DETECTORS
from loganomaly.models.autoencoder import Autoencoder
from loganomaly.models.base import Batch
from loganomaly.models.iforest import IsolationForest
from loganomaly.models.nn import Adam, glorot
from loganomaly.models.sequence import NextEvent
from loganomaly.models.zscore import RobustZ


def blob(n: int = 600, d: int = 8, seed: int = 0) -> tuple[Batch, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, d))
    outlier = np.full((5, d), 8.0)
    seqs = [[1, 2, 3, 1, 2, 3] for _ in range(n)]
    return Batch(z, seqs), outlier


@pytest.mark.parametrize("name", ["zscore", "iforest", "autoencoder"])
def test_outliers_score_higher(name: str) -> None:
    batch, outlier = blob()
    model = DETECTORS[name]() if name == "zscore" else DETECTORS[name](seed=1)
    model.fit(batch)
    normal = model.score(batch)
    odd = model.score(Batch(outlier, [[] for _ in outlier]))
    assert odd.min() > np.quantile(normal, 0.95)


@pytest.mark.parametrize("name", ["zscore", "iforest", "autoencoder", "sequence"])
def test_state_roundtrip_gives_identical_scores(name: str) -> None:
    batch, _ = blob(300)
    model = DETECTORS[name]() if name == "zscore" else DETECTORS[name](seed=2)
    if name == "sequence":
        model.epochs = 2  # type: ignore[attr-defined]
    model.fit(batch)
    clone = DETECTORS[name].from_state(model.params(), model.arrays())
    assert np.allclose(model.score(batch), clone.score(batch))


def test_training_is_deterministic() -> None:
    batch, _ = blob(300)
    a, b = Autoencoder(seed=4, epochs=5), Autoencoder(seed=4, epochs=5)
    a.fit(batch)
    b.fit(batch)
    assert np.array_equal(a.score(batch), b.score(batch))
    c = Autoencoder(seed=5, epochs=5)
    c.fit(batch)
    assert not np.array_equal(a.score(batch), c.score(batch))


def test_autoencoder_gradients_match_finite_differences() -> None:
    rng = np.random.default_rng(0)
    model = Autoencoder(hidden=5, code=3, seed=0)
    model._init(6)
    x = rng.normal(size=(7, 6))
    _, grads = model._gradients(x)
    eps = 1e-6
    for name, grad in grads.items():
        flat = model.w[name].reshape(-1)
        for i in rng.choice(flat.size, size=min(4, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + eps
            up = model._gradients(x)[0]
            flat[i] = old - eps
            down = model._gradients(x)[0]
            flat[i] = old
            assert grad.reshape(-1)[i] == pytest.approx((up - down) / (2 * eps), rel=1e-4, abs=1e-8)


def test_autoencoder_learns() -> None:
    batch, _ = blob(400)
    model = Autoencoder(epochs=1, seed=0)
    model.fit(batch)
    early = model.score(batch).mean()
    model = Autoencoder(epochs=60, seed=0)
    model.fit(batch)
    assert model.score(batch).mean() < early


def test_adam_moves_against_the_gradient() -> None:
    w = {"a": np.array([1.0, -1.0])}
    adam = Adam(w, 0.1)
    adam.step({"a": np.array([1.0, -1.0])})
    assert w["a"][0] < 1.0
    assert w["a"][1] > -1.0


def test_glorot_scale() -> None:
    w = glorot(np.random.default_rng(0), 100, 100)
    assert w.shape == (100, 100)
    assert abs(w.std() - np.sqrt(2 / 200)) < 0.01


def test_isolation_forest_matches_sklearn() -> None:
    """Our forest and scikit-learn's should rank the same points as odd."""
    from scipy.stats import spearmanr
    from sklearn.ensemble import IsolationForest as Reference

    rng = np.random.default_rng(1)
    z = np.vstack([rng.normal(size=(500, 5)), rng.normal(loc=5, size=(10, 5))])
    ours = IsolationForest(seed=0)
    ours.fit(Batch(z, []))
    reference = Reference(n_estimators=200, max_samples=256, random_state=0).fit(z)
    rho = spearmanr(ours.score(Batch(z, [])), -reference.score_samples(z)).statistic
    assert rho > 0.9
    assert ours.score(Batch(z, []))[-10:].min() > np.quantile(ours.score(Batch(z, []))[:500], 0.9)


def test_isolation_forest_handles_constant_data() -> None:
    model = IsolationForest(trees=5, sample_size=16)
    model.fit(Batch(np.zeros((50, 3)), []))
    assert np.isfinite(model.score(Batch(np.ones((2, 3)), []))).all()


def test_sequence_model_flags_a_broken_pattern() -> None:
    normal = [[0, 1, 2, 3, 0, 1, 2, 3] for _ in range(200)]
    model = NextEvent(epochs=30, seed=0)
    model.fit(Batch(np.zeros((200, 1)), normal))
    broken = [[3, 3, 3, 0, 2, 2, 1, 3]]
    assert (
        model.score(Batch(np.zeros((1, 1)), broken))[0]
        > 3 * model.score(Batch(np.zeros((1, 1)), normal[:1]))[0]
    )
    top = model.explain(broken[0], 2)
    assert len(top) == 2
    _context, actual, p_actual, expected, p_expected = top[0]
    assert 0 <= p_actual <= p_expected <= 1
    assert actual != expected or p_actual == p_expected


def test_sequence_model_handles_empty_windows() -> None:
    model = NextEvent(epochs=1)
    model.fit(Batch(np.zeros((3, 1)), [[0, 1], [], [1]]))
    assert model.score(Batch(np.zeros((1, 1)), [[]]))[0] == 0.0
    assert model.explain([]) == []


def test_zscore_scores_extremes_higher() -> None:
    batch, outlier = blob()
    model = RobustZ()
    model.fit(batch)
    assert model.score(Batch(outlier, [[]] * 5)).min() > model.score(batch).max()
