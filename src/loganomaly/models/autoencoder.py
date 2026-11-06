"""An autoencoder: squeeze each window through a narrow layer and see how well it comes back.

The network is trained on normal windows only, so it learns to rebuild what normal looks like.
A window it cannot rebuild is unlike anything it has seen, and the rebuild error is the score.
Layers are d -> 32 -> 8 -> 32 -> d with tanh in between, trained with Adam on the mean squared
error, stopping when a held-out tenth of the windows stops improving.

Inputs go through arcsinh first: close to the identity for ordinary values, logarithmic for the
extreme ones, so one huge column does not wreck training.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import Batch, Detector, Floats
from .nn import Adam, glorot

_LAYERS = ("1", "2", "3", "4")


class Autoencoder(Detector):
    name = "autoencoder"

    def __init__(
        self,
        hidden: int = 32,
        code: int = 8,
        epochs: int = 120,
        lr: float = 2e-3,
        batch: int = 128,
        patience: int = 10,
        seed: int = 0,
    ) -> None:
        self.hidden, self.code = hidden, code
        self.epochs, self.lr, self.batch = epochs, lr, batch
        self.patience, self.seed = patience, seed
        self.w: dict[str, Floats] = {}
        self.epochs_run = 0

    @staticmethod
    def _input(z: Floats) -> Floats:
        return np.arcsinh(z)

    def _init(self, d: int) -> None:
        rng = np.random.default_rng(self.seed)
        sizes = [(d, self.hidden), (self.hidden, self.code), (self.code, self.hidden),
                 (self.hidden, d)]  # fmt: skip
        self.w = {}
        for name, (fan_in, fan_out) in zip(_LAYERS, sizes, strict=True):
            self.w["W" + name] = glorot(rng, fan_in, fan_out)
            self.w["b" + name] = np.zeros(fan_out)

    def _forward(self, x: Floats) -> tuple[list[Floats], Floats]:
        w = self.w
        a1 = np.tanh(x @ w["W1"] + w["b1"])
        a2 = np.tanh(a1 @ w["W2"] + w["b2"])
        a3 = np.tanh(a2 @ w["W3"] + w["b3"])
        out = a3 @ w["W4"] + w["b4"]
        return [x, a1, a2, a3], out

    def _gradients(self, x: Floats) -> tuple[float, dict[str, Floats]]:
        """Loss and gradients of the mean squared error on one batch."""
        acts, out = self._forward(x)
        x_in, a1, a2, a3 = acts
        n, d = x.shape
        diff = out - x
        loss = float(np.mean(diff**2))
        d_out = 2.0 * diff / (n * d)
        grads: dict[str, Floats] = {"W4": a3.T @ d_out, "b4": d_out.sum(axis=0)}
        d3 = (d_out @ self.w["W4"].T) * (1 - a3**2)
        grads |= {"W3": a2.T @ d3, "b3": d3.sum(axis=0)}
        d2 = (d3 @ self.w["W3"].T) * (1 - a2**2)
        grads |= {"W2": a1.T @ d2, "b2": d2.sum(axis=0)}
        d1 = (d2 @ self.w["W2"].T) * (1 - a1**2)
        grads |= {"W1": x_in.T @ d1, "b1": d1.sum(axis=0)}
        return loss, grads

    def fit(self, batch: Batch) -> None:
        x = self._input(batch.z)
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(x))
        held = max(1, len(x) // 10) if len(x) >= 20 else 0
        valid, train = x[order[:held]], x[order[held:]]
        self._init(x.shape[1])
        adam = Adam(self.w, self.lr)
        best, best_weights, stale = np.inf, {k: v.copy() for k, v in self.w.items()}, 0
        for epoch in range(self.epochs):
            shuffled = train[rng.permutation(len(train))]
            for start in range(0, len(shuffled), self.batch):
                _, grads = self._gradients(shuffled[start : start + self.batch])
                adam.step(grads)
            self.epochs_run = epoch + 1
            current = float(np.mean((self._forward(valid)[1] - valid) ** 2)) if held else 0.0
            if current < best - 1e-6:
                best, best_weights, stale = current, {k: v.copy() for k, v in self.w.items()}, 0
            else:
                stale += 1
                if held and stale >= self.patience:
                    break
        if held:
            self.w = best_weights

    def errors(self, batch: Batch) -> Floats:
        """Squared rebuild error per window and column."""
        x = self._input(batch.z)
        return np.asarray((self._forward(x)[1] - x) ** 2)

    def score(self, batch: Batch) -> Floats:
        return np.asarray(self.errors(batch).mean(axis=1))

    def params(self) -> dict[str, Any]:
        return {"hidden": self.hidden, "code": self.code, "epochs": self.epochs, "lr": self.lr,
                "batch": self.batch, "patience": self.patience, "seed": self.seed,
                "epochs_run": self.epochs_run}  # fmt: skip

    def arrays(self) -> dict[str, NDArray[Any]]:
        return dict(self.w)

    @classmethod
    def from_state(cls, params: dict[str, Any], arrays: dict[str, NDArray[Any]]) -> Autoencoder:
        known = ("hidden", "code", "epochs", "lr", "batch", "patience", "seed")
        model = cls(**{k: params[k] for k in known})
        model.epochs_run = params["epochs_run"]
        model.w = {k: np.asarray(v, dtype=np.float64) for k, v in arrays.items()}
        return model
