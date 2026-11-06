"""The little bit of neural-network machinery the two neural detectors share."""

from __future__ import annotations

import numpy as np

from .base import Floats


class Adam:
    """Adam optimiser (Kingma and Ba, 2015) over a dict of named arrays, updated in place."""

    def __init__(
        self, params: dict[str, Floats], lr: float, b1: float = 0.9, b2: float = 0.999
    ) -> None:
        self.params = params
        self.lr, self.b1, self.b2 = lr, b1, b2
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict[str, Floats]) -> None:
        self.t += 1
        for key, grad in grads.items():
            self.m[key] = self.b1 * self.m[key] + (1 - self.b1) * grad
            self.v[key] = self.b2 * self.v[key] + (1 - self.b2) * grad * grad
            m_hat = self.m[key] / (1 - self.b1**self.t)
            v_hat = self.v[key] / (1 - self.b2**self.t)
            self.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)


def glorot(rng: np.random.Generator, fan_in: int, fan_out: int) -> Floats:
    """Weights drawn so signals keep their size from layer to layer."""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))
