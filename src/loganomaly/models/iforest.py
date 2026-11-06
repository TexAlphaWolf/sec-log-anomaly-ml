"""Isolation Forest in plain NumPy (Liu, Ting and Zhou, 2008).

Anomalies are few and different, so random splits isolate them in fewer steps. A tree is
grown on a small random sample by picking a random column and a random cut until each point
sits alone; the score is 2^(-average depth / c(n)), near 1 for points isolated quickly.

Written out here, rather than imported, for two reasons: the detector then needs nothing but
NumPy to run on a small server, and a model can be saved as plain arrays instead of a pickle,
which would run code from the file when loaded. `tests/test_iforest.py` checks it against
scikit-learn's implementation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import Batch, Detector, Floats

_EULER = 0.5772156649015329


def _c(n: Floats) -> Floats:
    """Average path length of an unsuccessful search in a binary search tree of n points."""
    n = np.asarray(n, dtype=np.float64)
    out = np.zeros_like(n)
    big = n > 2
    out[big] = 2.0 * (np.log(n[big] - 1.0) + _EULER) - 2.0 * (n[big] - 1.0) / n[big]
    out[n == 2] = 1.0
    return out


class IsolationForest(Detector):
    name = "iforest"

    def __init__(self, trees: int = 200, sample_size: int = 256, seed: int = 0) -> None:
        self.trees = trees
        self.sample_size = sample_size
        self.seed = seed
        self.psi = sample_size
        self.feature = np.zeros((0, 0), dtype=np.int16)
        self.threshold = np.zeros((0, 0))
        self.left = np.zeros((0, 0), dtype=np.int16)
        self.right = np.zeros((0, 0), dtype=np.int16)
        self.size = np.zeros((0, 0), dtype=np.int32)

    # ------------------------------------------------------------------ training

    def _grow(self, x: Floats, rng: np.random.Generator, max_depth: int) -> list[list[float]]:
        """One tree as rows of [feature, threshold, left, right, size]; feature -1 is a leaf."""
        nodes: list[list[float]] = []

        def build(index: NDArray[np.intp], depth: int) -> int:
            node_id = len(nodes)
            nodes.append([-1, 0.0, -1, -1, len(index)])
            if depth >= max_depth or len(index) <= 1:
                return node_id
            sub = x[index]
            low, high = sub.min(axis=0), sub.max(axis=0)
            usable = np.flatnonzero(high > low)
            if len(usable) == 0:
                return node_id
            column = int(rng.choice(usable))
            cut = float(rng.uniform(low[column], high[column]))
            go_left = sub[:, column] < cut
            if go_left.all() or not go_left.any():
                return node_id
            nodes[node_id][0] = column
            nodes[node_id][1] = cut
            nodes[node_id][2] = build(index[go_left], depth + 1)
            nodes[node_id][3] = build(index[~go_left], depth + 1)
            return node_id

        build(np.arange(len(x)), 0)
        return nodes

    def fit(self, batch: Batch) -> None:
        x = batch.z
        rng = np.random.default_rng(self.seed)
        self.psi = min(self.sample_size, len(x))
        max_depth = math.ceil(math.log2(max(self.psi, 2)))
        grown = [
            self._grow(x[rng.choice(len(x), self.psi, replace=False)], rng, max_depth)
            for _ in range(self.trees)
        ]
        width = max(len(tree) for tree in grown)
        shape = (self.trees, width)
        self.feature = np.full(shape, -1, dtype=np.int16)
        self.threshold = np.zeros(shape)
        self.left = np.zeros(shape, dtype=np.int16)
        self.right = np.zeros(shape, dtype=np.int16)
        self.size = np.zeros(shape, dtype=np.int32)
        for t, tree in enumerate(grown):
            for i, (feat, cut, left, right, size) in enumerate(tree):
                self.feature[t, i] = feat
                self.threshold[t, i] = cut
                self.left[t, i] = left
                self.right[t, i] = right
                self.size[t, i] = size

    # ------------------------------------------------------------------- scoring

    def score(self, batch: Batch) -> Floats:
        x = batch.z
        n = len(x)
        rows = np.arange(n)
        total = np.zeros(n)
        for t in range(len(self.feature)):
            feature, threshold = self.feature[t], self.threshold[t]
            left, right = self.left[t], self.right[t]
            node = np.zeros(n, dtype=np.intp)
            depth = np.zeros(n)
            while True:
                column = feature[node]
                leaf = column < 0
                if leaf.all():
                    break
                value = x[rows, np.where(leaf, 0, column)]
                step = np.where(value < threshold[node], left[node], right[node])
                node = np.where(leaf, node, step)
                depth += ~leaf
            total += depth + _c(self.size[t][node])
        mean_depth = total / len(self.feature)
        return np.asarray(2.0 ** (-mean_depth / float(_c(np.array([self.psi]))[0] or 1.0)))

    # --------------------------------------------------------------- persistence

    def params(self) -> dict[str, Any]:
        return {"trees": self.trees, "sample_size": self.sample_size, "seed": self.seed,
                "psi": self.psi}  # fmt: skip

    def arrays(self) -> dict[str, NDArray[Any]]:
        return {"feature": self.feature, "threshold": self.threshold, "left": self.left,
                "right": self.right, "size": self.size}  # fmt: skip

    @classmethod
    def from_state(cls, params: dict[str, Any], arrays: dict[str, NDArray[Any]]) -> IsolationForest:
        forest = cls(params["trees"], params["sample_size"], params["seed"])
        forest.psi = params["psi"]
        forest.feature = arrays["feature"]
        forest.threshold = arrays["threshold"]
        forest.left = arrays["left"]
        forest.right = arrays["right"]
        forest.size = arrays["size"]
        return forest
