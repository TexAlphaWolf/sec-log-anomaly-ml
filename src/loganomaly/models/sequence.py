"""A next-event model: given the last few events, what comes next?

Inspired by DeepLog (Du et al., 2017). Instead of an LSTM, a feed-forward network reads a fixed
context of the previous four event kinds: each kind has a learned 8-number embedding, the four
embeddings are joined and pass through one tanh layer to a softmax over every token. Trained on
normal windows, it learns the grammar of normal logs: a failed password is usually followed by
another, or by a login; a burst of 404s at machine speed is rare. The score of a window is the
average surprise (negative log probability) of its events.

It sees order and tempo, which the other detectors do not. It does not see how many: a window
of a thousand perfectly ordinary events is a boring sequence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..events import VOCAB
from .base import Batch, Detector, Floats
from .nn import Adam, glorot

BOS = VOCAB - 1  # the token that marks "start of window"


class NextEvent(Detector):
    name = "sequence"

    def __init__(
        self,
        context: int = 4,
        embed: int = 8,
        hidden: int = 48,
        epochs: int = 12,
        lr: float = 3e-3,
        batch: int = 512,
        seed: int = 0,
    ) -> None:
        self.context, self.embed, self.hidden = context, embed, hidden
        self.epochs, self.lr, self.batch, self.seed = epochs, lr, batch, seed
        self.w: dict[str, Floats] = {}

    # ----------------------------------------------------------------- examples

    def _examples(
        self, tokens: list[list[int]]
    ) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]:
        """(contexts, targets, window of each) for every event of every window."""
        k = self.context
        contexts: list[list[int]] = []
        targets: list[int] = []
        owners: list[int] = []
        for owner, seq in enumerate(tokens):
            padded = [BOS] * k + seq
            for i, target in enumerate(seq):
                contexts.append(padded[i : i + k])
                targets.append(target)
                owners.append(owner)
        shape = (-1, k)
        return (
            np.array(contexts, dtype=np.intp).reshape(shape),
            np.array(targets, dtype=np.intp),
            np.array(owners, dtype=np.intp),
        )

    # ------------------------------------------------------------------- network

    def _init(self) -> None:
        rng = np.random.default_rng(self.seed)
        k, e, h = self.context, self.embed, self.hidden
        self.w = {
            "E": rng.normal(0.0, 0.1, size=(VOCAB, e)),
            "W1": glorot(rng, k * e, h),
            "b1": np.zeros(h),
            "W2": glorot(rng, h, VOCAB),
            "b2": np.zeros(VOCAB),
        }

    def _log_probs(self, ctx: NDArray[np.intp]) -> tuple[Floats, Floats, Floats]:
        w = self.w
        emb = w["E"][ctx].reshape(len(ctx), -1)
        hid = np.tanh(emb @ w["W1"] + w["b1"])
        logits = hid @ w["W2"] + w["b2"]
        logits = logits - logits.max(axis=1, keepdims=True)
        log_p = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        return emb, hid, log_p

    def _gradients(
        self, ctx: NDArray[np.intp], target: NDArray[np.intp]
    ) -> tuple[float, dict[str, Floats]]:
        n = len(ctx)
        emb, hid, log_p = self._log_probs(ctx)
        loss = float(-log_p[np.arange(n), target].mean())
        d_logits = np.exp(log_p)
        d_logits[np.arange(n), target] -= 1.0
        d_logits /= n
        d_hid = (d_logits @ self.w["W2"].T) * (1 - hid**2)
        d_emb = (d_hid @ self.w["W1"].T).reshape(n, self.context, self.embed)
        d_table = np.zeros_like(self.w["E"])
        np.add.at(d_table, ctx, d_emb)
        grads = {"E": d_table, "W1": emb.T @ d_hid, "b1": d_hid.sum(axis=0),
                 "W2": hid.T @ d_logits, "b2": d_logits.sum(axis=0)}  # fmt: skip
        return loss, grads

    def fit(self, batch: Batch) -> None:
        ctx, target, _ = self._examples(batch.tokens)
        self._init()
        if len(ctx) == 0:
            return
        rng = np.random.default_rng(self.seed + 1)
        adam = Adam(self.w, self.lr)
        for _ in range(self.epochs):
            order = rng.permutation(len(ctx))
            for start in range(0, len(order), self.batch):
                pick = order[start : start + self.batch]
                _, grads = self._gradients(ctx[pick], target[pick])
                adam.step(grads)

    # ------------------------------------------------------------------- scoring

    def surprises(self, tokens: list[list[int]]) -> tuple[Floats, NDArray[np.intp]]:
        """Surprise of every event, and the window each belongs to."""
        ctx, target, owner = self._examples(tokens)
        if len(ctx) == 0:
            return np.zeros(0), owner
        log_p = self._log_probs(ctx)[2]
        return -log_p[np.arange(len(ctx)), target], owner

    def score(self, batch: Batch) -> Floats:
        surprise, owner = self.surprises(batch.tokens)
        count = np.bincount(owner, minlength=len(batch))
        total = np.bincount(owner, weights=surprise, minlength=len(batch))
        return np.asarray(total / np.maximum(count, 1))

    def explain(
        self, seq: list[int], top: int = 3
    ) -> list[tuple[list[int], int, float, int, float]]:
        """The most surprising events of one window, as (context, actual kind, its probability,
        the kind the model expected instead, that kind's probability)."""
        ctx, target, _ = self._examples([seq])
        if len(ctx) == 0:
            return []
        log_p = self._log_probs(ctx)[2]
        surprise = -log_p[np.arange(len(ctx)), target]
        found = []
        for i in np.argsort(surprise)[::-1][:top]:
            expected = int(np.argmax(log_p[i]))
            found.append(
                (
                    [int(t) for t in ctx[i]],
                    int(target[i]),
                    float(np.exp(log_p[i, target[i]])),
                    expected,
                    float(np.exp(log_p[i, expected])),
                )
            )
        return found

    def params(self) -> dict[str, Any]:
        return {"context": self.context, "embed": self.embed, "hidden": self.hidden,
                "epochs": self.epochs, "lr": self.lr, "batch": self.batch,
                "seed": self.seed}  # fmt: skip

    def arrays(self) -> dict[str, NDArray[Any]]:
        return dict(self.w)

    @classmethod
    def from_state(cls, params: dict[str, Any], arrays: dict[str, NDArray[Any]]) -> NextEvent:
        model = cls(**params)
        model.w = {k: np.asarray(v, dtype=np.float64) for k, v in arrays.items()}
        return model
