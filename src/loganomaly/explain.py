"""Saying why: the columns that made a window stand out, or the events that surprised the model."""

from __future__ import annotations

import math

import numpy as np

from .events import describe_token
from .features import FEATURES, applies
from .models import Autoencoder, Batch, NextEvent
from .pipeline import Analysis
from .store import Bundle

_REASONS = 4


def _shown(index: int, value: float) -> float:
    """A column value in human terms: counts are stored as log1p."""
    return math.expm1(value) if FEATURES[index].log else value


def _fmt(value: float) -> str:
    return f"{value:.0f}" if abs(value - round(value)) < 0.05 else f"{value:.2f}"


def _label(index: int) -> str:
    name = FEATURES[index].name
    return f"{name[2:]} events" if name.startswith("n_") else name.replace("_", " ")


def _column_reasons(bundle: Bundle, analysis: Analysis, row: int, order: np.ndarray) -> list[str]:
    """The columns, in `order`, whose value falls outside the range normal windows stay in."""
    window = analysis.windows[row]
    ranges = bundle.info.get("ranges")
    found: list[str] = []
    together: list[str] = []
    for index in (int(i) for i in order):
        if not applies(FEATURES[index], window):
            continue
        value = float(analysis.raw[row, index])
        low, high = ranges[index] if ranges else (-math.inf, math.inf)
        shown = _fmt(_shown(index, value))
        if value > high + 1e-9:
            text = f"{_label(index)}: {shown} (normally at most {_fmt(_shown(index, high))})"
        elif value < low - 1e-9:
            text = f"{_label(index)}: {shown} (normally at least {_fmt(_shown(index, low))})"
        else:
            if len(together) < 3:
                together.append(f"{_label(index)} {shown}")
            continue
        found.append(text)
        if len(found) >= _REASONS:
            break
    if found:
        return found
    # Nothing is out of range alone: the combination is what the model did not expect.
    return (
        [f"no single value is out of range; unusual together: {', '.join(together)}"]
        if together
        else []
    )


def reasons(bundle: Bundle, analysis: Analysis, row: int, detector: str) -> list[str]:
    """Plain-language reasons that window `row` scored high for `detector`."""
    model = bundle.detectors.get(detector)
    if isinstance(model, NextEvent):
        found = []
        for context, actual, p_actual, expected, p_expected in model.explain(
            analysis.batch.tokens[row], 2
        ):
            found.append(
                f"{describe_token(actual)} after {describe_token(context[-1])} has probability"
                f" {p_actual:.3f}; expected {describe_token(expected)} ({p_expected:.2f})"
            )
        return found
    if isinstance(model, Autoencoder):
        errors = model.errors(Batch(analysis.batch.z[row : row + 1], []))[0]
        return _column_reasons(bundle, analysis, row, np.argsort(errors)[::-1])
    z = np.abs(analysis.batch.z[row])
    return _column_reasons(bundle, analysis, row, np.argsort(z)[::-1])
