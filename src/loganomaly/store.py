"""Saving and loading a trained model: a JSON file and a NumPy archive, never a pickle.

A pickle runs code when it is loaded, so a model downloaded from somewhere would be a way to
run something on your machine. `weights.npz` is loaded with `allow_pickle=False`, and
`model.json` is plain data. The JSON records the archive's SHA-256 and every load checks it, which
catches a truncated copy or a file swapped for another; it is not a signature, so keep models
you did not train out of the way all the same.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .events import KINDS
from .features import FEATURE_NAMES, Tracker
from .models import DETECTORS, Detector, Scaler

SCHEMA = 1


class ModelError(Exception):
    """The model directory is missing, damaged or from another version."""


@dataclass
class Bundle:
    """Everything detection needs: scaling, history, detectors and their thresholds."""

    window: int
    quantile: float
    scaler: Scaler
    tracker: Tracker
    detectors: dict[str, Detector]
    thresholds: dict[str, float]
    info: dict[str, Any] = field(default_factory=dict)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(bundle: Bundle, directory: str | Path) -> None:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "scaler/center": bundle.scaler.center,
        "scaler/scale": bundle.scaler.scale,
    }
    detectors: dict[str, Any] = {}
    for name, detector in bundle.detectors.items():
        detectors[name] = detector.params()
        for key, value in detector.arrays().items():
            arrays[f"{name}/{key}"] = value
    archive = out / "weights.npz"
    np.savez_compressed(archive, **arrays)
    document = {
        "schema": SCHEMA,
        "version": __version__,
        "window": bundle.window,
        "quantile": bundle.quantile,
        "features": list(FEATURE_NAMES),
        "kinds": list(KINDS),
        "thresholds": bundle.thresholds,
        "detectors": detectors,
        "tracker": bundle.tracker.to_json(),
        "info": bundle.info,
        "weights_sha256": _digest(archive),
    }
    (out / "model.json").write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")


def load(directory: str | Path) -> Bundle:
    root = Path(directory)
    meta, archive = root / "model.json", root / "weights.npz"
    if not meta.is_file() or not archive.is_file():
        raise ModelError(f"{root} is not a model directory (model.json and weights.npz expected)")
    try:
        document = json.loads(meta.read_text(encoding="utf-8"))
    except ValueError as error:
        raise ModelError(f"{meta} is not valid JSON: {error}") from error
    if document.get("schema") != SCHEMA:
        raise ModelError(f"model schema {document.get('schema')} is not supported (need {SCHEMA})")
    if document.get("features") != list(FEATURE_NAMES) or document.get("kinds") != list(KINDS):
        raise ModelError("the model was trained with other features: train it again")
    if _digest(archive) != document.get("weights_sha256"):
        raise ModelError("weights.npz does not match model.json (damaged or replaced)")
    try:
        with np.load(archive, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        detectors: dict[str, Detector] = {}
        for name, params in document["detectors"].items():
            prefix = f"{name}/"
            own = {k[len(prefix) :]: v for k, v in arrays.items() if k.startswith(prefix)}
            detectors[name] = DETECTORS[name].from_state(params, own)
        return Bundle(
            window=int(document["window"]),
            quantile=float(document["quantile"]),
            scaler=Scaler(arrays["scaler/center"], arrays["scaler/scale"]),
            tracker=Tracker.from_json(document["tracker"]),
            detectors=detectors,
            thresholds={k: float(v) for k, v in document["thresholds"].items()},
            info=document.get("info", {}),
        )
    except (KeyError, ValueError, TypeError) as error:
        raise ModelError(f"model files are inconsistent: {error!r}") from error
