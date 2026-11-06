"""Convert canonical sampler batches into one-second EEG foundation-model patches.

Filtering, referencing and resampling belong at corpus conversion time. This
adapter preserves the requested electrode order and refuses missing/bad channels;
CBraMod's patch corruption mask is not a channel-presence attention mask.
"""
from __future__ import annotations

import numpy as np

from . import canonical


def channel_ids(names: list[str]) -> np.ndarray:
    """Resolve an explicit, unique list of canonical EEG electrode names."""
    table = canonical.canonical_names()
    types = canonical.canonical_types()
    if not names or len(set(names)) != len(names):
        raise ValueError("provide a nonempty list of unique channel names")
    lookup = {name: i for i, name in enumerate(table) if types[i] == "eeg"}
    try:
        return np.asarray([lookup[name] for name in names], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"unknown canonical EEG channel: {exc.args[0]}") from exc


def model_patches(batch: dict, names: list[str], *, sfreq: float, unit: str = "V"):
    """Return float32 (batch, channels, seconds, 200), in microvolts / 100.

    Accepts NumPy batches from WindowSampler(canonical_space=True). For LaBraM,
    callers must additionally supply its own channel IDs (including CLS), which
    are a different vocabulary from ZarrFM's canonical IDs.
    """
    if not np.isclose(sfreq, 200.0, rtol=0, atol=1e-8):
        raise ValueError("resample recordings to 200 Hz before sampling")
    if unit not in ("V", "uV", "µV"):
        raise ValueError("unit must be V, uV or µV")
    x = np.asarray(batch["signals"])
    mask = np.asarray(batch["mask"], dtype=bool)
    if x.ndim != 3 or x.shape[1] != canonical.n_canonical():
        raise ValueError("expected canonical signals with shape (B, canonical_channels, T)")
    if mask.shape != x.shape[:2]:
        raise ValueError("mask must have shape (B, canonical_channels)")
    if x.shape[0] == 0 or x.shape[-1] == 0 or x.shape[-1] % 200:
        raise ValueError("windows must contain a positive integer number of seconds")
    ids = channel_ids(names)
    if not mask[:, ids].all():
        raise ValueError("requested channels include missing or bad electrodes")
    selected = x[:, ids, :]
    if not np.isfinite(selected).all():
        raise ValueError("signals contain non-finite values")
    scale = 1e4 if unit == "V" else 0.01
    selected = np.ascontiguousarray(selected, dtype=np.float32) * np.float32(scale)
    return selected.reshape(x.shape[0], len(ids), x.shape[-1] // 200, 200)
