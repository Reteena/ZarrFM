from __future__ import annotations

import numpy as np

FORMAT = "zarrfm"
VERSION = "0.1.0"

SIGNALS = "signals"
CHANNELS = "channels"
EVENTS = "events"
SEGMENTS = "segments"
PROVENANCE = "provenance"
EVENT_VOCAB = "vocab"

EVENT_REVISION_ATTR = "events_revision"
EVENT_REVISION_PREFIX = "r"
REVISION_ATTRS = ("n_events", "digest", "vocab")
TOMBSTONE_ATTR = "superseded"
STEP_REVISION_KEY = "events_revision"

ROOT_ATTRS = ("format", "version", "subject", "session", "task", "t0", "created_by",
              "line_freq", "primary_stream", "canonical_version", "events_revision")

STREAM_FILTER_ATTRS = ("highpass", "lowpass")

DIG_MIN, DIG_MAX = -32768, 32767

TARGET_CHUNK_BYTES = 1024**2

MIN_CHUNK_SECONDS = 2.0
MAX_CHUNK_SECONDS = 64.0

MIN_CHUNK_BYTES = 256 * 1024
MAX_CHUNK_BYTES = 4 * 1024**2

MAX_CHUNK_FRACTION = 8

MIN_CHUNK_SAMPLES = 128
MAX_CHUNK_SAMPLES = 1 << 20

TARGET_SHARD_BYTES = 64 * 1024**2

MERGE_BREAK_EVEN_BYTES = 2 * 1024**2

SPAN_BUDGET_BYTES = 64 * 1024**2

STORAGE_DTYPES = {"int16": np.int16, "float32": np.float32}

COMPRESSION_PROFILES = ("fast", "max")
DEFAULT_COMPRESSION = "fast"

CHANNEL_FIELDS = {
    "name": "str",
    "type": "str",
    "unit": "str",
    "gain": "float64",
    "offset": "float64",
    "bad": "bool",
    "canonical_id": "int32",
    "reference": "str",
}
CHANNEL_FIELDS_V02 = ("canonical_id", "reference")
POSITION_FIELD = "position"

CHANNEL_STRING_FIELDS = ("name", "type", "unit", "reference")
CHANNEL_ARRAY_FIELDS = {"gain": "float64", "offset": "float64", "bad": "bool",
                        "canonical_id": "int32"}

EVENT_FIELDS = {
    "onset": "float64",
    "duration": "float64",
    "label": "str",
    "value": "int64",
    "label_id": "int32",
}
EVENT_FIELDS_V02 = ("label_id",)

EVENT_ARRAY_FIELDS = {"onset": "float64", "duration": "float64", "value": "int64",
                      "label_id": "int32"}

SEGMENT_FIELDS = {
    "start_sample": "int64",
    "n_samples": "int64",
    "t0_offset": "float64",
}


def parse_version(v) -> tuple[int, int, int]:
    parts = str(v or "0").split(".")
    out = []
    for i in range(3):
        try:
            out.append(int(parts[i]))
        except (IndexError, ValueError):
            out.append(0)
    return tuple(out)


def has_v02_fields(version) -> bool:
    return parse_version(version) >= (0, 2, 0)


def has_v03_layout(version) -> bool:
    return parse_version(version) >= (0, 3, 0)


def revision_path(k: int) -> str:
    return f"{EVENTS}/{EVENT_REVISION_PREFIX}{int(k)}"


def event_digest(table) -> str:
    import hashlib
    import json as _json

    h = hashlib.sha256()
    parts = [
        np.ascontiguousarray(table["onset"], dtype="<f8").tobytes(),
        np.ascontiguousarray(table["duration"], dtype="<f8").tobytes(),
        np.ascontiguousarray(table["value"], dtype="<i8").tobytes(),
        np.ascontiguousarray(table["label_id"], dtype="<i4").tobytes(),
        _json.dumps([str(v) for v in table["vocab"]], ensure_ascii=False,
                    separators=(",", ":")).encode("utf-8"),
    ]
    for part in parts:
        h.update(len(part).to_bytes(8, "little"))
        h.update(part)
    return "sha256:" + h.hexdigest()


def prev_pow2(n: int) -> int:
    n = int(n)
    if n < 1:
        return 1
    return 1 << (n.bit_length() - 1)


def next_pow2(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def choose_chunk_samples(
    n_channels: int, itemsize: int, n_samples: int, sfreq: float = 256.0
) -> int:
    row = max(n_channels * itemsize, 1)
    sfreq = max(float(sfreq), 1e-9)

    t = prev_pow2(max(TARGET_CHUNK_BYTES // row, 1))
    t = max(next_pow2(MIN_CHUNK_SECONDS * sfreq), t)
    t = min(max(prev_pow2(MAX_CHUNK_SECONDS * sfreq), 1), t)
    while t * row > MAX_CHUNK_BYTES and t > 1:
        t //= 2
    while t * row < MIN_CHUNK_BYTES and t < MAX_CHUNK_SAMPLES:
        t *= 2
    t = max(MIN_CHUNK_SAMPLES, min(MAX_CHUNK_SAMPLES, t))
    if n_samples:
        t = min(t, max(MIN_CHUNK_SAMPLES, prev_pow2(int(n_samples) // MAX_CHUNK_FRACTION)))
        t = max(1, min(t, int(n_samples)))
    return t


def choose_shard_samples(n_channels: int, itemsize: int, chunk_samples: int, n_samples: int) -> int:
    chunk_samples = max(1, int(chunk_samples))
    n_chunks_in_shard = choose_shard_factor(n_channels, itemsize, chunk_samples)
    n_chunks_total = -(-max(int(n_samples), 1) // chunk_samples)
    return chunk_samples * max(1, min(n_chunks_in_shard, n_chunks_total))


def choose_shard_factor(n_channels: int, itemsize: int, chunk_samples: int) -> int:
    chunk_bytes = max(n_channels * itemsize * chunk_samples, 1)
    return max(1, int(TARGET_SHARD_BYTES // chunk_bytes))


def compute_scaling(
    phys_min: np.ndarray, phys_max: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    phys_min = np.asarray(phys_min, dtype=np.float64).ravel()
    phys_max = np.asarray(phys_max, dtype=np.float64).ravel()
    if phys_min.shape != phys_max.shape:
        raise ValueError("phys_min and phys_max must have the same shape")
    if np.any(phys_max < phys_min):
        raise ValueError("phys_max must be >= phys_min for every channel")

    span = phys_max - phys_min
    flat = span <= 0

    gain = np.where(flat, 1.0, span) / (DIG_MAX - DIG_MIN)
    offset = phys_max - gain * DIG_MAX

    if np.any(flat):
        nominal = np.maximum(np.abs(phys_max), 1.0) * 1e-9
        gain = np.where(flat, nominal, gain)
        offset = np.where(flat, phys_max, offset)
    return gain, offset


def quantize(
    data: np.ndarray, gain: np.ndarray, offset: np.ndarray
) -> tuple[np.ndarray, int]:
    g = np.asarray(gain, dtype=np.float64).reshape(-1, 1)
    o = np.asarray(offset, dtype=np.float64).reshape(-1, 1)
    codes = np.rint((np.asarray(data, dtype=np.float64) - o) / g)
    n_clipped = int(np.count_nonzero((codes < DIG_MIN) | (codes > DIG_MAX)))
    np.clip(codes, DIG_MIN, DIG_MAX, out=codes)
    return codes.astype(np.int16), n_clipped


def dequantize(codes: np.ndarray, gain, offset, dtype=np.float32) -> np.ndarray:
    dtype = np.dtype(dtype)
    out = np.asarray(codes).astype(dtype)
    out *= np.asarray(gain, dtype=dtype).reshape(-1, 1)
    out += np.asarray(offset, dtype=dtype).reshape(-1, 1)
    return out


def json_safe(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return str(value)
