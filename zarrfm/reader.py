from __future__ import annotations

import datetime as _dt
import warnings
from typing import Sequence

import numpy as np
import zarr

import json

from . import _storeio, canonical, spec
from .errors import ZarrFMIntegrityError, StaleMetadataWarning
from .writer import read_str_column


def merge_spans(
    starts: np.ndarray, width: int, tolerance: int, max_span: int
) -> list[tuple[int, int]]:
    spans: list[list[int]] = []
    for s in starts:
        s = int(s)
        if (
            spans
            and s - spans[-1][1] <= tolerance
            and (s + width) - spans[-1][0] <= max_span
        ):
            spans[-1][1] = max(spans[-1][1], s + width)
        else:
            spans.append([s, s + width])
    return [(lo, hi) for lo, hi in spans]


def _open_root(path):
    if isinstance(path, zarr.Group):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return zarr.open_group(store=path.store, path=path.path, mode="r", zarr_format=3)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return zarr.open_consolidated(str(path), mode="r", zarr_format=3)
    except (KeyError, ValueError, TypeError):
        return zarr.open_group(str(path), mode="r", zarr_format=3)


def decode_revision(grp, *, verify: bool = True) -> dict:
    a = dict(grp.attrs)
    vocab_raw = a.get("vocab", [])
    if not isinstance(vocab_raw, list):
        raise ZarrFMIntegrityError(f"{grp.path}: vocab attr is not a list")
    vocab = [str(v) for v in vocab_raw]
    if len(set(vocab)) != len(vocab):
        raise ZarrFMIntegrityError(f"{grp.path}: vocab has duplicate labels")
    try:
        cols = {f: np.asarray(grp[f][:], dtype=d) for f, d in spec.EVENT_ARRAY_FIELDS.items()}
    except Exception as e:
        raise ZarrFMIntegrityError(f"{grp.path}: cannot decode event columns: {type(e).__name__}: {e}") from e
    n = a.get("n_events", -1)
    if isinstance(n, bool) or not isinstance(n, int):
        raise ZarrFMIntegrityError(f"{grp.path}: n_events attr {n!r} is not an integer")
    if verify:
        lengths = {f: int(v.shape[0]) for f, v in cols.items()}
        if any(l != n for l in lengths.values()):
            raise ZarrFMIntegrityError(
                f"{grp.path}: column lengths {lengths} do not match n_events={n}; "
                "torn edit, or a reader that outlived an edit (refresh())"
            )
        if cols["label_id"].size and (
            cols["label_id"].min() < 0 or cols["label_id"].max() >= len(vocab)
        ):
            raise ZarrFMIntegrityError(f"{grp.path}: label_id indexes outside vocab")
        digest = spec.event_digest({**cols, "vocab": vocab})
        if digest != a.get("digest"):
            raise ZarrFMIntegrityError(
                f"{grp.path}: digest mismatch -- the table on disk is not the one that "
                "was committed (torn edit, tampering, or a stale reader; refresh())"
            )
    ids = cols["label_id"]
    label = np.array([vocab[i] if 0 <= i < len(vocab) else "" for i in ids], dtype=object)
    return {**cols, "label": label, "vocab": vocab}


def decode_legacy_events(grp) -> dict:
    onset_arr = grp.get("onset")
    if onset_arr is not None and int(onset_arr.shape[0]) > 1:
        chunk_key = f"{onset_arr.path}/c/0"
        if not _storeio.exists(onset_arr.store, chunk_key):
            raise ZarrFMIntegrityError(
                f"{grp.path}: onset chunk {chunk_key} is missing; the table would read as "
                "zeros (lost or truncated store)"
            )
    label = np.array(read_str_column(grp, "label") or [], dtype=object)
    vocab = read_str_column(grp, spec.EVENT_VOCAB)
    if vocab is not None and grp.get("label_id") is not None:
        label_id = np.asarray(grp["label_id"][:], dtype=np.int32)
        if label_id.size and (label_id.min() < 0 or label_id.max() >= len(vocab)):
            raise ZarrFMIntegrityError(f"{grp.path}: label_id indexes outside vocab")
    else:
        vocab = sorted(set(label.tolist()))
        lookup = {v: i for i, v in enumerate(vocab)}
        label_id = np.array([lookup[l] for l in label], dtype=np.int32)
    out = {
        "onset": np.asarray(grp["onset"][:], dtype=np.float64),
        "duration": np.asarray(grp["duration"][:], dtype=np.float64),
        "label": label,
        "value": np.asarray(grp["value"][:], dtype=np.int64),
        "label_id": label_id,
        "vocab": vocab,
    }
    lengths = {k: len(v) for k, v in out.items() if k != "vocab"}
    if len(set(lengths.values())) > 1:
        raise ZarrFMIntegrityError(f"{grp.path}: event columns have mismatched lengths {lengths}")
    return out


def _empty_events() -> dict:
    out = {k: np.array([], dtype=d) for k, d in spec.EVENT_ARRAY_FIELDS.items()}
    out["label"] = np.array([], dtype=object)
    out["vocab"] = []
    return out


class ZarrFMReader:
    def __init__(self, path):
        self._src = path
        self.root = path if isinstance(path, zarr.Group) else _open_root(path)
        self.path = self.root.path if isinstance(path, zarr.Group) else str(path)
        self.attrs = dict(self.root.attrs)
        if self.attrs.get("format") != spec.FORMAT:
            raise ValueError(
                f"{self.path!r} is not an zarrfm store (format={self.attrs.get('format')!r})"
            )
        self._cache: dict[str, dict] = {}

    def refresh(self) -> None:
        self.root = _open_root(self._src)
        self.attrs = dict(self.root.attrs)
        self._cache.clear()

    @property
    def events_revision(self) -> int | None:
        k = self.attrs.get(spec.EVENT_REVISION_ATTR)
        if k is None:
            return None
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ZarrFMIntegrityError(
                f"root attr {spec.EVENT_REVISION_ATTR}={k!r} is not a non-negative integer"
            )
        return k

    def _store_prefix(self) -> str:
        return f"{self.root.path}/" if self.root.path else ""


    @property
    def version(self) -> tuple[int, int, int]:
        return spec.parse_version(self.attrs.get("version"))

    @property
    def streams(self) -> list[str]:
        return sorted(self.root[spec.SIGNALS].array_keys())

    @property
    def primary_stream(self) -> str:
        p = self.attrs.get("primary_stream")
        return p if p else self.streams[0]

    @property
    def t0(self) -> _dt.datetime | None:
        t = self.attrs.get("t0")
        return _dt.datetime.fromisoformat(t) if t else None

    def _stream(self, stream: str | None) -> dict:
        if stream is None:
            names = self.streams
            if len(names) != 1:
                raise ValueError(f"specify a stream; store has {names}")
            stream = names[0]
        if stream in self._cache:
            return self._cache[stream]
        try:
            arr = self.root[f"{spec.SIGNALS}/{stream}"]
        except KeyError as e:
            raise KeyError(f"unknown stream {stream!r}; have {self.streams}") from e
        ch = self.root[f"{spec.CHANNELS}/{stream}"]
        names = read_str_column(ch, "name")
        types = read_str_column(ch, "type")
        units = read_str_column(ch, "unit")
        if names is None or types is None or units is None:
            raise KeyError(f"channels/{stream} has no name/type/unit table")
        if names and all(n == "" for n in names) and not isinstance(ch.attrs.get("name"), list):
            raise ZarrFMIntegrityError(
                f"channels/{stream}: the channel table changed under this reader (the store "
                "was migrated); call refresh()"
            )
        if ch.get("canonical_id") is not None:
            cid = np.asarray(ch["canonical_id"][:], dtype=np.int32)
            ref = read_str_column(ch, "reference") or [""] * len(names)
        else:
            cid, ref = canonical.map_channels(names, types)
        info = {
            "name": stream,
            "array": arr,
            "attrs": dict(arr.attrs),
            "names": names,
            "types": types,
            "units": units,
            "gain": np.asarray(ch["gain"][:], dtype=np.float64),
            "offset": np.asarray(ch["offset"][:], dtype=np.float64),
            "bad": np.asarray(ch["bad"][:], dtype=bool),
            "position": np.asarray(ch[spec.POSITION_FIELD][:], dtype=np.float64),
            "canonical_id": cid,
            "reference": ref,
        }
        self._cache[stream] = info
        return info

    def info(self, stream: str | None = None) -> dict:
        st = self._stream(stream)
        a = st["attrs"]
        return {
            "stream": st["name"],
            "sfreq": float(a["sfreq"]),
            "n_channels": int(st["array"].shape[0]),
            "n_samples": int(st["array"].shape[1]),
            "duration_s": int(st["array"].shape[1]) / float(a["sfreq"]),
            "unit": a.get("unit", "V"),
            "storage": a.get("storage", "int16"),
            "compression": a.get("compression", "fast"),
            "t0_offset": float(a.get("t0_offset", 0.0)),
            "n_clipped": int(a.get("n_clipped", 0)),
            "chunks": tuple(st["array"].chunks),
            "shards": tuple(st["array"].shards) if st["array"].shards else None,
            "highpass": a.get("highpass"),
            "lowpass": a.get("lowpass"),
            "n_canonical": int(np.unique(st["canonical_id"][st["canonical_id"] >= 0]).size),
            "n_mapped": int(np.count_nonzero(st["canonical_id"] >= 0)),
        }

    def ch_names(self, stream: str | None = None) -> list[str]:
        return list(self._stream(stream)["names"])

    def ch_types(self, stream: str | None = None) -> list[str]:
        return list(self._stream(stream)["types"])

    def bad_channels(self, stream: str | None = None) -> list[str]:
        st = self._stream(stream)
        return [n for n, b in zip(st["names"], st["bad"]) if b]

    def bad_mask(self, stream: str | None = None) -> np.ndarray:
        return self._stream(stream)["bad"].copy()

    def positions(self, stream: str | None = None) -> np.ndarray:
        return self._stream(stream)["position"].copy()

    def canonical_ids(self, stream: str | None = None) -> np.ndarray:
        return self._stream(stream)["canonical_id"].copy()

    def references(self, stream: str | None = None) -> list[str]:
        return list(self._stream(stream)["reference"])

    def sfreq(self, stream: str | None = None) -> float:
        return float(self._stream(stream)["attrs"]["sfreq"])

    def n_samples(self, stream: str | None = None) -> int:
        return int(self._stream(stream)["array"].shape[1])


    def _resolve_picks(self, st: dict, picks) -> np.ndarray:
        n = len(st["names"])
        if picks is None:
            return np.arange(n)
        if isinstance(picks, (str, int, np.integer)):
            picks = [picks]
        index = {name: i for i, name in enumerate(st["names"])}
        out = []
        for p in picks:
            if isinstance(p, (int, np.integer)):
                if not -n <= int(p) < n:
                    raise IndexError(f"channel index {p} out of range for {n} channels")
                out.append(int(p) % n)
            elif p in index:
                out.append(index[p])
            else:
                matched = [i for i, t in enumerate(st["types"]) if t == p]
                if not matched:
                    raise KeyError(f"no channel named or typed {p!r}")
                out.extend(matched)
        return np.asarray(out, dtype=int)

    def _sample_range(self, st: dict, tmin, tmax) -> tuple[int, int]:
        sfreq = float(st["attrs"]["sfreq"])
        n = int(st["array"].shape[1])
        start = 0 if tmin is None else int(np.floor(float(tmin) * sfreq))
        stop = n if tmax is None else int(np.ceil(float(tmax) * sfreq))
        start, stop = max(0, start), min(n, stop)
        return start, max(start, stop)


    def get_data(
        self,
        stream: str | None = None,
        picks: Sequence | str | int | None = None,
        tmin: float | None = None,
        tmax: float | None = None,
        *,
        raw: bool = False,
        dtype=np.float32,
    ) -> np.ndarray:
        st = self._stream(stream)
        idx = self._resolve_picks(st, picks)
        start, stop = self._sample_range(st, tmin, tmax)
        return self._read(st, idx, start, stop, raw=raw, dtype=dtype)

    def read_samples(
        self, stream: str | None, start: int, stop: int, *, dtype=np.float32
    ) -> np.ndarray:
        st = self._stream(stream)
        n = int(st["array"].shape[1])
        start, stop = max(0, int(start)), min(n, int(stop))
        return self._read(st, None, start, max(start, stop), raw=False, dtype=dtype)

    def _read(self, st, idx, start, stop, *, raw, dtype):
        block = st["array"][:, start:stop]
        if idx is not None:
            block = block[idx]
        if raw:
            return np.asarray(block)
        if st["attrs"].get("storage") == "float32":
            return np.asarray(block).astype(dtype, copy=False)
        gain, offset = st["gain"], st["offset"]
        if idx is not None:
            gain, offset = gain[idx], offset[idx]
        return spec.dequantize(block, gain, offset, dtype=dtype)

    def get_canonical(
        self,
        stream: str | None = None,
        tmin: float | None = None,
        tmax: float | None = None,
        *,
        n_canonical: int | None = None,
        dtype=np.float32,
    ) -> tuple[np.ndarray, np.ndarray]:
        st = self._stream(stream)
        start, stop = self._sample_range(st, tmin, tmax)
        x = self._read(st, None, start, stop, raw=False, dtype=dtype)
        out, present, source = canonical.project(
            x, st["canonical_id"], n_canonical, bad=st["bad"]
        )
        good = present & ~np.where(source >= 0, st["bad"][np.maximum(source, 0)], True)
        return out, good

    def times(
        self, stream: str | None = None, tmin: float | None = None, tmax: float | None = None
    ) -> np.ndarray:
        st = self._stream(stream)
        sfreq = float(st["attrs"]["sfreq"])
        start, stop = self._sample_range(st, tmin, tmax)
        return np.arange(start, stop, dtype=np.float64) / sfreq

    def epochs(
        self,
        onsets: Sequence[float],
        stream: str | None = None,
        tmin: float = -0.2,
        tmax: float = 0.8,
        picks=None,
        dtype=np.float32,
        span_budget_bytes: int | None = None,
    ) -> np.ndarray:
        st = self._stream(stream)
        sfreq = float(st["attrs"]["sfreq"])
        n = int(st["array"].shape[1])
        idx = self._resolve_picks(st, picks)
        width = int(round((tmax - tmin) * sfreq))
        if width <= 0:
            raise ValueError("tmax must be greater than tmin")

        starts = np.round((np.asarray(onsets, dtype=np.float64) + tmin) * sfreq).astype(int)
        keep = (starts >= 0) & (starts + width <= n)
        self.epoch_mask = keep

        kept = starts[keep]
        out = np.empty((kept.size, idx.size, width), dtype=np.dtype(dtype))
        if kept.size == 0:
            return out

        row_bytes = max(int(st["array"].shape[0]) * st["array"].dtype.itemsize, 1)
        tolerance = max(
            int(st["array"].chunks[1]), int(spec.MERGE_BREAK_EVEN_BYTES // row_bytes)
        )
        budget = spec.SPAN_BUDGET_BYTES if span_budget_bytes is None else int(span_budget_bytes)
        max_span = max(width, int(budget // row_bytes))
        order = np.argsort(kept, kind="stable")
        spans = merge_spans(kept[order], width, tolerance, max_span)

        gain, offset = st["gain"][idx], st["offset"][idx]
        scale = st["attrs"].get("storage") != "float32"
        cursor = 0
        for lo, hi in spans:
            block = np.asarray(st["array"][:, lo:hi])[idx]
            while cursor < kept.size:
                s = int(kept[order[cursor]])
                if s < lo or s + width > hi:
                    break
                win = block[:, s - lo : s - lo + width]
                out[order[cursor]] = (
                    spec.dequantize(win, gain, offset, dtype=dtype)
                    if scale
                    else win.astype(dtype, copy=False)
                )
                cursor += 1
        return out


    def events(self) -> dict:
        k = self.events_revision
        if k is not None:
            grp = self.root.get(spec.revision_path(k))
            if grp is None:
                raise ZarrFMIntegrityError(
                    f"{spec.EVENT_REVISION_ATTR}={k} names no group in this reader's "
                    "snapshot; the store was edited under it -- call refresh()"
                )
            return decode_revision(grp, verify=True)

        grp = self.root.get(spec.EVENTS)
        if grp is None:
            return _empty_events()
        key = f"{self._store_prefix()}{spec.EVENTS}/onset/zarr.json"
        raw = _storeio.read_key(self.root.store, key)
        if not raw:
            raise ZarrFMIntegrityError(
                "events layout changed under this reader (the store was migrated or "
                "edited); call refresh()"
            )
        actual = int(json.loads(raw).get("shape", [0])[0])
        if actual != int(grp["onset"].shape[0]):
            warnings.warn(
                f"consolidated metadata is stale for {key} (snapshot {grp['onset'].shape[0]}"
                f", on disk {actual}); reading the event table authoritatively -- "
                "run zarrfm.repair() to reconsolidate",
                StaleMetadataWarning,
                stacklevel=2,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                grp = zarr.open_group(
                    store=self.root.store, path=f"{self._store_prefix()}{spec.EVENTS}",
                    mode="r", use_consolidated=False, zarr_format=3,
                )
        return decode_legacy_events(grp)

    def event_matrix(self, stream: str | None = None) -> np.ndarray:
        st = self._stream(stream if stream is not None else self.primary_stream)
        sfreq = float(st["attrs"]["sfreq"])
        t0_offset = float(st["attrs"].get("t0_offset", 0.0))
        ev = self.events()
        sample = np.round((ev["onset"] - t0_offset) * sfreq).astype(np.int64)
        dur = np.round(ev["duration"] * sfreq).astype(np.int64)
        return np.stack([sample, dur, ev["label_id"].astype(np.int64)], axis=1)

    def segments(self) -> dict | None:
        grp = self.root.get(spec.SEGMENTS)
        if grp is None:
            return None
        return {k: np.asarray(grp[k][:]) for k in spec.SEGMENT_FIELDS}

    @property
    def is_continuous(self) -> bool:
        return self.root.get(spec.SEGMENTS) is None


    def provenance(self) -> dict:
        key = f"{self._store_prefix()}{spec.PROVENANCE}/zarr.json"
        try:
            raw = _storeio.read_key(self.root.store, key)
            if raw:
                return dict(json.loads(raw).get("attributes", {}))
        except (ValueError, OSError):
            pass

        grp = self.root.get(spec.PROVENANCE)
        return {} if grp is None else dict(grp.attrs)


    def nbytes_stored(self) -> int:
        prefix = self.root.path
        return _storeio.store_bytes(self.root.store, f"{prefix}/" if prefix else "")

    def content_hash(self) -> str:
        prefix = self.root.path
        return _storeio.content_hash(self.root.store, f"{prefix}/" if prefix else "")

    def __repr__(self) -> str:
        bits = ", ".join(
            f"{s}({self.info(s)['n_channels']}ch @ {self.info(s)['sfreq']:g}Hz, "
            f"{self.info(s)['duration_s']:.1f}s)"
            for s in self.streams
        )
        subj = self.attrs.get("subject") or "?"
        return f"<ZarrFMReader subject={subj} {bits}>"
