from __future__ import annotations

import datetime as _dt
import warnings
from typing import Iterable, Sequence

import numpy as np
import zarr
from zarr.codecs import BloscCodec

from . import _storeio, canonical, spec


SHUFFLES = ("shuffle", "bitshuffle")


def choose_shuffle(codes: np.ndarray, compression: str = spec.DEFAULT_COMPRESSION) -> str:
    from numcodecs import Blosc

    buf = np.ascontiguousarray(codes)
    if compression == "max" and buf.dtype == np.int16:
        flat = buf.ravel().astype(np.int32)
        buf = np.diff(flat, prepend=flat[:1]).astype(np.int16)
    sizes = {s: len(Blosc(cname="zstd", clevel=5, shuffle=getattr(Blosc, s.upper())).encode(buf))
             for s in SHUFFLES}
    return min(SHUFFLES, key=lambda s: (sizes[s], SHUFFLES.index(s)))


def _codecs(compression: str, shuffle: str = "shuffle"):
    if compression not in spec.COMPRESSION_PROFILES:
        raise ValueError(
            f"compression must be one of {spec.COMPRESSION_PROFILES}, got {compression!r}"
        )
    if shuffle not in SHUFFLES:
        raise ValueError(f"shuffle must be one of {SHUFFLES}, got {shuffle!r}")
    compressors = [BloscCodec(cname="zstd", clevel=5, shuffle=shuffle)]
    if compression == "max":
        try:
            from zarr.codecs.numcodecs import Delta
        except ImportError:
            try:
                from numcodecs.zarr3 import Delta
            except ImportError as e:
                raise ImportError("compression='max' needs numcodecs") from e
        return [Delta(dtype="int16")], compressors
    return None, compressors


TRIAL_SAMPLES = 65536


def _step_offset(lo, hi, step):
    zero_fits = (lo / step >= spec.DIG_MIN) & (hi / step <= spec.DIG_MAX)
    return np.where(zero_fits, 0.0, lo - step * spec.DIG_MIN)


def _resolve_shuffle(shuffle, data, storage, lo, hi, step, compression=spec.DEFAULT_COMPRESSION) -> str:
    if shuffle != "auto":
        return shuffle
    block = np.asarray(data[:, :TRIAL_SAMPLES])
    if storage == "int16":
        lo = np.asarray(lo, dtype=np.float64)
        gain, offset = spec.compute_scaling(lo, np.asarray(hi, dtype=np.float64))
        if step is not None:
            step_a = np.broadcast_to(np.asarray(step, dtype=np.float64), gain.shape)
            use = np.isfinite(step_a)
            gain = np.where(use, step_a, gain)
            offset = np.where(use, _step_offset(lo, np.asarray(hi, dtype=np.float64), step_a), offset)
        block, _ = spec.quantize(block, gain, offset)
    else:
        block = block.astype(spec.STORAGE_DTYPES[storage], copy=False)
    return choose_shuffle(block, compression)


def _quantisation_step(st: dict, ch_names=None) -> dict:
    gain = np.asarray(st["gain"], dtype=np.float64)
    return {"name": "quantise", "step": [float(g) for g in gain],
            "max_abs_error": [float(g / 2) for g in gain],
            "unit": st["attrs"].get("unit") if isinstance(st.get("attrs"), dict) else None,
            "channels": list(ch_names) if ch_names is not None else None}


def _iso_utc(t0) -> str | None:
    if t0 is None:
        return None
    if isinstance(t0, str):
        try:
            t0 = _dt.datetime.fromisoformat(t0.replace("Z", "+00:00"))
        except ValueError as e:
            raise ValueError(f"t0 {t0!r} is not an ISO 8601 timestamp") from e
    if isinstance(t0, (int, float)):
        t0 = _dt.datetime.fromtimestamp(float(t0), tz=_dt.timezone.utc)
    if isinstance(t0, _dt.datetime):
        if t0.tzinfo is None:
            t0 = t0.replace(tzinfo=_dt.timezone.utc)
        return t0.astimezone(_dt.timezone.utc).isoformat()
    raise TypeError(f"cannot interpret {t0!r} as a timestamp")


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")


def _set_attrs(node, mapping: dict) -> None:
    node.attrs.update({str(k): spec.json_safe(v) for k, v in mapping.items()})


def write_str_column(grp, name: str, values) -> None:
    grp.attrs[name] = [str(v) for v in values]


def read_str_column(grp, name: str) -> list[str] | None:
    v = grp.attrs.get(name)
    if isinstance(v, list):
        return [str(x) for x in v]
    arr = grp.get(name)
    if arr is None:
        return None
    return [str(x) for x in arr[:]]


def _write_column(grp, name: str, values, dtype: str, overwrite: bool = False):
    if dtype == "str":
        write_str_column(grp, name, values)
        return None
    n = len(values)
    a = grp.create_array(
        name=name, shape=(n,), chunks=(max(n, 1),), dtype=dtype, overwrite=overwrite
    )
    if n:
        a[:] = values
    return a


def build_event_table(
    onset: Iterable[float],
    duration: Iterable[float] | None = None,
    label: Iterable[str] | None = None,
    value: Iterable[int] | None = None,
    vocab: Sequence[str] | None = None,
) -> dict:
    onset = np.asarray(list(onset), dtype=np.float64)
    n = onset.size
    labels = [str(x) for x in (label if label is not None else [""] * n)]
    duration = np.asarray(
        list(duration) if duration is not None else np.zeros(n), dtype=np.float64
    )
    value = np.asarray(list(value) if value is not None else np.zeros(n), dtype=np.int64)
    for k, v in (("duration", duration), ("label", labels), ("value", value)):
        if len(v) != n:
            raise ValueError(f"events/{k} has {len(v)} entries, expected {n}")
    if n and not np.all(np.isfinite(onset)):
        raise ValueError("events/onset must be finite")
    if n and np.any(onset < 0):
        raise ValueError("events/onset must be >= 0 (seconds from root t0)")
    if n and (not np.all(np.isfinite(duration)) or np.any(duration < 0)):
        raise ValueError("events/duration must be finite and >= 0")

    if vocab is None:
        vocab = sorted(set(labels))
    vocab = [str(v) for v in vocab]
    if len(set(vocab)) != len(vocab):
        raise ValueError("vocab has duplicate labels")
    lookup = {v: i for i, v in enumerate(vocab)}
    try:
        label_id = np.array([lookup[l] for l in labels], dtype=np.int32)
    except KeyError as e:
        raise ValueError(f"label {e.args[0]!r} is not in the supplied vocab") from e

    order = np.argsort(onset, kind="stable") if n and np.any(np.diff(onset) < 0) else np.arange(n)
    return {
        "onset": onset[order],
        "duration": duration[order],
        "value": value[order],
        "label_id": label_id[order],
        "vocab": vocab,
        "label": np.array(labels, dtype=object)[order],
    }


def revision_attrs(table: dict) -> dict:
    return {
        "n_events": int(len(table["onset"])),
        "digest": spec.event_digest(table),
        "vocab": [str(v) for v in table["vocab"]],
    }


def write_revision_columns(grp, table: dict) -> None:
    for field, dtype in spec.EVENT_ARRAY_FIELDS.items():
        _write_column(grp, field, table[field], dtype)


def write_revision(root, k: int, table: dict, *, overwrite: bool = False):
    events = root.require_group(spec.EVENTS)
    grp = events.create_group(f"{spec.EVENT_REVISION_PREFIX}{int(k)}", overwrite=overwrite)
    grp.attrs.put(revision_attrs(table))
    write_revision_columns(grp, table)
    return grp


class ZarrFMWriter:
    def __init__(
        self,
        path: str | None = None,
        *,
        group=None,
        subject: str | None = None,
        session: str | None = None,
        task: str | None = None,
        t0=None,
        line_freq: float | None = None,
        primary_stream: str | None = None,
        overwrite: bool = False,
        extra: dict | None = None,
    ):
        if (path is None) == (group is None):
            raise ValueError("give exactly one of path= or group=")
        if group is not None:
            self.path = None
            self.root = group
            self._owns_store = False
        else:
            self.path = str(path)
            self.root = zarr.create_group(store=self.path, overwrite=overwrite)
            self._owns_store = True
        self.root.create_group(spec.SIGNALS)
        self.root.create_group(spec.CHANNELS)
        _set_attrs(
            self.root,
            {
                "format": spec.FORMAT,
                "version": spec.VERSION,
                "subject": subject,
                "session": session,
                "task": task,
                "t0": _iso_utc(t0),
                "line_freq": line_freq,
                "created_by": f"zarrfm {spec.VERSION}",
                "primary_stream": primary_stream,
                "canonical_version": canonical.CANONICAL_VERSION,
                **(extra or {}),
            },
        )
        self._primary = primary_stream
        self._streams: dict[str, dict] = {}
        self._closed = False


    def add_stream(
        self,
        name: str,
        *,
        n_channels: int,
        n_samples: int,
        sfreq: float,
        ch_names: Sequence[str] | None = None,
        ch_types: Sequence[str] | str = "eeg",
        unit: str | Sequence[str] = "V",
        phys_range=None,
        storage: str = "int16",
        positions: np.ndarray | None = None,
        bad: Sequence[bool] | Sequence[str] | None = None,
        canonical_id: Sequence[int] | None = None,
        reference: str | Sequence[str] | None = None,
        t0_offset: float = 0.0,
        chunk_samples: int | None = None,
        shard_factor: int | None = None,
        compression: str = spec.DEFAULT_COMPRESSION,
        highpass: float | None = None,
        lowpass: float | None = None,
        step=None,
        shuffle: str = "shuffle",
    ):
        if self._closed:
            raise RuntimeError("writer is closed")
        if name in self._streams:
            raise ValueError(f"stream {name!r} already exists")
        if storage not in spec.STORAGE_DTYPES:
            raise ValueError(f"storage must be one of {sorted(spec.STORAGE_DTYPES)}")
        if sfreq <= 0:
            raise ValueError("sfreq must be positive")
        n_channels, n_samples = int(n_channels), int(n_samples)

        dtype = spec.STORAGE_DTYPES[storage]
        itemsize = np.dtype(dtype).itemsize

        if storage == "int16":
            if phys_range is None:
                raise ValueError(
                    "int16 storage needs phys_range=(min, max); pass explicit limits, "
                    "or use from_array/from_mne which measure the data first, or "
                    "storage='float32'."
                )
            lo, hi = phys_range
            lo = np.broadcast_to(np.asarray(lo, dtype=np.float64), (n_channels,))
            hi = np.broadcast_to(np.asarray(hi, dtype=np.float64), (n_channels,))
            gain, offset = spec.compute_scaling(lo, hi)
            if step is not None:
                step_a = np.broadcast_to(np.asarray(step, dtype=np.float64), (n_channels,))
                use = np.isfinite(step_a)
                if np.any(step_a[use] <= 0):
                    raise ValueError("step must be positive")
                need = (hi - lo) / np.where(use, step_a, 1.0)
                too_fine = use & (need > spec.DIG_MAX - spec.DIG_MIN)
                if np.any(too_fine):
                    worst = int(np.argmax(np.where(too_fine, need, 0)))
                    raise ValueError(
                        f"step is too fine for {int(too_fine.sum())} channel(s): channel "
                        f"{worst} spans {hi[worst] - lo[worst]:.6g}, which needs a step of at "
                        f"least {(hi[worst] - lo[worst]) / (spec.DIG_MAX - spec.DIG_MIN):.6g}"
                    )
                gain = np.where(use, step_a, gain)
                offset = np.where(use, _step_offset(lo, hi, step_a), offset)
        else:
            gain = np.ones(n_channels, dtype=np.float64)
            offset = np.zeros(n_channels, dtype=np.float64)

        chunk_t = int(
            chunk_samples
            or spec.choose_chunk_samples(n_channels, itemsize, n_samples, sfreq)
        )
        chunk_t = max(1, min(chunk_t, max(n_samples, 1)))
        if shard_factor is None:
            shard_t = spec.choose_shard_samples(n_channels, itemsize, chunk_t, n_samples)
        else:
            n_chunks_total = -(-max(n_samples, 1) // chunk_t)
            shard_t = chunk_t * max(1, min(int(shard_factor), n_chunks_total))

        signals = self.root[spec.SIGNALS]
        filters, compressors = _codecs(compression, shuffle)
        if storage != "int16" and filters is not None:
            filters, compression = None, "fast"
        create_kw = {} if filters is None else {"filters": filters}
        arr = signals.create_array(
            name=name,
            shape=(n_channels, n_samples),
            chunks=(n_channels, chunk_t),
            shards=(n_channels, shard_t),
            dtype=dtype,
            compressors=compressors,
            dimension_names=("channel", "time"),
            **create_kw,
        )
        _set_attrs(
            arr,
            {
                "sfreq": float(sfreq),
                "n_channels": n_channels,
                "n_samples": n_samples,
                "unit": unit if isinstance(unit, str) else max(set(unit), key=list(unit).count),
                "storage": storage,
                "t0_offset": float(t0_offset),
                "compression": compression,
                "shuffle": shuffle,
                "n_clipped": 0,
                "highpass": None if highpass is None else float(highpass),
                "lowpass": None if lowpass is None else float(lowpass),
            },
        )

        self._write_channel_table(
            name, n_channels, ch_names, ch_types, unit, gain, offset, positions, bad,
            canonical_id, reference,
        )
        self._streams[name] = {
            "array": arr,
            "gain": gain,
            "offset": offset,
            "storage": storage,
            "n_clipped": 0,
            "n_channels": n_channels,
            "n_samples": n_samples,
        }
        if self._primary is None:
            self._primary = name
            _set_attrs(self.root, {"primary_stream": name})
        return arr

    def _write_channel_table(
        self, stream, n_channels, ch_names, ch_types, unit, gain, offset, positions, bad,
        canonical_id, reference,
    ):
        if ch_names is None:
            ch_names = [f"CH{i:03d}" for i in range(n_channels)]
        ch_names = [str(x) for x in ch_names]
        if len(ch_names) != n_channels:
            raise ValueError(f"ch_names has {len(ch_names)} entries, expected {n_channels}")
        if len(set(ch_names)) != n_channels:
            dupes = sorted({n for n in ch_names if ch_names.count(n) > 1})
            raise ValueError(
                f"channel names must be unique; duplicated: {dupes}. Picks by name would "
                "otherwise resolve silently to one of them."
            )
        if isinstance(ch_types, str):
            ch_types = [ch_types] * n_channels
        ch_types = [str(x).lower() for x in ch_types]
        if len(ch_types) != n_channels:
            raise ValueError(f"ch_types has {len(ch_types)} entries, expected {n_channels}")

        if bad is None or len(bad) == 0:
            bad_mask = np.zeros(n_channels, dtype=bool)
        elif isinstance(next(iter(bad)), str):
            bad_set = set(bad)
            bad_mask = np.array([n in bad_set for n in ch_names], dtype=bool)
        else:
            bad_mask = np.asarray(bad, dtype=bool)
            if bad_mask.shape != (n_channels,):
                raise ValueError("bad mask length does not match n_channels")

        inferred_ids, inferred_refs = canonical.map_channels(ch_names, ch_types)
        if canonical_id is None:
            canonical_id = inferred_ids
        canonical_id = np.asarray(canonical_id, dtype=np.int32)
        if canonical_id.shape != (n_channels,):
            raise ValueError(f"canonical_id must have {n_channels} entries")
        if reference is None:
            reference = inferred_refs
        elif isinstance(reference, str):
            reference = [reference] * n_channels
        reference = [str(r) for r in reference]
        if len(reference) != n_channels:
            raise ValueError(f"reference must have {n_channels} entries")

        grp = self.root[spec.CHANNELS].create_group(stream)
        units = [unit] * n_channels if isinstance(unit, str) else [str(u) for u in unit]
        if len(units) != n_channels:
            raise ValueError(f"unit has {len(units)} entries, expected {n_channels}")

        grp.attrs.put(
            {"name": ch_names, "type": ch_types, "unit": units, "reference": reference}
        )
        values = {
            "gain": np.asarray(gain, dtype=np.float64),
            "offset": np.asarray(offset, dtype=np.float64),
            "bad": bad_mask,
            "canonical_id": canonical_id,
        }
        for field, dtype in spec.CHANNEL_ARRAY_FIELDS.items():
            a = grp.create_array(
                name=field, shape=(n_channels,), chunks=(n_channels,), dtype=dtype
            )
            a[:] = values[field]

        if positions is None:
            pos = np.full((n_channels, 3), np.nan, dtype=np.float64)
        else:
            pos = np.asarray(positions, dtype=np.float64)
            if pos.shape != (n_channels, 3):
                raise ValueError(f"positions must be ({n_channels}, 3), got {pos.shape}")
        a = grp.create_array(
            name=spec.POSITION_FIELD,
            shape=(n_channels, 3),
            chunks=(n_channels, 3),
            dtype="float64",
        )
        a[:] = pos


    def write_block(self, stream: str, start: int, data: np.ndarray) -> None:
        st = self._streams.get(stream)
        if st is None:
            raise KeyError(f"unknown stream {stream!r}")
        data = np.asarray(data)
        if data.ndim != 2 or data.shape[0] != st["n_channels"]:
            raise ValueError(
                f"expected block of shape ({st['n_channels']}, t), got {data.shape}"
            )
        start = int(start)
        stop = start + data.shape[1]
        if start < 0 or stop > st["n_samples"]:
            raise ValueError(
                f"block [{start}:{stop}) does not fit in {st['n_samples']} samples"
            )

        if st["storage"] == "int16":
            codes, n_clipped = spec.quantize(data, st["gain"], st["offset"])
            st["n_clipped"] += n_clipped
            st["array"][:, start:stop] = codes
        else:
            st["array"][:, start:stop] = data.astype(np.float32)

    def write_codes(self, stream: str, start: int, codes: np.ndarray, n_clipped: int = 0) -> None:
        st = self._streams.get(stream)
        if st is None:
            raise KeyError(f"unknown stream {stream!r}")
        if st["storage"] != "int16":
            raise ValueError("write_codes is only for int16 streams")
        codes = np.asarray(codes)
        if codes.dtype != np.int16 or codes.ndim != 2 or codes.shape[0] != st["n_channels"]:
            raise ValueError(
                f"expected int16 codes of shape ({st['n_channels']}, t), got "
                f"{codes.dtype} {codes.shape}"
            )
        start = int(start)
        stop = start + codes.shape[1]
        if start < 0 or stop > st["n_samples"]:
            raise ValueError(f"block [{start}:{stop}) does not fit in {st['n_samples']} samples")
        st["n_clipped"] += int(n_clipped)
        st["array"][:, start:stop] = codes

    def block_samples(self, stream: str) -> int:
        st = self._streams.get(stream)
        if st is None:
            raise KeyError(f"unknown stream {stream!r}")
        arr = st["array"]
        return int(arr.shards[1] if arr.shards else arr.chunks[1])

    def write_stream(self, stream: str, data: np.ndarray, block_samples: int | None = None) -> None:
        data = np.asarray(data)
        step = int(block_samples or self.block_samples(stream))
        for s in range(0, data.shape[1], step):
            self.write_block(stream, s, data[:, s : s + step])


    def set_events(
        self,
        onset: Iterable[float],
        duration: Iterable[float] | None = None,
        label: Iterable[str] | None = None,
        value: Iterable[int] | None = None,
        vocab: Sequence[str] | None = None,
    ) -> None:
        table = build_event_table(onset, duration, label, value, vocab)
        write_revision(self.root, 0, table, overwrite=True)
        _set_attrs(self.root, {spec.EVENT_REVISION_ATTR: 0})

    def set_segments(self, start_sample, n_samples, t0_offset=None) -> None:
        start_sample = np.asarray(list(start_sample), dtype=np.int64)
        n_samples = np.asarray(list(n_samples), dtype=np.int64)
        n = start_sample.size
        if n_samples.size != n:
            raise ValueError("start_sample and n_samples must be the same length")
        t0_offset = (
            np.zeros(n, dtype=np.float64)
            if t0_offset is None
            else np.asarray(list(t0_offset), dtype=np.float64)
        )
        cols = {"start_sample": start_sample, "n_samples": n_samples, "t0_offset": t0_offset}
        grp = self.root.require_group(spec.SEGMENTS)
        for field, dtype in spec.SEGMENT_FIELDS.items():
            _write_column(grp, field, cols[field], dtype, overwrite=True)


    def set_provenance(
        self,
        *,
        source: dict | None = None,
        pipeline: Sequence[dict] | None = None,
        software: dict | None = None,
        **extra,
    ) -> None:
        grp = self.root.require_group(spec.PROVENANCE)
        pipeline = [dict(step) for step in (pipeline or [])]
        for step in pipeline:
            if "name" not in step:
                raise ValueError("every pipeline step needs a 'name'")
        software = dict(software or {})
        software.setdefault("zarrfm", spec.VERSION)
        try:
            software.setdefault("zarr", zarr.__version__)
            software.setdefault("numpy", np.__version__)
        except AttributeError:
            pass

        _set_attrs(
            grp,
            {
                "source": dict(source or {}),
                "pipeline": pipeline,
                "software": software,
                "written": _now_iso(),
                **extra,
            },
        )


    def close(self) -> None:
        if self._closed:
            return
        prefix = f"{self.root.path}/" if self.root.path else ""
        for name, st in self._streams.items():
            extra = {}
            if st["n_clipped"]:
                extra["n_clipped"] = st["n_clipped"]
            extra["content_hash"] = _storeio.content_hash(
                self.root.store, f"{prefix}{spec.SIGNALS}/{name}/"
            )
            _set_attrs(st["array"], extra)
        if self.root.get(spec.EVENTS) is None:
            self.set_events([])
        if self.root.get(spec.PROVENANCE) is None:
            self.set_provenance()
        if self._owns_store:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                zarr.consolidate_metadata(self.path)
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def from_array(
    data: np.ndarray,
    path: str | None = None,
    *,
    sfreq: float,
    group=None,
    stream: str = "eeg",
    storage: str = "int16",
    block_samples: int | None = None,
    step=None,
    shuffle: str = "auto",
    **kwargs,
):
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"expected (n_channels, n_times), got {data.shape}")
    stream_kw = {
        k: kwargs.pop(k)
        for k in ("ch_names", "ch_types", "unit", "positions", "bad", "t0_offset",
                  "chunk_samples", "shard_factor", "compression", "canonical_id",
                  "reference")
        if k in kwargs
    }
    events = kwargs.pop("events", None)
    provenance = kwargs.pop("provenance", None)
    overwrite = kwargs.pop("overwrite", True)
    if group is not None:
        w = ZarrFMWriter(group=group, **kwargs)
    else:
        w = ZarrFMWriter(path, overwrite=overwrite, **kwargs)
    with w:
        w.add_stream(
            stream,
            n_channels=data.shape[0],
            n_samples=data.shape[1],
            sfreq=sfreq,
            storage=storage,
            phys_range=(data.min(axis=1), data.max(axis=1)) if storage == "int16" else None,
            step=step if storage == "int16" else None,
            shuffle=_resolve_shuffle(shuffle, data, storage, data.min(axis=1), data.max(axis=1), step,
                                     stream_kw.get("compression", spec.DEFAULT_COMPRESSION)),
            **stream_kw,
        )
        w.write_stream(stream, data, block_samples=block_samples)
        if events is not None:
            w.set_events(**events)
        if provenance is None and step is not None and storage == "int16":
            provenance = {"source": {"format": "array"}, "pipeline": []}
        if provenance is not None:
            if step is not None and storage == "int16":
                provenance = dict(provenance)
                provenance["pipeline"] = list(provenance.get("pipeline") or []) + [
                    _quantisation_step(w._streams[stream], stream_kw.get("ch_names"))]
            w.set_provenance(**provenance)
    return path if group is None else group.path
