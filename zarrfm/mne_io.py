from __future__ import annotations

import datetime as _dt
import warnings

import numpy as np

from . import spec
from .reader import ZarrFMReader
from .writer import ZarrFMWriter


def _require_mne():
    try:
        import mne
    except ImportError as e:
        raise ImportError("the MNE bridge needs mne: pip install 'zarrfm[mne]'") from e
    return mne


_VOLT_TYPES = {"eeg", "eog", "ecg", "emg", "seeg", "ecog", "dbs", "bio"}

READ_BLOCK_BYTES = 32 * 1024**2

_UNKNOWN_SUBJECT = {"", "x", "xx", "xxx", "unknown", "anonymous", "anon", "n/a", "none",
                    "surrogate", "surrogate name", "patient", "subject"}

_TO_SPEC_TYPE = {"temperature": "temp", "bio": "misc", "syst": "misc", "ias": "misc",
                 "exci": "misc", "chpi": "misc", "hbo": "misc", "hbr": "misc",
                 "fnirs_cw_amplitude": "misc", "fnirs_od": "misc", "mag": "misc",
                 "grad": "misc", "ref_meg": "misc", "csd": "eeg", "dipole": "misc",
                 "gof": "misc", "eyegaze": "misc", "pupil": "misc"}
_TO_MNE_TYPE = {"temp": "temperature", "seeg": "seeg", "ecog": "ecog", "dbs": "dbs",
                "resp": "resp", "gsr": "gsr", "stim": "stim", "misc": "misc", "eeg": "eeg",
                "eog": "eog", "ecg": "ecg", "emg": "emg"}


def _channel_units(ch_types) -> list[str]:
    return ["V" if t in _VOLT_TYPES else "n/a" for t in ch_types]


def _native_rates(raw) -> np.ndarray | None:
    extras = getattr(raw, "_raw_extras", None)
    if not extras or "n_samps" not in extras[0] or "record_length" not in extras[0]:
        return None
    ex = extras[0]
    try:
        n_samps = np.asarray(ex["n_samps"], dtype=np.float64)
        sel = np.asarray(ex.get("sel", np.arange(n_samps.size)), dtype=int)
        rec_len = float(np.asarray(ex["record_length"]).ravel()[0])
        rates = n_samps[sel] / rec_len
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
        return None
    names = list(ex.get("ch_names", []))
    if len(names) == rates.size and names != list(raw.ch_names):
        idx = {n: i for i, n in enumerate(names)}
        try:
            rates = rates[[idx[n] for n in raw.ch_names]]
        except KeyError:
            pass

    return rates


def _native_scaling(raw) -> tuple[np.ndarray, np.ndarray] | None:
    extras = getattr(raw, "_raw_extras", None)
    if not extras:
        return None
    ex = extras[0]
    keys = ("cal", "offsets", "units", "digital_max")
    if any(k not in ex for k in keys):
        return None
    try:
        cal = np.asarray(ex["cal"], dtype=np.float64)
        off = np.asarray(ex["offsets"], dtype=np.float64)
        units = np.asarray(ex["units"], dtype=np.float64)
        dmax = np.asarray(ex["digital_max"], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    n = len(raw.ch_names)
    if not (cal.size == off.size == units.size == dmax.size):
        return None
    names = list(ex.get("ch_names", []))
    if len(names) == cal.size and names != list(raw.ch_names):
        idx = {nm: i for i, nm in enumerate(names)}
        try:
            pick = np.array([idx[nm] for nm in raw.ch_names])
        except KeyError:
            return None
        cal, off, units, dmax = cal[pick], off[pick], units[pick], dmax[pick]
    elif cal.size != n:
        return None
    if np.any(dmax > spec.DIG_MAX) or np.any(cal <= 0) or not np.all(np.isfinite(cal)):
        return None
    gain = cal * units
    offset = off * units
    return offset + gain * spec.DIG_MIN, offset + gain * spec.DIG_MAX


def _filters(info, sfreq: float) -> tuple[float | None, float | None]:
    hp = info.get("highpass")
    lp = info.get("lowpass")
    hp = float(hp) if hp is not None and hp > 0 else None
    lp = float(lp) if lp is not None and 0 < lp < sfreq / 2 else None
    return hp, lp


def from_mne(
    raw,
    path: str,
    *,
    stream: str = "eeg",
    storage: str = "int16",
    block_samples: int | None = None,
    overwrite: bool = True,
    subject: str | None = None,
    session: str | None = None,
    task: str | None = None,
    line_freq: float | None = None,
    event_id: dict | None = None,
    mixed_rates: str = "error",
    preserve_codes: bool = True,
    step=None,
    shuffle: str = "auto",
    **kwargs,
) -> str:
    mne = _require_mne()
    info = raw.info
    n_channels = len(raw.ch_names)
    n_samples = int(raw.n_times)
    sfreq = float(info["sfreq"])
    mne_types = list(raw.get_channel_types())
    ch_types = [_TO_SPEC_TYPE.get(t, t) for t in mne_types]
    first_time = float(getattr(raw, "first_samp", 0)) / sfreq

    provenance_steps: list[dict] = []

    rates = _native_rates(raw)
    if rates is not None and np.unique(np.round(rates, 6)).size > 1:
        aligned = rates.size == len(raw.ch_names)
        detail = {float(r): ([n for n, rr in zip(raw.ch_names, rates) if rr == r] if aligned
                             else int(np.count_nonzero(rates == r)))
                  for r in np.unique(rates)}
        if mixed_rates != "resample":
            counts = {k: (len(v) if isinstance(v, list) else v) for k, v in detail.items()}
            raise ValueError(
                "this Raw was read from a file with mixed native sampling rates "
                f"{counts} and MNE resampled every "
                f"channel to {sfreq:g} Hz on load; ZarrFM keeps one stream per native "
                "rate. Use zarrfm.from_edf(path, store) for the file, or "
                "pass mixed_rates='resample' to store the resampled channels anyway."
            )
        for rate, names in detail.items():
            if rate != sfreq:
                provenance_steps.append({"name": "resample", "channels": names,
                                         "from_hz": rate, "to_hz": sfreq, "by": "mne on read",
                                         "deterministic": False})

    positions = np.full((n_channels, 3), np.nan, dtype=np.float64)
    for i, ch in enumerate(info["chs"]):
        loc = np.asarray(ch["loc"][:3], dtype=np.float64)
        if np.all(np.isfinite(loc)) and np.any(loc != 0):
            positions[i] = loc

    meas_date = info.get("meas_date")
    if meas_date is not None and meas_date > _dt.datetime.now(_dt.timezone.utc):
        warnings.warn(
            f"meas_date {meas_date.isoformat()} is in the future -- an anonymised or "
            "surrogate date (EDF+ two-digit years map 1985-2084); stored as given",
            stacklevel=2,
        )
    subject_info = info.get("subject_info") or {}
    if subject is None:
        his = subject_info.get("his_id")
        subject = his if his and str(his).strip().lower() not in _UNKNOWN_SUBJECT else None
    if line_freq is None:
        line_freq = info.get("line_freq")

    stream_kw = {
        k: kwargs.pop(k)
        for k in ("compression", "chunk_samples", "shard_factor", "unit")
        if k in kwargs
    }
    read_block = int(block_samples or max(1024, READ_BLOCK_BYTES // max(n_channels * 8, 1)))

    lo = np.full(n_channels, np.inf)
    hi = np.full(n_channels, -np.inf)
    integer = np.ones(n_channels, dtype=bool)
    if storage == "int16":
        for start in range(0, n_samples, read_block):
            block = raw.get_data(start=start, stop=min(start + read_block, n_samples))
            np.minimum(lo, block.min(axis=1), out=lo)
            np.maximum(hi, block.max(axis=1), out=hi)
            integer &= np.all(block == np.rint(block), axis=1)
        lo = np.where(np.isfinite(lo), lo, 0.0)
        hi = np.where(np.isfinite(hi), hi, 0.0)

    scaling = "observed"
    native = _native_scaling(raw) if (storage == "int16" and preserve_codes) else None
    if native is not None:
        nlo, nhi = native
        fits = (lo >= nlo - 1e-12) & (hi <= nhi + 1e-12)
        lo, hi = np.where(fits, nlo, lo), np.where(fits, nhi, hi)
        scaling = "native" if bool(np.all(fits)) else "mixed"
    integer_channels: list[str] = []
    if storage == "int16":
        integer &= (lo >= spec.DIG_MIN) & (hi <= spec.DIG_MAX)
        if native is not None:
            integer &= ~fits
        if integer.any():
            lo = np.where(integer, float(spec.DIG_MIN), lo)
            hi = np.where(integer, float(spec.DIG_MAX), hi)
            integer_channels = [n for n, f in zip(raw.ch_names, integer) if f]
    step_a = None
    if step is not None and storage == "int16":
        step_a = np.broadcast_to(np.asarray(step, dtype=np.float64), (n_channels,)).copy()
        keep_exact = integer | (fits if native is not None else False)
        step_a[keep_exact] = np.nan

    hp, lp = _filters(info, sfreq)
    from .writer import TRIAL_SAMPLES, _resolve_shuffle
    chosen = _resolve_shuffle(shuffle, raw.get_data(start=0, stop=min(n_samples, TRIAL_SAMPLES)),
                              storage, lo, hi, step_a,
                              stream_kw.get("compression", spec.DEFAULT_COMPRESSION)) if shuffle == "auto" else shuffle
    with ZarrFMWriter(
        path,
        subject=subject,
        session=session,
        task=task,
        t0=meas_date,
        line_freq=line_freq,
        overwrite=overwrite,
        **kwargs,
    ) as w:
        stream_kw.setdefault("unit", _channel_units(ch_types))
        w.add_stream(
            stream,
            n_channels=n_channels,
            n_samples=n_samples,
            sfreq=sfreq,
            ch_names=list(raw.ch_names),
            ch_types=ch_types,
            storage=storage,
            phys_range=(lo, hi) if storage == "int16" else None,
            positions=positions,
            bad=list(info["bads"]),
            t0_offset=first_time,
            highpass=hp,
            lowpass=lp,
            step=step_a,
            shuffle=chosen,
            **stream_kw,
        )
        st = w._streams[stream]

        shard = w.block_samples(stream)
        for s0 in range(0, n_samples, shard):
            s1 = min(s0 + shard, n_samples)
            if storage == "int16":
                buf = np.empty((n_channels, s1 - s0), dtype=np.int16)
                clipped = 0
                for b0 in range(s0, s1, read_block):
                    b1 = min(b0 + read_block, s1)
                    codes, nc = spec.quantize(raw.get_data(start=b0, stop=b1), st["gain"], st["offset"])
                    buf[:, b0 - s0 : b1 - s0] = codes
                    clipped += nc
                w.write_codes(stream, s0, buf, clipped)
            else:
                for b0 in range(s0, s1, read_block):
                    b1 = min(b0 + read_block, s1)
                    w.write_block(stream, b0, raw.get_data(start=b0, stop=b1))

        ann = raw.annotations
        dropped = 0
        if len(ann):
            onset = np.asarray(ann.onset, dtype=np.float64) - first_time + first_time
            rel = onset - first_time
            keep = rel < n_samples / sfreq
            dropped = int((~keep).sum())
            labels = [str(d) for d in ann.description]
            value = None
            if event_id:
                value = [int(event_id.get(l, 0)) for l in labels]
            w.set_events(
                onset=onset[keep],
                duration=np.asarray(ann.duration, dtype=np.float64)[keep],
                label=[l for l, k in zip(labels, keep) if k],
                value=None if value is None else [v for v, k in zip(value, keep) if k],
            )

        filenames = [str(f) for f in (getattr(raw, "filenames", None) or []) if f]
        source = {
            "file": filenames[0] if filenames else None,
            "format": type(raw).__name__,
            "reader": f"mne {mne.__version__}",
        }
        orig_units = getattr(raw, "_orig_units", None)
        if orig_units:
            source["orig_units"] = sorted({str(u) for u in orig_units.values()})
        if mne_types != ch_types:
            source["mne_channel_types"] = sorted({t for t in mne_types if _TO_SPEC_TYPE.get(t, t) != t})
        step = {"name": "from_mne", "scaling": scaling,
                "read_block_samples": read_block}
        if integer_channels:
            step["integer_channels"] = integer_channels
        if step_a is not None and np.isfinite(step_a).any():
            from .writer import _quantisation_step
            q = _quantisation_step(st, raw.ch_names)
            q["channels"] = [n for n, s in zip(raw.ch_names, step_a) if np.isfinite(s)]
            q["step"] = [float(s) for s in step_a if np.isfinite(s)]
            q["max_abs_error"] = [float(s / 2) for s in step_a if np.isfinite(s)]
            provenance_steps.append(q)
        if dropped:
            step["dropped_events_past_end"] = dropped
        w.set_provenance(source=source, pipeline=provenance_steps + [step])
    return path


def to_mne(source, stream: str | None = None, *, preload: bool = True):
    mne = _require_mne()

    reader = source if isinstance(source, ZarrFMReader) else ZarrFMReader(source)
    st = reader._stream(stream)
    name = st["name"]
    sinfo = reader.info(name)

    data = reader.get_data(name, dtype=np.float64)
    info = mne.create_info(
        ch_names=reader.ch_names(name),
        sfreq=reader.sfreq(name),
        ch_types=[_TO_MNE_TYPE.get(t, "misc") for t in reader.ch_types(name)],
    )
    with info._unlock():
        info["line_freq"] = reader.attrs.get("line_freq")
        if sinfo.get("highpass") is not None:
            info["highpass"] = float(sinfo["highpass"])
        if sinfo.get("lowpass") is not None:
            info["lowpass"] = float(sinfo["lowpass"])
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    bads = reader.bad_channels(name)
    if bads:
        raw.info["bads"] = bads

    pos = reader.positions(name)
    ch_pos = {
        n: p
        for n, p, t in zip(reader.ch_names(name), pos, reader.ch_types(name))
        if np.all(np.isfinite(p)) and t == "eeg"
    }
    if ch_pos:
        montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
        raw.set_montage(montage, on_missing="ignore", verbose="ERROR")

    t0 = reader.t0
    t0_offset = float(sinfo.get("t0_offset", 0.0))
    if t0 is not None:
        raw.set_meas_date(t0 + _dt.timedelta(seconds=t0_offset))

    ev = reader.events()
    if ev["onset"].size:
        raw.set_annotations(
            mne.Annotations(
                onset=ev["onset"] - t0_offset,
                duration=ev["duration"],
                description=[str(x) for x in ev["label"]],
            ),
            verbose="ERROR",
        )
    return raw
