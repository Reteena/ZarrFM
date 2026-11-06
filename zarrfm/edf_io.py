from __future__ import annotations

import datetime as _dt
import os
import re
import warnings

import numpy as np

from . import canonical, spec
from .writer import ZarrFMWriter


def _require_edfio():
    try:
        import edfio
    except ImportError as e:
        raise ImportError("from_edf needs edfio: pip install 'zarrfm[dev]' or pip install edfio") from e
    return edfio


_VOLT_UNITS = {"v": 1.0, "mv": 1e-3, "uv": 1e-6, "µv": 1e-6, "μv": 1e-6, "nv": 1e-9}

_TYPE_BY_PREFIX = {
    "eeg": "eeg", "eog": "eog", "ecg": "ecg", "ekg": "ecg", "emg": "emg", "seeg": "seeg",
    "ecog": "ecog", "dbs": "dbs", "resp": "resp", "gsr": "gsr", "temp": "temp",
}
_TYPE_BY_CUE = (
    (re.compile(r"\b(resp|airflow|thorax|thor|abdom|abd|nasal|thermistor|flow|snore|pleth|spo2|sao2)\b", re.I), "resp"),
    (re.compile(r"\b(temp|thermo|rectal)\b", re.I), "temp"),
    (re.compile(r"\b(marker|trigger|event|status|stim|button)\b", re.I), "stim"),
    (re.compile(r"(\b(emg|chin|submental|leg|tibialis|lat|rat|rleg|lleg)\b|chin\d)", re.I), "emg"),
    (re.compile(r"(\b(eog|roc|loc|e1|e2)\b|eog\d)", re.I), "eog"),
    (re.compile(r"\b(ecg|ekg)\b", re.I), "ecg"),
    (re.compile(r"\b(gsr|eda)\b", re.I), "gsr"),
)
_PREFILTER_HP = re.compile(r"HP\s*:\s*([0-9]+(?:[.,][0-9]+)?)\s*(?:Hz)?", re.I)
_PREFILTER_LP = re.compile(r"LP\s*:\s*([0-9]+(?:[.,][0-9]+)?)\s*(?:Hz)?", re.I)


def _start_datetime(edf):
    try:
        d, t = edf.startdate, edf.starttime
    except Exception:
        return None, "anonymised in the EDF+ header"
    if d == _dt.date(1985, 1, 1):
        return None, "EDF placeholder 1985-01-01 (unknown)"
    return _dt.datetime.combine(d, t, tzinfo=_dt.timezone.utc), "from the EDF header"


def _infer_type(label: str, unit: str) -> str:
    head = label.strip().split(" ", 1)[0].lower()
    if head in _TYPE_BY_PREFIX:
        return _TYPE_BY_PREFIX[head]
    for rx, t in _TYPE_BY_CUE:
        if rx.search(label):
            return t
    if unit.strip().lower() in _VOLT_UNITS:
        return "eeg"
    return "misc"


def _unit_scale(dimension: str) -> tuple[str, float]:
    d = dimension.strip()
    key = d.lower()
    if key in _VOLT_UNITS:
        return "V", _VOLT_UNITS[key]
    return (d if d else "n/a"), 1.0


def _dedup(labels: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    counts = {l: labels.count(l) for l in labels}
    out = []
    for l in labels:
        if counts[l] > 1:
            k = seen.get(l, 0)
            seen[l] = k + 1
            out.append(f"{l}-{k}")
        else:
            out.append(l)
    return out


def _filters(prefilters: list[str]) -> tuple[float | None, float | None]:
    hps, lps = set(), set()
    for s in prefilters:
        m = _PREFILTER_HP.search(s or "")
        hps.add(float(m.group(1).replace(",", ".")) if m else None)
        m = _PREFILTER_LP.search(s or "")
        lps.add(float(m.group(1).replace(",", ".")) if m else None)
    hp = hps.pop() if len(hps) == 1 else None
    lp = lps.pop() if len(lps) == 1 else None
    return (hp if hp else None), (lp if lp else None)


def _stream_name(rate: float, types: list[str], is_primary: bool) -> str:
    if is_primary:
        return "eeg"
    common = max(sorted(set(types)), key=types.count)
    kind = common if types.count(common) * 2 > len(types) else "aux"
    return f"{kind}_{rate:g}hz".replace(".", "p")


def from_edf(
    path: str,
    store: str,
    *,
    annotations: str | None = None,
    subject: str | None = None,
    session: str | None = None,
    task: str | None = None,
    line_freq: float | None = None,
    event_id: dict | None = None,
    overwrite: bool = True,
    compression: str = spec.DEFAULT_COMPRESSION,
    shuffle: str = "auto",
    **writer_kwargs,
) -> str:
    from .writer import TRIAL_SAMPLES, choose_shuffle

    edfio = _require_edfio()
    with open(path, "rb") as fh:
        magic = fh.read(8)
    if magic[:1] == b"\xff" or magic[1:8].upper() == b"BIOSEMI":
        raise NotImplementedError(
            f"{path} is a BDF (24-bit BioSemi) file: its codes do not fit int16, so they "
            "cannot be stored natively. Use zarrfm.from_mne(mne.io.read_raw_bdf(path), "
            "store), which re-quantises from the observed range."
        )
    edf = edfio.read_edf(path, lazy_load_data=True)
    if not edf.is_continuous:
        raise NotImplementedError(
            f"{path} is a discontinuous EDF+D file; its records are not contiguous in time "
            "and would be stored as one continuous stream with no segments. Not supported yet."
        )
    signals = [s for s in edf.signals if s.label.strip().upper() != "EDF ANNOTATIONS"]
    if not signals:
        raise ValueError(f"{path} carries no signals")

    labels = _dedup([s.label.strip() for s in signals])
    if labels != [s.label.strip() for s in signals]:
        dupes = sorted({s.label.strip() for s in signals if
                        [x.label.strip() for x in signals].count(s.label.strip()) > 1})
        warnings.warn(f"duplicate channel labels {dupes} renamed with running numbers", stacklevel=2)

    rates = sorted({float(s.sampling_frequency) for s in signals}, reverse=True)
    t0, t0_note = _start_datetime(edf)
    duration_s = float(edf.duration)

    type_of = {i: _infer_type(s.label, s.physical_dimension) for i, s in enumerate(signals)}
    rate_of = {i: float(s.sampling_frequency) for i, s in enumerate(signals)}
    eeg_count = {r: sum(1 for i in type_of if rate_of[i] == r and type_of[i] == "eeg") for r in rates}
    primary_rate = max(rates, key=lambda r: (eeg_count[r], r))

    with ZarrFMWriter(
        store, subject=subject, session=session, task=task, t0=t0, line_freq=line_freq,
        overwrite=overwrite, **writer_kwargs,
    ) as w:
        streams_written = []
        for rate in rates:
            idx = [i for i, s in enumerate(signals) if float(s.sampling_frequency) == rate]
            group = [signals[i] for i in idx]
            names = [labels[i] for i in idx]
            units_scale = [_unit_scale(s.physical_dimension) for s in group]
            units = [u for u, _ in units_scale]
            types = [type_of[i] for i in idx]
            lo, hi, flip, degenerate = [], [], [], []
            for s, (_, scale) in zip(group, units_scale):
                if s.digital_max == s.digital_min:
                    g, o = 1.0, 0.0
                    degenerate.append(True)
                else:
                    g = (s.physical_max - s.physical_min) / (s.digital_max - s.digital_min)
                    o = s.physical_max - g * s.digital_max
                    degenerate.append(False)
                a, b = scale * (o + g * spec.DIG_MIN), scale * (o + g * spec.DIG_MAX)
                lo.append(min(a, b))
                hi.append(max(a, b))
                flip.append(g < 0)
            if any(degenerate):
                warnings.warn(
                    f"{[n for n, d in zip(names, degenerate) if d]}: digital_min == "
                    "digital_max in the EDF header; codes stored with gain 1 and the "
                    "channel flagged bad", stacklevel=2,
                )
            n_samples = int(round(duration_s * rate))
            hp, lp = _filters([s.prefiltering for s in group])
            name = _stream_name(rate, types, is_primary=(rate == primary_rate))

            def read_codes(s0: int, s1: int) -> np.ndarray:
                buf = np.empty((len(group), s1 - s0), dtype=np.int16)
                for row, s in enumerate(group):
                    seg = s.get_digital_slice(s0 / rate, s1 / rate)
                    if flip[row]:
                        seg = (-seg.astype(np.int32) - 1).astype(np.int16)
                    buf[row, : seg.size] = seg[: s1 - s0]
                    if seg.size < s1 - s0:
                        buf[row, seg.size:] = seg[-1] if seg.size else 0
                return buf

            chosen = shuffle
            if shuffle == "auto":
                chosen = choose_shuffle(read_codes(0, min(n_samples, TRIAL_SAMPLES)), compression)
            w.add_stream(
                name, n_channels=len(group), n_samples=n_samples, sfreq=rate,
                ch_names=names, ch_types=types, unit=units, phys_range=(np.array(lo), np.array(hi)),
                highpass=hp, lowpass=lp, compression=compression, shuffle=chosen,
                bad=degenerate if any(degenerate) else None,
            )
            if rate == primary_rate and w._primary != name:
                w._primary = name
                from .writer import _set_attrs
                _set_attrs(w.root, {"primary_stream": name})
            shard = w.block_samples(name)
            for s0 in range(0, n_samples, shard):
                s1 = min(s0 + shard, n_samples)
                w.write_codes(name, s0, read_codes(s0, s1))
            streams_written.append({
                "stream": name, "rate_hz": rate, "channels": names,
                "transducer": [s.transducer_type for s in group],
                "prefiltering": [s.prefiltering for s in group],
                "physical_dimension": [s.physical_dimension for s in group],
            })

        onset, dur, text = [], [], []
        for a in edf.annotations:
            onset.append(float(a.onset)); dur.append(float(a.duration or 0.0)); text.append(str(a.text))
        ann_file = None
        if annotations is not None:
            ann = edfio.read_edf(annotations)
            ann_file = str(annotations)
            ann_t0, ann_note = _start_datetime(ann)
            if t0 is not None and ann_t0 is not None:
                shift = (ann_t0 - t0).total_seconds()
            else:
                shift = 0.0
                warnings.warn(
                    "annotation file or signal file has an anonymised start time; assuming "
                    "the two started together", stacklevel=2,
                )
            for a in ann.annotations:
                onset.append(float(a.onset) + shift); dur.append(float(a.duration or 0.0)); text.append(str(a.text))
        onset_a = np.asarray(onset, dtype=np.float64)
        keep = (onset_a >= 0) & (onset_a < duration_s) & np.isfinite(onset_a)
        dropped = int((~keep).sum())
        value = None
        if event_id:
            value = [int(event_id.get(t, 0)) for t, k in zip(text, keep) if k]
        w.set_events(
            onset=onset_a[keep], duration=np.asarray(dur, dtype=np.float64)[keep],
            label=[t for t, k in zip(text, keep) if k], value=value,
        )

        rec = edf.recording
        source = {
            "file": os.path.abspath(path),
            "format": "EDF+" if edf.reserved.strip().upper().startswith("EDF+") else "EDF",
            "reader": "edfio",
            "equipment": getattr(rec, "equipment_code", None),
            "startdate": t0_note,
            "timezone": "unknown; EDF has none, stored as UTC",
            "streams": streams_written,
        }
        if ann_file:
            source["annotations_file"] = ann_file
        step = {"name": "from_edf", "scaling": "native", "n_streams": len(rates)}
        if dropped:
            step["dropped_events_past_end"] = dropped
        w.set_provenance(source=source, pipeline=[step])
    return store
