from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import os
import shutil
import sys
import tempfile
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import zarrfm
from zarrfm.synth import synth_eeg


def _size(path: str) -> tuple[int, int]:
    if os.path.isfile(path):
        return os.path.getsize(path), 1
    total = n = 0
    for d, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(d, f))
            n += 1
    return total, n


def _measure(fn, repeat=5, close=False):
    def release(value):
        if close and hasattr(value, "close"):
            value.close()
    release(fn())
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        value = fn()
        times.append(time.perf_counter() - t)
        release(value)
        del value
    return times


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--sfreq", type=float, default=256.0)
    ap.add_argument("--keep", action="store_true", help="keep the generated stores")
    ap.add_argument("--edf", help="benchmark a real EDF recording instead of synthesising one")
    ap.add_argument("--native-edf", action="store_true", help="read the original EDF with the selected channels; skip EDF re-export")
    ap.add_argument("--annotations", help="paired annotation file, e.g. Sleep-EDF hypnogram")
    ap.add_argument("--picks", nargs="+", help="native EDF channel names to include before loading")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--output", type=Path, help="save JSON and CSV, including all repeat timings")
    args = ap.parse_args()
    if args.native_edf and not args.edf:
        ap.error("--native-edf requires --edf")
    if args.repeats < 1:
        ap.error("repeats must be positive")

    import mne

    tmp = tempfile.mkdtemp(prefix="zarrfm-bench-")
    if args.edf:
        print(f"loading {args.edf} ...")
        raw = mne.io.read_raw_edf(args.edf, preload=True, verbose="ERROR",
                                  **({"include": args.picks} if args.picks else {}))
        if args.picks and set(raw.ch_names) != set(args.picks):
            raise ValueError("requested EDF channels were not all found")
        if args.annotations:
            raw.set_annotations(mne.read_annotations(args.annotations))
        data = raw.get_data()
        onset_source = "paired annotations" if args.annotations else "embedded EDF annotations"
        onsets = np.asarray(raw.annotations.onset, dtype=np.float64) - raw.first_samp / raw.info["sfreq"]
        if onsets.size < 50:
            onset_source = "regular five-second positions (fewer than 50 annotated onsets)"
            onsets = np.arange(1.0, raw.times[-1] - 1.0, 5.0)
        meta = {
            "sfreq": float(raw.info["sfreq"]),
            "ch_names": list(raw.ch_names),
            "events": {"onset": onsets},
        }
        args.channels, args.sfreq = len(raw.ch_names), meta["sfreq"]
        args.minutes = raw.times[-1] / 60.0
    else:
        onset_source = "synthetic event table"
        print(f"synthesising {args.channels} ch x {args.minutes:g} min @ {args.sfreq:g} Hz ...")
        data, meta = synth_eeg(
            n_channels=args.channels, duration_s=args.minutes * 60, sfreq=args.sfreq, n_events=200
        )
        info = mne.create_info(meta["ch_names"], meta["sfreq"], meta["ch_types"])
        raw = mne.io.RawArray(data, info, verbose="ERROR")
        raw.set_annotations(
            mne.Annotations(
                meta["events"]["onset"], meta["events"]["duration"], meta["events"]["label"]
            ),
            verbose="ERROR",
        )
    n_samples = data.shape[1]
    raw_bytes = data.shape[0] * n_samples * 2

    paths = {
        "zarrfm": os.path.join(tmp, "study.zarr"),
        "edf": str(Path(args.edf).resolve()) if args.native_edf else os.path.join(tmp, "study.edf"),
        "fif": os.path.join(tmp, "study_raw.fif"),
    }

    print("writing ...")
    write_s = {}
    t = time.perf_counter()
    zarrfm.from_mne(raw, paths["zarrfm"])
    write_s["zarrfm"] = time.perf_counter() - t

    t = time.perf_counter()
    if not args.native_edf:
        mne.export.export_raw(paths["edf"], raw, fmt="edf", overwrite=True, verbose="ERROR")
    write_s["edf"] = 0.0 if args.native_edf else time.perf_counter() - t

    t = time.perf_counter()
    try:
        raw.save(paths["fif"], overwrite=True, verbose="ERROR")
    except RuntimeError as e:
        if "meas_date" not in str(e):
            raise
        print("  note: FIF cannot store this meas_date; saving the FIF copy without it")
        raw.copy().set_meas_date(None).save(paths["fif"], overwrite=True, verbose="ERROR")
    write_s["fif"] = time.perf_counter() - t

    rng = np.random.default_rng(0)
    window_samples = int(round(10 * args.sfreq))
    if n_samples < window_samples:
        raise ValueError("recording is shorter than the ten-second benchmark window")
    starts = rng.integers(0, n_samples - window_samples + 1, size=8)
    width = int(round(.8 * args.sfreq))
    candidates = np.asarray(meta["events"]["onset"])
    epoch_starts = np.round((candidates - .2) * args.sfreq).astype(int)
    valid = (epoch_starts >= 0) & (epoch_starts + width <= n_samples)
    onsets = candidates[valid][:50]
    epoch_starts = epoch_starts[valid][:50]
    if not len(onsets):
        raise ValueError("no complete event epochs available")
    pick = meta["ch_names"][len(meta["ch_names"]) // 2]

    def zarr_open():
        return zarrfm.ZarrFMReader(paths["zarrfm"])

    def edf_open():
        reader = mne.io.read_raw_edf(paths["edf"], preload=False, verbose="ERROR",
                                   **({"include": args.picks} if args.native_edf and args.picks else {}))
        if args.native_edf and args.annotations:
            reader.set_annotations(mne.read_annotations(args.annotations))
        return reader

    def fif_open():
        return mne.io.read_raw_fif(paths["fif"], preload=False, verbose="ERROR")

    zr, er, fr = zarr_open(), edf_open(), fif_open()

    def mne_window(r):
        return [r.get_data(start=int(i), stop=int(i) + window_samples) for i in starts]

    def mne_channel(r):
        i = r.ch_names.index(pick)
        return r.get_data(picks=[i], start=0, stop=n_samples)

    def mne_epochs(r):
        return np.stack([r.get_data(start=int(i), stop=int(i) + width) for i in epoch_starts])

    benches = {
        "open store": (lambda: zarr_open(), lambda: edf_open(), lambda: fif_open()),
        "8 x 10 s window (all ch)": (
            lambda: [zr.read_samples("eeg", int(i), int(i) + window_samples, dtype=np.float64) for i in starts],
            lambda: mne_window(er),
            lambda: mne_window(fr),
        ),
        "1 channel, full duration": (
            lambda: zr.get_data("eeg", picks=[pick], dtype=np.float64),
            lambda: mne_channel(er),
            lambda: mne_channel(fr),
        ),
        f"{len(onsets)} epochs (-0.2..0.6 s)": (
            lambda: zr.epochs(onsets, "eeg", -0.2, 0.6, dtype=np.float64),
            lambda: mne_epochs(er),
            lambda: mne_epochs(fr),
        ),
        "full read": (
            lambda: zr.get_data("eeg", dtype=np.float64),
            lambda: er.get_data(),
            lambda: fr.get_data(),
        ),
    }

    print(f"\nuncompressed int16 signal: {raw_bytes / 1e6:.1f} MB")
    print(f"latencies are medians of {args.repeats} warm runs; float64 output in every format\n")
    print(f"{'':28} {'zarrfm':>14} {'edf':>14} {'fif':>14}")
    print("-" * 72)

    row = {}
    for fmt in ("zarrfm", "edf", "fif"):
        row[fmt] = _size(paths[fmt])
    print(
        f"{'size on disk':28} "
        + " ".join(f"{row[f][0] / 1e6:>11.1f} MB" for f in ("zarrfm", "edf", "fif"))
    )
    print(
        f"{'vs raw int16':28} "
        + " ".join(f"{raw_bytes / row[f][0]:>13.2f}x" for f in ("zarrfm", "edf", "fif"))
    )
    print(
        f"{'files':28} "
        + " ".join(f"{row[f][1]:>14d}" for f in ("zarrfm", "edf", "fif"))
    )
    print(
        f"{'write time':28} "
        + " ".join(f"{write_s[f]:>12.2f} s" for f in ("zarrfm", "edf", "fif"))
    )
    print("-" * 72)

    def _rel(other: float, ours: float) -> str:
        if ours <= 0 or other <= 0:
            return "-"
        r = other / ours
        return f"{r:.1f}x faster" if r >= 1 else f"{1 / r:.1f}x slower"

    measurements, timing_rows = {}, []
    for label, functions in benches.items():
        measurements[label] = {}
        for fmt, fn in zip(("zarrfm", "edf", "fif"), functions):
            samples = _measure(fn, args.repeats, close=label == "open store")
            measurements[label][fmt] = {"samples_s": samples, "median_s": float(np.median(samples))}
            timing_rows.append({"operation": label, "format": fmt, "repeats": args.repeats,
                "median_ms": np.median(samples) * 1000,
                "min_ms": min(samples) * 1000, "max_ms": max(samples) * 1000})
        zt, et, ft = [measurements[label][fmt]["median_s"] for fmt in ("zarrfm", "edf", "fif")]
        print(
            f"{label:28} {zt * 1e3:>11.1f} ms {et * 1e3:>11.1f} ms {ft * 1e3:>11.1f} ms"
            f"   zarrfm is {_rel(et, zt)} than edf, {_rel(ft, zt)} than fif"
        )

    # Validate shapes and real values outside the timed regions. EDF export may
    # requantize and FIF uses float32 storage; report those errors explicitly.
    fidelity = {}
    expected_windows = [data[:, int(i):int(i) + window_samples] for i in starts]
    expected_epochs = np.stack([data[:, int(i):int(i) + width] for i in epoch_starts])
    for fmt, windows_fn, epochs_fn in zip(("zarrfm", "edf", "fif"),
            benches["8 x 10 s window (all ch)"], benches[f"{len(onsets)} epochs (-0.2..0.6 s)"]):
        windows, epochs = windows_fn(), epochs_fn()
        assert epochs.dtype == np.float64 and epochs.shape == expected_epochs.shape
        errors = [float(np.max(np.abs(epochs - expected_epochs)))]
        for actual, expected in zip(windows, expected_windows):
            assert actual.dtype == np.float64 and actual.shape == expected.shape
            assert np.isfinite(actual).all()
            errors.append(float(np.max(np.abs(actual - expected))))
        if fmt == "zarrfm":
            tolerance = .51 * float(np.max(np.abs(zr._stream("eeg")["gain"]))) + 1e-15
        elif fmt == "edf":
            extras = er._raw_extras[0]
            tolerance = .51 * float(np.max(np.abs(extras["cal"] * extras["units"]))) + 1e-12
        else:
            tolerance = float(2 * float(np.max(np.abs(data))) * np.finfo(np.float32).eps + 1e-15)
        assert max(errors) <= tolerance, f"{fmt}: values exceed its storage-precision tolerance"
        fidelity[fmt] = {"max_abs_error_vs_source_V": max(errors), "matching_shapes": True,
                         "storage_precision_tolerance_V": tolerance, "within_tolerance": True,
                         "output_dtype": "float64"}
    print("\nfidelity on tested windows/epochs:", json.dumps(fidelity))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"generated_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
            "source_edf": str(Path(args.edf).resolve()) if args.edf else None,
            "source_sha256": hashlib.sha256(Path(args.edf).read_bytes()).hexdigest() if args.edf else None,
            "annotations": str(Path(args.annotations).resolve()) if args.annotations else None,
            "annotations_sha256": hashlib.sha256(Path(args.annotations).read_bytes()).hexdigest() if args.annotations else None,
            "epoch_onset_source": onset_source,
            "channels": meta["ch_names"], "sfreq": args.sfreq, "n_samples": n_samples,
            "duration_s": n_samples / args.sfreq, "repeats": args.repeats,
            "cache": "warm page cache; one unmeasured warmup call per operation/format",
            "output_dtype": "float64", "seed": 0, "window_start_samples": starts.tolist(),
            "window_samples": window_samples, "epoch_onsets_s": onsets.tolist(),
            "epoch_start_samples": epoch_starts.tolist(), "epoch_samples": width,
            "epoch_definition_s": [-.2,.6], "timings": measurements, "fidelity": fidelity,
            "write_s": write_s, "storage_bytes": {k:v[0] for k,v in row.items()},
            "stores": paths, "stores_kept": args.keep,
            "edf_comparator": "original EDF with selected native-rate channels and paired annotations" if args.native_edf else "EDF re-export of the same selected channels, as in bench.py",
            "native_edf": args.native_edf,
            "scope": "one recording; format latency, not corpus-wide or training throughput",
            "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "versions": {"python": sys.version, "numpy": np.__version__, "mne": mne.__version__},
            "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        with args.output.with_suffix(".csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(timing_rows[0]))
            writer.writeheader(); writer.writerows(timing_rows)
        print("Saved:", args.output, args.output.with_suffix(".csv"))
    er.close(); fr.close(); raw.close()
    if args.keep:
        print(f"stores kept in {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
