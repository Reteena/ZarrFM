from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import shutil
import sys
import tempfile
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import zarr
from zarr.storage import LocalStore, WrapperStore

import zarrfm
from zarrfm import canonical
from zarrfm.dataset import DatasetReader, DatasetWriter
from zarrfm.sampler import WindowSampler
from zarrfm.synth import synth_eeg

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_results", "ml.json")
CHANNELS, SFREQ, MINUTES, N_REC = 64, 256.0, 10.0, 4
WINDOW_S = 4.0


def _median(fn, repeat=3, warm=1):
    for _ in range(warm):
        fn()
    t = []
    for _ in range(repeat):
        s = time.perf_counter()
        fn()
        t.append(time.perf_counter() - s)
    return float(np.median(t))


SAVE_SUFFIX = ""


def _save(exp: str, payload) -> None:
    exp = exp + SAVE_SUFFIX
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    data = {}
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            data = json.load(f)
    data[exp] = payload
    with open(RESULTS, "w") as f:
        json.dump(data, f, indent=1)


def _table(headers, rows):
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
    print()


def _scalp_names(n):
    names = [x for x, t in zip(canonical.canonical_names(), canonical.canonical_types())
             if t == "eeg"][1:]
    return names[:n] + [f"EXG{i}" for i in range(max(0, n - len(names)))]


def load_edf_signals(paths):
    global CHANNELS, SFREQ, MINUTES, N_REC
    import mne

    out = []
    for path in paths:
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        data = raw.get_data()
        onsets = np.asarray(raw.annotations.onset, dtype=np.float64) - raw.first_time
        if onsets.size < 20:
            onsets = np.arange(1.0, raw.times[-1] - 5.0, 5.0)
        meta = {"ch_names": list(raw.ch_names), "sfreq": float(raw.info["sfreq"]),
                "events": {"onset": onsets, "duration": np.zeros(onsets.size),
                           "label": ["event"] * onsets.size},
                "raw": raw, "edf_path": os.path.abspath(path)}
        out.append((data, meta))
    CHANNELS = out[0][0].shape[0]
    SFREQ = out[0][1]["sfreq"]
    MINUTES = out[0][0].shape[1] / SFREQ / 60.0
    N_REC = len(out)
    for data, meta in out:
        if data.shape[0] != CHANNELS or meta["sfreq"] != SFREQ:
            raise ValueError("all --edf files must share channel count and sampling rate")
    return out


def make_signals(n_rec=N_REC, minutes=MINUTES, channels=CHANNELS, sfreq=SFREQ):
    out = []
    for i in range(n_rec):
        data, meta = synth_eeg(n_channels=channels, duration_s=minutes * 60, sfreq=sfreq,
                               seed=i, n_events=int(minutes * 20))
        out.append((data, meta))
    return out


def build_corpus(path, signals, chunk_samples=None, sfreq=None):
    sfreq = SFREQ if sfreq is None else sfreq
    with DatasetWriter(path, name="bench", overwrite=True) as d:
        for i, (data, meta) in enumerate(signals):
            names = meta.get("ch_names") or _scalp_names(data.shape[0])
            kw = {"chunk_samples": chunk_samples} if chunk_samples else {}
            d.add_recording_from_array(
                f"rec-{i:03d}", data, sfreq=sfreq, subject=f"sub-{i // 2:03d}",
                session=f"ses-{i % 2}", ch_names=names, events=meta["events"], **kw,
            )
        d.assign_splits({"train": 1.0}, level="subject")
    return path


def bench_metadata(tmp, n_rec=200, latencies=(0.0, 0.001, 0.010)):
    print(f"## metadata: {n_rec} recordings, 3 representations, injected per-request latency\n")
    K = canonical.n_canonical()
    names = [f"EEG {n.upper()}-REF" for n in _scalp_names(19)]
    root = os.path.join(tmp, "bids")
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(0)

    splits = {}
    n_files = 0
    for i in range(n_rec):
        sub, ses = f"sub-{i // 2:03d}", f"ses-{i % 2}"
        d = os.path.join(root, sub, ses, "eeg")
        os.makedirs(d, exist_ok=True)
        stem = os.path.join(d, f"{sub}_{ses}_task-rest")
        bad = set(rng.choice(names, size=rng.integers(0, 3), replace=False).tolist())
        with open(stem + "_eeg.json", "w") as f:
            json.dump({"SamplingFrequency": SFREQ, "RecordingDuration": 600.0,
                       "EEGChannelCount": 19, "EEGReference": "REF", "PowerLineFrequency": 50,
                       "TaskName": "rest", "SoftwareFilters": "n/a"}, f)
        with open(stem + "_channels.tsv", "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["name", "type", "units", "status", "status_description"])
            for n in names:
                w.writerow([n, "EEG", "uV", "bad" if n in bad else "good", "n/a"])
        with open(stem + "_events.tsv", "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["onset", "duration", "trial_type", "value"])
            for k in range(200):
                w.writerow([f"{3.0 * k + 1:.4f}", "0", "target" if k % 3 else "standard", k % 3])
        splits[f"{sub}_{ses}"] = "train" if i % 5 else "test"
        n_files += 3
    with open(os.path.join(root, "splits.json"), "w") as f:
        json.dump(splits, f)

    def open_delayed(p, lat, mode="r", **kw):
        if lat:
            time.sleep(lat)
        return open(p, mode, **kw)

    def read_sidecars(lat):
        with open_delayed(os.path.join(root, "splits.json"), lat) as f:
            spl = json.load(f)
        present = np.zeros((n_rec, K), dtype=bool)
        sfreq = np.zeros(n_rec)
        n_samples = np.zeros(n_rec, dtype=np.int64)
        split = np.zeros(n_rec, dtype=np.int8)
        vocab = {}
        for i in range(n_rec):
            sub, ses = f"sub-{i // 2:03d}", f"ses-{i % 2}"
            stem = os.path.join(root, sub, ses, "eeg", f"{sub}_{ses}_task-rest")
            with open_delayed(stem + "_eeg.json", lat) as f:
                meta = json.load(f)
            sfreq[i] = meta["SamplingFrequency"]
            n_samples[i] = int(meta["RecordingDuration"] * sfreq[i])
            with open_delayed(stem + "_channels.tsv", lat, newline="") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))
            ids, _ = canonical.map_channels([r["name"] for r in rows],
                                            [r["type"].lower() for r in rows])
            good = np.array([r["status"] == "good" for r in rows])
            present[i, ids[(ids >= 0) & good]] = True
            with open_delayed(stem + "_events.tsv", lat, newline="") as f:
                ev = list(csv.DictReader(f, delimiter="\t"))
            for r in ev:
                vocab.setdefault(r["trial_type"], len(vocab))
            split[i] = 0 if spl[f"{sub}_{ses}"] == "train" else 1
        return present, sfreq, n_samples, split

    dpath = os.path.join(tmp, "meta.zarr")
    with DatasetWriter(dpath, overwrite=True) as d:
        for i in range(n_rec):
            data = rng.standard_normal((19, int(2 * SFREQ))) * 1e-5
            d.add_recording_from_array(
                f"sub-{i // 2:03d}_ses-{i % 2}", data, sfreq=SFREQ, subject=f"sub-{i // 2:03d}",
                session=f"ses-{i % 2}", ch_names=names,
                bad=rng.choice(names, size=rng.integers(0, 3), replace=False).tolist(),
                events={"onset": 3.0 * np.arange(200) + 1, "duration": np.zeros(200),
                        "label": ["target" if k % 3 else "standard" for k in range(200)]},
            )
        d.assign_splits({"train": 0.8, "test": 0.2})
    zarr_files = sum(len(f) for _, _, f in os.walk(dpath))

    def open_ds(lat):
        store = LatencyStore(LocalStore(dpath, read_only=True), lat)
        return DatasetReader(zarr.open_consolidated(store=store, mode="r")), store

    def read_per_recording(lat):
        ds, store = open_ds(lat)
        present = np.zeros((n_rec, K), dtype=bool)
        sfreq = np.zeros(n_rec)
        n_samples = np.zeros(n_rec, dtype=np.int64)
        for i in range(n_rec):
            r = ds.recording(i)
            ids = r.canonical_ids()
            good = ~r.bad_mask()
            present[i, ids[(ids >= 0) & good]] = True
            sfreq[i] = r.sfreq()
            n_samples[i] = r.n_samples()
            r.events()
        return store.requests

    def read_index(lat):
        ds, store = open_ds(lat)
        _ = ds.canonical_present, ds.index["sfreq"], ds.index["n_samples"], ds.index["split"]
        return store.requests

    rows, payload = [], {}
    for label, fn, files in (
        ("BIDS sidecars (json + 2 tsv, stdlib parsers)", read_sidecars, n_files + 1),
        ("Zarr, per-recording channel tables", read_per_recording, zarr_files),
        ("Zarr, dataset index/ tensors", read_index, zarr_files),
    ):
        row = [label, files]
        payload[label] = {"files": files}
        for lat in latencies:
            t = _median(lambda: fn(lat), repeat=3 if lat < 0.01 else 1)
            req = fn(lat) if fn is not read_sidecars else n_files + 1
            row.append(f"{t * 1e3:.0f} ms ({req} req)")
            payload[label][f"lat_{lat * 1e3:g}ms"] = {"total_s": t, "requests": int(req),
                                                      "per_recording_us": t / n_rec * 1e6}
        rows.append(row)
    _table(["representation", "files in store"] + [f"{l * 1e3:g} ms/request" for l in latencies], rows)
    _save("metadata", {"n_rec": n_rec, "latencies_ms": [l * 1e3 for l in latencies], "rows": payload})


def _run_sampler(ds, **kw):
    s = WindowSampler(ds, "train", window_s=WINDOW_S, with_events=False, **kw)
    n = 0
    for b in s.batches(32):
        n += b["signals"].shape[0]
    return n, s.stats["reads"]


def bench_chunks(tmp, signals, chunk_list=(512, 1024, 2048, 4096, 8192, 16384, 32768)):
    print(f"## chunks: {CHANNELS} ch, {SFREQ:g} Hz, {N_REC} x {MINUTES:g} min, "
          f"{WINDOW_S:g} s windows\n")
    rows, payload = [], []
    win_bytes = CHANNELS * int(WINDOW_S * SFREQ) * 4
    for chunk in chunk_list:
        path = build_corpus(os.path.join(tmp, f"chunk{chunk}.zarr"), signals, chunk_samples=chunk)
        ds = DatasetReader(path)
        res = {}
        for mode in (False, "block", "window"):
            n, reads = _run_sampler(ds, shuffle=mode)
            t = _median(lambda: _run_sampler(ds, shuffle=mode), repeat=3)
            res[str(mode)] = {"windows_per_s": n / t, "reads": reads, "mb_per_s": n * win_bytes / t / 1e6}
        kb = CHANNELS * chunk * 2 // 1024
        rows.append([chunk, f"{kb} KiB", f"{chunk / SFREQ:.0f} s"]
                    + [f"{res[m]['windows_per_s']:.0f}" for m in ("False", "block", "window")]
                    + [res["block"]["reads"], res["window"]["reads"]])
        payload.append({"chunk_samples": chunk, "chunk_kib": kb, **res})
        shutil.rmtree(path, ignore_errors=True)
    _table(["chunk (samples)", "bytes", "duration", "sequential win/s", "block-shuffle win/s",
            "random win/s", "reads (block)", "reads (random)"], rows)
    _save("chunks", payload)


def bench_formats(tmp, signals, n_windows=512):
    print(f"## formats: {n_windows} random {WINDOW_S:g} s windows, all channels, 1 process\n")
    import mne

    data, meta = signals[0]
    n_samples = data.shape[1]
    W = int(WINDOW_S * SFREQ)
    rng = np.random.default_rng(0)
    starts = rng.integers(0, n_samples - W, size=n_windows)
    names = meta.get("ch_names") or _scalp_names(CHANNELS)
    real = meta.get("raw") is not None
    if real:
        raw = meta["raw"]
    else:
        info = mne.create_info(names, SFREQ, ["eeg"] * CHANNELS)
        raw = mne.io.RawArray(data, info, verbose="ERROR")

    paths = {k: os.path.join(tmp, f"fmt.{k}") for k in ("edf", "fif", "npy", "h5", "zarr")}
    if real:
        paths["edf"] = meta["edf_path"]
    else:
        mne.export.export_raw(paths["edf"], raw, fmt="edf", overwrite=True, verbose="ERROR")
    try:
        raw.save(paths["fif"], overwrite=True, verbose="ERROR")
    except RuntimeError as e:
        if "meas_date" not in str(e):
            raise
        raw.copy().set_meas_date(None).save(paths["fif"], overwrite=True, verbose="ERROR")
    np.save(paths["npy"], data.astype(np.float32))
    try:
        import h5py
    except ImportError:
        h5py = None
    if h5py is not None:
        with h5py.File(paths["h5"], "w") as f:
            f.create_dataset("eeg", data=data.astype(np.float32), chunks=(CHANNELS, 1024))
        with h5py.File(paths["h5"] + "gz", "w") as f:
            f.create_dataset("eeg", data=data.astype(np.float32), chunks=(CHANNELS, 1024),
                             compression="gzip", compression_opts=4)
    if real:
        zarrfm.from_mne(raw, paths["zarr"])
    else:
        zarrfm.from_array(data, paths["zarr"], sfreq=SFREQ, ch_names=names)

    def size(p):
        if os.path.isfile(p):
            return os.path.getsize(p)
        return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs)

    readers = {}
    edf = mne.io.read_raw_edf(paths["edf"], preload=False, verbose="ERROR")
    fif = mne.io.read_raw_fif(paths["fif"], preload=False, verbose="ERROR")
    npy = np.load(paths["npy"], mmap_mode="r")
    zr = zarrfm.ZarrFMReader(paths["zarr"])
    readers["EDF (MNE, preload=False)"] = (
        lambda: [edf.get_data(start=int(s), stop=int(s) + W) for s in starts], "edf")
    readers["FIF (MNE, preload=False)"] = (
        lambda: [fif.get_data(start=int(s), stop=int(s) + W) for s in starts], "fif")
    readers["NPY float32 memmap"] = (
        lambda: [np.array(npy[:, int(s):int(s) + W]) for s in starts], "npy")
    if h5py is not None:
        h5 = h5py.File(paths["h5"], "r")["eeg"]
        h5gz = h5py.File(paths["h5"] + "gz", "r")["eeg"]
        readers["HDF5 float32, chunked"] = (lambda: [h5[:, int(s):int(s) + W] for s in starts], "h5")
        readers["HDF5 float32, gzip-4"] = (lambda: [h5gz[:, int(s):int(s) + W] for s in starts], "h5gz")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        pa = pq = None
    if pa is not None:
        paths["parquet"] = os.path.join(tmp, "fmt.parquet")
        n_grid = n_samples // W
        rows_per_group = 16
        blobs = [data[:, i * W:(i + 1) * W].astype(np.float32).tobytes() for i in range(n_grid)]
        table = pa.table({"start": pa.array(np.arange(n_grid) * W, type=pa.int64()),
                          "x": pa.array(blobs, type=pa.binary())})
        pq.write_table(table, paths["parquet"], compression="zstd", row_group_size=rows_per_group)
        pf = pq.ParquetFile(paths["parquet"])
        grid_idx = np.minimum(starts // W, n_grid - 1)

        def read_parquet():
            out = []
            for g in grid_idx:
                rg = pf.read_row_group(int(g) // rows_per_group, columns=["x"])
                blob = rg.column("x")[int(g) % rows_per_group].as_py()
                out.append(np.frombuffer(blob, dtype=np.float32).reshape(CHANNELS, W))
            return out

        readers["Parquet, materialised 4 s windows (zstd)"] = (read_parquet, "parquet")
    readers["ZarrFM, one read per window"] = (
        lambda: [zr.read_samples("eeg", int(s), int(s) + W) for s in starts], "zarr")
    readers["ZarrFM, block-merged (epochs)"] = (
        lambda: zr.epochs(starts / SFREQ, "eeg", 0.0, WINDOW_S), "zarr")

    win_bytes = CHANNELS * W * 4
    rows, payload = [], {}
    for label, (fn, key) in readers.items():
        t = _median(fn, repeat=3)
        p = paths[key] if key != "h5gz" else paths["h5"] + "gz"
        rows.append([label, f"{size(p) / 1e6:.1f} MB", f"{n_windows / t:.0f}",
                     f"{n_windows * win_bytes / t / 1e6:.0f}", f"{t / n_windows * 1e3:.2f} ms"])
        payload[label] = {"size_mb": size(p) / 1e6, "windows_per_s": n_windows / t,
                          "mb_per_s": n_windows * win_bytes / t / 1e6}
    _table(["format", "size", "windows/s", "MB/s (float32 out)", "latency/window"], rows)
    _save("formats", payload)


class LatencyStore(WrapperStore):
    def __init__(self, store, latency_s: float):
        super().__init__(store)
        self.latency_s = latency_s
        self.requests = 0

    async def get(self, key, prototype, byte_range=None):
        self.requests += 1
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        return await self._store.get(key, prototype, byte_range)


def bench_latency(tmp, signals, latencies=(0.0, 0.001, 0.010, 0.030)):
    print(f"## latency: injected per-request delay (simulated backends), "
          f"{WINDOW_S:g} s windows, default chunks\n")
    path = build_corpus(os.path.join(tmp, "lat.zarr"), signals)
    rows, payload = [], []
    for lat in latencies:
        store = LatencyStore(LocalStore(path, read_only=True), lat)
        root = zarr.open_consolidated(store=store, mode="r")
        ds = DatasetReader(root)
        res = {}
        for mode in ("block", "window"):
            store.requests = 0
            n, reads = _run_sampler(ds, shuffle=mode)
            req = store.requests
            t = _median(lambda: _run_sampler(ds, shuffle=mode), repeat=2)
            res[mode] = {"windows_per_s": n / t, "requests": req}
        rows.append([f"{lat * 1e3:.0f} ms", f"{res['block']['windows_per_s']:.0f}",
                     res["block"]["requests"], f"{res['window']['windows_per_s']:.0f}",
                     res["window"]["requests"]])
        payload.append({"latency_ms": lat * 1e3, **res})
    _table(["per-request latency", "block-shuffle win/s", "requests", "random win/s", "requests"], rows)
    _save("latency", payload)


def bench_workers(tmp, signals, worker_counts=(0, 1, 2, 4, 8)):
    try:
        import torch
        from zarrfm.sampler import torch_dataset
    except ImportError:
        print("## workers: torch not installed, skipped\n")
        return
    print(f"## workers: torch DataLoader, batch 32, {WINDOW_S:g} s windows, all recordings; "
          "canonical space (112 slots) vs physical space (64 ch)\n")
    path = build_corpus(os.path.join(tmp, "workers.zarr"), signals)
    rows, payload = [], []
    W = int(WINDOW_S * SFREQ)
    for nw in worker_counts:
        res = {}
        for space, K in (("canonical", canonical.n_canonical()), ("physical", CHANNELS)):
            ds = torch_dataset(path, batch_size=32, split=None, window_s=WINDOW_S,
                               with_events=False, canonical_space=(space == "canonical"))
            loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=nw,
                                                 persistent_workers=nw > 0)

            def epoch():
                n = 0
                for b in loader:
                    n += b["signals"].shape[0]
                return n

            n = epoch()
            t = _median(epoch, repeat=2)
            res[space] = {"windows_per_s": n / t, "mb_per_s": n / t * K * W * 4 / 1e6}
        rows.append([nw, f"{res['canonical']['windows_per_s']:.0f}", f"{res['canonical']['mb_per_s']:.0f}",
                     f"{res['physical']['windows_per_s']:.0f}", f"{res['physical']['mb_per_s']:.0f}"])
        payload.append({"num_workers": nw, **res})
    _table(["num_workers", "canonical win/s", "MB/s", "physical win/s", "MB/s"], rows)
    _save("workers", payload)


def bench_gpu(tmp, signals, batch=64, n_batches=50):
    try:
        import torch
    except ImportError:
        print("## gpu: torch not installed, skipped\n")
        return
    if not torch.cuda.is_available():
        print("## gpu: no CUDA device visible, skipped\n")
        return
    print(f"## gpu: {torch.cuda.get_device_name(0)}, batch {batch} x ({canonical.n_canonical()}, "
          f"{int(WINDOW_S * SFREQ)}) float32\n")
    path = build_corpus(os.path.join(tmp, "gpu.zarr"), signals)
    ds = DatasetReader(path)
    dev = torch.device("cuda:0")
    K, W = canonical.n_canonical(), int(WINDOW_S * SFREQ)
    batch_bytes = batch * K * W * 4

    def stream_batches(pin: bool, mask_mult: bool):
        s = WindowSampler(ds, "train", window_s=WINDOW_S, with_events=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n = 0
        for b in s.batches(batch):
            x = torch.from_numpy(b["signals"])
            m = torch.from_numpy(b["mask"])
            if pin:
                x, m = x.pin_memory(), m.pin_memory()
            x = x.to(dev, non_blocking=pin)
            m = m.to(dev, non_blocking=pin)
            if mask_mult:
                x = x * m[:, :, None]
            n += 1
            if n >= n_batches:
                break
        torch.cuda.synchronize()
        return n, time.perf_counter() - t0

    rows, payload = [], {}
    for label, pin, mm in (("pageable copy", False, False), ("pinned, non_blocking", True, False),
                           ("pinned + X ⊙ mask on device", True, True)):
        stream_batches(pin, mm)
        n, t = stream_batches(pin, mm)
        rows.append([label, f"{n * batch / t:.0f}", f"{n * batch_bytes / t / 1e9:.2f} GB/s"])
        payload[label] = {"windows_per_s": n * batch / t, "gb_per_s": n * batch_bytes / t / 1e9}
    x = torch.empty((batch, K, W), dtype=torch.float32).pin_memory()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_batches):
        x.to(dev, non_blocking=True)
    torch.cuda.synchronize()
    t = time.perf_counter() - t0
    rows.append(["(H2D copy only, pinned)", f"{n_batches * batch / t:.0f}", f"{n_batches * batch_bytes / t / 1e9:.2f} GB/s"])
    payload["h2d_only"] = {"windows_per_s": n_batches * batch / t, "gb_per_s": n_batches * batch_bytes / t / 1e9}
    _table(["path", "windows/s", "throughput"], rows)
    _save("gpu", payload)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="*", default=["all"],
                    choices=["all", "metadata", "chunks", "formats", "latency", "workers", "gpu"])
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--edf", nargs="*", help="real EDF recording(s) instead of synthetic signals")
    args = ap.parse_args()
    exps = {"metadata", "chunks", "formats", "latency", "workers", "gpu"} if "all" in args.exp else set(args.exp)

    tmp = tempfile.mkdtemp(prefix="zarrfm-ml-")
    print(f"# bench_ml  (python {sys.version.split()[0]}, zarr {zarr.__version__}, "
          f"numpy {np.__version__}, {os.cpu_count()} cpus)\n")
    signals = None
    if args.edf:
        global SAVE_SUFFIX
        signals = load_edf_signals(args.edf)
        SAVE_SUFFIX = ":real:" + os.path.splitext(os.path.basename(args.edf[0]))[0]
        print(f"real recording(s): {N_REC} x {CHANNELS} ch x {MINUTES:.1f} min @ {SFREQ:g} Hz\n")
    elif exps - {"metadata"}:
        print(f"synthesising {N_REC} x {CHANNELS} ch x {MINUTES:g} min @ {SFREQ:g} Hz ...\n")
        signals = make_signals()
    if "metadata" in exps:
        bench_metadata(tmp)
    if "chunks" in exps:
        bench_chunks(tmp, signals)
    if "formats" in exps:
        bench_formats(tmp, signals)
    if "latency" in exps:
        bench_latency(tmp, signals)
    if "workers" in exps:
        bench_workers(tmp, signals)
    if "gpu" in exps:
        bench_gpu(tmp, signals)
    if args.keep:
        print(f"stores kept in {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"results appended to {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
