from __future__ import annotations

import argparse
import glob
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

import zarrfm
from zarrfm.reader import ZarrFMReader


STEPS_UV = (0.01, 0.03, 0.1)


def _entropy(codes: np.ndarray) -> float:
    h = np.bincount(codes.astype(np.int32).ravel() + 32768, minlength=65536)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())


def _lattice_step(x: np.ndarray) -> float | None:
    u = np.unique(x)
    if u.size < 3:
        return None
    d = np.diff(u)
    step = d[d > 0].min()
    levels = (u[-1] - u[0]) / step
    if levels > 2 ** 24:
        return None
    k = (u - u[0]) / step
    return float(step) if np.allclose(k, np.rint(k), atol=1e-3) else None


def _tree_bytes(path: str) -> int:
    return sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs)


def measure_run(raw, scratch: str) -> dict:
    from numcodecs import Blosc, Zstd

    x = raw.get_data()
    C, N = x.shape
    types = raw.get_channel_types()
    sig = np.array([t != "stim" for t in types])
    rec = {"n_channels": C, "n_samples": N, "sfreq": float(raw.info["sfreq"]),
           "duration_s": N / float(raw.info["sfreq"]), "types": dict(zip(*np.unique(types, return_counts=True))),
           "float64_bytes": x.nbytes, "float32_bytes": x.nbytes // 2,
           "n_annotations": len(raw.annotations)}
    rec["types"] = {k: int(v) for k, v in rec["types"].items()}
    steps = [_lattice_step(x[i]) for i in range(C)]
    rec["lattice_channels"] = int(sum(s is not None for s in steps))
    rec["lattice_step_uV"] = [None if s is None else s * 1e6 for s in steps]
    rec["rms_uV"] = float(np.sqrt(np.mean(x[sig] ** 2)) * 1e6)
    rec["range_uV"] = [float(x[sig].min() * 1e6), float(x[sig].max() * 1e6)]

    variants = [("int16_fast", {"storage": "int16", "compression": "fast"}),
                ("int16_max", {"storage": "int16", "compression": "max"}),
                ("float32_fast", {"storage": "float32", "compression": "fast"})]
    variants += [(f"int16_step_{s}uV", {"storage": "int16", "compression": "fast", "step": s * 1e-6})
                 for s in STEPS_UV]
    for name, kw in variants:
        store = os.path.join(scratch, name + ".zarr")
        t = time.perf_counter()
        zarrfm.from_mne(raw, store, **kw)
        t_write = time.perf_counter() - t
        r = ZarrFMReader(store)
        st = r._stream(None)
        t = time.perf_counter()
        y = r.get_data(dtype=np.float64)
        t_read = time.perf_counter() - t
        err = np.abs(y - x)
        out = {"store_bytes": _tree_bytes(store), "t_write": round(t_write, 3), "t_read": round(t_read, 4),
               "max_abs_err_uV": float(err.max() * 1e6),
               "max_abs_err_signal_uV": float(err[sig].max() * 1e6),
               "n_clipped": int(st["attrs"].get("n_clipped", 0)),
               "exact": bool(np.array_equal(y, x))}
        if kw["storage"] == "int16":
            codes = r.get_data(raw=True)
            out["step_uV_median_signal"] = float(np.median(st["gain"][sig]) * 1e6)
            out["step_uV_stim"] = [float(g * 1e6) for g in st["gain"][~sig]] if (~sig).any() else []
            out["entropy_bits"] = _entropy(codes[sig])
            out["used_bits"] = float(np.mean([np.log2(np.ptp(c.astype(np.int32)) + 1) for c in codes[sig]]))
            out["stim_roundtrip_exact"] = bool(np.array_equal(y[~sig], x[~sig])) if (~sig).any() else None
            out["stim_roundtrip_after_rint"] = bool(np.array_equal(np.rint(y[~sig]), x[~sig])) if (~sig).any() else None
        rec[name] = out
        shutil.rmtree(store, ignore_errors=True)

    x32 = x.astype(np.float32)
    rec["float32_lossless"] = bool(np.array_equal(x32.astype(np.float64), x))
    t = time.perf_counter()
    rec["npy32_zstd5_bytes"] = len(Zstd(level=5).encode(np.ascontiguousarray(x32).tobytes()))
    rec["t_zstd5"] = round(time.perf_counter() - t, 3)
    rec["npy32_blosc_shuffle_zstd5_bytes"] = len(Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE).encode(np.ascontiguousarray(x32)))
    rec["npy64_blosc_shuffle_zstd5_bytes"] = len(Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE).encode(np.ascontiguousarray(x)))
    fif = os.path.join(scratch, "run_raw.fif")
    raw.save(fif, overwrite=True, verbose="ERROR")
    rec["fif_bytes"] = os.path.getsize(fif)
    os.remove(fif)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BNCI2014_009")
    ap.add_argument("--subject", type=int, default=1)
    args = ap.parse_args()
    import moabb.datasets as md

    ds = getattr(md, args.dataset)()
    data = ds.get_data(subjects=[args.subject])
    root = os.environ.get("MNE_DATASETS_BNCI_PATH") or os.environ.get("MNE_DATA") or ""
    src_files = sorted(glob.glob(os.path.join(root, "**", "*.mat"), recursive=True)) if root else []
    scratch = tempfile.mkdtemp(prefix="moabb_")
    results = {"dataset": args.dataset, "subject": args.subject, "runs": {},
               "source_files": [(os.path.relpath(p, root), os.path.getsize(p)) for p in src_files]}
    rows = []
    try:
        for sess, runs in data[args.subject].items():
            for run, raw in runs.items():
                rec = measure_run(raw, scratch)
                results["runs"][f"{sess}/{run}"] = rec
                f = rec["int16_fast"]
                rows.append((f"{sess}/{run}", rec["n_channels"], f"{rec['duration_s']:.0f}", rec["lattice_channels"],
                             f"{rec['float32_bytes'] / 1e6:.2f}", f"{rec['fif_bytes'] / 1e6:.2f}",
                             f"{rec['npy32_zstd5_bytes'] / 1e6:.2f}", f"{rec['npy32_blosc_shuffle_zstd5_bytes'] / 1e6:.2f}",
                             f"{rec['float32_fast']['store_bytes'] / 1e6:.2f}",
                             f"{f['store_bytes'] / 1e6:.2f}", f"{rec['int16_max']['store_bytes'] / 1e6:.2f}",
                             f"{f['step_uV_median_signal']:.3f}", f"{f['max_abs_err_signal_uV']:.3f}",
                             f"{rec['rms_uV']:.1f}", f"{f['entropy_bits']:.2f}", f['n_clipped'],
                             *[f"{rec[f'int16_step_{s}uV']['store_bytes'] / 1e6:.2f} ({rec[f'int16_step_{s}uV']['entropy_bits']:.1f} b, err {rec[f'int16_step_{s}uV']['max_abs_err_signal_uV']:.3f})" for s in STEPS_UV]))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    headers = ["session/run", "ch", "s", "lattice ch", "float32 MB", "FIF MB", "f32+zstd5", "f32+shuffle+zstd5",
               "zarrfm f32", "zarrfm int16 fast", "int16 max", "step uV", "max err uV", "rms uV", "H0 bits", "clipped",
               *[f"int16 step {s} uV" for s in STEPS_UV]]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        print("| " + " | ".join(str(v) for v in r) + " |")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_results",
                       f"moabb_{args.dataset}_s{args.subject}.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
