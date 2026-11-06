"""Matched NPY/ZarrFM input and runtime experiment with official CBraMod.

Uses the reconstruction head, with optional backward passes. No downstream
accuracy is measured. Preprocessing and store creation are excluded from timing.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

from zarrfm import canonical
from zarrfm.dataset import DatasetReader, DatasetWriter, validate_dataset
from zarrfm.foundation import channel_ids, model_patches
from zarrfm.sampler import WindowSampler, collate
from zarrfm.synth import synth_eeg


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_signals(args):
    signals, metadata = [], []
    if args.edf:
        import mne
        for path in args.edf:
            raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
            raw.pick("eeg", exclude="bads")
            original_rate = float(raw.info["sfreq"])
            raw.crop(tmax=min(raw.times[-1], args.seconds), include_tmax=False)
            # Continuous filtering precedes windowing to avoid per-window edges.
            high = min(75., original_rate / 2 - 1.)
            if high <= 0.3:
                raise ValueError("source sampling rate is too low")
            raw.filter(0.3, high, verbose="ERROR")
            if args.notch < original_rate / 2 and args.notch < high:
                raw.notch_filter([args.notch], verbose="ERROR")
            raw.resample(200, verbose="ERROR")
            ids, _ = canonical.map_channels(raw.ch_names, ["eeg"] * len(raw.ch_names))
            names = [canonical.canonical_names()[i] if i >= 0 else None for i in ids]
            if None in names or len(set(names)) != len(names):
                raise ValueError("EDF requires unique mapped scalp electrodes; select a montage first")
            signals.append((raw.get_data().astype(np.float32), names))
            metadata.append({"path": str(Path(path).resolve()), "sha256": digest(path),
                             "source_sfreq": original_rate, "bandpass_hz": [0.3, high],
                             "notch_hz": args.notch if args.notch < high else None,
                             "reference": "source recording reference retained"})
    else:
        for i in range(args.recordings):
            data, meta = synth_eeg(n_channels=19, duration_s=args.seconds,
                                   sfreq=200, seed=args.seed + i, n_events=0)
            signals.append((data.astype(np.float32), meta["ch_names"]))
            metadata.append({"synthetic_seed": args.seed + i})
    common = set(signals[0][1]).intersection(*(set(n) for _, n in signals))
    names = [n for n in signals[0][1] if n in common]
    if not names:
        raise ValueError("recordings have no common EEG electrodes")
    # A fixed montage avoids treating absent electrodes as measured zero EEG.
    signals = [np.ascontiguousarray(x[[n.index(c) for c in names]]) for x, n in signals]
    if any(x.shape[-1] < args.window_seconds * 200 for x in signals):
        raise ValueError("each recording must contain at least one full window")
    return signals, names, metadata


def build_inputs(root, signals, names, args):
    paths, errors = {}, {}
    for i, x in enumerate(signals):
        np.save(root / f"recording-{i}.npy", x)
    for storage in ("float32", "int16"):
        path = root / f"{storage}.zarr"
        with DatasetWriter(str(path), name="CBraMod storage experiment") as writer:
            for i, x in enumerate(signals):
                writer.add_recording_from_array(
                    f"rec-{i:03d}", x, sfreq=200, subject=f"sub-{i:03d}",
                    ch_names=names, unit="V", storage=storage, chunk_samples=2000,
                    provenance={"source": {"kind": "edf" if args.edf else "synthetic"},
                                "pipeline": [{"name": "bench_cbramod", "sfreq": 200}]})
            writer.assign_splits({"train": 1.0}, level="subject", seed=args.seed)
        problems = validate_dataset(str(path))
        if problems:
            raise RuntimeError(f"invalid corpus: {problems}")
        ds = DatasetReader(str(path))
        max_error, bound = 0., 0.
        for i, source in enumerate(signals):
            reader = ds.recording(i)
            got = reader.get_data("eeg", dtype=np.float32)
            max_error = max(max_error, float(np.max(np.abs(got - source))))
            if storage == "int16":
                bound = max(bound, float(reader._stream("eeg")["gain"].max() / 2))
            elif not np.array_equal(got, source):
                raise AssertionError("float32 ZarrFM changed source samples")
        # Float32 decode rounding adds at most a small floating-point allowance.
        if storage == "int16" and max_error > bound + 1e-10:
            raise AssertionError("quantisation error exceeds gain/2")
        paths[storage] = path
        errors[storage] = {"max_abs_error_volts": max_error, "gain_half_volts": bound}
    return paths, errors


def input_factories(root, paths, signals, names, args):
    ids = channel_ids(names)
    arrays = [np.load(root / f"recording-{i}.npy", mmap_mode="r") for i in range(len(signals))]
    window = args.window_seconds * 200

    def npy():
        items = []
        for rec, x in enumerate(arrays):
            for start in range(0, x.shape[-1] - window + 1, window):
                projected = np.zeros((canonical.n_canonical(), window), dtype=np.float32)
                projected[ids] = x[:, start:start + window]
                mask = np.zeros(canonical.n_canonical(), dtype=bool)
                mask[ids] = True
                items.append({"signals": projected, "mask": mask, "recording": rec, "start": start})
                if len(items) == args.batch_size:
                    yield collate(items)
                    items = []
        if items:
            yield collate(items)

    factories = {"npy": npy}
    for storage, path in paths.items():
        ds = DatasetReader(str(path))
        sampler = WindowSampler(ds, "train", window_samples=window,
                                canonical_space=True, shuffle=False, with_events=False)
        factories[f"zarr_{storage}"] = lambda s=sampler: s.batches(args.batch_size)
    return factories


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_pass(factory, model, names, device, args, *, compute):
    model.eval()  # Fixed dropout for matched outputs and backward comparison.
    count, load_s, compute_s = 0, 0., 0.
    iterator = iter(factory())
    sync(device)
    started = time.perf_counter()
    for _ in range(args.max_batches):
        t = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        patches = model_patches(batch, names, sfreq=200, unit="V")
        load_s += time.perf_counter() - t
        if compute:
            t = time.perf_counter()
            x = torch.from_numpy(patches).to(device)
            model.zero_grad(set_to_none=True)
            context = nullcontext() if args.backward else torch.inference_mode()
            with context:
                output = model(x)
                if args.backward:
                    loss = torch.nn.functional.mse_loss(output, x)
                    loss.backward()
            sync(device)
            compute_s += time.perf_counter() - t
        count += patches.shape[0]
    elapsed = time.perf_counter() - started
    if not count:
        raise ValueError("no windows sampled")
    return {"windows": count, "elapsed_s": elapsed, "windows_per_s": count / elapsed,
            "load_and_adapter_s": load_s, "transfer_and_model_s": compute_s}


def compare_outputs(factories, model, names, device, args):
    reference = iter(factories["npy"]())
    others = {k: iter(fn()) for k, fn in factories.items() if k != "npy"}
    report = {k: {"max_input_abs_error": 0., "max_output_abs_error": 0.} for k in others}
    gradients_checked = False
    for _, base in zip(range(args.max_batches), reference):
        patches = model_patches(base, names, sfreq=200)
        x = torch.from_numpy(patches).to(device)
        with torch.inference_mode():
            expected = model(x)
        for key, iterator in others.items():
            batch = next(iterator)
            for field in ("recording", "start"):
                np.testing.assert_array_equal(batch[field], base[field])
            candidate = model_patches(batch, names, sfreq=200)
            if key == "zarr_float32":
                np.testing.assert_array_equal(candidate, patches)
            with torch.inference_mode():
                output = model(torch.from_numpy(candidate).to(device))
            if not torch.isfinite(output).all():
                raise AssertionError("non-finite model output")
            if key == "zarr_float32":
                torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
            report[key]["max_input_abs_error"] = max(report[key]["max_input_abs_error"],
                                                     float(np.max(np.abs(candidate - patches))))
            report[key]["max_output_abs_error"] = max(report[key]["max_output_abs_error"],
                                                      float((output - expected).abs().max()))
        if not gradients_checked:
            model.zero_grad(set_to_none=True)
            torch.nn.functional.mse_loss(model(x), x).backward()
            grads = [p.grad for p in model.parameters() if p.requires_grad]
            if any(g is None or not torch.isfinite(g).all() for g in grads):
                raise AssertionError("missing or non-finite model gradients")
            model.zero_grad(set_to_none=True)
            gradients_checked = True
    return {"backward_gradients_finite": gradients_checked, "comparisons": report}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-repo", type=Path, required=True, help="official CBraMod checkout")
    weights = p.add_mutually_exclusive_group(required=True)
    weights.add_argument("--checkpoint", type=Path)
    weights.add_argument("--random-init", action="store_true", help="explicit architecture-only smoke test")
    p.add_argument("--edf", nargs="+", type=Path)
    p.add_argument("--seconds", type=int, default=64, help="duration per recording")
    p.add_argument("--recordings", type=int, default=2, help="synthetic recordings")
    p.add_argument("--notch", type=int, choices=(50, 60), default=60)
    p.add_argument("--window-seconds", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=8)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--backward", action="store_true", help="time forward + backward, without optimizer updates")
    p.add_argument("--output", type=Path, default=Path("bench_results/cbramod.json"))
    args = p.parse_args()
    if any(getattr(args, k) <= 0 for k in ("seconds", "recordings", "window_seconds", "batch_size", "max_batches", "repeats")):
        p.error("duration, counts and batch sizes must be positive")
    sys.path.insert(0, str(args.model_repo.resolve()))
    from models.cbramod import CBraMod
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model = CBraMod()
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.to(device).eval()
    signals, names, metadata = load_signals(args)
    revision = subprocess.check_output(["git", "-C", str(args.model_repo), "rev-parse", "HEAD"], text=True).strip()
    result = {"experiment": "CBraMod storage/runtime integration; no downstream accuracy",
              "data_kind": "real_edf" if args.edf else "synthetic", "sources": metadata,
              "model_revision": revision, "checkpoint_sha256": digest(args.checkpoint) if args.checkpoint else None,
              "weights": "pretrained" if args.checkpoint else "random_initialization",
              "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "python": platform.python_version(), "torch": torch.__version__,
              "channels": names, "sfreq": 200, "input_unit": "microvolts / 100",
              "window_seconds": args.window_seconds, "batch_size": args.batch_size,
              "max_batches": args.max_batches, "repeats": args.repeats, "seed": args.seed,
              "mode": "eval_forward_backward_no_optimizer" if args.backward else "inference",
              "cache": "warm OS cache; synchronous reads/compute; setup excluded",
              "timings": {}}
    with tempfile.TemporaryDirectory(prefix="zarrfm-cbramod-") as tmp:
        root = Path(tmp)
        paths, result["roundtrip"] = build_inputs(root, signals, names, args)
        factories = input_factories(root, paths, signals, names, args)
        result["fidelity"] = compare_outputs(factories, model, names, device, args)
        result["storage_bytes"] = {"npy": sum(f.stat().st_size for f in root.glob("*.npy"))}
        for storage, path in paths.items():
            result["storage_bytes"][f"zarr_{storage}"] = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        for compute in (False, True):
            phase = "end_to_end" if compute else "loader_only"
            timings = {key: [] for key in factories}
            for fn in factories.values():
                run_pass(fn, model, names, device, args, compute=compute)
            keys = list(factories)
            for repeat in range(args.repeats):
                for key in keys[repeat % len(keys):] + keys[:repeat % len(keys)]:
                    timings[key].append(run_pass(factories[key], model, names, device, args, compute=compute))
            result["timings"][phase] = {k: {"runs": v, "median_windows_per_s": float(np.median([r["windows_per_s"] for r in v]))} for k, v in timings.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['data_kind']}, {result['weights']}, {result['gpu'] or device}")
    for phase, timings in result["timings"].items():
        print(phase + ": " + ", ".join(f"{k}={v['median_windows_per_s']:.1f} windows/s" for k, v in timings.items()))
    print(f"Fidelity: {result['fidelity']}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
