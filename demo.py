from __future__ import annotations

import os
import shutil
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import zarrfm
from zarrfm import canonical
from zarrfm.dataset import DatasetReader, DatasetWriter, validate_dataset
from zarrfm.sampler import WindowSampler
from zarrfm.synth import synth_eeg

STORE = "demo_study.zarr"
CORPUS = "demo_corpus.zarr"


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


def main() -> int:
    shutil.rmtree(STORE, ignore_errors=True)

    section("1. Write one store")
    data, meta = synth_eeg(n_channels=19, duration_s=600.0, sfreq=256.0, n_events=300)
    zarrfm.from_array(
        data,
        STORE,
        sfreq=meta["sfreq"],
        ch_names=meta["ch_names"],
        ch_types=meta["ch_types"],
        positions=meta["positions"],
        events=meta["events"],
        subject="sub-01",
        session="ses-01",
        task="oddball",
    )
    n_files = sum(len(f) for _, _, f in os.walk(STORE))
    print(f"   {data.shape[0]} channels x {data.shape[1]} samples -> {STORE}")
    print(f"   signals, channel table, montage, events, provenance: {n_files} files, one store")

    section("2. Validate against the spec")
    problems = zarrfm.validate(STORE)
    print("   conforming" if not problems else "\n".join(f"   - {p}" for p in problems))

    section("3. Open and inspect (one metadata read)")
    r = zarrfm.ZarrFMReader(STORE)
    print(f"   {r}")
    info = r.info("eeg")
    for k in ("sfreq", "n_channels", "n_samples", "duration_s", "storage", "compression"):
        print(f"   {k:12} {info[k]}")
    print(f"   {'chunks':12} {info['chunks']}  ({info['chunks'][1] / info['sfreq']:.0f} s per chunk)")
    print(f"   {'shards':12} {info['shards']}")
    print(f"   {'on disk':12} {r.nbytes_stored() / 1e6:.1f} MB  "
          f"(raw int16 {data.size * 2 / 1e6:.1f} MB, float32 {data.size * 4 / 1e6:.1f} MB)")
    print("                -> int16 is the size win; compression adds little at 256 Hz")

    section("4. Read a window -- no time vector stored, it is computed")
    win = r.get_data("eeg", picks=["Cz", "O1", "Fp1"], tmin=10.0, tmax=20.0)
    t = r.times("eeg", 10.0, 20.0)
    print(f"   picks x samples: {win.shape}, t = {t[0]:.2f}..{t[-1]:.2f} s")
    print(f"   Cz  peak-to-peak {np.ptp(win[0]) * 1e6:6.1f} uV")
    print(f"   O1  peak-to-peak {np.ptp(win[1]) * 1e6:6.1f} uV   <- posterior alpha")
    print(f"   Fp1 peak-to-peak {np.ptp(win[2]) * 1e6:6.1f} uV   <- frontal, catches blinks")

    section("5. Events travel in the same store")
    ev = r.events()
    labels, counts = np.unique(ev["label"].astype(str), return_counts=True)
    print(f"   {ev['onset'].size} events: " + ", ".join(f"{c}x {l}" for l, c in zip(labels, counts)))

    section("6. Epoch around events (scattered windows, merged reads)")
    targets = ev["onset"][ev["label"].astype(str) == "target"]
    standards = ev["onset"][ev["label"].astype(str) == "standard"]
    ep_t = r.epochs(targets, "eeg", tmin=-0.2, tmax=0.6, picks=["Cz"])
    ep_s = r.epochs(standards, "eeg", tmin=-0.2, tmax=0.6, picks=["Cz"])
    print(f"   target epochs   {ep_t.shape}")
    print(f"   standard epochs {ep_s.shape}")
    diff = (ep_t.mean(0)[0] - ep_s.mean(0)[0]) * 1e6
    times = np.linspace(-0.2, 0.6, diff.size)
    peak = int(np.argmax(np.abs(diff)))
    baseline = diff[times < 0].std()
    print(f"   target-minus-standard peaks at {times[peak] * 1e3:.0f} ms, {diff[peak]:+.2f} uV"
          f"  ({abs(diff[peak]) / baseline:.1f}x the pre-stimulus SD)")

    section("7. Fidelity of the int16 round trip")
    got = r.get_data("eeg")
    gain = r._stream("eeg")["gain"]
    print(f"   max error   {np.abs(got - data).max() * 1e9:.2f} nV")
    print(f"   bound gain/2 {gain.max() / 2 * 1e9:.2f} nV  (amplifier noise floor is ~1000x larger)")

    section("8. Hand off to MNE")
    raw = zarrfm.to_mne(r)
    positioned = sum(int(np.any(c["loc"][:3] != 0)) for c in raw.info["chs"])
    print(f"   {raw}")
    print(f"   {positioned}/{len(raw.ch_names)} channels carry montage positions")
    print(f"   {len(raw.annotations)} annotations restored")

    section("8b. Edit after close(): append events, flag a channel")
    before = r.events_revision
    n = zarrfm.append_events(STORE, [15.25, 42.0], duration=[0.3, 0.3],
                              label=["blink", "blink"], note="visual inspection")
    bads = zarrfm.set_bad(STORE, ["T7"])
    try:
        r.events()
    except zarrfm.ZarrFMIntegrityError:
        print("   the reader opened before the edit raises rather than returning stale data")
    r.refresh()
    print(f"   events revision {before} -> {r.events_revision}: {n} events, vocab {r.events()['vocab']}")
    print(f"   bad channels now {bads}; provenance logged "
          f"{[s['name'] for s in r.provenance()['pipeline'][-2:]]}")
    print(f"   validate after editing: {'conforming' if not zarrfm.validate(STORE) else zarrfm.validate(STORE)}")

    section("9. Canonical channel space")
    ids = r.canonical_ids("eeg")
    print(f"   {int((ids >= 0).sum())}/{ids.size} channels map into the {canonical.n_canonical()}-slot "
          f"canonical table v{canonical.CANONICAL_VERSION}")
    print("   " + ", ".join(f"{n}->{canonical.canonical_names()[i]}#{i}"
                            for n, i in zip(r.ch_names("eeg")[:4], ids[:4])))
    X, mask = r.get_canonical("eeg", tmin=0.0, tmax=1.0)
    print(f"   get_canonical -> X {X.shape}, mask {mask.shape} with {int(mask.sum())} good slots; "
          "no strings on this path")

    section("10. A corpus: many recordings, one index")
    shutil.rmtree(CORPUS, ignore_errors=True)
    names = [f"EEG {n.upper()}-REF" for n in meta["ch_names"]]
    with DatasetWriter(CORPUS, name="demo-corpus") as d:
        for s in range(6):
            for ses in range(2):
                x, m = synth_eeg(n_channels=19, duration_s=60.0, sfreq=256.0,
                                 seed=10 * s + ses, n_events=20)
                d.add_recording_from_array(
                    f"sub-{s:02d}_ses-{ses:02d}", x, sfreq=256.0, subject=f"sub-{s:02d}",
                    session=f"ses-{ses:02d}", task="oddball", ch_names=names,
                    positions=m["positions"], events=m["events"],
                    bad=["EEG T7-REF"] if s == 1 else None,
                    provenance={"source": {"dataset": "synthetic", "seed": 10 * s + ses},
                                "pipeline": [{"name": "synth_eeg", "lowpass_hz": m["lowpass_hz"]},
                                             {"name": "quantise", "storage": "int16"}]},
                )
        d.assign_splits({"train": 0.7, "validation": 0.15, "test": 0.15}, level="subject", seed=1)
        d.set_provenance(sources=[{"dataset": "synthetic", "license": "CC0"}])
    problems = validate_dataset(CORPUS, verify_hashes=True)
    ds = DatasetReader(CORPUS)
    print(f"   {ds}")
    print("   validate_dataset (with checksums): " + ("conforming" if not problems else str(problems)))
    print(f"   index/ tensors: {sorted(ds.index)[:6]} ...")
    for name in ds.split_names:
        subs = sorted(set(ds.subject_of[ds.split(name)]))
        print(f"   {name:10} {ds.split(name).size:2d} recordings, subjects {subs}")
    print(f"   canonical_present: {ds.canonical_present.shape} bool -- montage coverage without opening a recording")
    print(f"   event vocab (dataset-wide): {ds.vocab}")

    section("11. Sharded window sampler")
    total = WindowSampler(ds, "train", window_s=4.0).n_windows_total
    seen = []
    for rank in range(2):
        for w in range(2):
            smp = WindowSampler(ds, "train", window_s=4.0, seed=0, rank=rank, world_size=2,
                                worker_id=w, num_workers=2)
            keys = [(it["recording"], it["start"]) for it in smp]
            seen.extend(keys)
            print(f"   rank {rank} worker {w}: {len(keys):3d} windows, {smp.stats['reads']} reads")
    print(f"   {len(seen)} windows over 4 shards, {len(set(seen))} unique, {total} expected "
          f"-> {'disjoint and complete' if len(seen) == len(set(seen)) == total else 'MISMATCH'}")
    batch = next(WindowSampler(ds, "train", window_s=4.0).batches(32))
    print("   batch: " + ", ".join(f"{k} {tuple(v.shape)} {v.dtype}" for k, v in batch.items()))

    print(f"\nstores left at ./{STORE} and ./{CORPUS}  (rm -rf *.zarr to clean up)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
