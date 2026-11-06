from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ROOT = os.path.join(HERE, "data", "real", "physionet")
OUT = os.path.join(HERE, "bench_results", "physionet")
RECORDS = os.path.join(OUT, "records.jsonl")
MANIFEST = os.path.join(ROOT, "fetched.txt")
FETCH_DONE = os.path.join(ROOT, "fetch.done")
MIRROR = "https://physionet-open.s3.amazonaws.com"
ORIGIN = "https://physionet.org/files"

DATASETS = {
    "shhpsgdb": ("1.0.0", 0.09, "Sleep Heart Health PSG sample, 1 file"),
    "eegmat": ("1.0.0", 0.18, "EEG during mental arithmetic, 23 ch 500 Hz"),
    "ltrsvp": ("1.0.0", 0.75, "long-term RSVP, 8 ch 2048 Hz"),
    "ucddb": ("1.0.0", 1.3, "St Vincent's sleep apnoea PSG (.rec) + Holter ECG (.edf)"),
    "sleepbrl": ("1.0.0", 1.5, "sleep bruxism PSG"),
    "psg-ipa": ("1.0.0", 2.6, "PSG inter-scorer agreement: 20 PSGs + 500 annotation-only scorer EDFs"),
    "eegmmidb": ("1.0.0", 3.4, "EEGBCI motor imagery, 64 ch 160 Hz"),
    "eeg-eye-gaze-for-fls-tasks": ("1.0.0", 7.8, "EEG + eye gaze, surgical training"),
    "sleep-edfx": ("1.0.0", 8.1, "Sleep-EDF expanded: mixed-rate PSG + hypnogram EDFs"),
    "hmc-sleep-staging": ("1.1", 15.7, "Haaglanden PSG + scoring EDFs"),
    "siena-scalp-eeg": ("1.0.0", 20.3, "Siena scalp EEG, 512 Hz epilepsy"),
    "eeg-eye-gaze-data": ("1.0.0", 33.2, "EEG + eye gaze, laparoscopic tasks"),
    "capslpdb": ("1.0.0", 40.1, "CAP sleep database, mixed-rate PSG"),
    "chbmit": ("1.0.0", 42.6, "CHB-MIT scalp EEG, 23 ch 256 Hz"),
    "bigp3bci": ("1.0.0", 44.6, "bigP3BCI, P300 speller EEG"),
}

PAIRING = {
    "sleep-edfx": (re.compile(r"^(sleep-(?:cassette|telemetry)/(?:SC|ST)\d{4})\w\d-PSG\.edf$"),
                   lambda m, names: [n for n in names if n.startswith(m.group(1)) and n.endswith("-Hypnogram.edf")]),
    "hmc-sleep-staging": (re.compile(r"^(recordings/SN\d+)\.edf$"),
                          lambda m, names: [n for n in names if n == m.group(1) + "_sleepscoring.edf"]),
}


def _is_signal_file(name: str) -> bool:
    return name.lower().endswith((".edf", ".rec"))


def _sums(slug: str, ver: str) -> dict[str, str]:
    d = os.path.join(ROOT, slug, ver)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "SHA256SUMS.txt")
    if not os.path.exists(path):
        for base in (MIRROR, ORIGIN):
            try:
                with urllib.request.urlopen(f"{base}/{slug}/{ver}/SHA256SUMS.txt", timeout=600) as r:
                    data = r.read()
                break
            except Exception as e:
                print(f"  SHA256SUMS from {base}: {e}", flush=True)
        else:
            raise RuntimeError(f"no SHA256SUMS.txt for {slug}/{ver}")
        with open(path, "wb") as f:
            f.write(data)
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                out[parts[1].strip()] = parts[0]
    return out


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _download(url: str, dest: str, sha: str, retries: int = 6) -> tuple[int, float]:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.isfile(dest) and _sha256_file(dest) == sha:
        return os.path.getsize(dest), 0.0
    if os.path.exists(dest) and _sha256_file(dest) == sha:
        return os.path.getsize(dest), 0.0
    last = None
    for k in range(retries):
        part = dest + ".part"
        h = hashlib.sha256()
        n = 0
        t = time.perf_counter()
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
                for blk in iter(lambda: r.read(1 << 20), b""):
                    f.write(blk)
                    h.update(blk)
                    n += len(blk)
            if h.hexdigest() != sha:
                raise IOError(f"sha256 mismatch after {n} bytes")
            os.replace(part, dest)
            return n, time.perf_counter() - t
        except Exception as e:
            last = e
            time.sleep(min(60, 2 ** k))
    raise IOError(f"{url}: {last}")


def cmd_fetch(args) -> int:
    os.makedirs(ROOT, exist_ok=True)
    done = set()
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            done = {ln.strip() for ln in f if ln.strip()}
    if os.path.exists(FETCH_DONE):
        os.remove(FETCH_DONE)
    failures = []
    slugs = args.datasets or list(DATASETS)
    for slug in slugs:
        ver = DATASETS[slug][0]
        sums = _sums(slug, ver)
        want = [(rel, sha) for rel, sha in sums.items() if _is_signal_file(rel)]
        # The manifest is an index, not proof the bytes still exist. Verify
        # every destination on resume, including files previously marked done.
        todo = want
        print(f"[{slug}] {len(want)} EDF files to verify/download", flush=True)
        t0 = time.perf_counter()
        got = 0
        with ThreadPoolExecutor(args.jobs) as ex, open(MANIFEST, "a") as man:
            futs = {ex.submit(_download, f"{MIRROR}/{slug}/{ver}/{urllib.parse.quote(rel)}",
                              os.path.join(ROOT, slug, ver, rel), sha): rel for rel, sha in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                rel = futs[fut]
                try:
                    n, _ = fut.result()
                    got += n
                    key = f"{slug}/{ver}/{rel}"
                    if key not in done:
                        man.write(key + "\n")
                        man.flush()
                        done.add(key)
                except Exception as e:
                    failures.append((f"{slug}/{ver}/{rel}", str(e)))
                    print(f"  FAILED {rel}: {e}", flush=True)
                if i % 100 == 0 or i == len(todo):
                    dt = time.perf_counter() - t0
                    print(f"  {i}/{len(todo)}  {got / 1e9:.2f} GB  {got / 1e6 / max(dt, 1e-9):.1f} MB/s",
                          flush=True)
    with open(os.path.join(ROOT, "fetch_failed.txt"), "w") as f:
        for rel, err in failures:
            f.write(f"{rel}\t{err}\n")
    with open(FETCH_DONE, "w") as f:
        f.write(json.dumps({"datasets": slugs, "failures": len(failures), "time": time.time()}))
    print(f"fetch done, {len(failures)} failures", flush=True)
    return 1 if failures else 0


PROFILES = ("fast", "max")


def _worker_init():
    import warnings
    warnings.filterwarnings("ignore")
    try:
        import numcodecs.blosc
        numcodecs.blosc.set_nthreads(2)
    except Exception:
        pass

    try:
        import zarr
        zarr.config.set({"threading.max_workers": 2})
    except Exception:
        pass


def _entropy_bits(hist: np.ndarray) -> float:
    n = hist.sum()
    if n == 0:
        return 0.0
    p = hist[hist > 0] / n
    return float(-(p * np.log2(p)).sum())


def _store_bytes(store: str) -> tuple[int, int, int]:
    sig = other = n = 0
    for dp, _, fns in os.walk(store):
        in_signals = "/signals/" in (dp + "/")
        for fn in fns:
            sz = os.path.getsize(os.path.join(dp, fn))
            n += 1
            if in_signals and fn != "zarr.json":
                sig += sz
            else:
                other += sz
    return sig, other, n


def measure(job: dict) -> dict:
    import edfio

    import zarrfm
    from zarrfm.reader import ZarrFMReader

    slug, ver, rel = job["dataset"], job["version"], job["rel"]
    path = os.path.join(ROOT, slug, ver, rel)
    rec = {"dataset": slug, "version": ver, "file": rel, "edf_bytes": os.path.getsize(path),
           "load1": os.getloadavg()[0], "status": "ok",
           "annotations": job.get("annotations")}
    try:
        with open(path, "rb") as fh:
            magic = fh.read(8)
        if magic[:1] == b"\xff" or magic[1:8].upper() == b"BIOSEMI":
            rec["status"] = "unsupported_bdf"
            return rec
        edf = edfio.read_edf(path, lazy_load_data=True)
        signals = [s for s in edf.signals if s.label.strip().upper() != "EDF ANNOTATIONS"]
        rec["n_signals"] = len(signals)
        rec["n_annotations"] = len(edf.annotations)
        rec["duration_s"] = float(edf.duration)
        rec["edf_plus"] = edf.reserved.strip().upper().startswith("EDF+")
        rec["continuous"] = bool(edf.is_continuous)
        if not signals:
            rec["status"] = "annotation_only"
            return rec
        if not edf.is_continuous:
            rec["status"] = "unsupported_edf_d"
            return rec
        rates = sorted({float(s.sampling_frequency) for s in signals}, reverse=True)
        rec["rates_hz"] = rates
        rec["declared_bits"] = [float(np.log2(abs(s.digital_max - s.digital_min) + 1)) for s in signals]

        ann = os.path.join(ROOT, slug, ver, job["annotations"]) if job.get("annotations") else None
        scratch = tempfile.mkdtemp(prefix="physionet_", dir=job["scratch"])
        try:
            for profile in PROFILES:
                store = os.path.join(scratch, f"{profile}.zarr")
                t = time.perf_counter()
                zarrfm.from_edf(path, store, annotations=ann, compression=profile)
                t_write = time.perf_counter() - t
                sig_b, other_b, n_obj = _store_bytes(store)
                r = ZarrFMReader(store)
                streams = {}
                total_samples = 0
                t_read = 0.0
                for name in r.streams:
                    st = r._stream(name)
                    arr = st["array"]
                    C, N = arr.shape
                    total_samples += C * N
                    sfreq = float(st["attrs"]["sfreq"])
                    info = {"n_channels": C, "n_samples": N, "sfreq": sfreq,
                            "chunk_samples": int(arr.chunks[1]),
                            "shard_samples": int(arr.shards[1]) if arr.shards else None,
                            "types": {}}
                    for tname in st["types"]:
                        info["types"][tname] = info["types"].get(tname, 0) + 1
                    src = [s for s in signals if float(s.sampling_frequency) == sfreq]
                    verify = profile == "fast"
                    if verify:
                        hist = np.zeros((C, 65536), dtype=np.int64)
                        mismatch = 0
                        padded = 0
                        flip = []
                        for s in src:
                            if s.digital_max == s.digital_min:
                                flip.append(False)
                                continue
                            g = (s.physical_max - s.physical_min) / (s.digital_max - s.digital_min)
                            flip.append(g < 0)
                    block = max(1, (16 << 20) // (2 * C))
                    for s0 in range(0, N, block):
                        s1 = min(N, s0 + block)
                        t = time.perf_counter()
                        codes = np.asarray(arr[:, s0:s1])
                        t_read += time.perf_counter() - t
                        if verify:
                            for row, s in enumerate(src):
                                seg = s.get_digital_slice(s0 / sfreq, s1 / sfreq)
                                if flip[row]:
                                    seg = (-seg.astype(np.int32) - 1).astype(np.int16)
                                m = min(seg.size, s1 - s0)
                                mismatch += int(np.count_nonzero(codes[row, :m] != seg[:m]))
                                padded += (s1 - s0) - m
                            off = codes.astype(np.int32) + 32768
                            for row in range(C):
                                hist[row] += np.bincount(off[row], minlength=65536)
                    if verify:
                        ent = [round(_entropy_bits(h), 3) for h in hist]
                        used = []
                        for h in hist:
                            nz = np.flatnonzero(h)
                            used.append(round(float(np.log2(nz[-1] - nz[0] + 1)) if nz.size else 0.0, 3))
                        info.update({"entropy_bits": ent, "used_bits": used,
                                     "mismatch": mismatch, "padded_samples": padded,
                                     "flipped": int(sum(flip))})
                    streams[name] = info
                rec[profile] = {"store_bytes": sig_b + other_b, "signal_bytes": sig_b,
                                "other_bytes": other_b, "n_objects": n_obj,
                                "t_write": round(t_write, 4), "t_read": round(t_read, 4),
                                "streams": streams}
                rec["total_samples"] = total_samples
                rec["n_events"] = int(len(r.events()["onset"]))
                shutil.rmtree(store, ignore_errors=True)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        rec["bit_exact"] = all(s.get("mismatch", 1) == 0 for s in rec["fast"]["streams"].values())

        t = time.perf_counter()
        ztmp = os.path.join(job["scratch"], f"z{os.getpid()}.zst")
        p = subprocess.run(["zstd", "-5", "-q", "-T1", "-f", "-o", ztmp, path], capture_output=True, text=True)
        rec["zstd5_edf_bytes"] = os.path.getsize(ztmp) if p.returncode == 0 else None
        rec["zstd5_stderr"] = p.stderr.strip()[:200] if p.returncode != 0 else None
        if os.path.exists(ztmp):
            os.remove(ztmp)
        rec["t_zstd5"] = round(time.perf_counter() - t, 4)
    except Exception as e:
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-2000:]
    return rec


COMPANION = {
    "sleep-edfx": re.compile(r"-Hypnogram\.edf$"),
    "hmc-sleep-staging": re.compile(r"_sleepscoring\.edf$"),
}


def _jobs_from_manifest(datasets: list[str] | None, fetch_done: bool) -> list[dict]:
    if not os.path.exists(MANIFEST):
        return []
    with open(MANIFEST) as f:
        entries = list(dict.fromkeys(ln.strip() for ln in f if ln.strip()))
    by_ds: dict[tuple[str, str], list[str]] = {}
    for e in entries:
        slug, ver, rel = e.split("/", 2)
        by_ds.setdefault((slug, ver), []).append(rel)
    jobs = []
    for (slug, ver), rels in by_ds.items():
        if datasets and slug not in datasets:
            continue
        names = set(rels)
        rule = PAIRING.get(slug)
        comp = COMPANION.get(slug)
        for rel in rels:
            if comp and comp.search(rel):
                continue
            if not os.path.isfile(os.path.join(ROOT, slug, ver, rel)):
                if fetch_done:
                    raise FileNotFoundError(f"manifest references missing recording {slug}/{ver}/{rel}; "
                                            f"run fetch --datasets {slug} to repair it")
                continue
            ann = None
            if rule:
                m = rule[0].match(rel)
                if m:
                    cands = [c for c in rule[1](m, names) if c in names]
                    cands = [c for c in cands if os.path.isfile(os.path.join(ROOT, slug, ver, c))]
                    if len(cands) == 1:
                        ann = cands[0]
                    elif not fetch_done:
                        continue
                    else:
                        raise ValueError(f"{slug}/{ver}/{rel}: expected exactly one downloaded annotation companion, "
                                         f"found {len(cands)}; run fetch --datasets {slug} to repair it")
            jobs.append({"dataset": slug, "version": ver, "rel": rel, "annotations": ann})
    return jobs


def cmd_run(args) -> int:
    import multiprocessing as mp

    if not args.follow and not _jobs_from_manifest(args.datasets, os.path.exists(FETCH_DONE)):
        selected = " ".join(args.datasets or DATASETS)
        raise ValueError(f"no downloaded signal recordings for {selected}; "
                         f"run fetch --datasets {selected} first")
    os.makedirs(OUT, exist_ok=True)
    scratch = args.scratch or tempfile.mkdtemp(prefix="physionet_stores_")
    os.makedirs(scratch, exist_ok=True)
    done = set()
    if os.path.exists(RECORDS):
        with open(RECORDS) as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                    done.add((r["dataset"], r["file"]))
                except Exception:
                    pass

    ctx = mp.get_context("spawn")
    total_files = total_bytes = 0
    t_start = time.perf_counter()
    with ctx.Pool(args.workers, initializer=_worker_init, maxtasksperchild=64) as pool, \
            open(RECORDS, "a") as out:
        while True:
            fetch_done = os.path.exists(FETCH_DONE)
            jobs = [j for j in _jobs_from_manifest(args.datasets, fetch_done)
                    if (j["dataset"], j["rel"]) not in done]
            if not jobs:
                if fetch_done or not args.follow:
                    break
                time.sleep(30)
                continue
            for j in jobs:
                j["scratch"] = scratch
            print(f"[run] {len(jobs)} files queued (fetch {'done' if fetch_done else 'in progress'})", flush=True)
            for rec in pool.imap_unordered(measure, jobs, chunksize=1):
                done.add((rec["dataset"], rec["file"]))
                rec.pop("traceback_full", None)
                out.write(json.dumps(rec) + "\n")
                out.flush()
                total_files += 1
                total_bytes += rec["edf_bytes"]
                if rec["status"] != "ok":
                    print(f"  {rec['dataset']}/{rec['file']}: {rec['status']} {rec.get('error', '')}", flush=True)
                if total_files % 100 == 0:
                    dt = time.perf_counter() - t_start
                    print(f"  {total_files} files, {total_bytes / 1e9:.1f} GB EDF, "
                          f"{total_bytes / 1e6 / dt:.0f} MB/s", flush=True)
            if not args.follow:
                break
    print(f"[run] finished: {total_files} files, {total_bytes / 1e9:.1f} GB", flush=True)
    return 0


def _q(xs, q):
    return float(np.quantile(xs, q)) if len(xs) else float("nan")


def cmd_summary(args) -> int:
    recs = []
    with open(RECORDS) as f:
        for ln in f:
            recs.append(json.loads(ln))
    per = {}
    for r in recs:
        d = per.setdefault(r["dataset"], {"files": 0, "status": {}, "edf_bytes": 0, "ok": []})
        d["files"] += 1
        d["status"][r["status"]] = d["status"].get(r["status"], 0) + 1
        if r["status"] == "ok":
            d["edf_bytes"] += r["edf_bytes"]
            d["ok"].append(r)

    def agg(rs: list[dict]) -> dict:
        edf = sum(r["edf_bytes"] for r in rs)
        fast = sum(r["fast"]["store_bytes"] for r in rs)
        mx = sum(r["max"]["store_bytes"] for r in rs)
        zs = sum(r["zstd5_edf_bytes"] or 0 for r in rs)
        samples = sum(r["total_samples"] for r in rs)
        fast_sig = sum(r["fast"]["signal_bytes"] for r in rs)
        max_sig = sum(r["max"]["signal_bytes"] for r in rs)
        other = sum(r["fast"]["other_bytes"] for r in rs)
        ent_w = used_w = decl_w = 0.0
        nsamp = 0
        for r in rs:
            for st in r["fast"]["streams"].values():
                n = st["n_samples"]
                ent_w += n * sum(st["entropy_bits"])
                used_w += n * sum(st["used_bits"])
                nsamp += n * st["n_channels"]
            decl_w += r["total_samples"] / len(r["declared_bits"]) * sum(r["declared_bits"]) if r["declared_bits"] else 0
        ratios = [r["edf_bytes"] / r["fast"]["store_bytes"] for r in rs]
        ratios_max = [r["edf_bytes"] / r["max"]["store_bytes"] for r in rs]
        ratios_z = [r["edf_bytes"] / r["zstd5_edf_bytes"] for r in rs if r["zstd5_edf_bytes"]]
        wmb = [r["edf_bytes"] / 1e6 / r["fast"]["t_write"] for r in rs if r["fast"]["t_write"] > 0]
        rmb = [2 * r["total_samples"] / 1e6 / r["fast"]["t_read"] for r in rs if r["fast"]["t_read"] > 0]
        rmb_max = [2 * r["total_samples"] / 1e6 / r["max"]["t_read"] for r in rs if r["max"]["t_read"] > 0]
        chans = sum(st["n_channels"] for r in rs for st in r["fast"]["streams"].values())
        rates = sorted({x for r in rs for x in r["rates_hz"]})
        return {
            "files": len(rs), "edf_gb": edf / 1e9, "fast_gb": fast / 1e9, "max_gb": mx / 1e9,
            "zstd5_gb": zs / 1e9, "hours": sum(r["duration_s"] for r in rs) / 3600,
            "channels_total": chans, "rates_hz": rates,
            "ratio_fast": edf / fast if fast else None, "ratio_max": edf / mx if mx else None,
            "ratio_zstd5": edf / zs if zs else None,
            "ratio_fast_p10_p50_p90": [_q(ratios, .1), _q(ratios, .5), _q(ratios, .9)],
            "ratio_max_p50": _q(ratios_max, .5), "ratio_zstd5_p50": _q(ratios_z, .5),
            "bits_per_sample_entropy": ent_w / nsamp if nsamp else None,
            "bits_per_sample_used": used_w / nsamp if nsamp else None,
            "bits_per_sample_declared": decl_w / samples if samples else None,
            "bits_per_sample_fast": 8 * fast_sig / samples if samples else None,
            "bits_per_sample_max": 8 * max_sig / samples if samples else None,
            "metadata_overhead_pct": 100 * other / fast if fast else None,
            "bit_exact": sum(1 for r in rs if r.get("bit_exact")),
            "not_bit_exact": [r["file"] for r in rs if not r.get("bit_exact")][:20],
            "padded_samples": sum(st.get("padded_samples", 0) for r in rs for st in r["fast"]["streams"].values()),
            "write_mb_s_p50": _q(wmb, .5), "read_mb_s_fast_p50": _q(rmb, .5),
            "read_mb_s_max_p50": _q(rmb_max, .5),
            "load1_p50": _q([r["load1"] for r in rs], .5),
        }

    summary = {"datasets": {}, "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    all_ok = []
    for slug in DATASETS:
        if slug not in per:
            continue
        d = per[slug]
        s = {"note": DATASETS[slug][2], "files_seen": d["files"], "status": d["status"]}
        if d["ok"]:
            s.update(agg(d["ok"]))
            all_ok.extend(d["ok"])
        errs = {}
        for r in recs:
            if r["dataset"] == slug and r["status"] == "error":
                errs[r["error"]] = errs.get(r["error"], 0) + 1
        s["errors"] = errs
        summary["datasets"][slug] = s
    summary["total"] = agg(all_ok) if all_ok else {}
    summary["total"]["status"] = {}
    for r in recs:
        summary["total"]["status"][r["status"]] = summary["total"]["status"].get(r["status"], 0) + 1
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    lines = ["| database | files ok / seen | hours | EDF GB | zstd-5 GB (×) | fast GB (×) | max GB (×) | "
             "bits: declared / used / H0 / fast / max | bit-exact | write MB/s | read MB/s fast/max |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]

    def row(name, s):
        if "files" not in s:
            return f"| {name} | 0 / {s['files_seen']} | | | | | | | | | |"
        return ("| {n} | {f} / {seen} | {h:.0f} | {edf:.2f} | {z:.2f} ({rz:.2f}×) | {fa:.2f} ({rf:.2f}×) | "
                "{mx:.2f} ({rm:.2f}×) | {bd:.1f} / {bu:.1f} / {be:.2f} / {bf:.2f} / {bm:.2f} | {bx}/{f} | "
                "{w:.0f} | {rr:.0f}/{rrm:.0f} |").format(
            n=name, f=s["files"], seen=s.get("files_seen", s["files"]), h=s["hours"], edf=s["edf_gb"],
            z=s["zstd5_gb"], rz=s["ratio_zstd5"] or 0, fa=s["fast_gb"], rf=s["ratio_fast"],
            mx=s["max_gb"], rm=s["ratio_max"], bd=s["bits_per_sample_declared"] or 0,
            bu=s["bits_per_sample_used"] or 0, be=s["bits_per_sample_entropy"] or 0,
            bf=s["bits_per_sample_fast"], bm=s["bits_per_sample_max"], bx=s["bit_exact"],
            w=s["write_mb_s_p50"], rr=s["read_mb_s_fast_p50"], rrm=s["read_mb_s_max_p50"])

    for slug, s in summary["datasets"].items():
        lines.append(row(slug, s))
    if summary["total"]:
        lines.append(row("**all**", {**summary["total"], "files_seen": len(recs)}))
    md = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write(md)
    print(md)
    print(json.dumps(summary["total"].get("status"), indent=0))
    return 0


def cmd_bytype(args) -> int:
    import edfio

    from zarrfm.edf_io import _infer_type

    recs = [json.loads(ln) for ln in open(RECORDS) if ln.strip()]
    agg: dict = {}
    per_ds: dict = {}
    for r in recs:
        if r["status"] != "ok":
            continue
        edf = edfio.read_edf(os.path.join(ROOT, r["dataset"], r["version"], r["file"]), lazy_load_data=True)
        sig = [s for s in edf.signals if s.label.strip().upper() != "EDF ANNOTATIONS"]
        for st in r["fast"]["streams"].values():
            src = [s for s in sig if float(s.sampling_frequency) == st["sfreq"]]
            if len(src) != st["n_channels"]:
                continue
            for s, h, u in zip(src, st["entropy_bits"], st["used_bits"]):
                t = _infer_type(s.label, s.physical_dimension)
                n = st["n_samples"]
                for a in (agg.setdefault(t, {}), per_ds.setdefault(r["dataset"], {}).setdefault(t, {})):
                    a["channels"] = a.get("channels", 0) + 1
                    a["samples"] = a.get("samples", 0) + n
                    a["H"] = a.get("H", 0.0) + n * h
                    a["used"] = a.get("used", 0.0) + n * u
                    a["decl"] = a.get("decl", 0.0) + n * float(np.log2(abs(s.digital_max - s.digital_min) + 1))

    def fin(a):
        return {"channels": a["channels"], "gsamples": a["samples"] / 1e9,
                "declared_bits": a["decl"] / a["samples"], "used_bits": a["used"] / a["samples"],
                "entropy_bits": a["H"] / a["samples"]}

    out = {"by_type": {t: fin(a) for t, a in agg.items()},
           "by_dataset_type": {d: {t: fin(a) for t, a in x.items()} for d, x in per_ds.items()}}
    with open(os.path.join(OUT, "by_type.json"), "w") as f:
        json.dump(out, f, indent=1)
    _table(["type", "channels", "Gsamples", "declared", "used bits", "H0"],
           [(t, v["channels"], f"{v['gsamples']:.1f}", f"{v['declared_bits']:.1f}", f"{v['used_bits']:.2f}",
             f"{v['entropy_bits']:.2f}") for t, v in sorted(out["by_type"].items(), key=lambda kv: -kv[1]["gsamples"])])
    return 0


def _table(headers, rows):
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--datasets", nargs="*")
    p.set_defaults(fn=cmd_fetch)
    p = sub.add_parser("run")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--follow", action="store_true")
    p.add_argument("--scratch")
    p.add_argument("--datasets", nargs="*")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("summary")
    p.set_defaults(fn=cmd_summary)
    p = sub.add_parser("bytype")
    p.set_defaults(fn=cmd_bytype)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
