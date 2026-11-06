import argparse
import hashlib
from pathlib import Path

import pytest

import bench_physionet as bench


def paths(monkeypatch, tmp_path):
    for name, value in {"ROOT": tmp_path / "raw", "MANIFEST": tmp_path / "fetched.txt",
                        "FETCH_DONE": tmp_path / "fetch.done"}.items():
        monkeypatch.setattr(bench, name, str(value))


def test_resume_repairs_missing_files_even_when_manifest_says_done(monkeypatch, tmp_path):
    paths(monkeypatch, tmp_path)
    payload = b"real verified bytes"
    source = tmp_path / "source.edf"
    source.write_bytes(payload)
    rel = "sleep-cassette/SC4001E0-PSG.edf"
    key = f"sleep-edfx/1.0.0/{rel}"
    Path(bench.MANIFEST).write_text(key + "\n")
    expected = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(bench, "_sums", lambda slug, version: {rel: expected})
    original = bench._download
    calls = []
    def local_download(url, dest, sha):
        calls.append(dest)
        return original(source.as_uri(), dest, sha)
    monkeypatch.setattr(bench, "_download", local_download)
    args = argparse.Namespace(datasets=["sleep-edfx"], jobs=1)
    assert bench.cmd_fetch(args) == 0
    assert (Path(bench.ROOT) / key).read_bytes() == payload
    assert Path(bench.MANIFEST).read_text().splitlines() == [key]
    # A second resume must use the checksum-verified local file, not the URL.
    source.unlink()
    assert bench.cmd_fetch(args) == 0
    assert len(calls) == 2


def test_fetch_reports_failure(monkeypatch, tmp_path):
    paths(monkeypatch, tmp_path)
    monkeypatch.setattr(bench, "_sums", lambda slug, version: {"bad.edf": "unused"})
    def fail(*args):
        raise OSError("download failed")
    monkeypatch.setattr(bench, "_download", fail)
    assert bench.cmd_fetch(argparse.Namespace(datasets=["sleep-edfx"], jobs=1)) == 1


def test_sleep_pairs_require_real_hypnogram_and_skip_annotation_only_jobs(monkeypatch, tmp_path):
    paths(monkeypatch, tmp_path)
    base = Path(bench.ROOT) / "sleep-edfx/1.0.0/sleep-cassette"
    base.mkdir(parents=True)
    psg = "sleep-cassette/SC4001E0-PSG.edf"
    hyp = "sleep-cassette/SC4001EC-Hypnogram.edf"
    Path(bench.MANIFEST).write_text(f"sleep-edfx/1.0.0/{psg}\nsleep-edfx/1.0.0/{hyp}\n")
    (base / Path(psg).name).write_bytes(b"signal")
    assert bench._jobs_from_manifest(["sleep-edfx"], fetch_done=False) == []
    with pytest.raises(ValueError, match="annotation companion"):
        bench._jobs_from_manifest(["sleep-edfx"], fetch_done=True)
    (base / Path(hyp).name).write_bytes(b"annotations")
    jobs = bench._jobs_from_manifest(["sleep-edfx"], fetch_done=True)
    assert len(jobs) == 1
    assert jobs[0]["annotations"] == hyp


def test_missing_dataset_is_not_a_successful_empty_benchmark(monkeypatch, tmp_path):
    paths(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="no downloaded signal recordings"):
        bench.cmd_run(argparse.Namespace(datasets=["sleep-edfx"], follow=False))
