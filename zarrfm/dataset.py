from __future__ import annotations

import datetime as _dt
import hashlib
import json
import uuid
import warnings
from typing import Iterable, Mapping, Sequence

import numpy as np
import zarr

from zarr.storage import LocalStore

from . import _storeio, canonical, spec
from .reader import ZarrFMReader, _open_root
from .validate import check_consolidated, validate_recording
from .writer import (ZarrFMWriter, _set_attrs, _write_column, build_event_table,
                     read_str_column, write_revision)

DATASET_FORMAT = "zarrfm-dataset"
RECORDINGS = "recordings"
INDEX = "index"
SUBJECTS = "subjects"
SPLITS = "splits"
VOCAB = "vocab"
CANONICAL = "canonical"

SPLIT_LEVELS = ("subject", "session", "recording")
DEFAULT_SPLITS = {"train": 0.8, "validation": 0.1, "test": 0.1}

INDEX_FIELDS = {
    "recording_id": "str",
    "subject_idx": "int32",
    "session": "str",
    "task": "str",
    "primary_stream": "str",
    "n_channels": "int32",
    "n_samples": "int64",
    "sfreq": "float64",
    "duration_s": "float64",
    "n_events": "int64",
    "split": "int8",
    "nbytes": "int64",
    "content_hash": "str",
}
UNASSIGNED = -1


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")


def hash_unit(seed, key: str) -> float:
    h = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


def assign_splits(
    units: Sequence[str],
    fractions: Mapping[str, float] = DEFAULT_SPLITS,
    seed=0,
) -> dict[str, str]:
    names = list(fractions)
    fr = np.asarray([float(fractions[n]) for n in names], dtype=np.float64)
    if np.any(fr < 0) or fr.sum() <= 0:
        raise ValueError("split fractions must be non-negative and not all zero")
    edges = np.cumsum(fr / fr.sum())
    out = {}
    for u in units:
        k = int(np.searchsorted(edges, hash_unit(seed, str(u)), side="right"))
        out[str(u)] = names[min(k, len(names) - 1)]
    return out


def consolidate_two_level(path: str, recording_ids) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for rid in recording_ids:
            zarr.consolidate_metadata(path, path=f"{RECORDINGS}/{rid}")
        zarr.consolidate_metadata(path)
    store = LocalStore(path)
    raw = _storeio.read_key(store, "zarr.json")
    doc = json.loads(raw)
    cm = doc.get("consolidated_metadata", {}).get("metadata", {})
    pruned = {k: v for k, v in cm.items() if not k.startswith(f"{RECORDINGS}/")}
    if RECORDINGS in pruned:
        pruned[RECORDINGS].pop("consolidated_metadata", None)
    doc["consolidated_metadata"]["metadata"] = pruned
    _storeio.write_key(store, "zarr.json", json.dumps(doc).encode("utf-8"))


class DatasetWriter:
    def __init__(
        self,
        path: str,
        *,
        name: str | None = None,
        description: str | None = None,
        dataset_id: str | None = None,
        overwrite: bool = False,
        extra: dict | None = None,
    ):
        self.path = str(path)
        self.root = zarr.create_group(store=self.path, overwrite=overwrite)
        self.root.create_group(RECORDINGS)
        _set_attrs(
            self.root,
            {
                "format": DATASET_FORMAT,
                "version": spec.VERSION,
                "dataset_id": dataset_id or str(uuid.uuid4()),
                "name": name,
                "description": description,
                "created_by": f"zarrfm {spec.VERSION}",
                "created": _now_iso(),
                "canonical_version": canonical.CANONICAL_VERSION,
                **(extra or {}),
            },
        )
        self._recordings: list[dict] = []
        self._open: dict[str, ZarrFMWriter] = {}
        self._subject_cols: dict[str, dict] = {}
        self._split_names: list[str] = list(DEFAULT_SPLITS)
        self._split_policy: dict = {"level": "subject", "seed": 0, "fractions": None,
                                    "explicit": False}
        self._split_of: dict[str, str] = {}
        self._closed = False


    def add_recording(
        self,
        recording_id: str,
        *,
        subject: str,
        session: str | None = None,
        task: str | None = None,
        **writer_kwargs,
    ) -> ZarrFMWriter:
        if self._closed:
            raise RuntimeError("dataset writer is closed")
        rid = str(recording_id)
        if "/" in rid or not rid:
            raise ValueError(f"recording_id {rid!r} must be a non-empty single path component")
        if any(r["id"] == rid for r in self._recordings):
            raise ValueError(f"recording {rid!r} already exists")
        grp = self.root[RECORDINGS].create_group(rid)
        w = ZarrFMWriter(group=grp, subject=str(subject), session=session, task=task,
                          **writer_kwargs)
        self._recordings.append(
            {"id": rid, "subject": str(subject), "session": session or "", "task": task or "",
             "group": grp}
        )
        self._open[rid] = w
        return w

    def add_recording_from_array(
        self, recording_id: str, data: np.ndarray, *, sfreq: float, subject: str,
        session: str | None = None, task: str | None = None, stream: str = "eeg",
        storage: str = "int16", events: dict | None = None, provenance: dict | None = None,
        writer_kwargs: dict | None = None, **stream_kwargs,
    ) -> None:
        w = self.add_recording(recording_id, subject=subject, session=session, task=task,
                               **(writer_kwargs or {}))
        data = np.asarray(data)
        w.add_stream(
            stream, n_channels=data.shape[0], n_samples=data.shape[1], sfreq=sfreq,
            storage=storage,
            phys_range=(data.min(axis=1), data.max(axis=1)) if storage == "int16" else None,
            **stream_kwargs,
        )
        w.write_stream(stream, data)
        if events is not None:
            w.set_events(**events)
        if provenance is not None:
            w.set_provenance(**provenance)
        w.close()


    def set_subject_info(self, info: Mapping[str, Mapping[str, object]]) -> None:
        for sid, cols in info.items():
            self._subject_cols.setdefault(str(sid), {}).update(dict(cols))


    def assign_splits(
        self,
        fractions: Mapping[str, float] = DEFAULT_SPLITS,
        *,
        level: str = "subject",
        seed=0,
    ) -> dict[str, str]:
        if level not in SPLIT_LEVELS:
            raise ValueError(f"level must be one of {SPLIT_LEVELS}")
        self._split_names = list(fractions)
        self._split_policy = {"level": level, "seed": seed,
                              "fractions": {k: float(v) for k, v in fractions.items()},
                              "explicit": False}
        units = {r["id"]: self._unit(r, level) for r in self._recordings}
        by_unit = assign_splits(sorted(set(units.values())), fractions, seed)
        self._split_of = {rid: by_unit[u] for rid, u in units.items()}
        return dict(self._split_of)

    def set_splits(self, splits: Mapping[str, Iterable[str]], *, level: str = "subject") -> None:
        if level not in SPLIT_LEVELS:
            raise ValueError(f"level must be one of {SPLIT_LEVELS}")
        self._split_names = list(splits)
        self._split_policy = {"level": level, "seed": None, "fractions": None,
                              "explicit": True}
        self._split_of = {}
        known = {r["id"] for r in self._recordings}
        for name, ids in splits.items():
            for rid in ids:
                rid = str(rid)
                if rid not in known:
                    raise KeyError(f"split {name!r} names unknown recording {rid!r}")
                if rid in self._split_of:
                    raise ValueError(f"recording {rid!r} appears in more than one split")
                self._split_of[rid] = name

    @staticmethod
    def _unit(rec: dict, level: str) -> str:
        if level == "subject":
            return rec["subject"]
        if level == "session":
            return f"{rec['subject']}/{rec['session']}"
        return rec["id"]


    def set_provenance(self, *, sources: Sequence[dict] | None = None,
                       pipeline: Sequence[dict] | None = None,
                       software: dict | None = None, **extra) -> None:
        grp = self.root.require_group(spec.PROVENANCE)
        software = dict(software or {})
        software.setdefault("zarrfm", spec.VERSION)
        software.setdefault("zarr", zarr.__version__)
        software.setdefault("numpy", np.__version__)
        _set_attrs(grp, {"sources": [dict(s) for s in (sources or [])],
                         "pipeline": [dict(s) for s in (pipeline or [])],
                         "software": software, "written": _now_iso(), **extra})


    def close(self) -> None:
        if self._closed:
            return
        for w in self._open.values():
            w.close()
        self._open.clear()
        if not self._split_of:
            self.assign_splits(DEFAULT_SPLITS, level="subject", seed=0)
        self._write_vocab()
        self._write_subjects()
        self._write_splits()
        self._write_canonical()
        self._write_index()
        if self.root.get(spec.PROVENANCE) is None:
            self.set_provenance()
        _set_attrs(self.root, {"n_recordings": len(self._recordings)})
        consolidate_two_level(self.path, [r["id"] for r in self._recordings])
        self._closed = True

    def _write_vocab(self) -> None:
        labels: set[str] = set()
        readers = {}
        for rec in self._recordings:
            r = ZarrFMReader(rec["group"])
            readers[rec["id"]] = r
            labels.update(r.events()["vocab"])
        vocab = sorted(labels)
        lookup = {v: i for i, v in enumerate(vocab)}
        for rec in self._recordings:
            ev = readers[rec["id"]].events()
            table = build_event_table(ev["onset"], ev["duration"], list(ev["label"]),
                                      ev["value"], vocab)
            write_revision(rec["group"], 0, table, overwrite=True)
        grp = self.root.require_group(VOCAB)
        _write_column(grp, "event_labels", np.array(vocab, dtype=object), "str", overwrite=True)
        self._vocab = vocab

    def _write_subjects(self) -> None:
        ids = sorted({r["subject"] for r in self._recordings})
        grp = self.root.require_group(SUBJECTS)
        _write_column(grp, "id", np.array(ids, dtype=object), "str", overwrite=True)
        cols: dict[str, list] = {}
        for sid in ids:
            for k, v in self._subject_cols.get(sid, {}).items():
                cols.setdefault(k, [])
        for k in cols:
            vals = [self._subject_cols.get(sid, {}).get(k) for sid in ids]
            if all(isinstance(v, (int, float, np.integer, np.floating)) or v is None for v in vals):
                arr = np.array([np.nan if v is None else float(v) for v in vals], dtype=np.float64)
                _write_column(grp, k, arr, "float64", overwrite=True)
            else:
                arr = np.array(["" if v is None else str(v) for v in vals], dtype=object)
                _write_column(grp, k, arr, "str", overwrite=True)
        self._subject_ids = ids

    def _write_splits(self) -> None:
        grp = self.root.require_group(SPLITS)
        _write_column(grp, "names", np.array(self._split_names, dtype=object), "str",
                      overwrite=True)
        _set_attrs(grp, self._split_policy)

    def _write_canonical(self) -> None:
        grp = self.root.require_group(CANONICAL)
        names = canonical.canonical_names()
        _write_column(grp, "name", np.array(names, dtype=object), "str", overwrite=True)
        _write_column(grp, "type", np.array(canonical.canonical_types(), dtype=object), "str",
                      overwrite=True)
        pos = canonical.canonical_positions()
        a = grp.create_array(name="position", shape=pos.shape, chunks=pos.shape,
                             dtype="float64", overwrite=True)
        a[:] = pos
        _set_attrs(grp, {"version": canonical.CANONICAL_VERSION})

    def _write_index(self) -> None:
        R = len(self._recordings)
        K = canonical.n_canonical()
        cols: dict[str, list] = {k: [] for k in INDEX_FIELDS}
        present = np.zeros((R, K), dtype=bool)
        subj_idx = {s: i for i, s in enumerate(self._subject_ids)}
        split_idx = {s: i for i, s in enumerate(self._split_names)}
        for i, rec in enumerate(self._recordings):
            r = ZarrFMReader(rec["group"])
            ps = r.primary_stream
            info = r.info(ps)
            ids = r.canonical_ids(ps)
            good = ids[(ids >= 0) & ~r.bad_mask(ps)]
            present[i, good] = True
            cols["recording_id"].append(rec["id"])
            cols["subject_idx"].append(subj_idx[rec["subject"]])
            cols["session"].append(rec["session"])
            cols["task"].append(rec["task"])
            cols["primary_stream"].append(ps)
            cols["n_channels"].append(info["n_channels"])
            cols["n_samples"].append(info["n_samples"])
            cols["sfreq"].append(info["sfreq"])
            cols["duration_s"].append(info["duration_s"])
            cols["n_events"].append(int(r.events()["onset"].size))
            cols["split"].append(split_idx.get(self._split_of.get(rec["id"]), UNASSIGNED))
            cols["nbytes"].append(r.nbytes_stored())
            cols["content_hash"].append(r.content_hash())
        grp = self.root.require_group(INDEX)
        for k, dtype in INDEX_FIELDS.items():
            vals = np.array(cols[k], dtype=object) if dtype == "str" else np.asarray(cols[k], dtype=dtype)
            _write_column(grp, k, vals, dtype, overwrite=True)
        a = grp.create_array(name="canonical_present", shape=(R, K), chunks=(max(R, 1), K),
                             dtype="bool", overwrite=True)
        if R:
            a[:] = present

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class DatasetReader:
    def __init__(self, path):
        self.root = _open_root(path)
        self.path = str(path)
        self.attrs = dict(self.root.attrs)
        if self.attrs.get("format") != DATASET_FORMAT:
            raise ValueError(f"{path!r} is not an zarrfm dataset store")
        idx = self.root[INDEX]
        self.index: dict[str, np.ndarray] = {}
        for k, dtype in INDEX_FIELDS.items():
            if dtype == "str":
                self.index[k] = np.array(read_str_column(idx, k) or [], dtype=object)
            else:
                self.index[k] = np.asarray(idx[k][:])
        self.canonical_present = np.asarray(idx["canonical_present"][:], dtype=bool)
        self.split_names = read_str_column(self.root[SPLITS], "names") or []
        self.split_policy = {k: v for k, v in self.root[SPLITS].attrs.items() if k != "names"}
        self.vocab = read_str_column(self.root[VOCAB], "event_labels") or []
        self.subject_ids = read_str_column(self.root[SUBJECTS], "id") or []
        self._readers: dict[int, ZarrFMReader] = {}

    def __len__(self) -> int:
        return int(self.index["recording_id"].size)

    @property
    def recording_ids(self) -> list[str]:
        return list(self.index["recording_id"])

    @property
    def subject_of(self) -> np.ndarray:
        return np.array(self.subject_ids, dtype=object)[self.index["subject_idx"]]

    def split(self, name: str) -> np.ndarray:
        if name not in self.split_names:
            raise KeyError(f"no split {name!r}; have {self.split_names}")
        return np.flatnonzero(self.index["split"] == self.split_names.index(name))

    def recording(self, which) -> ZarrFMReader:
        if isinstance(which, str):
            hits = np.flatnonzero(self.index["recording_id"] == which)
            if hits.size == 0:
                raise KeyError(f"no recording {which!r}")
            which = int(hits[0])
        which = int(which)
        if which not in self._readers:
            rid = self.index["recording_id"][which]
            sub = f"{RECORDINGS}/{rid}"
            if self.root.path:
                sub = f"{self.root.path}/{sub}"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                grp = zarr.open_group(store=self.root.store, path=sub, mode="r")
            self._readers[which] = ZarrFMReader(grp)
        return self._readers[which]

    def canonical_table(self) -> dict:
        grp = self.root[CANONICAL]
        return {
            "name": read_str_column(grp, "name") or [],
            "type": read_str_column(grp, "type") or [],
            "position": np.asarray(grp["position"][:]),
            "version": grp.attrs.get("version"),
        }

    def subject_info(self) -> dict[str, np.ndarray]:
        grp = self.root[SUBJECTS]
        out = {k: np.asarray(grp[k][:]) for k in grp.array_keys()}
        for k, v in grp.attrs.items():
            if isinstance(v, list):
                out[k] = np.array([str(x) for x in v], dtype=object)
        return out

    def provenance(self) -> dict:
        grp = self.root.get(spec.PROVENANCE)
        return {} if grp is None else dict(grp.attrs)

    def verify(self, which) -> bool:
        r = self.recording(which)
        i = int(which) if not isinstance(which, str) else int(
            np.flatnonzero(self.index["recording_id"] == which)[0])
        return r.content_hash() == str(self.index["content_hash"][i])

    def __repr__(self) -> str:
        hours = float(self.index["duration_s"].sum()) / 3600.0
        return (f"<DatasetReader {self.attrs.get('name') or '?'}: {len(self)} recordings, "
                f"{len(self.subject_ids)} subjects, {hours:.1f} h, splits={self.split_names}>")


def validate_dataset(path: str, *, verify_hashes: bool = False) -> list[str]:
    problems: list[str] = []
    try:
        root = zarr.open_group(str(path), mode="r", use_consolidated=False, zarr_format=3)
    except Exception as e:
        return [f"cannot open store: {type(e).__name__}: {e}"]
    attrs = dict(root.attrs)
    if attrs.get("format") != DATASET_FORMAT:
        problems.append(f"root attr format={attrs.get('format')!r}, expected {DATASET_FORMAT!r}")
    if not attrs.get("dataset_id"):
        problems.append("root attr dataset_id is missing")
    check_consolidated(path, root, problems, skip_prefix=f"{RECORDINGS}/")

    for name in (RECORDINGS, INDEX, SUBJECTS, SPLITS, VOCAB, CANONICAL, spec.PROVENANCE):
        if root.get(name) is None:
            problems.append(f"missing {name}/ group")
    if any(p.startswith("missing") for p in problems):
        return problems

    recs = root[RECORDINGS]
    rec_ids = sorted(recs.group_keys())
    idx = root[INDEX]
    lengths = {}
    str_cols: dict[str, list[str]] = {}
    for k, dtype in INDEX_FIELDS.items():
        if dtype == "str":
            col = read_str_column(idx, k)
            if col is None:
                problems.append(f"missing index/{k}")
            else:
                str_cols[k] = col
                lengths[k] = len(col)
        elif idx.get(k) is None:
            problems.append(f"missing index/{k}")
        else:
            lengths[k] = int(idx[k].shape[0])
    if len(set(lengths.values())) > 1:
        problems.append(f"index/ columns have mismatched lengths: {lengths}")
        return problems
    R = next(iter(lengths.values()), 0)
    index_ids = str_cols.get("recording_id", [])
    if sorted(index_ids) != rec_ids:
        problems.append("index/recording_id does not match the groups under recordings/")
    cp = idx.get("canonical_present")
    K = len(read_str_column(root[CANONICAL], "name") or [])
    if cp is None or tuple(cp.shape) != (R, K):
        problems.append(f"index/canonical_present must be ({R}, {K})")

    vocab = read_str_column(root[VOCAB], "event_labels") or []
    split_names = read_str_column(root[SPLITS], "names") or []
    policy = dict(root[SPLITS].attrs)
    level = policy.get("level")
    if level not in SPLIT_LEVELS:
        problems.append(f"splits/ attr level={level!r} is not one of {SPLIT_LEVELS}")
    split_col = np.asarray(idx["split"][:], dtype=np.int64)
    if split_col.size and (split_col.min() < UNASSIGNED or split_col.max() >= len(split_names)):
        problems.append("index/split indexes outside splits/names")

    subject_ids = read_str_column(root[SUBJECTS], "id") or []
    subj_idx = np.asarray(idx["subject_idx"][:], dtype=np.int64)
    if subj_idx.size and (subj_idx.min() < 0 or subj_idx.max() >= len(subject_ids)):
        problems.append("index/subject_idx indexes outside subjects/id")

    units: dict[str, set] = {}
    sessions = str_cols.get("session", [""] * R)
    for i in range(R):
        if split_col[i] == UNASSIGNED:
            continue
        subj = subject_ids[subj_idx[i]] if 0 <= subj_idx[i] < len(subject_ids) else "?"
        unit = {"subject": subj, "session": f"{subj}/{sessions[i]}", "recording": index_ids[i]}.get(
            level, index_ids[i])
        units.setdefault(unit, set()).add(int(split_col[i]))
    leaks = sorted(u for u, s in units.items() if len(s) > 1)
    if leaks:
        problems.append(f"split leakage at level={level!r}: {leaks[:5]}"
                        + (" ..." if len(leaks) > 5 else ""))

    hashes = str_cols.get("content_hash", [""] * R)
    n_samples = np.asarray(idx["n_samples"][:], dtype=np.int64)
    for i, rid in enumerate(index_ids):
        grp = recs.get(rid)
        if grp is None:
            continue
        problems.extend(validate_recording(grp, prefix=f"{RECORDINGS}/{rid}: "))
        check_consolidated(path, grp, problems, subpath=f"{RECORDINGS}/{rid}",
                           prefix=f"{RECORDINGS}/{rid}: ")
        rec_vocab = None
        live = grp.attrs.get(spec.EVENT_REVISION_ATTR)
        if live is not None:
            rgrp = grp.get(spec.revision_path(int(live)))
            if rgrp is not None:
                rec_vocab = rgrp.attrs.get(spec.EVENT_VOCAB)
        else:
            ev = grp.get(spec.EVENTS)
            if ev is not None:
                rec_vocab = read_str_column(ev, spec.EVENT_VOCAB)
        if rec_vocab is not None and [str(x) for x in rec_vocab] != vocab:
            problems.append(f"{RECORDINGS}/{rid}: events vocab differs from vocab/event_labels")
        ps = grp.attrs.get("primary_stream")
        sig = grp.get(f"{spec.SIGNALS}/{ps}") if ps else None
        if sig is not None and int(sig.shape[1]) != int(n_samples[i]):
            problems.append(f"{RECORDINGS}/{rid}: index/n_samples disagrees with the signal array")
        if verify_hashes:
            got = _storeio.content_hash(root.store, f"{RECORDINGS}/{rid}/")
            if got != hashes[i]:
                problems.append(f"{RECORDINGS}/{rid}: content hash mismatch (corrupt or edited)")
    return problems
