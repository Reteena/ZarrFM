from __future__ import annotations

import contextlib
import json
import os
import uuid
import warnings
from typing import Iterable, Sequence

import numpy as np
import zarr

from . import _storeio, spec
from .errors import EditConflictError, ZarrFMIntegrityError
from .reader import decode_legacy_events, decode_revision, _empty_events
from .writer import (
    _now_iso,
    build_event_table,
    read_str_column,
    revision_attrs,
    write_revision_columns,
)

_LEGACY_EVENT_KEYS = tuple(f"{spec.EVENTS}/{f}" for f in (*spec.EVENT_FIELDS, spec.EVENT_VOCAB))


def _open_rw(path: str):
    return zarr.open_group(str(path), mode="r+", use_consolidated=False, zarr_format=3)


def _normalise(path) -> str:
    return os.path.realpath(str(path)).rstrip(os.sep)


def is_tombstone(doc: dict) -> bool:
    return bool(doc.get("attributes", {}).get(spec.TOMBSTONE_ATTR))


def tombstone_doc(by: int) -> bytes:
    return json.dumps(
        {"zarr_format": 3, "node_type": "group",
         "attributes": {spec.TOMBSTONE_ATTR: True, "superseded_by": int(by)}},
        indent=2,
    ).encode("utf-8")


@contextlib.contextmanager
def store_lock(path: str, enabled: bool = True):
    if not enabled:
        yield False
        return
    try:
        import fcntl
    except ImportError:
        yield False
        return
    lock_path = f"{_normalise(path)}.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
    except OSError as e:
        warnings.warn(
            f"cannot create the store lock {lock_path} ({e}); editing without it -- "
            "concurrent local editors will conflict instead of waiting",
            stacklevel=3,
        )
        yield False
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield True
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _nested_in_dataset(path: str) -> bool:
    parent = os.path.join(os.path.dirname(os.path.dirname(path)), "zarr.json")
    if not os.path.exists(parent):
        return False
    try:
        pattrs = json.loads(open(parent).read()).get("attributes", {})
    except (OSError, ValueError):
        return False
    return pattrs.get("format") == "zarrfm-dataset"


class ZarrFMEditor:
    def __init__(self, path: str, *, lock: bool = True):
        self.path = _normalise(path)
        if not os.path.exists(os.path.join(self.path, "zarr.json")):
            raise FileNotFoundError(f"{self.path} is not a zarr store")
        root = _open_rw(self.path)
        attrs = dict(root.attrs)
        if attrs.get("format") != spec.FORMAT:
            raise ValueError(f"{self.path!r} is not an zarrfm recording store")
        if not spec.has_v02_fields(attrs.get("version")):
            raise ValueError(
                "editing needs a 0.2.0+ store (canonical channel table); rewrite this "
                f"{attrs.get('version')} store with the current writer first"
            )
        if _nested_in_dataset(self.path):
            raise NotImplementedError(
                "this recording is nested in a dataset store; its index/ (n_events, "
                "canonical_present, content_hash) is derived from it and would go "
                "stale -- rebuild the dataset instead of editing in place"
            )
        self.store = root.store
        self._lock = lock
        self._closed = False


    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("editor is closed")

    def _current(self):
        root = _open_rw(self.path)
        attrs = dict(root.attrs)
        live = attrs.get(spec.EVENT_REVISION_ATTR)
        if live is not None:
            if isinstance(live, bool) or not isinstance(live, int) or live < 0:
                raise ZarrFMIntegrityError(
                    f"root attr {spec.EVENT_REVISION_ATTR}={live!r} is not a non-negative integer"
                )
        elif spec.has_v03_layout(attrs.get("version")):
            raise ZarrFMIntegrityError(
                f"a {attrs.get('version')} store without {spec.EVENT_REVISION_ATTR}: refusing "
                "to guess which revision is live"
            )
        return root, attrs, live

    @property
    def events_revision(self) -> int | None:
        return self._current()[2]

    def _read_live_table(self, root, attrs, live: int | None) -> dict:
        try:
            if live is not None:
                grp = root.get(spec.revision_path(live))
                if grp is None:
                    raise KeyError(spec.revision_path(live))
                return decode_revision(grp, verify=True)
            grp = root.get(spec.EVENTS)
            if grp is None or grp.get("onset") is None:
                return _empty_events()
            return decode_legacy_events(grp)
        except (KeyError, ZarrFMIntegrityError) as e:
            now = dict(_open_rw(self.path).attrs).get(spec.EVENT_REVISION_ATTR)
            if now != live:
                raise EditConflictError(
                    f"the store was edited concurrently (live revision {live} -> {now}); retry"
                ) from e
            if isinstance(e, ZarrFMIntegrityError):
                raise
            raise ZarrFMIntegrityError(f"live revision {live} cannot be read: {e}") from e

    def _log(self, root, step: dict) -> None:
        grp = root.require_group(spec.PROVENANCE)
        cur = dict(grp.attrs)
        pipeline = list(cur.get("pipeline", [])) + [spec.json_safe(step)]
        grp.attrs.put({**cur, "pipeline": pipeline, "written": _now_iso()})


    def _commit_table(self, table: dict, step: dict, root, attrs: dict, live: int | None) -> int:
        nk = 0 if live is None else live + 1
        rpath = spec.revision_path(nk)

        attrs_doc = {**revision_attrs(table), "claim": uuid.uuid4().hex}
        doc = {"zarr_format": 3, "node_type": "group", "attributes": attrs_doc}
        claim = json.dumps(doc, indent=2).encode("utf-8")
        if not _storeio.set_if_not_exists(self.store, f"{rpath}/zarr.json", claim):
            raise EditConflictError(
                f"{rpath} already exists: another editor committed on top of revision "
                f"{live} or is mid-edit (retry), or an interrupted edit left the claim "
                "behind (run zarrfm.repair(path); until then every edit from this base "
                "conflicts)"
            )

        try:
            grp = zarr.open_group(
                store=self.store, path=rpath, mode="r+", use_consolidated=False, zarr_format=3
            )
            write_revision_columns(grp, table)
        except Exception as e:
            if _storeio.read_key(self.store, f"{rpath}/zarr.json") != claim:
                raise EditConflictError(f"{rpath} was repaired away during the edit; retry") from e
            raise

        try:
            check = zarr.open_group(
                store=self.store, path=rpath, mode="r", use_consolidated=False, zarr_format=3
            )
            decode_revision(check, verify=True)
        except (ZarrFMIntegrityError, KeyError) as e:
            if _storeio.read_key(self.store, f"{rpath}/zarr.json") != claim:
                raise EditConflictError(f"{rpath} was repaired away during the edit; retry") from e
            raise

        superseded: list[str] = []
        if live is None:
            superseded.extend(self._migrate_channel_strings(root))
            superseded.extend(_LEGACY_EVENT_KEYS)
        else:
            superseded.append(spec.revision_path(live))

        self._log(root, {**step, spec.STEP_REVISION_KEY: nk, "at": _now_iso()})

        if _storeio.read_key(self.store, f"{rpath}/zarr.json") != claim:
            raise EditConflictError(f"{rpath} was taken over or repaired during the edit; retry")

        new_attrs = {**attrs, spec.EVENT_REVISION_ATTR: nk, "version": spec.VERSION}
        _storeio.commit_root(self.store, "", attributes=new_attrs, exclude=tuple(superseded),
                             live_revision=nk)

        for key in superseded:
            if key.startswith(f"{spec.EVENTS}/{spec.EVENT_REVISION_PREFIX}"):
                _storeio.write_key(self.store, f"{key}/zarr.json", tombstone_doc(nk))
                for col in spec.EVENT_ARRAY_FIELDS:
                    _storeio.delete_prefix(self.store, f"{key}/{col}")
            else:
                _storeio.delete_prefix(self.store, key)
        return int(len(table["onset"]))

    def _migrate_channel_strings(self, root) -> list[str]:
        superseded = []
        channels = root.get(spec.CHANNELS)
        if channels is None:
            return superseded
        for name, grp in channels.groups():
            cur = dict(grp.attrs)
            new = dict(cur)
            for field in spec.CHANNEL_STRING_FIELDS:
                if grp.get(field) is not None:
                    new[field] = read_str_column(grp, field)
                    superseded.append(f"{spec.CHANNELS}/{name}/{field}")
            if new != cur:
                grp.attrs.put(new)
        return superseded

    def append_events(
        self,
        onset: Iterable[float],
        duration: Iterable[float] | None = None,
        label: Iterable[str] | None = None,
        value: Iterable[int] | None = None,
        *,
        vocab: Sequence[str] | None = None,
        note: str | None = None,
    ) -> int:
        self._check_open()
        new_onset = np.asarray(list(onset), dtype=np.float64)
        n_new = new_onset.size
        new_labels = [str(x) for x in (label if label is not None else [""] * n_new)]
        new_dur = np.asarray(list(duration) if duration is not None else np.zeros(n_new), dtype=np.float64)
        new_val = np.asarray(list(value) if value is not None else np.zeros(n_new), dtype=np.int64)
        with store_lock(self.path, self._lock):
            root, attrs, live = self._current()
            base = self._read_live_table(root, attrs, live)
            if vocab is None:
                have = list(base["vocab"])
                vocab = have if set(new_labels) <= set(have) else sorted(set(have) | set(new_labels))
            table = build_event_table(
                np.concatenate([base["onset"], new_onset]),
                np.concatenate([base["duration"], new_dur]),
                list(base["label"]) + new_labels,
                np.concatenate([base["value"], new_val]),
                vocab,
            )
            step = {"name": "append_events", "n_added": int(n_new)}
            if note:
                step["note"] = str(note)
            return self._commit_table(table, step, root, attrs, live)

    def set_events(
        self,
        onset: Iterable[float],
        duration: Iterable[float] | None = None,
        label: Iterable[str] | None = None,
        value: Iterable[int] | None = None,
        *,
        vocab: Sequence[str] | None = None,
        note: str | None = None,
    ) -> int:
        self._check_open()
        table = build_event_table(onset, duration, label, value, vocab)
        step = {"name": "set_events", "n_events": int(len(table["onset"]))}
        if note:
            step["note"] = str(note)
        with store_lock(self.path, self._lock):
            root, attrs, live = self._current()
            return self._commit_table(table, step, root, attrs, live)


    def set_bad(
        self,
        channels,
        stream: str | None = None,
        *,
        bad: bool = True,
        note: str | None = None,
    ) -> list[str]:
        self._check_open()
        with store_lock(self.path, self._lock):
            root, attrs, live = self._current()
            streams = sorted(root[spec.SIGNALS].array_keys())
            if stream is None:
                if len(streams) != 1:
                    raise ValueError(f"specify a stream; store has {streams}")
                stream = streams[0]
            if stream not in streams:
                raise KeyError(f"unknown stream {stream!r}; have {streams}")
            grp = root[f"{spec.CHANNELS}/{stream}"]
            names = read_str_column(grp, "name") or []
            n = len(names)
            mask = np.asarray(grp["bad"][:], dtype=bool).copy()

            arr = np.asarray(channels)
            if arr.dtype == bool:
                if arr.shape != (n,):
                    raise ValueError(f"bool mask must have shape ({n},), got {arr.shape}")
                mask = arr.copy()
            else:
                index = {nm: i for i, nm in enumerate(names)}
                items = [channels] if isinstance(channels, (str, int, np.integer)) else list(channels)
                for c in items:
                    if isinstance(c, (int, np.integer)):
                        if not -n <= int(c) < n:
                            raise IndexError(f"channel index {c} out of range for {n} channels")
                        mask[int(c) % n] = bad
                    elif c in index:
                        mask[index[c]] = bad
                    else:
                        raise KeyError(f"no channel named {c!r} in stream {stream!r}")

            grp["bad"][:] = mask
            now_bad = [nm for nm, b in zip(names, mask) if b]
            step = {"name": "set_bad", "stream": stream, "channels": now_bad}
            if note:
                step["note"] = str(note)
            self._log(root, {**step, "at": _now_iso()})
            return now_bad


    def close(self) -> None:
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def append_events(path: str, onset, duration=None, label=None, value=None, *,
                  vocab=None, note=None, lock: bool = True) -> int:
    with ZarrFMEditor(path, lock=lock) as e:
        return e.append_events(onset, duration, label, value, vocab=vocab, note=note)


def set_bad(path: str, channels, stream: str | None = None, *, bad: bool = True,
            note=None, lock: bool = True) -> list[str]:
    with ZarrFMEditor(path, lock=lock) as e:
        return e.set_bad(channels, stream, bad=bad, note=note)


def repair(path: str, *, lock: bool = True) -> list[str]:
    path = _normalise(path)
    with store_lock(path, lock):
        root = _open_rw(path)
        attrs = dict(root.attrs)
        if attrs.get("format") != spec.FORMAT:
            raise ValueError(f"{path!r} is not an zarrfm recording store")
        live = attrs.get(spec.EVENT_REVISION_ATTR)
        if live is None and spec.has_v03_layout(attrs.get("version")):
            raise ZarrFMIntegrityError(
                f"{attrs.get('version')} store without {spec.EVENT_REVISION_ATTR}: refusing to "
                "guess which revision is live"
            )
        if live is not None and (isinstance(live, bool) or not isinstance(live, int) or live < 0):
            raise ZarrFMIntegrityError(f"{spec.EVENT_REVISION_ATTR}={live!r} is not a valid revision")
        actions: list[str] = []
        events = root.get(spec.EVENTS)
        if events is not None:
            if live is not None:
                grp = events.get(f"{spec.EVENT_REVISION_PREFIX}{live}")
                if grp is None:
                    raise ZarrFMIntegrityError(f"live revision {spec.revision_path(live)} is missing")
                decode_revision(grp, verify=True)
            numbers = []
            for name, grp in list(events.groups()):
                try:
                    j = int(name[len(spec.EVENT_REVISION_PREFIX):])
                except ValueError:
                    _storeio.delete_prefix(root.store, f"{spec.EVENTS}/{name}")
                    actions.append(f"deleted {spec.EVENTS}/{name} (not a revision)")
                    continue
                numbers.append(j)
                if live is not None and j == live:
                    continue
                key = f"{spec.EVENTS}/{name}"
                if live is None or j > live:
                    now = dict(_open_rw(path).attrs).get(spec.EVENT_REVISION_ATTR)
                    if now != live:
                        raise EditConflictError(
                            f"the store was edited during repair (live {live} -> {now}); rerun"
                        )
                    _storeio.delete_prefix(root.store, key)
                    actions.append(f"deleted {key}")
                    continue
                doc = json.loads(_storeio.read_key(root.store, f"{key}/zarr.json") or b"{}")
                if not is_tombstone(doc) or list(grp.array_keys()):
                    for col in list(grp.array_keys()):
                        _storeio.delete_prefix(root.store, f"{key}/{col}")
                    _storeio.write_key(root.store, f"{key}/zarr.json", tombstone_doc(live))
                    actions.append(f"tombstoned {key}")
            if live is not None:
                for j in range(0, live):
                    if j not in numbers:
                        key = spec.revision_path(j)
                        _storeio.write_key(root.store, f"{key}/zarr.json", tombstone_doc(live))
                        actions.append(f"recreated tombstone {key}")
            if live is not None:
                for name, _ in list(events.arrays()):
                    _storeio.delete_prefix(root.store, f"{spec.EVENTS}/{name}")
                    actions.append(f"deleted legacy {spec.EVENTS}/{name}")
        if live is not None:
            channels = root.get(spec.CHANNELS)
            for name, grp in (channels.groups() if channels is not None else []):
                for field in spec.CHANNEL_STRING_FIELDS:
                    if grp.get(field) is not None and isinstance(grp.attrs.get(field), list):
                        _storeio.delete_prefix(root.store, f"{spec.CHANNELS}/{name}/{field}")
                        actions.append(f"deleted legacy {spec.CHANNELS}/{name}/{field}")
        prov = root.get(spec.PROVENANCE)
        if prov is not None:
            cur = dict(prov.attrs)
            steps = list(cur.get("pipeline", []))
            kept = [st for st in steps if not _uncommitted_step(st, live)]
            if len(kept) != len(steps):
                prov.attrs.put({**cur, "pipeline": kept, "written": _now_iso()})
                actions.append(f"pruned {len(steps) - len(kept)} provenance step(s) of uncommitted edits")
        try:
            _storeio.commit_root(root.store, "", expect_revision=live)
        except RuntimeError as e:
            raise EditConflictError(str(e)) from e
        actions.append("reconsolidated")
        return actions


def _uncommitted_step(step, live) -> bool:
    if not isinstance(step, dict):
        return False
    rev = step.get(spec.STEP_REVISION_KEY)
    if rev is None or isinstance(rev, bool) or not isinstance(rev, int):
        return False
    return live is None or rev > live
