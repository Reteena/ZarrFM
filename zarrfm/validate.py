from __future__ import annotations

import warnings

import numpy as np
import zarr

from . import _storeio, canonical, spec
from .writer import read_str_column


def _node_doc(node) -> dict:
    d = node.metadata.to_dict()
    d.pop("consolidated_metadata", None)
    return d


def _live_revision(root):
    return dict(root.attrs).get(spec.EVENT_REVISION_ATTR)


def _valid_revision(k) -> bool:
    return isinstance(k, int) and not isinstance(k, bool) and k >= 0


def _is_revision_key(key: str) -> int | None:
    parts = key.split("/")
    if len(parts) >= 2 and parts[0] == spec.EVENTS and parts[1].startswith(spec.EVENT_REVISION_PREFIX):
        try:
            return int(parts[1][len(spec.EVENT_REVISION_PREFIX):])
        except ValueError:
            return None
    return None


def _walk(group, prefix: str = ""):
    for name, arr in group.arrays():
        yield f"{prefix}{name}", arr
    for name, sub in group.groups():
        key = f"{prefix}{name}"
        yield key, sub
        yield from _walk(sub, prefix=f"{key}/")


def check_consolidated(path: str, root, problems: list[str], *, subpath: str | None = None,
                       skip_prefix: str | None = None, prefix: str = "") -> None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cached = zarr.open_consolidated(str(path), path=subpath, mode="r", zarr_format=3)
    except Exception:
        problems.append(f"{prefix}metadata is not consolidated at {subpath or 'the root'}")
        return
    live = _live_revision(root)
    seen: set[str] = set()
    for key, node in _walk(root):
        if skip_prefix and key.startswith(skip_prefix):
            continue
        rev = _is_revision_key(key)
        if rev is not None and rev != live:
            continue
        if key == spec.PROVENANCE or key.startswith(f"{spec.PROVENANCE}/"):
            seen.add(key)
            continue
        seen.add(key)
        try:
            cnode = cached[key]
        except KeyError:
            problems.append(f"{prefix}{key} is missing from the consolidated metadata")
            continue
        if _node_doc(cnode) != _node_doc(node):
            problems.append(
                f"{prefix}consolidated metadata for {key} is stale; run zarrfm.repair()"
            )
    cm = cached.metadata.consolidated_metadata
    listed = list(cm.flattened_metadata) if cm is not None else []
    for key in listed:
        if skip_prefix and key.startswith(skip_prefix):
            continue
        if key not in seen and root.get(key) is None:
            problems.append(
                f"{prefix}consolidated metadata lists {key}, which does not exist; "
                "run zarrfm.repair()"
            )


def _check_onsets(onset: np.ndarray, bounds, problems: list[str], where: str) -> None:
    if onset.size and not np.all(np.isfinite(onset)):
        problems.append(f"{where}/onset has non-finite entries")
        return
    if onset.size > 1 and np.any(np.diff(onset) < 0):
        problems.append(f"{where}/onset is not non-decreasing")
    if bounds is not None and onset.size:
        lo, hi = bounds
        out = np.count_nonzero((onset < lo) | (onset >= hi))
        if out:
            problems.append(
                f"{where}/onset: {int(out)} event(s) lie outside the primary stream "
                f"[{lo:.3f}, {hi:.3f}) s from t0; their sample index is out of range"
            )


def _check_durations(duration: np.ndarray, problems: list[str], where: str) -> None:
    if duration.size and (not np.all(np.isfinite(duration)) or np.any(duration < 0)):
        problems.append(f"{where}/duration has negative or non-finite entries")


def _validate_legacy_events(events, v02: bool, bounds, problems: list[str], p: str) -> None:
    for name, _ in events.groups():
        problems.append(
            f"{p}events/{name} is an orphan revision on a {'0.2' if v02 else '0.1'} store "
            "(interrupted migration); run zarrfm.repair()"
        )
    lengths = {}
    required = [f for f in spec.EVENT_FIELDS if v02 or f not in spec.EVENT_FIELDS_V02]
    for field in required:
        sub = events.get(field)
        if sub is None:
            problems.append(f"{p}missing events/{field}")
        else:
            lengths[field] = int(sub.shape[0])
    if len(set(lengths.values())) > 1:
        problems.append(f"{p}events arrays have mismatched lengths: {lengths}")
    onset = events.get("onset")
    if onset is not None:
        _check_onsets(np.asarray(onset[:], dtype=np.float64), bounds, problems, f"{p}events")
    dur = events.get("duration")
    if dur is not None:
        _check_durations(np.asarray(dur[:], dtype=np.float64), problems, f"{p}events")
    vocab = read_str_column(events, spec.EVENT_VOCAB)
    if v02 and vocab is None:
        problems.append(f"{p}missing events/{spec.EVENT_VOCAB}")
    lid = events.get("label_id")
    if vocab is not None and lid is not None and lid.shape[0]:
        ids = np.asarray(lid[:], dtype=np.int64)
        if ids.min() < 0 or ids.max() >= len(vocab):
            problems.append(f"{p}events/label_id indexes outside events/vocab")
        else:
            labels = read_str_column(events, "label")
            if labels is not None and any(vocab[i] != l for i, l in zip(ids, labels)):
                problems.append(f"{p}events/label_id disagrees with events/label")


def _validate_revisioned_events(root, events, live, bounds, problems: list[str], p: str) -> None:
    if live is None:
        problems.append(f"{p}root attr {spec.EVENT_REVISION_ATTR} is missing on a 0.3 store")
        return
    if not _valid_revision(live):
        problems.append(f"{p}root attr {spec.EVENT_REVISION_ATTR}={live!r} is not a non-negative integer")
        return
    if events is None:
        problems.append(f"{p}missing events/ group")
        return
    rname = f"{spec.EVENT_REVISION_PREFIX}{live}"
    numbers = []
    for name, grp in events.groups():
        try:
            j = int(name[len(spec.EVENT_REVISION_PREFIX):])
        except ValueError:
            problems.append(f"{p}events/{name} is not a revision group; run zarrfm.repair()")
            continue
        numbers.append(j)
        if j == live:
            continue
        a = dict(grp.attrs)
        if a.get(spec.TOMBSTONE_ATTR):
            if list(grp.array_keys()):
                problems.append(f"{p}events/{name} is a tombstone that still holds arrays; run zarrfm.repair()")
            continue
        problems.append(
            f"{p}events/{name} is not the live revision (interrupted or concurrent edit); "
            "run zarrfm.repair()"
        )
    missing = [j for j in range(live) if j not in numbers]
    if missing:
        problems.append(
            f"{p}events/ is missing tombstone(s) for revision(s) {missing[:8]}"
            f"{' ...' if len(missing) > 8 else ''}; run zarrfm.repair()"
        )
    for name, _ in events.arrays():
        problems.append(
            f"{p}legacy events/{name} coexists with revision {live}; run zarrfm.repair()"
        )
    grp = events.get(rname)
    if grp is None:
        problems.append(f"{p}{spec.EVENT_REVISION_ATTR}={live} but events/{rname} does not exist")
        return
    if dict(grp.attrs).get(spec.TOMBSTONE_ATTR):
        problems.append(f"{p}events/{rname} is live but marked superseded")
        return
    extra = sorted(set(grp.array_keys()) - set(spec.EVENT_ARRAY_FIELDS))
    if extra:
        problems.append(f"{p}events/{rname} carries arrays outside the spec: {extra}")
    a = dict(grp.attrs)
    n = a.get("n_events")
    vocab = a.get("vocab")
    if isinstance(n, bool) or not isinstance(n, int) or not isinstance(a.get("digest"), str) or not isinstance(vocab, list):
        problems.append(f"{p}events/{rname} lacks int n_events / str digest / list vocab attrs")
        return
    vocab = [str(v) for v in vocab]
    if len(set(vocab)) != len(vocab):
        problems.append(f"{p}events/{rname}: vocab has duplicate labels")
    cols = {}
    for field, dtype in spec.EVENT_ARRAY_FIELDS.items():
        sub = grp.get(field)
        if sub is None:
            problems.append(f"{p}missing events/{rname}/{field}")
            continue
        if int(sub.shape[0]) != n:
            problems.append(
                f"{p}events/{rname}/{field} has length {int(sub.shape[0])}, expected n_events={n}"
            )
        try:
            cols[field] = np.asarray(sub[:], dtype=dtype)
        except Exception as e:
            problems.append(f"{p}events/{rname}/{field} cannot be decoded: {type(e).__name__}: {e}")
    if len(cols) != len(spec.EVENT_ARRAY_FIELDS) or any(v.shape[0] != n for v in cols.values()):
        return
    ids = cols["label_id"]
    if ids.size and (ids.min() < 0 or ids.max() >= len(vocab)):
        problems.append(f"{p}events/{rname}/label_id indexes outside vocab")
    if spec.event_digest({**cols, "vocab": vocab}) != a["digest"]:
        problems.append(
            f"{p}events/{rname}: digest mismatch (torn edit or tampering); a reader will "
            "refuse this table"
        )
    _check_onsets(cols["onset"], bounds, problems, f"{p}events/{rname}")
    _check_durations(cols["duration"], problems, f"{p}events/{rname}")


def _validate_provenance_steps(root, live, problems: list[str], p: str) -> None:
    prov = root.get(spec.PROVENANCE)
    if prov is None:
        return
    for step in dict(prov.attrs).get("pipeline", []) or []:
        rev = step.get(spec.STEP_REVISION_KEY) if isinstance(step, dict) else None
        if isinstance(rev, int) and not isinstance(rev, bool) and (live is None or rev > live):
            problems.append(
                f"{p}provenance logs {step.get('name')} for revision {rev}, which is not "
                f"live ({live}): an uncommitted edit; run zarrfm.repair()"
            )


def _validate_signal_hashes(root, problems: list[str], p: str) -> None:
    prefix = f"{root.path}/" if root.path else ""
    signals = root.get(spec.SIGNALS)
    if signals is None:
        return
    for name, arr in signals.arrays():
        want = dict(arr.attrs).get("content_hash")
        if not want:
            continue
        got = _storeio.content_hash(root.store, f"{prefix}{spec.SIGNALS}/{name}/")
        if got != want:
            problems.append(
                f"{p}signals/{name}: content hash mismatch -- a shard was lost, truncated "
                "or altered (it would read as fill values)"
            )


def validate_recording(root, prefix: str = "", *, verify_hashes: bool = False) -> list[str]:
    problems: list[str] = []
    try:
        _validate_recording_rules(root, prefix, problems)
        if verify_hashes:
            _validate_signal_hashes(root, problems, prefix)
    except Exception as e:
        problems.append(f"{prefix}validation aborted: {type(e).__name__}: {e}")
    return problems


def _validate_recording_rules(root, prefix: str, problems: list[str]) -> None:
    p = prefix

    attrs = dict(root.attrs)
    if attrs.get("format") != spec.FORMAT:
        problems.append(f"{p}root attr format={attrs.get('format')!r}, expected {spec.FORMAT!r}")
    version = str(attrs.get("version", ""))
    if not version or not version.split(".")[0].isdigit():
        problems.append(f"{p}root attr version={attrs.get('version')!r} is not parseable")
    v02 = spec.has_v02_fields(version)
    live = _live_revision(root)
    v03 = spec.has_v03_layout(version) or live is not None
    if v03:
        _validate_provenance_steps(root, live if _valid_revision(live) else None, problems, p)

    signals = root.get(spec.SIGNALS)
    if signals is None:
        problems.append(f"{p}missing {spec.SIGNALS}/ group")
        return
    stream_names = sorted(signals.array_keys())
    if not stream_names:
        problems.append(f"{p}{spec.SIGNALS}/ contains no arrays")

    primary = attrs.get("primary_stream")
    if v02 and stream_names and primary not in stream_names:
        problems.append(f"{p}root attr primary_stream={primary!r} is not one of {stream_names}")
    if v02 and attrs.get("canonical_version") is None:
        problems.append(f"{p}root attr canonical_version is missing")

    channels = root.get(spec.CHANNELS)
    for name in stream_names:
        arr = signals[name]
        a = dict(arr.attrs)
        if arr.ndim != 2:
            problems.append(f"{p}signals/{name}: expected 2-D (channel, time), got {arr.shape}")
            continue
        n_ch, n_s = arr.shape

        sfreq = a.get("sfreq")
        if not isinstance(sfreq, (int, float)) or not sfreq > 0:
            problems.append(f"{p}signals/{name}: sfreq={sfreq!r} is not a positive number")
        if int(a.get("n_channels", -1)) != n_ch:
            problems.append(
                f"{p}signals/{name}: attr n_channels={a.get('n_channels')} != shape {n_ch}"
            )
        if int(a.get("n_samples", -1)) != n_s:
            problems.append(
                f"{p}signals/{name}: attr n_samples={a.get('n_samples')} != shape {n_s}"
            )
        storage = a.get("storage")
        if storage not in spec.STORAGE_DTYPES:
            problems.append(f"{p}signals/{name}: storage={storage!r} is not a known storage mode")
        elif np.dtype(arr.dtype) != np.dtype(spec.STORAGE_DTYPES[storage]):
            problems.append(
                f"{p}signals/{name}: dtype {arr.dtype} does not match storage={storage!r}"
            )
        if tuple(arr.chunks)[0] != n_ch and n_ch <= 128:
            problems.append(
                f"{p}signals/{name}: chunks {tuple(arr.chunks)} do not span the channel axis"
            )

        grp = None if channels is None else channels.get(name)
        if grp is None:
            problems.append(f"{p}missing channels/{name}/ for signals/{name}")
            continue
        if v03:
            for field in spec.CHANNEL_STRING_FIELDS:
                col = grp.attrs.get(field)
                if grp.get(field) is not None:
                    problems.append(
                        f"{p}channels/{name}/{field} is an array; 0.3 stores keep strings "
                        "in the group's attributes"
                    )
                elif not isinstance(col, list):
                    problems.append(f"{p}missing channels/{name} attr {field!r}")
                elif len(col) != n_ch:
                    problems.append(
                        f"{p}channels/{name} attr {field!r} has {len(col)} entries, expected {n_ch}"
                    )
            required = list(spec.CHANNEL_ARRAY_FIELDS) + [spec.POSITION_FIELD]
        else:
            required = [
                f for f in spec.CHANNEL_FIELDS if v02 or f not in spec.CHANNEL_FIELDS_V02
            ] + [spec.POSITION_FIELD]
        for field in required:
            sub = grp.get(field)
            if sub is None:
                problems.append(f"{p}missing channels/{name}/{field}")
                continue
            expected = (n_ch, 3) if field == spec.POSITION_FIELD else (n_ch,)
            if tuple(sub.shape) != expected:
                problems.append(
                    f"{p}channels/{name}/{field}: shape {tuple(sub.shape)}, expected {expected}"
                )
        names = read_str_column(grp, "name")
        if names is not None and len(names) == n_ch and len(set(names)) != n_ch:
            dupes = sorted({x for x in names if names.count(x) > 1})
            problems.append(f"{p}channels/{name}/name has duplicate entries: {dupes}")
        if grp.get("gain") is not None and tuple(grp["gain"].shape) == (n_ch,):
            gain = np.asarray(grp["gain"][:], dtype=np.float64)
            offset = np.asarray(grp["offset"][:], dtype=np.float64)
            if not np.all(np.isfinite(gain)) or np.any(gain == 0):
                problems.append(f"{p}channels/{name}/gain has zero or non-finite entries")
            if not np.all(np.isfinite(offset)):
                problems.append(f"{p}channels/{name}/offset has non-finite entries")
        types = read_str_column(grp, "type")
        if types is not None and len(types) == n_ch:
            unknown = sorted(set(types) - set(canonical.CHANNEL_TYPES))
            if unknown:
                problems.append(
                    f"{p}channels/{name}/type has values outside the spec vocabulary: {unknown}"
                )
        cid = grp.get("canonical_id")
        if cid is not None and tuple(cid.shape) == (n_ch,):
            ids = np.asarray(cid[:], dtype=np.int64)
            if np.any(ids < canonical.UNMAPPED):
                problems.append(f"{p}channels/{name}/canonical_id has values below -1")
            if attrs.get("canonical_version") == canonical.CANONICAL_VERSION and np.any(
                ids >= canonical.n_canonical()
            ):
                problems.append(
                    f"{p}channels/{name}/canonical_id exceeds the canonical table "
                    f"({canonical.n_canonical()} entries in v{canonical.CANONICAL_VERSION})"
                )
        if int(a.get("n_clipped", 0)) > 0:
            problems.append(
                f"{p}signals/{name}: {a['n_clipped']} samples saturated the int16 range "
                "(declared phys_range was too narrow)"
            )

    events = root.get(spec.EVENTS)
    bounds = None
    if stream_names:
        ps = primary if primary in stream_names else stream_names[0]
        pa = dict(signals[ps].attrs)
        try:
            off = float(pa.get("t0_offset", 0.0) or 0.0)
            bounds = (off, off + int(signals[ps].shape[1]) / float(pa.get("sfreq")))
        except (TypeError, ValueError, ZeroDivisionError):
            bounds = None

    if v03:
        _validate_revisioned_events(root, events, live, bounds, problems, p)
    elif events is not None:
        _validate_legacy_events(events, v02, bounds, problems, p)
    elif v02:
        problems.append(f"{p}missing events/ group")

    if v02 and root.get(spec.PROVENANCE) is None:
        problems.append(f"{p}missing {spec.PROVENANCE}/ group")

    segments = root.get(spec.SEGMENTS)
    if segments is not None:
        lengths = {
            f: int(segments[f].shape[0]) for f in spec.SEGMENT_FIELDS if segments.get(f) is not None
        }
        missing = set(spec.SEGMENT_FIELDS) - set(lengths)
        if missing:
            problems.append(f"{p}segments/ missing {sorted(missing)}")
        elif len(set(lengths.values())) > 1:
            problems.append(f"{p}segments arrays have mismatched lengths: {lengths}")

    return


def validate(path: str, *, verify_hashes: bool = False) -> list[str]:
    problems: list[str] = []
    try:
        root = zarr.open_group(str(path), mode="r", use_consolidated=False, zarr_format=3)
    except Exception as e:
        return [f"cannot open store: {type(e).__name__}: {e}"]
    try:
        check_consolidated(path, root, problems)
    except Exception as e:
        problems.append(f"consolidation check aborted: {type(e).__name__}: {e}")
    problems.extend(validate_recording(root, verify_hashes=verify_hashes))
    return problems


def assert_valid(path: str) -> None:
    problems = validate(path)
    if problems:
        raise AssertionError(
            f"{path} is not a conforming zarrfm store:\n  - " + "\n  - ".join(problems)
        )
