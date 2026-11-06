from __future__ import annotations

import hashlib
import json
import re

from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync


async def _list(store, prefix: str) -> list[str]:
    return [k async for k in store.list_prefix(prefix)]


def list_keys(store, prefix: str = "") -> list[str]:
    return sorted(sync(_list(store, prefix)))


async def _get(store, key: str) -> bytes:
    buf = await store.get(key, prototype=default_buffer_prototype())
    return b"" if buf is None else buf.to_bytes()


def read_key(store, key: str) -> bytes:
    return sync(_get(store, key))


async def _set(store, key: str, data: bytes) -> None:
    await store.set(key, default_buffer_prototype().buffer.from_bytes(data))


def write_key(store, key: str, data: bytes) -> None:
    sync(_set(store, key, data))


def store_bytes(store, prefix: str = "") -> int:
    return int(sync(store.getsize_prefix(prefix)))


def content_hash(store, prefix: str = "", *, skip_metadata: bool = True) -> str:
    h = hashlib.sha256()
    for key in list_keys(store, prefix):
        if skip_metadata and key.rsplit("/", 1)[-1] == "zarr.json":
            continue
        h.update(key.encode("utf-8"))
        h.update(b"\0")
        h.update(read_key(store, key))
        h.update(b"\0")
    return h.hexdigest()


async def _exists(store, key: str) -> bool:
    return bool(await store.exists(key))


def exists(store, key: str) -> bool:
    return sync(_exists(store, key))


async def _delete_prefix(store, prefix: str) -> None:
    await store.delete_dir(prefix)


def delete_prefix(store, prefix: str) -> None:
    sync(_delete_prefix(store, prefix))


async def _set_if_not_exists(store, key: str, data: bytes) -> None:
    await store.set_if_not_exists(key, default_buffer_prototype().buffer.from_bytes(data))


def set_if_not_exists(store, key: str, data: bytes) -> bool:
    sync(_set_if_not_exists(store, key, data))
    return read_key(store, key) == data


def _group_stub() -> dict:
    return {"kind": "inline", "must_understand": False, "metadata": {}}


_REVISION_RE = re.compile(r"^events/r(\d+)(?:/|$)")


def revision_of(rel: str) -> int | None:
    m = _REVISION_RE.match(rel)
    return int(m.group(1)) if m else None


def build_consolidated_doc(store, prefix: str = "", *, attributes: dict | None = None,
                           exclude: tuple[str, ...] = (), live_revision=None) -> dict:
    root_key = f"{prefix}zarr.json"
    doc = json.loads(read_key(store, root_key))
    if attributes is not None:
        doc["attributes"] = attributes
    if live_revision is None:
        live_revision = doc.get("attributes", {}).get("events_revision")
    flat: dict[str, dict] = {}
    for key in list_keys(store, prefix):
        if not key.endswith("zarr.json") or key == root_key:
            continue
        rel = key[len(prefix):-len("/zarr.json")]
        if any(rel == e or rel.startswith(e + "/") for e in exclude):
            continue
        rev = revision_of(rel)
        if rev is not None and rev != live_revision:
            continue
        raw = read_key(store, key)
        if not raw:
            continue
        try:
            node = json.loads(raw)
        except ValueError:
            continue
        node.pop("consolidated_metadata", None)
        node.setdefault("attributes", {})
        if node.get("node_type") == "group":
            node["consolidated_metadata"] = _group_stub()
        flat[rel] = node
    for rel in list(flat):
        parent = rel.rsplit("/", 1)[0] if "/" in rel else None
        if parent is not None and parent not in flat:
            del flat[rel]
    doc.pop("consolidated_metadata", None)
    doc["consolidated_metadata"] = {
        "kind": "inline",
        "must_understand": False,
        "metadata": {k: flat[k] for k in sorted(flat)},
    }
    return doc


def commit_root(store, prefix: str = "", *, attributes: dict | None = None,
                exclude: tuple[str, ...] = (), live_revision=None,
                expect_revision=None) -> None:
    doc = build_consolidated_doc(store, prefix, attributes=attributes, exclude=exclude,
                                 live_revision=live_revision)
    if expect_revision is not None or attributes is None:
        now = json.loads(read_key(store, f"{prefix}zarr.json")).get("attributes", {})
        current = now.get("events_revision")
        want = expect_revision if expect_revision is not None else doc["attributes"].get("events_revision")
        if current != want:
            raise RuntimeError(
                f"root events_revision moved ({want} -> {current}) while reconsolidating; "
                "another editor committed -- rerun"
            )
        if attributes is None:
            doc["attributes"] = now
    write_key(store, f"{prefix}zarr.json", json.dumps(doc, indent=2).encode("utf-8"))
