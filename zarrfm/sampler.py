from __future__ import annotations

import os
from typing import Iterator

import numpy as np

from . import canonical
from .dataset import DatasetReader

DEFAULT_BLOCK_BYTES = 8 * 1024**2


class WindowSampler:
    def __init__(
        self,
        dataset: DatasetReader,
        split: str | None = "train",
        *,
        window_s: float | None = None,
        window_samples: int | None = None,
        stride_s: float | None = None,
        stride_samples: int | None = None,
        stream: str | None = None,
        recordings=None,
        canonical_space: bool = True,
        n_canonical: int | None = None,
        shuffle: str | bool = "block",
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        worker_id: int = 0,
        num_workers: int = 1,
        block_bytes: int = DEFAULT_BLOCK_BYTES,
        dtype=np.float32,
        with_events: bool = True,
    ):
        self.ds = dataset
        self.stream = stream
        self.canonical_space = bool(canonical_space)
        self.n_canonical = int(n_canonical) if n_canonical else canonical.n_canonical()
        if shuffle is True:
            shuffle = "block"
        if shuffle not in ("block", "window", False):
            raise ValueError("shuffle must be 'block', 'window' or False")
        self.shuffle = shuffle
        self.seed = int(seed)
        self.epoch = 0
        self.rank, self.world_size = int(rank), int(world_size)
        self.worker_id, self.num_workers = int(worker_id), int(num_workers)
        self.dtype = np.dtype(dtype)
        self.with_events = with_events
        self.stats = {"reads": 0, "samples_read": 0, "windows": 0}

        if recordings is None:
            if split is None:
                recordings = np.arange(len(dataset))
            else:
                recordings = dataset.split(split)
        self.recordings = np.asarray(recordings, dtype=np.int64)
        idx = dataset.index
        rates = np.unique(idx["sfreq"][self.recordings]) if self.recordings.size else np.array([])
        if rates.size > 1:
            raise ValueError(
                f"recordings in this split have mixed sampling rates {rates.tolist()}; "
                "harmonise at conversion time, the loader will not resample"
            )
        self.sfreq = float(rates[0]) if rates.size else float("nan")
        if window_samples is None:
            if window_s is None:
                raise ValueError("give window_s or window_samples")
            if not rates.size:
                raise ValueError(f"split {split!r} has no recordings; give window_samples "
                                 "to build an empty sampler")
            window_samples = int(round(float(window_s) * self.sfreq))
        self.window = int(window_samples)
        if self.window <= 0:
            raise ValueError("window must be positive")
        if stride_samples is None:
            stride_samples = (
                self.window if stride_s is None else int(round(float(stride_s) * self.sfreq))
            )
        self.stride = max(1, int(stride_samples))
        self.block_bytes = int(block_bytes)
        self._blocks = self._enumerate()
        self._events: dict[int, np.ndarray] = {}


    def _stream_for(self, rec: int) -> str:
        return self.stream or str(self.ds.index["primary_stream"][rec])

    def _enumerate(self) -> list[tuple[int, int, int, np.ndarray]]:
        blocks = []
        idx = self.ds.index
        W, S = self.window, self.stride
        for rec in self.recordings:
            rec = int(rec)
            n = int(idx["n_samples"][rec])
            if n < W:
                continue
            starts = np.arange(0, n - W + 1, S, dtype=np.int64)
            r = self.ds.recording(rec)
            arr = r._stream(self._stream_for(rec))["array"]
            chunk_t = int(arr.chunks[1])
            row_bytes = int(arr.shape[0]) * arr.dtype.itemsize
            per_block = max(1, self.block_bytes // max(chunk_t * row_bytes, 1))
            block_t = chunk_t * per_block
            if arr.shards:
                block_t = min(block_t, int(arr.shards[1]))
            block_t = max(block_t, W)
            first = 0
            while first < n:
                lo, hi = first, min(first + block_t, n)
                sel = starts[(starts >= lo) & (starts < hi)]
                if sel.size:
                    blocks.append((rec, lo, hi, sel))
                first = hi
        return blocks

    @property
    def n_windows_total(self) -> int:
        return int(sum(b[3].size for b in self._blocks))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng([self.seed, self.epoch])

    def _shard_id(self) -> tuple[int, int]:
        return self.rank * self.num_workers + self.worker_id, self.world_size * self.num_workers

    def _my_blocks(self) -> list[tuple[int, int, int, np.ndarray]]:
        order = np.arange(len(self._blocks))
        if self.shuffle:
            order = self._rng().permutation(order)
        me, n = self._shard_id()
        return [self._blocks[i] for i in order[me::n]]

    def _my_windows(self) -> np.ndarray:
        table = np.concatenate(
            [np.stack([np.full(b[3].size, b[0]), b[3]], axis=1) for b in self._blocks]
        ) if self._blocks else np.zeros((0, 2), dtype=np.int64)
        order = self._rng().permutation(table.shape[0])
        me, n = self._shard_id()
        return table[order[me::n]]

    def __len__(self) -> int:
        if self.shuffle == "window":
            return int(self._my_windows().shape[0])
        return int(sum(b[3].size for b in self._my_blocks()))


    def _read(self, rec: int, lo: int, hi: int):
        r = self.ds.recording(rec)
        st = self._stream_for(rec)
        x = r.read_samples(st, lo, hi, dtype=self.dtype)
        self.stats["reads"] += 1
        self.stats["samples_read"] += int(x.shape[1])
        if not self.canonical_space:
            info = r._stream(st)
            return x, ~info["bad"]
        info = r._stream(st)
        out, present, source = canonical.project(
            x, info["canonical_id"], self.n_canonical, bad=info["bad"]
        )
        good = present & ~np.where(source >= 0, info["bad"][np.maximum(source, 0)], True)
        return out, good

    def _events_for(self, rec: int) -> np.ndarray:
        if rec not in self._events:
            r = self.ds.recording(rec)
            ev = r.event_matrix(self._stream_for(rec)) if self.with_events else np.zeros((0, 3), np.int64)
            self._events[rec] = ev[np.argsort(ev[:, 0], kind="stable")] if ev.size else ev
        return self._events[rec]

    def _window_events(self, rec: int, start: int) -> np.ndarray:
        ev = self._events_for(rec)
        if ev.shape[0] == 0:
            return ev
        lo = np.searchsorted(ev[:, 0], start, side="left")
        hi = np.searchsorted(ev[:, 0], start + self.window, side="left")
        out = ev[lo:hi].copy()
        out[:, 0] -= start
        return out

    def _item(self, rec, start, block, mask, lo) -> dict:
        s = int(start) - int(lo)
        item = {
            "signals": block[:, s : s + self.window],
            "mask": mask,
            "recording": int(rec),
            "start": int(start),
        }
        if self.with_events:
            item["events"] = self._window_events(int(rec), int(start))
        self.stats["windows"] += 1
        return item

    def __iter__(self) -> Iterator[dict]:
        if self.shuffle == "window":
            for rec, start in self._my_windows():
                block, mask = self._read(int(rec), int(start), int(start) + self.window)
                yield self._item(rec, start, block, mask, start)
            return
        rng = self._rng()
        for rec, lo, hi, starts in self._my_blocks():
            block, mask = self._read(rec, lo, min(hi + self.window - 1, int(self.ds.index["n_samples"][rec])))
            order = rng.permutation(starts.size) if self.shuffle else np.arange(starts.size)
            for k in order:
                yield self._item(rec, starts[k], block, mask, lo)

    def batches(self, batch_size: int, drop_last: bool = False) -> Iterator[dict]:
        buf: list[dict] = []
        for item in self:
            buf.append(item)
            if len(buf) == batch_size:
                yield collate(buf)
                buf = []
        if buf and not drop_last:
            yield collate(buf)


def collate(items: list[dict]) -> dict:
    out = {
        "signals": np.stack([it["signals"] for it in items]),
        "mask": np.stack([it["mask"] for it in items]),
        "recording": np.asarray([it["recording"] for it in items], dtype=np.int64),
        "start": np.asarray([it["start"] for it in items], dtype=np.int64),
    }
    if "events" in items[0]:
        n = np.asarray([it["events"].shape[0] for it in items], dtype=np.int64)
        e_max = int(n.max()) if n.size else 0
        ev = np.full((len(items), e_max, 3), -1, dtype=np.int64)
        for i, it in enumerate(items):
            ev[i, : it["events"].shape[0]] = it["events"]
        out["events"] = ev
        out["n_events"] = n
    return out


def _torch_context() -> tuple[int, int, int, int]:
    import torch

    rank, world = 0, 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    else:
        rank = int(os.environ.get("RANK", 0))
        world = int(os.environ.get("WORLD_SIZE", 1))
    wi = torch.utils.data.get_worker_info()
    if wi is None:
        return rank, world, 0, 1
    return rank, world, wi.id, wi.num_workers


def torch_dataset(path: str, batch_size: int = 32, **sampler_kwargs):
    import torch

    class _Windows(torch.utils.data.IterableDataset):
        def __init__(self):
            self.path, self.batch_size, self.kw = str(path), int(batch_size), dict(sampler_kwargs)
            self.epoch = 0

        def set_epoch(self, epoch: int) -> None:
            self.epoch = int(epoch)

        def __iter__(self):
            rank, world, wid, nw = _torch_context()
            sampler = WindowSampler(
                DatasetReader(self.path), rank=rank, world_size=world,
                worker_id=wid, num_workers=nw, **self.kw,
            )
            sampler.set_epoch(self.epoch)
            for batch in sampler.batches(self.batch_size):
                yield {
                    k: (torch.from_numpy(np.ascontiguousarray(v)) if isinstance(v, np.ndarray) else v)
                    for k, v in batch.items()
                }

    return _Windows()
