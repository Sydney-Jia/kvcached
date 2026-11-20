# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Device-independent residency contracts for native cache layouts.

An engine adapter supplies allocation identities, byte spans and live owners.
No model class, cache-spec class, tensor library or device API belongs here.
Spans describe native views: pages need not tile logical blocks, and groups
may alias an allocation. Ownership changes are atomic on the control thread.
Device completion is a separate owner; removing a request does not complete
the kernels that previously consumed its state.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from collections import OrderedDict
from itertools import product
from math import prod
from typing import Callable, Iterable, Mapping


# Diagnostic only: re-expand every block and compare with the incremental
# ledger. Never leave it on in a throughput run (it costs the full pass).
_LEDGER_CHECK = os.getenv("KVCACHED_NATIVE_LEDGER_CHECK", "0").strip() == "1"


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


@dataclass(frozen=True)
class Storage:
    """One unique native allocation, rounded only at the VMM boundary."""

    key: str
    size_bytes: int
    page_bytes: int

    def __post_init__(self):
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("Storage needs a non-empty identity")
        _integer(self.size_bytes, "size_bytes", 1)
        _integer(self.page_bytes, "page_bytes", 1)

    @property
    def reservation_bytes(self) -> int:
        return ((self.size_bytes + self.page_bytes - 1)
                // self.page_bytes * self.page_bytes)


@dataclass(frozen=True)
class BlockSpan:
    """A byte interval at offset + block_id * stride for one cache group.

    A logical block may have several spans (K/V, state components, heads,
    virtual kernel blocks). Holes inside a span are conservatively retained.
    group_id identifies an engine cache group, NOT a VMM allocation.
    """

    storage: str
    group_id: int
    offset: int
    stride: int
    length: int
    num_blocks: int

    def __post_init__(self):
        _integer(self.group_id, "group_id")
        _integer(self.offset, "offset")
        _integer(self.stride, "stride", 1)
        _integer(self.length, "length", 1)
        _integer(self.num_blocks, "num_blocks", 1)

    def interval(self, block_id: int) -> tuple[int, int]:
        _integer(block_id, "block_id")
        if block_id >= self.num_blocks:
            raise ValueError(f"Block {block_id} exceeds {self.num_blocks} blocks")
        start = self.offset + block_id * self.stride
        return start, start + self.length


def block_first_spans(*, storage: str, group_id: int, offset: int,
                      shape: tuple[int, ...], strides: tuple[int, ...],
                      itemsize: int, num_blocks: int,
                      max_segments: int = 65536) -> tuple[BlockSpan, ...]:
    """Translate a declared blocks-first native view; never guess its block axis.

    Uses the same blocks-first contract as native vLLM's KVBlockZeroer, also
    covering recurrent-state components. Splitting one scheduler block into
    several kernel blocks preserves each segment's address and stride.
    The adapter must separately account for hidden views or external owners.
    """
    _integer(itemsize, "itemsize", 1)
    _integer(num_blocks, "num_blocks", 1)
    _integer(max_segments, "max_segments", 1)
    if not shape or len(shape) != len(strides):
        raise ValueError("Shape and strides must have the same positive rank")
    for size in shape:
        _integer(size, "dimension", 1)
    for stride in strides:
        _integer(stride, "tensor stride")
    if not strides[0] or shape[0] % num_blocks:
        raise ValueError("View does not expose a positive integral blocks-first axis")
    ratio = shape[0] // num_blocks
    if not ratio:
        raise ValueError("View contains no complete logical blocks")
    block_stride = strides[0] * itemsize
    outer = [d for d in range(1, len(shape))
             if shape[d] > 1 and strides[d] * itemsize >= block_stride]
    inner = [d for d in range(1, len(shape)) if d not in outer]
    count = ratio * prod(shape[d] for d in outer)
    if count > max_segments:
        raise ValueError(f"Native view needs {count} segments; explicit layout adapter required")
    length = itemsize + sum((shape[d] - 1) * strides[d] * itemsize for d in inner)
    result = []
    for indices in product(*(range(shape[d]) for d in outer)):
        outer_offset = sum(index * strides[d] * itemsize for index, d in zip(indices, outer))
        for subblock in range(ratio):
            result.append(BlockSpan(
                storage, group_id, offset + outer_offset + subblock * block_stride,
                block_stride * ratio, length, num_blocks))
    return tuple(result)


class Layout:
    """Validated storage/view graph, independent of page/block geometry ratios."""

    def __init__(self, storages: Iterable[Storage], spans: Iterable[BlockSpan],
                 *, cache_page_entries: int = 65536):
        _integer(cache_page_entries, 'cache_page_entries')
        self.cache_page_entries = cache_page_entries
        # A speculative prefetch expands candidate blocks from another thread
        # while the engine thread keeps using the same immutable geometry.
        self._cache_lock = threading.Lock()
        self._block_pages = OrderedDict()
        self._cached_offsets = 0
        self._cache_hits = 0
        self._cache_misses = 0
        entries = tuple(storages)
        self.storages = {entry.key: entry for entry in entries}
        if not entries or len(entries) != len(self.storages):
            raise ValueError("Allocation identities must be non-empty and unique")
        self.spans = tuple(dict.fromkeys(spans))
        self.groups: dict[int, list[BlockSpan]] = {}
        self.block_counts: dict[int, int] = {}
        represented = set()
        for span in self.spans:
            if span.storage not in self.storages:
                raise ValueError(f"Unknown allocation {span.storage!r}")
            if span.interval(span.num_blocks - 1)[1] > self.storages[span.storage].size_bytes:
                raise ValueError("Native cache view exceeds its backing storage")
            if self.block_counts.setdefault(span.group_id, span.num_blocks) != span.num_blocks:
                raise ValueError("Components of one group disagree on logical block count")
            self.groups.setdefault(span.group_id, []).append(span)
            represented.add(span.storage)
        if represented != self.storages.keys():
            raise ValueError("Every allocation requires an explicit ownership description")

    def validate_blocks(self, blocks: Iterable[tuple[int, int]]) -> frozenset[tuple[int, int]]:
        result = frozenset(blocks)
        for group, block in result:
            _integer(group, "group_id")
            _integer(block, "block_id")
            if group not in self.groups or block >= self.block_counts[group]:
                raise ValueError(f"Unknown cache group/block {(group, block)!r}")
        return result

    def pages_for(self, blocks: Iterable[tuple[int, int]]) -> dict[str, frozenset[int]]:
        pages: dict[str, set[int]] = {key: set() for key in self.storages}
        for group, block in self.validate_blocks(blocks):
            for key, offsets in self._pages_for_block(group, block):
                pages[key].update(offsets)
        return {key: frozenset(offsets) for key, offsets in pages.items()}

    def _pages_for_block(self, group, block):
        identity = (group, block)
        with self._cache_lock:
            cached = self._block_pages.get(identity)
            if cached is not None:
                self._cache_hits += 1
                self._block_pages.move_to_end(identity)
                return cached
        self._cache_misses += 1
        pages = {}
        for span in self.groups[group]:
            allocation = self.storages[span.storage]
            start, end = span.interval(block)
            first = start // allocation.page_bytes * allocation.page_bytes
            stop = (end + allocation.page_bytes - 1) // allocation.page_bytes * allocation.page_bytes
            pages.setdefault(span.storage, set()).update(range(first, stop, allocation.page_bytes))
        result = tuple((key, frozenset(offsets)) for key, offsets in pages.items())
        size = sum(len(offsets) for _, offsets in result)
        # Geometry is fixed for this validated Layout. Cache only immutable
        # byte offsets, never owners, physical mappings or prefix validity.
        # Bound by offset count as well as the finite group/block domain.
        if size <= self.cache_page_entries:
            with self._cache_lock:
                while self._block_pages and self._cached_offsets + size > self.cache_page_entries:
                    _, removed = self._block_pages.popitem(last=False)
                    self._cached_offsets -= sum(len(offsets) for _, offsets in removed)
                if identity not in self._block_pages:
                    self._block_pages[identity] = result
                    self._cached_offsets += size
        return result

    def cache_status(self):
        return dict(hits=self._cache_hits, misses=self._cache_misses,
                    blocks=len(self._block_pages), page_offsets=self._cached_offsets,
                    max_page_offsets=self.cache_page_entries)


class ResidencyOwners:
    """Retain engine-owned state and in-flight execution independently.

    Adapters publish the native engine's ownership, including cached prefixes
    and copy endpoints. This class does not replace its cache policy. Execution
    owners are released only after their device completion callback succeeds.
    All methods are serialized by the caller's engine/control execution thread.
    """

    def __init__(self, layout: Layout):
        self.layout = layout
        self._owners: dict[str, frozenset[tuple[int, int]]] = {}
        self._executions: dict[str, tuple[frozenset[tuple[int, int]], Callable[[], bool]]] = {}
        self.revision = 0
        # Page bookkeeping is incremental: one block can span hundreds of
        # pages across tens of spans, so re-expanding every owner whenever a
        # single block changes costs tens of milliseconds on each scheduler
        # step. ``_pair_refs`` counts owners plus in-flight executions per
        # (group, block); ``_page_refs`` counts references per physical page.
        # Readers may run on another thread (idle reclaimer, admission
        # prefetch), so every mutation and the page snapshot share one lock.
        self._ledger_lock = threading.RLock()
        self._pair_refs: dict[tuple[int, int], int] = {}
        self._block_pages: dict[
            tuple[int, int], tuple[tuple[str, tuple[int, ...]], ...]] = {}
        self._page_refs: dict[str, dict[int, int]] = {}
        self._required: dict[str, frozenset[int]] | None = None
        self._required_dirty = True

    def _pages_of_pair(self, pair: tuple[int, int]) -> tuple[tuple[str, tuple[int, ...]], ...]:
        """Physical pages touched by one (group, block), expanded once."""
        cached = self._block_pages.get(pair)
        if cached is None:
            pages = self.layout.pages_for((pair,))
            cached = tuple((key, tuple(sorted(offsets)))
                           for key, offsets in pages.items() if offsets)
            self._block_pages[pair] = cached
        return cached

    def _acquire_pair(self, pair: tuple[int, int]) -> None:
        count = self._pair_refs.get(pair, 0)
        self._pair_refs[pair] = count + 1
        if count:
            return
        for key, pages in self._pages_of_pair(pair):
            refs = self._page_refs.setdefault(key, {})
            for page in pages:
                refs[page] = refs.get(page, 0) + 1
        self._required_dirty = True

    def _release_pair(self, pair: tuple[int, int]) -> None:
        count = self._pair_refs.get(pair, 0)
        if count > 1:
            self._pair_refs[pair] = count - 1
            return
        self._pair_refs.pop(pair, None)
        for key, pages in self._pages_of_pair(pair):
            refs = self._page_refs.get(key)
            if not refs:
                continue
            for page in pages:
                remaining = refs.get(page, 0) - 1
                if remaining > 0:
                    refs[page] = remaining
                else:
                    refs.pop(page, None)
        self._required_dirty = True

    @staticmethod
    def _check_owner(owner: str) -> None:
        if not isinstance(owner, str) or not owner:
            raise ValueError("Owner needs a non-empty identity")

    def replace(self, updates: Mapping[str, Iterable[tuple[int, int]]],
                removed: Iterable[str] = ()) -> None:
        remove = tuple(removed)
        for owner in (*updates, *remove):
            self._check_owner(owner)
        if set(updates).intersection(remove):
            raise ValueError("Cannot remove and replace the same owner")
        validated = {owner: self.layout.validate_blocks(blocks)
                     for owner, blocks in updates.items()}
        with self._ledger_lock:
            current = self._owners.copy()
            for owner in remove:
                if owner not in current:
                    raise ValueError(f"Cannot release unknown owner {owner!r}")
                del current[owner]
            current.update(validated)
            if current != self._owners:
                for owner in set(self._owners) | set(current):
                    old = self._owners.get(owner)
                    new = current.get(owner)
                    if old == new:
                        continue
                    if old:
                        for pair in old - (new or frozenset()):
                            self._release_pair(pair)
                    if new:
                        for pair in new - (old or frozenset()):
                            self._acquire_pair(pair)
                self._owners = current
                self.revision += 1

    def begin_execution(self, identity: str, blocks: Iterable[tuple[int, int]],
                        completed: Callable[[], bool]) -> None:
        self._check_owner(identity)
        if identity in self._executions or not callable(completed):
            raise ValueError("Execution identity must be unique and completion callable")
        validated = self.layout.validate_blocks(blocks)
        with self._ledger_lock:
            self._executions[identity] = (validated, completed)
            for pair in validated:
                self._acquire_pair(pair)
            self.revision += 1

    def collect_completed(self) -> int:
        # Query all fences before mutating the ledger; a failed query leaves
        # every execution protected, including those queried earlier.
        with self._ledger_lock:
            done = [key for key, (_, completed)
                    in self._executions.items() if completed()]
            for key in done:
                blocks, _ = self._executions.pop(key)
                for pair in blocks:
                    self._release_pair(pair)
            if done:
                self.revision += 1
            return len(done)

    def _full_pages(self) -> dict[str, frozenset[int]]:
        """Reference implementation: expand every owned block from scratch."""
        blocks = set().union(*self._owners.values()) if self._owners else set()
        for blockset, _ in self._executions.values():
            blocks.update(blockset)
        return self.layout.pages_for(blocks)

    def required_pages(self) -> dict[str, frozenset[int]]:
        with self._ledger_lock:
            if self._required is None or self._required_dirty:
                required = {key: frozenset(refs)
                            for key, refs in self._page_refs.items() if refs}
                for key in self.layout.storages:
                    required.setdefault(key, frozenset())
                if _LEDGER_CHECK:
                    truth = self._full_pages()
                    if any(truth[key] != required[key] for key in truth):
                        print("KVCACHED_LEDGER_MISMATCH " + json.dumps({
                            "extra": {key: sorted(set(required[key]) - truth[key])[:4]
                                      for key in truth
                                      if required[key] - truth[key]},
                            "missing": {key: sorted(truth[key] - required[key])[:4]
                                        for key in truth if truth[key] - required[key]},
                            "owners": len(self._owners),
                        }, sort_keys=True), flush=True)
                        # Publishing the full set keeps a live request's pages
                        # mapped; a mismatch must never retire them.
                        required = truth
                self._required = required
                self._required_dirty = False
            # Values are immutable frozensets; a shallow copy is enough to
            # keep callers from adding storages to the cached mapping.
            return dict(self._required)

    def snapshot(self) -> dict:
        return {"revision": self.revision, "owners": len(self._owners),
                "inflight_executions": len(self._executions),
                "required_pages": {key: sorted(pages) for key, pages in self.required_pages().items()}}
