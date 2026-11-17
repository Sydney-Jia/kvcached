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
from itertools import product
from math import prod
from typing import Callable, Iterable, Mapping


# Diagnostic only: re-expand every block and compare with the incremental
# ledger.  Never leave it on in a throughput run (it costs the full pass).
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
                 block_sizes: Mapping[int, int] | None = None):
        entries = tuple(storages)
        self.storages = {entry.key: entry for entry in entries}
        if not entries or len(entries) != len(self.storages):
            raise ValueError("Allocation identities must be non-empty and unique")
        self.spans = tuple(dict.fromkeys(spans))
        self.groups: dict[int, list[BlockSpan]] = {}
        self.block_counts: dict[int, int] = {}
        self.block_sizes = dict(block_sizes or {})
        for group_id, size in self.block_sizes.items():
            _integer(group_id, "group_id")
            _integer(size, "block_size", 1)
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
            for span in self.groups[group]:
                allocation = self.storages[span.storage]
                start, end = span.interval(block)
                first = start // allocation.page_bytes * allocation.page_bytes
                stop = (end + allocation.page_bytes - 1) // allocation.page_bytes * allocation.page_bytes
                pages[span.storage].update(range(first, stop, allocation.page_bytes))
        return {key: frozenset(offsets) for key, offsets in pages.items()}

    def pages_per_block_upper(self, group_id: int) -> int:
        """Conservative pages touched by one block, independent of alignment.

        A block interval has a fixed length but can start at any page offset,
        so it may straddle one page more than ``ceil(length / page)``. Overlaps
        between spans are deliberately not coalesced: the result is an upper
        bound for admission, never a layout identity.
        """
        if group_id not in self.groups:
            raise ValueError(f"Unknown cache group {group_id!r}")
        total = 0
        for span in self.groups[group_id]:
            page = self.storages[span.storage].page_bytes
            total += (span.length + page - 1) // page + 1
        return total

    def block_bytes_upper(self, group_id: int) -> int:
        """Conservative physical bytes needed for one block of a group."""
        page = min(self.storages[span.storage].page_bytes
                   for span in self.groups[group_id])
        return self.pages_per_block_upper(group_id) * page

    def average_block_bytes(self, group_id: int) -> int:
        """Mean unique-page bytes of one block in a group.

        Hybrid layouts alias many spans onto the same pages, so neither the
        sum of span lengths nor the maximum block is representative.  Admission
        uses the mean and relies on the exact block-pool check for peaks.
        """
        cached = getattr(self, '_average_block_bytes', None)
        if cached is None:
            cached = {}
            self._average_block_bytes = cached
        if group_id in cached:
            return cached[group_id]
        if group_id not in self.groups:
            raise ValueError(f"Unknown cache group {group_id!r}")
        total = 0
        count = self.block_counts[group_id]
        for block in range(count):
            pages: dict[str, set[int]] = {key: set() for key in self.storages}
            for span in self.groups[group_id]:
                allocation = self.storages[span.storage]
                start, end = span.interval(block)
                first = start // allocation.page_bytes * allocation.page_bytes
                stop = ((end + allocation.page_bytes - 1)
                        // allocation.page_bytes * allocation.page_bytes)
                pages[span.storage].update(
                    range(first, stop, allocation.page_bytes))
            total += sum(len(offsets) * self.storages[key].page_bytes
                         for key, offsets in pages.items())
        cached[group_id] = total // max(count, 1)
        return cached[group_id]

    def max_block_bytes(self, group_id: int) -> int:
        """Largest exact page union of any one block in a group.

        ``pages_per_block_upper`` deliberately sums spans without coalescing
        overlaps, which can overstate a hybrid K/V/state block by several
        times and make admission far too conservative.  This walks the real
        block IDs once and caches the largest unique-page set.
        """
        cached = getattr(self, '_max_block_bytes', None)
        if cached is None:
            cached = {}
            self._max_block_bytes = cached
        if group_id in cached:
            return cached[group_id]
        if group_id not in self.groups:
            raise ValueError(f"Unknown cache group {group_id!r}")
        best = 0
        for block in range(self.block_counts[group_id]):
            pages: dict[str, set[int]] = {key: set() for key in self.storages}
            for span in self.groups[group_id]:
                allocation = self.storages[span.storage]
                start, end = span.interval(block)
                first = start // allocation.page_bytes * allocation.page_bytes
                stop = ((end + allocation.page_bytes - 1)
                        // allocation.page_bytes * allocation.page_bytes)
                pages[span.storage].update(
                    range(first, stop, allocation.page_bytes))
            total = sum(len(offsets) * self.storages[key].page_bytes
                        for key, offsets in pages.items())
            best = max(best, total)
        cached[group_id] = best
        return best


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
        # step.  ``_pair_refs`` counts owners plus in-flight executions per
        # (group, block); ``_page_refs`` counts references per physical page.
        # The idle reclaimer reads this ledger from its own thread while the
        # engine thread publishes a new ownership snapshot, so every mutation
        # and the page snapshot share one lock.  Returning a half-applied
        # ledger would retire pages that a running request still reads.
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

    def has_owner(self, owner: str) -> bool:
        self._check_owner(owner)
        return owner in self._owners

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

    def remove_blocks(self, owner: str, blocks: Iterable[tuple[int, int]]) -> int:
        """Remove individual blocks from one owner, preserving the others.

        Idle reclamation needs to drop only the cold prefix blocks it selected,
        not the entire prefix owner.  Returns the number of removed entries.
        """
        self._check_owner(owner)
        remove = self.layout.validate_blocks(blocks)
        with self._ledger_lock:
            current = self._owners.get(owner)
            if current is None:
                return 0
            remaining = current - remove
            if remaining == current:
                return 0
            for pair in current - remaining:
                self._release_pair(pair)
            if remaining:
                self._owners[owner] = remaining
            else:
                del self._owners[owner]
            self.revision += 1
            return len(current) - len(remaining)

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
