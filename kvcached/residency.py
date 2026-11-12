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

from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Callable, Iterable, Mapping


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

    def __init__(self, storages: Iterable[Storage], spans: Iterable[BlockSpan]):
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
            for span in self.groups[group]:
                allocation = self.storages[span.storage]
                start, end = span.interval(block)
                first = start // allocation.page_bytes * allocation.page_bytes
                stop = (end + allocation.page_bytes - 1) // allocation.page_bytes * allocation.page_bytes
                pages[span.storage].update(range(first, stop, allocation.page_bytes))
        return {key: frozenset(offsets) for key, offsets in pages.items()}


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
        current = self._owners.copy()
        for owner in remove:
            if owner not in current:
                raise ValueError(f"Cannot release unknown owner {owner!r}")
            del current[owner]
        current.update(validated)
        self._owners = current
        self.revision += 1

    def begin_execution(self, identity: str, blocks: Iterable[tuple[int, int]],
                        completed: Callable[[], bool]) -> None:
        self._check_owner(identity)
        if identity in self._executions or not callable(completed):
            raise ValueError("Execution identity must be unique and completion callable")
        self._executions[identity] = (self.layout.validate_blocks(blocks), completed)
        self.revision += 1

    def collect_completed(self) -> int:
        # Query all fences before mutating the ledger; a failed query leaves
        # every execution protected, including those queried earlier.
        done = [key for key, (_, completed) in self._executions.items() if completed()]
        for key in done:
            del self._executions[key]
        if done:
            self.revision += 1
        return len(done)

    def required_pages(self) -> dict[str, frozenset[int]]:
        blocks = set().union(*self._owners.values())
        for execution, _ in self._executions.values():
            blocks.update(execution)
        return self.layout.pages_for(blocks)

    def snapshot(self) -> dict:
        return {"revision": self.revision, "owners": len(self._owners),
                "inflight_executions": len(self._executions),
                "required_pages": {key: sorted(pages) for key, pages in self.required_pages().items()}}
