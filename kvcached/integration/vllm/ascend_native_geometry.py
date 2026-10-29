# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Byte ownership for independent native cache backings sharing one block pool.

This module has no framework/device imports. A native manager block is not a
VMM page: its byte interval can intersect several pages, and several live
blocks can keep the same page resident. Aliased cache groups retain one backing.
"""

from dataclasses import dataclass
from typing import Any, Iterable


VMM_PAGE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class NativeBacking:
    group_id: int
    size_bytes: int
    block_stride: int
    num_blocks: int
    layers: tuple[str, ...]
    page_bytes: int = VMM_PAGE_BYTES

    @property
    def reservation_bytes(self) -> int:
        return ((self.size_bytes + self.page_bytes - 1) // self.page_bytes) * self.page_bytes

    def page_offsets(self, block_ids: Iterable[int]) -> set[int]:
        pages: set[int] = set()
        for block in block_ids:
            if type(block) is not int or not 0 <= block < self.num_blocks:
                raise ValueError(f"Block ID {block!r} is outside the native pool")
            start = block * self.block_stride
            end = start + self.block_stride
            first = start // self.page_bytes * self.page_bytes
            last = (end + self.page_bytes - 1) // self.page_bytes * self.page_bytes
            pages.update(range(first, last, self.page_bytes))
        return pages

    def all_page_offsets(self) -> set[int]:
        return set(range(0, self.reservation_bytes, self.page_bytes))


def native_backings(config: Any) -> tuple[NativeBacking, ...]:
    """Validate the DeepSeek native descriptor contract before creating storage.

    Each descriptor allocates an independent slot. All listed layers alias that
    allocation (possibly using different subviews inside each block). Do not
    generalize this to layouts whose descriptors share one global allocation.
    """
    if type(config.num_blocks) is not int or config.num_blocks <= 0:
        raise ValueError("A positive native block count is required")
    result = []
    seen: set[str] = set()
    for group_id, descriptor in enumerate(config.kv_cache_tensors):
        names = tuple(descriptor.layers)
        if not names or len(set(names)) != len(names) or seen.intersection(names):
            raise ValueError("Each native layer must belong to exactly one backing")
        if getattr(descriptor, "host_resident", False):
            raise ValueError("Host-resident caches are outside this adapter")
        if descriptor.offset != 0 or descriptor.layer_stride != 0:
            raise ValueError("Expected independent zero-offset, aliased native backings")
        stride = descriptor.block_stride
        if type(stride) is not int or stride <= 0 or descriptor.size != config.num_blocks * stride:
            raise ValueError("Native backing size must equal block count times byte stride")
        seen.update(names)
        result.append(NativeBacking(group_id, descriptor.size, stride, config.num_blocks, names))
    if not result:
        raise ValueError("No native cache backings")
    return tuple(result)


class NativeLiveBlocks:
    """Conservative worker ownership with prefix retention explicitly disabled.

    Retain unscheduled requests. Apply authoritative block-table replacements
    before appended blocks. The caller synchronizes the device before releasing
    old pages; parsing a scheduler message alone does not retire in-flight work.
    """

    def __init__(self, num_groups: int, num_blocks: int):
        self.num_groups = num_groups
        self.num_blocks = num_blocks
        self.requests: dict[str, tuple[list[int], ...]] = {}

    def _groups(self, groups: Any) -> tuple[list[int], ...]:
        if len(groups) != self.num_groups:
            raise ValueError("Scheduler cache-group count changed")
        result = tuple(list(group) for group in groups)
        for group in result:
            for block in group:
                if type(block) is not int or not 0 <= block < self.num_blocks:
                    raise ValueError(f"Invalid native block ID: {block!r}")
        return result

    def update(self, output: Any) -> set[int]:
        # Validate in a copy, so malformed updates cannot partially retire state.
        requests = {key: tuple(list(g) for g in groups) for key, groups in self.requests.items()}
        removed = set(output.finished_req_ids) | set(getattr(output, "preempted_req_ids", ()) or ())
        for req_id in removed:
            requests.pop(req_id, None)
        for request in output.scheduled_new_reqs:
            requests[request.req_id] = self._groups(request.block_ids)
        replacements = getattr(output, "block_table_updates", None) or {}
        for req_id, groups in replacements.items():
            if req_id in requests:
                requests[req_id] = self._groups(groups)
        cached = output.scheduled_cached_reqs
        if len(cached.req_ids) != len(cached.new_block_ids):
            raise ValueError("Mismatched cached request/block updates")
        for req_id, groups in zip(cached.req_ids, cached.new_block_ids):
            if req_id not in requests:
                raise ValueError(f"Unknown cached request: {req_id}")
            if groups is None or req_id in replacements:
                continue
            added = self._groups(groups)
            if req_id in getattr(cached, "resumed_req_ids", ()):
                requests[req_id] = added
            else:
                requests[req_id] = tuple(old + new for old, new in zip(requests[req_id], added))
        transient = set(output.new_block_ids_to_zero or ())
        for copy in getattr(output, "kv_cache_block_copies", ()) or ():
            # Refuse an unknown copy contract instead of guessing tuple fields.
            transient.add(copy.src_block_id)
            transient.add(copy.dst_block_id)
        if any(type(b) is not int or not 0 <= b < self.num_blocks for b in transient):
            raise ValueError("Invalid zero/copy block ID")
        self.requests = requests
        return self.live_blocks() | transient

    def live_blocks(self) -> set[int]:
        # Native padding can read block zero even when it has no request owner.
        return {0} | {block for groups in self.requests.values() for group in groups for block in group}
