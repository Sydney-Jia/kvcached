# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Read native vLLM ownership without replacing its allocation/cache policy.

The scheduler adapter passes the native group-id decoder explicitly. Keeping
this reader free of framework imports makes its ownership invariants testable
independently of a particular model, device or vLLM installation.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class NativeOwnership:
    version: int
    epoch: int
    num_blocks: int
    num_groups: int
    owners: dict[str, tuple[tuple[int, int], ...]]


def dummy_state_owners(block_counts, num_requests):
    """Conservative native MRV2 dummy-slot contract, independent of cache type.

    Null/padding uses slot zero; stateful dummy requests may use 1..N.
    Protect these slots in every group before native metadata preparation,
    which can itself zero state. The execution fence, not Python return,
    determines when a later scheduler step may retire the backing pages.
    """
    if type(num_requests) is not int or num_requests < 0:
        raise ValueError("Dummy request count must be a nonnegative integer")
    if not block_counts or any(num_requests >= count for count in block_counts.values()):
        raise ValueError("Dummy state slots exceed native cache capacity")
    return tuple((group, block) for group in block_counts
                 for block in range(num_requests + 1))


def snapshot_ownership(coordinator, output, *, epoch: int,
                       decode_group: Callable) -> NativeOwnership:
    managers = tuple(coordinator.single_type_managers)
    if not managers or epoch < 1:
        raise ValueError("Native residency requires groups and a positive epoch")
    pool = coordinator.block_pool
    if any(manager.block_pool is not pool for manager in managers):
        raise ValueError("Independent native block pools need distinct block identities")
    blocks = pool.blocks
    count = len(blocks)
    groups = len(managers)
    owners = {key: set() for key in
              ("requests", "prefix", "native-pins", "native-watchers", "null", "step-io")}
    attributed = Counter()

    def add(owner, group, block):
        if (type(group) is not int or not 0 <= group < groups
                or type(block) is not int or not 0 <= block < count):
            raise ValueError(f"Invalid native owner group/block {(group, block)!r}")
        owners[owner].add((group, block))

    def all_groups(owner, block):
        for group in range(groups):
            add(owner, group, block)

    for group, manager in enumerate(managers):
        for request_blocks in manager.req_to_blocks.values():
            for block in request_blocks:
                add("requests", group, block.block_id)
                attributed[block.block_id] += 1

    # Refcount zero does NOT mean disposable: prefix hashes and native reuse
    # watchers can still make this block readable by a later request/copy.
    aliases = pool.cached_block_hashes_by_block
    for index, block in enumerate(blocks):
        if block.block_id != index:
            raise ValueError("Native block identity is not indexed by pool position")
        hashes = set(aliases.get(index, ()))
        if block.block_hash is not None:
            hashes.add(block.block_hash)
        for key in hashes:
            add("prefix", decode_group(key), index)
        if block.is_null:
            all_groups("null", index)
        elif block.ref_cnt > attributed[index]:
            # Native manager pins (e.g. CoW staging) do not always belong to a
            # request table. Retain all possible views until native unpinning.
            all_groups("native-pins", index)
    for block in pool._reuse_watchers:
        all_groups("native-watchers", block)
    # Native zero/copy kernels can address components of every cache group,
    # even if the destination will eventually belong to only one group.
    for block in output.new_block_ids_to_zero or ():
        all_groups("step-io", block)
    for copy in output.kv_cache_block_copies or ():
        all_groups("step-io", copy.src_block_id)
        all_groups("step-io", copy.dst_block_id)
    return NativeOwnership(1, epoch, count, groups,
                           {key: tuple(sorted(value)) for key, value in owners.items()})
