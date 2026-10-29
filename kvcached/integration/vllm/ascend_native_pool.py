# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Native-layout storage backed by kvcached's existing transactional VMM.

The adapter owns only physical residency. Native attention reshaping, cache
group assignment and logical block IDs remain with the serving engine.
"""

from uuid import uuid4

from kvcached.integration.vllm.ascend_native_geometry import native_backings


class NativeKVPool:
    def __init__(self, config, device, operations, synchronize, instance_name,
                 wrap_storage=lambda tensor: tensor):
        self.backings = native_backings(config)
        self.num_blocks = config.num_blocks
        self.operations = operations
        self.synchronize = synchronize
        self.mapped = {b.group_id: set() for b in self.backings}
        self.raw = {}
        self.elastic = False
        self.poisoned = False
        self.grown_bytes = 0
        self.reclaimed_bytes = 0
        operations.init_kvcached(str(device), self.backings[0].page_bytes, False, instance_name)
        for backing in self.backings:
            tensors = operations.create_kv_tensors(
                backing.reservation_bytes, 1, str(device), 1, 1,
                backing.group_id, True, backing.page_bytes,
            )
            if len(tensors) != 1:
                raise RuntimeError("Expected exactly one elastic tensor per native backing")
            # ACLNN operators inspect native storage descriptors, which generic
            # stable-ABI from_blob tensors do not carry. Wrap before any view.
            self.raw[backing.group_id] = wrap_storage(tensors[0])[:backing.size_bytes]

    def views_by_layer(self):
        return {name: self.raw[b.group_id] for b in self.backings for name in b.layers}

    def desired_pages(self, blocks):
        # A global block ID is owned by one cache group. Mapping the union in
        # every slot is conservative and preserves all cross-group aliases.
        blocks = set(blocks) | {0}
        return {b.group_id: b.page_offsets(blocks) for b in self.backings}

    def ensure(self, blocks=None, *, full=False):
        if self.poisoned:
            raise RuntimeError("Native elastic pool is poisoned after failed rollback")
        desired = ({b.group_id: b.all_page_offsets() for b in self.backings}
                   if full else self.desired_pages(blocks))
        added = {}
        try:
            for group_id, target in desired.items():
                missing = sorted(target - self.mapped[group_id])
                if not missing:
                    continue
                success, attached = self.operations.map_to_kv_tensors_with_result(missing, group_id)
                if not success:
                    raise MemoryError(f"kvcached could not map native backing {group_id}")
                if set(attached) != set(missing):
                    # Divergent accounting is unsafe even if all missing pages
                    # happen to be backed. Do not continue serving silently.
                    self.poisoned = True
                    raise RuntimeError("Native VMM mapping disagrees with ownership ledger")
                added[group_id] = set(attached)
                self.mapped[group_id].update(attached)
            # Fresh physical pages can contain data from another allocation.
            # Initialize only newly mapped intervals, never a live shared page.
            for group_id, offsets in added.items():
                backing = self.backings[group_id]
                runs = []
                for offset in sorted(offsets):
                    if runs and runs[-1][1] == offset:
                        runs[-1][1] += backing.page_bytes
                    else:
                        runs.append([offset, offset + backing.page_bytes])
                for start, end in runs:
                    self.raw[group_id][start:min(end, backing.size_bytes)].zero_()
        except Exception:
            try:
                # These pages have not been exposed to model execution yet.
                # An initialization kernel may have been enqueued before an
                # exception, so finish it before undoing the new mappings.
                if added:
                    self.synchronize()
                for group_id, offsets in added.items():
                    if not self.operations.unmap_from_kv_tensors(sorted(offsets), group_id):
                        raise RuntimeError("Could not roll back new native KV mappings")
                    self.mapped[group_id].difference_update(offsets)
            except Exception as rollback:
                self.poisoned = True
                raise RuntimeError("Native KV growth rollback failed; stop this worker") from rollback
            raise
        grown = sum(len(offsets) for offsets in added.values()) * self.backings[0].page_bytes
        self.grown_bytes += grown
        return grown

    def reclaim(self, live_blocks):
        if self.poisoned:
            raise RuntimeError("Cannot reclaim a poisoned native elastic pool")
        keep = self.desired_pages(live_blocks)
        retiring = {g: sorted(pages - keep[g]) for g, pages in self.mapped.items() if pages - keep[g]}
        if not retiring:
            return 0
        # A request disappearing from a CPU block table does not complete prior
        # device operations. The barrier must precede every physical retirement.
        self.synchronize()
        tx = 'ascend-native-' + uuid4().hex
        prepared = []
        try:
            for group_id, offsets in retiring.items():
                if not self.operations.prepare_unmap_from_kv_tensors(offsets, tx, group_id):
                    raise RuntimeError("Native KV unmap preparation failed")
                prepared.append(group_id)
        except Exception:
            for group_id in reversed(prepared):
                if not self.operations.abort_unmap_from_kv_tensors(tx, group_id):
                    self.poisoned = True
                    raise RuntimeError("Native KV retirement rollback failed")
            raise
        try:
            for group_id in prepared:
                if not self.operations.commit_unmap_from_kv_tensors(tx, group_id):
                    raise RuntimeError("Native KV retirement commit failed")
                self.mapped[group_id].difference_update(retiring[group_id])
        except Exception:
            # Some groups may already have released their physical handles;
            # preserve evidence and fail closed rather than invent a rollback.
            self.poisoned = True
            raise
        reclaimed = sum(len(offsets) for offsets in retiring.values()) * self.backings[0].page_bytes
        self.reclaimed_bytes += reclaimed
        return reclaimed

    def status(self):
        return {'elastic': self.elastic, 'poisoned': self.poisoned,
                'grown_bytes_total': self.grown_bytes,
                'reclaimed_bytes_total': self.reclaimed_bytes,
                'num_blocks': self.num_blocks,
                'reserved_bytes': sum(b.reservation_bytes for b in self.backings),
                'mapped_bytes': sum(len(self.mapped[b.group_id]) * b.page_bytes for b in self.backings),
                'backings': [{'group_id': b.group_id, 'block_stride': b.block_stride,
                              'layers': list(b.layers), 'size_bytes': b.size_bytes,
                              'mapped_pages': len(self.mapped[b.group_id])} for b in self.backings]}
