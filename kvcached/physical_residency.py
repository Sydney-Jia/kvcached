# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Transactional physical residency over the existing kvcached VMM API.

This layer accepts byte-page sets, not model names or logical block tables.
The serving adapter supplies device completion and initialization callbacks.
"""

from uuid import uuid4


class PhysicalResidency:
    def __init__(self, operations, synchronize, initialize, fatal_errors=(), *, timed=None):
        self.operations = operations
        self.synchronize = synchronize
        self.initialize = initialize
        self.fatal_errors = fatal_errors
        self._timer = timed
        self.storages = {}
        self.pool_ids = {}
        self.mapped = {}
        # Pages mapped by a speculative prepare are not zeroed until a worker
        # actually consumes the block. Keeping them in a separate set lets a
        # later ensure() initialize them exactly once and lets prefetch stay a
        # map-only, fence-free operation.
        self.uninitialized = {}
        self.poisoned = False
        self.grown_bytes = 0
        self.reclaimed_bytes = 0

    def _stage(self, name, operation, *args):
        if self._timer is None:
            return operation(*args)
        return self._timer(name, operation, *args)

    def register(self, storage, pool_id):
        if storage.key in self.storages or pool_id in self.pool_ids.values():
            raise ValueError("Storage and VMM pool identities must be unique")
        self.storages[storage.key] = storage
        self.pool_ids[storage.key] = pool_id
        self.mapped[storage.key] = set()
        self.uninitialized[storage.key] = set()

    def full(self):
        return {key: set(range(0, storage.reservation_bytes, storage.page_bytes))
                for key, storage in self.storages.items()}

    def _validate(self, desired):
        if self.poisoned:
            raise RuntimeError("Physical residency is poisoned; worker restart required")
        if desired.keys() != self.storages.keys():
            raise ValueError("A complete storage-page snapshot is required")
        desired = {key: set(pages) for key, pages in desired.items()}
        for key, pages in desired.items():
            storage = self.storages[key]
            end, granularity = storage.reservation_bytes, storage.page_bytes
            if any(type(page) is not int or page < 0
                   or page >= end or page % granularity
                   for page in pages):
                raise ValueError("Invalid physical page offset")
        return desired

    def ensure(self, desired):
        desired = self._validate(desired)
        added = {}
        try:
            for key, target in desired.items():
                missing = sorted(target - self.mapped[key])
                if not missing:
                    continue
                success, attached = self._stage(
                    'map', self.operations.map_to_kv_tensors_with_result, missing, self.pool_ids[key])
                if not success:
                    raise MemoryError(f"Unable to back storage {key}")
                if set(attached) != set(missing):
                    self.poisoned = True
                    raise RuntimeError("VMM and page ledger disagree")
                added[key] = set(attached)
                self.mapped[key].update(attached)
                self.uninitialized[key].update(attached)
            pending = {key: sorted(pages & self.uninitialized[key])
                       for key, pages in desired.items()}
            pending = {key: pages for key, pages in pending.items() if pages}
            for key, pages in pending.items():
                self._stage('initialize', self.initialize, key, pages)
            for key, pages in pending.items():
                self.uninitialized[key].difference_update(pages)
        except Exception as error:
            # Backend quarantine means a map or its rollback has uncertain
            # state. Rolling back other newly added pages does not repair it.
            if isinstance(error, self.fatal_errors):
                self.poisoned = True
            # Initialization may have enqueued device operations before failing.
            # Never remove their backing until completion has been established.
            try:
                if added:
                    self.synchronize()
                for key, pages in added.items():
                    if not self._stage('rollback_map', self.operations.unmap_from_kv_tensors,
                                       sorted(pages), self.pool_ids[key]):
                        raise RuntimeError("Failed to undo newly mapped pages")
                    self.mapped[key].difference_update(pages)
                    self.uninitialized[key].difference_update(pages)
            except Exception as rollback:
                self.poisoned = True
                raise RuntimeError("Growth rollback failed; worker restart required") from rollback
            raise
        total = sum(len(pages) * self.storages[key].page_bytes for key, pages in added.items())
        self.grown_bytes += total
        return total

    def prefetch(self, desired):
        """Map pages ahead of use; initialization stays with the consumer.

        Deliberately map-only: no device fence is performed here, so this may
        run while the previous step still executes. Bytes are accounted in the
        shared budget exactly like ensure(), and a consumer that later calls
        ensure() also zeroes them through the uninitialized ledger.
        """
        desired = self._validate(desired)
        total = 0
        for key, target in desired.items():
            missing = sorted(target - self.mapped[key])
            if not missing:
                continue
            success, attached = self._stage(
                'prefetch_map', self.operations.map_to_kv_tensors_with_result,
                missing, self.pool_ids[key])
            if not success:
                raise MemoryError(f"Unable to back storage {key}")
            if set(attached) != set(missing):
                self.poisoned = True
                raise RuntimeError("VMM and page ledger disagree")
            self.mapped[key].update(attached)
            self.uninitialized[key].update(attached)
            total += len(attached) * self.storages[key].page_bytes
        self.grown_bytes += total
        return total

    def reclaim(self, required):
        required = self._validate(required)
        retiring = {key: sorted(pages - required[key]) for key, pages in self.mapped.items()
                    if pages - required[key]}
        if not retiring:
            return 0
        # CPU lifetime updates do not complete prior eager or graph kernels.
        try:
            self.synchronize()
        except Exception:
            self.poisoned = True
            raise
        transaction = 'residency-' + uuid4().hex
        prepared = []
        try:
            for key, pages in retiring.items():
                if not self._stage('prepare_unmap', self.operations.prepare_unmap_from_kv_tensors,
                                   pages, transaction, self.pool_ids[key]):
                    raise RuntimeError("Physical retirement preparation failed")
                prepared.append(key)
        except Exception as error:
            if isinstance(error, self.fatal_errors):
                self.poisoned = True
            try:
                for key in reversed(prepared):
                    if not self._stage('abort_unmap', self.operations.abort_unmap_from_kv_tensors,
                                       transaction, self.pool_ids[key]):
                        raise RuntimeError("Physical retirement rollback failed")
            except Exception as rollback:
                self.poisoned = True
                raise RuntimeError("Retirement rollback failed; worker restart required") from rollback
            raise
        try:
            for key in prepared:
                if not self._stage('commit_unmap', self.operations.commit_unmap_from_kv_tensors,
                                   transaction, self.pool_ids[key]):
                    raise RuntimeError("Physical retirement commit failed")
                self.mapped[key].difference_update(retiring[key])
                self.uninitialized[key].difference_update(retiring[key])
        except Exception:
            # Released physical handles cannot be restored by inventing data.
            self.poisoned = True
            raise
        total = sum(len(pages) * self.storages[key].page_bytes for key, pages in retiring.items())
        self.reclaimed_bytes += total
        return total

    def transition(self, required):
        """Retire unowned pages before growth, without transient double backing.

        The caller supplies *all* current owners, including copy endpoints and
        in-flight users. reclaim() fences prior device work before releasing
        anything. A failed growth leaves retained contents intact and rolls
        back only newly mapped pages; retired, unowned contents are not restored.
        This is not an atomic device-wide or cross-rank transaction.
        """
        required = self._validate(required)
        retired = self.reclaim(required)
        grown = self.ensure(required)
        return {"reclaimed_bytes": retired, "grown_bytes": grown}

    def status(self):
        return {"poisoned": self.poisoned,
                "reserved_bytes": sum(s.reservation_bytes for s in self.storages.values()),
                "mapped_bytes": sum(len(p) * self.storages[k].page_bytes for k, p in self.mapped.items()),
                "grown_bytes_total": self.grown_bytes,
                "reclaimed_bytes_total": self.reclaimed_bytes}
