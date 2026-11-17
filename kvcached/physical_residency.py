# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Transactional physical residency over the existing kvcached VMM API.

This layer accepts byte-page sets, not model names or logical block tables.
The serving adapter supplies device completion and initialization callbacks.
"""

import time
from uuid import uuid4

from kvcached.integration.vllm import native_profile


class PhysicalResidency:
    def __init__(self, operations, synchronize, initialize, fatal_errors=()):
        self.operations = operations
        self.synchronize = synchronize
        self.initialize = initialize
        self.fatal_errors = fatal_errors
        self.storages = {}
        self.pool_ids = {}
        self.mapped = {}
        # Pages mapped by speculative prefetch are not zeroed until a worker
        # actually consumes them; initialize them in ensure(), never in the
        # prefetch thread (device work must stay on the worker's stream).
        self.uninitialized = {}
        self.poisoned = False
        self.grown_bytes = 0
        self.reclaimed_bytes = 0

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
            if any(type(page) is not int or page < 0
                   or page >= storage.reservation_bytes or page % storage.page_bytes
                   for page in pages):
                raise ValueError("Invalid physical page offset")
        return desired

    def ensure(self, desired):
        started = time.perf_counter()
        try:
            return self._ensure(desired)
        finally:
            native_profile.record(
                "residency.ensure", time.perf_counter() - started)

    def _map_missing(self, desired):
        """Map pages absent from the ledger; return the newly mapped pages.

        Caller holds the physical lock.  This is deliberately separate from
        initialization so a background prefetcher can move pages into the VMM
        without enqueueing device work on the worker's stream.
        """
        added = {}
        try:
            for key, target in desired.items():
                missing = sorted(target - self.mapped[key])
                if not missing:
                    continue
                map_started = time.perf_counter()
                success, attached = self.operations.map_to_kv_tensors_with_result(
                    missing, self.pool_ids[key])
                native_profile.record(
                    "residency.map", time.perf_counter() - map_started,
                    nbytes=len(missing) * self.storages[key].page_bytes)
                if not success:
                    raise MemoryError(f"Unable to back storage {key}")
                if set(attached) != set(missing):
                    self.poisoned = True
                    raise RuntimeError("VMM and page ledger disagree")
                added[key] = set(attached)
                self.mapped[key].update(attached)
                self.uninitialized[key].update(attached)
        except Exception as error:
            # Backend quarantine means a map or its rollback has uncertain
            # state. Rolling back other newly added pages does not repair it.
            if isinstance(error, self.fatal_errors):
                self.poisoned = True
            try:
                if added:
                    self.synchronize()
                for key, pages in added.items():
                    if not self.operations.unmap_from_kv_tensors(sorted(pages), self.pool_ids[key]):
                        raise RuntimeError("Failed to undo newly mapped pages")
                    self.mapped[key].difference_update(pages)
                    self.uninitialized[key].difference_update(pages)
            except Exception as rollback:
                self.poisoned = True
                raise RuntimeError("Growth rollback failed; worker restart required") from rollback
            raise
        return added

    def _initialize_pending(self, desired):
        pending = {
            key: set(pages).intersection(self.uninitialized[key])
            for key, pages in desired.items()
        }
        pending = {key: pages for key, pages in pending.items() if pages}
        if not pending:
            return 0
        init_started = time.perf_counter()
        try:
            for key, pages in pending.items():
                self.initialize(key, pages)
        finally:
            native_profile.record(
                "residency.initialize", time.perf_counter() - init_started)
        for key, pages in pending.items():
            self.uninitialized[key].difference_update(pages)
        return sum(len(pages) * self.storages[key].page_bytes
                   for key, pages in pending.items())

    def _ensure(self, desired):
        validate_started = time.perf_counter()
        desired = self._validate(desired)
        native_profile.record(
            "residency.ensure_validate", time.perf_counter() - validate_started)
        added = self._map_missing(desired)
        try:
            initialized = self._initialize_pending(desired)
        except Exception:
            if added:
                try:
                    self.synchronize()
                    for key, pages in added.items():
                        if not self.operations.unmap_from_kv_tensors(
                                sorted(pages), self.pool_ids[key]):
                            raise RuntimeError("Failed to undo newly mapped pages")
                        self.mapped[key].difference_update(pages)
                        self.uninitialized[key].difference_update(pages)
                except Exception as rollback:
                    self.poisoned = True
                    raise RuntimeError(
                        "Growth rollback failed; worker restart required") from rollback
            raise
        total = sum(len(pages) * self.storages[key].page_bytes for key, pages in added.items())
        self.grown_bytes += total
        return max(total, initialized)

    def prefetch(self, desired):
        """Map pages ahead of use without initializing them on the device."""
        started = time.perf_counter()
        total = 0
        try:
            desired = self._validate(desired)
            added = self._map_missing(desired)
            total = sum(len(pages) * self.storages[key].page_bytes
                        for key, pages in added.items())
            self.grown_bytes += total
            return total
        finally:
            native_profile.record(
                "residency.prefetch", time.perf_counter() - started,
                nbytes=total)

    def reclaim(self, required, eligible=None):
        started = time.perf_counter()
        reclaimed = 0
        try:
            reclaimed = self._reclaim(required, eligible=eligible)
        finally:
            native_profile.record(
                "residency.reclaim", time.perf_counter() - started,
                nbytes=reclaimed)
        return reclaimed

    def _reclaim(self, required, eligible=None):
        validate_started = time.perf_counter()
        required = self._validate(required)
        retiring = {key: sorted(pages - required[key]) for key, pages in self.mapped.items()
                    if pages - required[key]}
        if eligible is not None:
            retiring = {key: sorted(set(pages) & set(eligible.get(key, ())))
                        for key, pages in retiring.items()}
            retiring = {key: pages for key, pages in retiring.items() if pages}
        native_profile.record(
            "residency.reclaim_validate", time.perf_counter() - validate_started)
        if not retiring:
            return 0
        total = sum(len(pages) * self.storages[key].page_bytes for key, pages in retiring.items())
        # CPU lifetime updates do not complete prior eager or graph kernels.
        fence_started = time.perf_counter()
        try:
            self.synchronize()
        except Exception:
            self.poisoned = True
            raise
        finally:
            native_profile.record(
                "residency.fence", time.perf_counter() - fence_started)
        transaction = 'residency-' + uuid4().hex
        prepared = []
        prepare_started = time.perf_counter()
        try:
            for key, pages in retiring.items():
                if not self.operations.prepare_unmap_from_kv_tensors(
                        pages, transaction, self.pool_ids[key]):
                    raise RuntimeError("Physical retirement preparation failed")
                prepared.append(key)
            native_profile.record(
                "residency.prepare_unmap", time.perf_counter() - prepare_started)
        except Exception as error:
            if isinstance(error, self.fatal_errors):
                self.poisoned = True
            try:
                for key in reversed(prepared):
                    if not self.operations.abort_unmap_from_kv_tensors(transaction, self.pool_ids[key]):
                        raise RuntimeError("Physical retirement rollback failed")
            except Exception as rollback:
                self.poisoned = True
                raise RuntimeError("Retirement rollback failed; worker restart required") from rollback
            raise
        commit_started = time.perf_counter()
        try:
            for key in prepared:
                if not self.operations.commit_unmap_from_kv_tensors(transaction, self.pool_ids[key]):
                    raise RuntimeError("Physical retirement commit failed")
                self.mapped[key].difference_update(retiring[key])
                self.uninitialized[key].difference_update(retiring[key])
            native_profile.record(
                "residency.commit_unmap", time.perf_counter() - commit_started)
        except Exception:
            # Released physical handles cannot be restored by inventing data.
            self.poisoned = True
            raise
        self.reclaimed_bytes += total
        return total

    def status(self):
        return {"poisoned": self.poisoned,
                "reserved_bytes": sum(s.reservation_bytes for s in self.storages.values()),
                "mapped_bytes": sum(len(p) * self.storages[k].page_bytes for k, p in self.mapped.items()),
                "uninitialized_bytes": sum(len(p) * self.storages[k].page_bytes
                                           for k, p in self.uninitialized.items()),
                "grown_bytes_total": self.grown_bytes,
                "reclaimed_bytes_total": self.reclaimed_bytes}
