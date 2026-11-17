# SPDX-License-Identifier: Apache-2.0
"""Bounded retention of already-backed, unowned pages during active execution.

This is a physical-page policy, not prefix validity or a logical allocation.
It never creates backing, removes required pages, or extends the byte limit.
The caller supplies completion fencing before releasing any selected-out page.
"""

from kvcached.native_capacity import optional_nonnegative, page_bytes


class SparePageCache:
    def __init__(self):
        self.clock = 0
        self.recent = {}
        self.required = {}

    def clear(self):
        self.recent.clear()
        self.required.clear()

    def select(self, storages, mapped, required, spare_bytes, limit_bytes=None):
        optional_nonnegative(spare_bytes, 'spare_bytes')
        optional_nonnegative(limit_bytes, 'limit_bytes')
        if spare_bytes is None:
            raise ValueError('spare_bytes must be bounded')
        live = page_bytes(storages, required)
        page_bytes(storages, mapped)
        room = spare_bytes if limit_bytes is None else min(spare_bytes, max(0, limit_bytes - live))
        self.clock += 1
        target = {key: set(pages) for key, pages in required.items()}
        # Recency matters only for spare pages. Record their last transition
        # out of the required set instead of rebuilding a per-page timestamp
        # dictionary for the much larger active working set on every call.
        recent = {}
        for key in storages:
            previous = self.required.get(key, ())
            for page in mapped[key] - required[key]:
                identity = (key, page)
                recent[identity] = (self.clock if page in previous
                                    else self.recent.get(identity, 0))
        self.required = {key: frozenset(pages) for key, pages in required.items()}
        self.recent = recent
        candidates = list(recent)
        if sum(storages[key].page_bytes for key, _ in candidates) > room:
            candidates.sort(key=lambda item: (-self.recent[item], item))
        for key, page in candidates:
            size = storages[key].page_bytes
            if size <= room:
                target[key].add(page)
                room -= size
        return target
