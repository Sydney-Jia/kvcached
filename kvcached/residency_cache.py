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

    def clear(self):
        self.recent.clear()

    def select(self, storages, mapped, required, spare_bytes, limit_bytes=None):
        optional_nonnegative(spare_bytes, 'spare_bytes')
        optional_nonnegative(limit_bytes, 'limit_bytes')
        if spare_bytes is None:
            raise ValueError('spare_bytes must be bounded')
        live = page_bytes(storages, required)
        page_bytes(storages, mapped)
        room = spare_bytes if limit_bytes is None else min(spare_bytes, max(0, limit_bytes - live))
        self.clock += 1
        self.recent = {
            (key, page): (self.clock if page in required[key]
                          else self.recent.get((key, page), 0))
            for key in storages for page in mapped[key] | required[key]
        }
        target = {key: set(pages) for key, pages in required.items()}
        candidates = [(key, page) for key in storages for page in mapped[key] - required[key]]
        if sum(storages[key].page_bytes for key, _ in candidates) > room:
            candidates.sort(key=lambda item: (-self.recent[item], item))
        for key, page in candidates:
            size = storages[key].page_bytes
            if size <= room:
                target[key].add(page)
                room -= size
        return target
