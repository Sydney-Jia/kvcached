# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Physical-capacity admission for the native V2 scheduler path.

The native worker owns the layout and VMM arena.  In the single-process V2
engine the scheduler's KVCacheManager.allocate_slots runs in the same process,
so the worker can install an admission gate here.

Admission has four layers:

* a conservative preflight before vLLM mutates request or prefix-cache state;
* exact page mapping after each group's ``get_new_blocks`` allocation;
* a rollback of the partial CPU-side allocation when the exact mapping misses;
* pressure relief that evicts replaceable prefix hashes and unmaps pages no
  current owner needs, so a full instance can make room instead of spinning.

A miss is returned to vLLM as ``None`` (a normal scheduling miss), preserving
its existing queue/preempt behavior instead of letting ``execute_model`` fail
later.  Running requests keep their existing blocks on a miss; only the new
request path frees and rebuilds its bookkeeping.
"""

import contextvars
import json
import os
import sys
import threading
import time

from kvcached.capacity_relief import is_capacity_rejection
from kvcached.integration.vllm import native_profile
from kvcached.utils import KVCachePoolExhausted

_GATE = None
_SCHEDULER_CONTEXT = None
_PATCHED = False
_POOL_PATCHED = False
_LOCK = threading.Lock()
_GROUP = contextvars.ContextVar("kvcached_native_group", default=None)


class NativeAdmissionGate:
    def __init__(self, arena, owners):
        self.arena = arena
        self.owners = owners
        self.coordinator = None
        self.scheduler = None
        # Serializes physical map/unmap with the idle reclaimer.  RLock keeps
        # ensure methods callable from paths that already hold it.
        self.physical_lock = threading.RLock()
        self._preflight_logged = set()
        self.preflight_misses = 0
        self.relief_events = 0
        self._limit_revision = 0
        self.block_priority = {}
        self._idle_thread = None
        self._idle_stop = threading.Event()
        # Speculative physical-map prefetch.  The queue is intentionally
        # bounded by bytes, not by request count, so it can never pin more
        # than MAP_AHEAD_BYTES of otherwise free capacity.
        self._prefetch_thread = None
        self._prefetch_stop = threading.Event()
        self._prefetch_blocks = {}
        self._prefetch_hits = 0
        self._prefetch_maps = 0
        self._prefetch_drops = 0
        self._prefetch_goal = 0
        self._prefetch_deadline = 0.0
        self._critical_pending = 0
        # Stall watchdog: if the engine has a request but the scheduler or
        # worker makes no progress for `STALL_S`, dump every thread stack so
        # the blocked call is identified instead of guessed at.
        self._watchdog_thread = None
        self._watchdog_stop = threading.Event()
        self._last_progress = time.monotonic()
        self._stall_reported = 0.0
        self._watchdog_key = None

    def bind_scheduler(self, coordinator, scheduler=None):
        self.coordinator = coordinator
        if scheduler is not None:
            self.scheduler = scheduler

    def start_idle_thread(self):
        """Engine cores block when idle, so no scheduler tick can reclaim."""
        configured = os.getenv("KVCACHED_NATIVE_IDLE_RECLAIM_S")
        if configured is None:
            # Device work from a background thread can stall the engine behind
            # the physical lock, so time-based donor shrink is explicit-only.
            # A request-driven miss path still reclaims normally.
            return
        try:
            interval = float(configured)
        except ValueError:
            interval = 30.0
        if interval <= 0 or self._idle_thread is not None:
            return
        self._idle_thread = threading.Thread(
            target=self._idle_loop, args=(interval,),
            name="kvcached-idle-reclaim", daemon=True)
        self._idle_thread.start()

    def _idle_loop(self, interval):
        # Bulk unmapping is slow (driver page release) and serializes with the
        # peer instance through the shared budget lock, so an idle shrink is
        # split into small chunks with a short pause between them.  Each chunk
        # bounds how long a concurrent admission in either engine can wait.
        try:
            gap = max(0.0, float(os.getenv(
                "KVCACHED_NATIVE_IDLE_RECLAIM_GAP_S", "2") or 0))
        except ValueError:
            gap = 2.0
        due = time.monotonic()
        while not self._idle_stop.wait(1.0):
            scheduler = self.scheduler
            try:
                active = scheduler is not None and (
                    len(getattr(scheduler, "running", ())) > 0
                    or len(getattr(scheduler, "waiting", ())) > 0)
            except TypeError:
                active = True
            now = time.monotonic()
            if active:
                due = now + interval
                continue
            if now < due:
                continue
            reclaimed = 0
            try:
                reclaimed = self.reclaim_idle()
            except Exception as error:
                print("KVCACHED_NATIVE_IDLE_RECLAIM_FAILED "
                      f"error={str(error)[:200]}", flush=True)
            now = time.monotonic()
            # Keep shrinking in the next small chunk while work remains;
            # otherwise wait a full idle interval before looking again.
            due = now + (gap if reclaimed else interval)

    def critical_enter(self):
        self._critical_pending += 1

    def critical_exit(self):
        self._critical_pending = max(0, self._critical_pending - 1)

    def note_progress(self):
        """Mark one completed scheduler/worker loop iteration."""
        self._last_progress = time.monotonic()

    def start_watchdog(self):
        if os.getenv("KVCACHED_NATIVE_WATCHDOG", "0").strip().lower() in (
                "", "0", "false", "no"):
            return
        if self._watchdog_thread is not None:
            return
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name="kvcached-watchdog", daemon=True)
        self._watchdog_thread.start()

    def _watchdog_loop(self):
        try:
            stall_s = float(os.getenv("KVCACHED_NATIVE_STALL_S", "60") or 60)
        except ValueError:
            stall_s = 60.0
        interval = max(1.0, min(5.0, stall_s / 4.0))
        while not self._watchdog_stop.wait(interval):
            scheduler = self.scheduler
            try:
                running = len(getattr(scheduler, "running", ()))
                waiting = len(getattr(scheduler, "waiting", ()))
            except TypeError:
                continue
            now = time.monotonic()
            # A queued request is not progress, but its arrival restarts the
            # stall clock: the engine was legitimately idle before it came in.
            key = (running, waiting)
            if key != self._watchdog_key:
                self._watchdog_key = key
                self._last_progress = now
            if not (running or waiting):
                continue
            idle = now - self._last_progress
            if idle < stall_s or now - self._stall_reported < 30.0:
                continue
            self._stall_reported = now
            print("KVCACHED_NATIVE_STALL " + json.dumps({
                "idle_s": round(idle, 1),
                "running": running,
                "waiting": waiting,
                "prefetch_blocks": len(self._prefetch_blocks),
                "critical_pending": self._critical_pending,
                "ts": round(time.time(), 3),
            }, sort_keys=True), flush=True)
            try:
                import faulthandler
                faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            except Exception as error:  # pragma: no cover - diagnostics only
                print(f"KVCACHED_NATIVE_STALL_DUMP_FAILED {error}", flush=True)

    def start_prefetch_thread(self):
        """Map free KV blocks ahead of allocation on a background thread.

        The prefetch window is byte-bounded and uses map-only residency; the
        worker still initializes pages when a block is actually consumed.
        """
        if os.getenv("KVCACHED_NATIVE_MAP_AHEAD", "0").strip().lower() in (
                "", "0", "false", "no"):
            return
        if self._prefetch_thread is not None:
            return
        try:
            interval = float(os.getenv(
                "KVCACHED_NATIVE_MAP_AHEAD_INTERVAL_MS", "20")) / 1000.0
        except ValueError:
            interval = 0.020
        interval = max(0.001, interval)
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_loop, args=(interval,),
            name="kvcached-map-prefetch", daemon=True)
        self._prefetch_thread.start()

    def _prefetch_loop(self, interval):
        while not self._prefetch_stop.wait(interval):
            try:
                self._prefetch_step()
            except Exception as error:  # prefetch is best-effort
                print("KVCACHED_NATIVE_PREFETCH_FAILED "
                      f"error={str(error)[:200]}", flush=True)

    def _prefetch_step(self):
        coordinator = self.coordinator
        if coordinator is None:
            return
        try:
            max_blocks = max(1, int(os.getenv(
                "KVCACHED_NATIVE_MAP_AHEAD_BLOCKS", "4")))
            max_bytes = max(0, int(os.getenv(
                "KVCACHED_NATIVE_MAP_AHEAD_BYTES", str(1 << 30))))
            ttl_s = float(os.getenv(
                "KVCACHED_NATIVE_MAP_AHEAD_TTL_S", "30"))
        except ValueError:
            max_blocks, max_bytes, ttl_s = 4, 1 << 30, 30.0
        now = time.monotonic()
        scheduler = self.scheduler
        try:
            active = scheduler is not None and (
                len(getattr(scheduler, "running", ())) > 0
                or len(getattr(scheduler, "waiting", ())) > 0)
        except TypeError:
            active = True
        with self.physical_lock:
            if not active:
                # Do not churn map/unmap while the engine is idle.  The idle
                # reclaimer owns donor shrink; prefetch is demand-driven.
                self._drop_expired_prefetch(now, ttl_s)
                return
            if self._critical_pending:
                # Let allocation/ensure own the VMM lock; prefetch is optional.
                return
            if now >= self._prefetch_deadline:
                self._prefetch_goal = 0
            target = min(max_blocks, self._prefetch_goal)
            remaining = target - len(self._prefetch_blocks)
            if remaining <= 0:
                # No active prefill window; leave existing pages in place.
                self._drop_expired_prefetch(now, ttl_s)
                return
            try:
                state = self.arena.operations.physical_budget_state()
                budget = int(state.get("budget_bytes", 0) or 0)
                charged = int(state.get("charged_bytes", 0) or 0)
                instance_charged = int(state.get(
                    "instance_charged_bytes", 0) or 0)
                instance_limit = int(state.get(
                    "instance_limit_bytes", 0) or 0)
            except Exception:
                return
            if budget <= 0 or charged >= budget * 0.90:
                return

            pool = coordinator.block_pool
            queue = pool.free_block_queue
            node = queue.fake_free_list_head.next_free_block
            tail = queue.fake_free_list_tail
            candidates = []
            while node is not None and node is not tail:
                block_id = getattr(node, "block_id", None)
                if (block_id is not None and not node.is_null
                        and not node.ref_cnt
                        and block_id not in self._prefetch_blocks):
                    candidates.append(block_id)
                    if len(candidates) >= remaining:
                        break
                node = node.next_free_block
            if not candidates:
                return

            layout = self.owners.layout
            blocks = {(group, block_id)
                      for group in layout.groups for block_id in candidates}
            try:
                pages = layout.pages_for(blocks)
            except (TypeError, ValueError):
                return
            required_bytes = sum(
                len(offsets) * layout.storages[key].page_bytes
                for key, offsets in pages.items())
            if required_bytes <= 0 or required_bytes > max_bytes:
                return
            if charged + required_bytes > budget * 0.95:
                return
            if instance_limit and instance_charged + required_bytes > instance_limit:
                return
            try:
                mapped = self.arena.prefetch(pages)
            except (MemoryError, RuntimeError):
                # Prefetch is optional; the synchronous allocation path still
                # owns capacity failure handling.
                return
            for block_id in candidates:
                self._prefetch_blocks[block_id] = now
            self._prefetch_maps += 1
            print("KVCACHED_NATIVE_PREFETCH "
                  f"blocks={len(candidates)} mapped_bytes={mapped} "
                  f"queued={len(self._prefetch_blocks)}", flush=True)

    def _release_prefetch(self, block_ids, reason):
        block_ids = [block_id for block_id in block_ids
                     if block_id in self._prefetch_blocks]
        if not block_ids:
            return 0
        layout = self.owners.layout
        pages = layout.pages_for(
            {(group, block_id) for group in layout.groups for block_id in block_ids})
        for block_id in block_ids:
            self._prefetch_blocks.pop(block_id, None)
        try:
            reclaimed = self.arena.residency.reclaim(
                self.owners.required_pages(), eligible=pages)
        except Exception:
            return 0
        self._prefetch_drops += len(block_ids)
        print("KVCACHED_NATIVE_PREFETCH_DROP "
              f"reason={reason} blocks={len(block_ids)} "
              f"reclaimed_bytes={reclaimed}", flush=True)
        return reclaimed

    def _drop_expired_prefetch(self, now, ttl_s):
        """Forget expired speculative blocks without touching the device.

        This runs on the prefetch thread.  A device fence or VMM unmap there
        can block the engine behind ``physical_lock`` for minutes, so expiry
        is bookkeeping only: the pages stop being protected and the next
        worker physical phase retires them via its ordinary reclaim.
        """
        if not self._prefetch_blocks or ttl_s <= 0:
            return
        expired = [block_id for block_id, stamp in self._prefetch_blocks.items()
                   if now - stamp >= ttl_s]
        if not expired:
            return
        if os.getenv("KVCACHED_NATIVE_PREFETCH_TTL_RECLAIM", "0").strip() == "1":
            # Legacy diagnostic mode: fence and unmap on this thread.
            self._release_prefetch(expired, "ttl")
            return
        for block_id in expired:
            self._prefetch_blocks.pop(block_id, None)
        self._prefetch_drops += len(expired)
        print("KVCACHED_NATIVE_PREFETCH_EXPIRE "
              f"blocks={len(expired)} queued={len(self._prefetch_blocks)}",
              flush=True)

    def note_prefill_blocks(self, blocks):
        """Request map-ahead for the next prefill chunk, not for decode."""
        if not self._prefetch_thread:
            return
        try:
            max_blocks = max(1, int(os.getenv(
                "KVCACHED_NATIVE_MAP_AHEAD_BLOCKS", "4")))
            window_s = float(os.getenv(
                "KVCACHED_NATIVE_MAP_AHEAD_WINDOW_S", "1.0"))
        except ValueError:
            max_blocks, window_s = 4, 1.0
        target = min(max_blocks, max(1, int(blocks)))
        now = time.monotonic()
        self._prefetch_goal = max(self._prefetch_goal, target)
        self._prefetch_deadline = now + max(0.05, window_s)

    def consume_prefetch(self, blocks):
        """Mark allocated blocks as prefetch hits and stop tracking them."""
        if not self._prefetch_blocks:
            return
        with self.physical_lock:
            for block in blocks:
                block_id = getattr(block, "block_id", None)
                if block_id is not None and self._prefetch_blocks.pop(
                        block_id, None) is not None:
                    self._prefetch_hits += 1
                    if self._prefetch_hits % 32 == 0:
                        print("KVCACHED_NATIVE_PREFETCH_HIT "
                              f"hits={self._prefetch_hits} "
                              f"maps={self._prefetch_maps} "
                              f"drops={self._prefetch_drops}", flush=True)

    def prefetch_pages(self):
        """Pages currently protected by speculative prefetch."""
        if not self._prefetch_blocks:
            return {}
        layout = self.owners.layout
        return layout.pages_for(
            {(group, block_id)
             for group in layout.groups for block_id in self._prefetch_blocks})

    def should_log_preflight(self, request_id):
        """Log one miss per request instead of once per scheduler pass."""
        if request_id in self._preflight_logged:
            return False
        if len(self._preflight_logged) > 4096:
            self._preflight_logged.clear()
        self._preflight_logged.add(request_id)
        return True

    def ensure_blocks(self, blocks):
        """Map physical backing for KVCacheBlocks; raise on capacity miss."""
        layout = self.owners.layout
        needed = set()
        for group_id, group_blocks in enumerate(blocks.blocks):
            for block in group_blocks:
                needed.add((group_id, block.block_id))
        if not needed:
            return
        required = layout.pages_for(needed)
        started = time.perf_counter()
        self.critical_enter()
        try:
            with self.physical_lock:
                self.arena.ensure(required)
        finally:
            self.critical_exit()
        native_profile.record(
            "admission.ensure_blocks", time.perf_counter() - started)

    def ensure_group_blocks(self, group_id, blocks):
        """Map one group's newly allocated KVCacheBlock objects."""
        needed = {(group_id, block.block_id) for block in blocks}
        if not needed:
            return
        required = self.owners.layout.pages_for(needed)
        started = time.perf_counter()
        self.critical_enter()
        try:
            with self.physical_lock:
                self.arena.ensure(required)
        finally:
            self.critical_exit()
        native_profile.record(
            "admission.ensure_group_blocks", time.perf_counter() - started)

    def estimate_bytes(self, num_new_tokens, num_lookahead_tokens=0):
        """Conservative bytes for a request's new blocks, or 0 if unknown."""
        layout = self.owners.layout
        block_sizes = getattr(layout, 'block_sizes', None)
        if (num_new_tokens <= 0 or not block_sizes
                or set(block_sizes) != set(layout.groups)):
            return 0
        per_group = []
        for group_id, block_size in layout.block_sizes.items():
            if group_id not in layout.groups:
                continue
            blocks = (num_new_tokens + num_lookahead_tokens
                      + block_size - 1) // block_size
            per_group.append(blocks * layout.average_block_bytes(group_id))
        if not per_group:
            return 0
        # Hybrid groups commonly alias one native allocation.  Summing them
        # would count the same pages once per group; use the largest group
        # instead when there is a single allocation, and sum distinct ones.
        return max(per_group) if len(layout.storages) == 1 else sum(per_group)

    def preflight(self, num_new_tokens, num_lookahead_tokens=0):
        started = time.perf_counter()
        try:
            return self._preflight(num_new_tokens, num_lookahead_tokens)
        finally:
            native_profile.record(
                "admission.preflight", time.perf_counter() - started)

    def _preflight(self, num_new_tokens, num_lookahead_tokens=0):
        """Conservative bytes check before vLLM mutates request state.

        The block pool remains the authority for exact pages; this check only
        avoids committing prefix-cache ownership and then rolling it back on
        the common capacity miss. A false positive queues the request for one
        more scheduler pass, which is the same external behavior as a block
        pool miss, so over-estimating is safe.
        """
        required = self.estimate_bytes(num_new_tokens, num_lookahead_tokens)
        if not required:
            return True, 0, 0
        try:
            state = self.arena.operations.physical_budget_state()
        except Exception:
            # Older builds or a non-accelerator backend: leave the exact
            # block-pool path to decide.
            return True, required, 0
        budget = int(state.get('budget_bytes', 0) or 0)
        if budget <= 0:
            return True, required, 0
        watermark = int(state.get('watermark_bytes', 0) or 0)
        charged = int(state.get('charged_bytes', 0) or 0)
        instance_charged = int(state.get('instance_charged_bytes', 0) or 0)
        instance_limit = int(state.get('instance_limit_bytes', 0) or 0)
        usable = max(budget - watermark, 0)
        gpu_available = max(usable - charged, 0)
        if instance_limit:
            instance_available = max(instance_limit - instance_charged, 0)
        else:
            instance_available = gpu_available
        available = min(gpu_available, instance_available)
        return required <= available, required, available

    def note_priority(self, request, blocks):
        """Remember the most important request priority that owned each block.

        vLLM uses lower numeric values for more important requests.
        """
        if os.getenv("KVCACHED_NATIVE_QOS", "1").strip() == "0":
            return
        priority = int(getattr(request, "priority", 0) or 0)
        groups = getattr(blocks, "blocks", blocks)
        try:
            for group_blocks in groups:
                for block in group_blocks:
                    block_id = getattr(block, "block_id", None)
                    if block_id is None:
                        continue
                    previous = self.block_priority.get(block_id, priority)
                    # vLLM priority semantics: lower value is more important.
                    self.block_priority[block_id] = min(previous, priority)
        except TypeError:
            # A single block/list is also accepted by callers/tests.
            block_id = getattr(blocks, "block_id", None)
            if block_id is not None:
                self.block_priority[block_id] = min(
                    self.block_priority.get(block_id, priority), priority)

    def forget_blocks(self, blocks):
        """Drop stale priority metadata when a block is handed out again."""
        for block in blocks:
            block_id = getattr(block, "block_id", None)
            if block_id is not None:
                self.block_priority.pop(block_id, None)

    def try_raise_limit(self, need_bytes=0):
        started = time.perf_counter()
        try:
            return self._try_raise_limit(need_bytes)
        finally:
            native_profile.record(
                "admission.try_raise_limit", time.perf_counter() - started)

    def _try_raise_limit(self, need_bytes=0):
        """Borrow unused shared budget by raising this instance's cap.

        Instance limits are ceilings, not reservations.  If the shared ledger
        has free bytes and this instance's logical layout can hold them, a
        higher cap lets the next preflight pass admit work without taking pages
        away from the other model.  The cap never exceeds the instance's own
        native reservation, so no runtime logical-block expansion is implied.
        """
        if os.getenv("KVCACHED_NATIVE_DYNAMIC_LIMIT", "1").strip() == "0":
            return False
        try:
            state = self.arena.operations.physical_budget_state()
            budget = int(state.get("budget_bytes", 0) or 0)
            watermark = int(state.get("watermark_bytes", 0) or 0)
            charged = int(state.get("charged_bytes", 0) or 0)
            current = int(state.get("instance_limit_bytes", 0) or 0)
            logical = int(self.arena.residency.status()["reserved_bytes"])
        except Exception:
            return False
        if budget <= 0 or current <= 0 or current >= (1 << 63):
            return False
        usable = max(budget - watermark, 0)
        free = max(usable - charged, 0)
        if not free:
            return False
        target = min(logical, charged + free)
        if target <= current:
            return False
        self._limit_revision += 1
        try:
            status = self.arena.operations.set_instance_budget_limit(
                int(target), self._limit_revision)
        except Exception as error:
            print("KVCACHED_NATIVE_DYNAMIC_LIMIT_FAILED "
                  f"target={target} error={str(error)[:160]}", flush=True)
            return False
        print("KVCACHED_NATIVE_DYNAMIC_LIMIT "
              f"from={current} to={target} need={need_bytes} "
              f"status={status}", flush=True)
        return True

    def reclaim_idle(self):
        started = time.perf_counter()
        reclaimed = 0
        try:
            reclaimed = self._reclaim_idle()
        finally:
            native_profile.record(
                "admission.idle_reclaim", time.perf_counter() - started,
                nbytes=reclaimed)
        return reclaimed

    def _reclaim_idle(self):
        """Release this instance's idle prefix cache, without a peer request.

        The normal relief path needs a local admission miss.  A full but idle
        donor has no miss, so its prefix cache would otherwise hold shared
        budget forever.  Only call this from the scheduler's no-running/
        no-waiting state: evict all free hashed blocks, drop the prefix owner,
        and let the existing physical reclaim fence/unmap them.
        """
        coordinator = self.coordinator
        if coordinator is None:
            return 0
        # Idle prefix is valuable.  Do not trade it away unless the shared
        # ledger is actually under pressure (or an explicit threshold is set
        # lower for a controlled shrink test).
        try:
            state_started = time.perf_counter()
            state = self.arena.operations.physical_budget_state()
            native_profile.record(
                "idle_reclaim.budget_state", time.perf_counter() - state_started)
            budget = int(state.get("budget_bytes", 0) or 0)
            charged = int(state.get("charged_bytes", 0) or 0)
            threshold = float(os.getenv(
                "KVCACHED_NATIVE_IDLE_RECLAIM_UTILIZATION", "0.8") or 0)
        except Exception:
            budget = charged = 0
            threshold = 0.8
        if budget > 0 and threshold > 0 and charged < budget * threshold:
            return 0
        try:
            max_bytes = max(0, int(os.getenv(
                "KVCACHED_NATIVE_IDLE_RECLAIM_MAX_BYTES",
                str(2 << 30))))
        except ValueError:
            max_bytes = 2 << 30
        # One chunk must stay short: every retired page is released by a
        # driver call while the cross-process budget lock is held, which stalls
        # a peer's admission map for as long as this chunk runs.
        try:
            batch_bytes = max(1 << 20, int(os.getenv(
                "KVCACHED_NATIVE_IDLE_RECLAIM_BATCH_BYTES",
                str(64 << 20))))
        except ValueError:
            batch_bytes = 64 << 20
        print("KVCACHED_NATIVE_IDLE_RECLAIM_START "
              f"charged={charged} budget={budget} max_bytes={max_bytes}",
              flush=True)
        with self.physical_lock:
            lock_wait = time.perf_counter()
            pool = coordinator.block_pool
            layout = self.owners.layout
            avg = max(layout.average_block_bytes(group)
                      for group in layout.groups)
            target = max(0, charged - int(budget * 0.70)) if budget else 0
            target = min(target, max_bytes, batch_bytes)
            target = max(target, avg)  # always make progress by one block
            queue = pool.free_block_queue
            node = queue.fake_free_list_head.next_free_block
            tail = queue.fake_free_list_tail
            candidates = []
            order = 0
            while node is not None and node is not tail:
                if (not node.is_null and not node.ref_cnt
                        and (node.block_hash is not None
                             or node.block_id in pool.cached_block_hashes_by_block)):
                    candidates.append((node, order))
                order += 1
                node = node.next_free_block
            native_profile.record(
                "idle_reclaim.queue_walk", time.perf_counter() - lock_wait)
            # Free queue is LRU-ordered; retire the coldest tail first.
            candidates.sort(key=lambda item: item[1], reverse=True)
            selected = []
            selected_bytes = 0
            for block, _ in candidates:
                selected.append(block)
                selected_bytes += avg
                if selected_bytes >= target:
                    break
            if not selected:
                return 0
            step = time.perf_counter()
            pool.evict_blocks({block.block_id for block in selected})
            native_profile.record(
                "idle_reclaim.evict", time.perf_counter() - step)
            selected_pairs = {
                (group, block.block_id)
                for group in layout.groups for block in selected
            }
            step = time.perf_counter()
            removed = self.owners.remove_blocks("prefix", selected_pairs)
            native_profile.record(
                "idle_reclaim.remove_owner", time.perf_counter() - step)
            step = time.perf_counter()
            eligible = layout.pages_for(selected_pairs)
            native_profile.record(
                "idle_reclaim.pages_for", time.perf_counter() - step)
            step = time.perf_counter()
            required = self.owners.required_pages()
            native_profile.record(
                "idle_reclaim.required_pages", time.perf_counter() - step)
            step = time.perf_counter()
            reclaimed = self.arena.residency.reclaim(required, eligible=eligible)
            native_profile.record(
                "idle_reclaim.physical", time.perf_counter() - step,
                nbytes=reclaimed)
            print(
                "KVCACHED_NATIVE_IDLE_RECLAIM "
                f"evicted={len(selected)} reclaimed={reclaimed} "
                f"prefix_removed={removed} target={target}",
                flush=True,
            )
            return reclaimed

    def relieve_capacity(self, need_bytes=0):
        started = time.perf_counter()
        selected = 0
        try:
            selected = self._relieve_capacity(need_bytes)
        finally:
            native_profile.record(
                "admission.relieve_capacity", time.perf_counter() - started,
                last_n=selected)
        return selected

    def _relieve_capacity(self, need_bytes=0):
        """Evict replaceable prefix blocks from the scheduler's free queue.

        Only the worker may unmap pages: the scheduler runs before the next
        ownership snapshot exists, and unmapping step-io/dummy pages there can
        make the following execute fail.  This method removes prefix hashes;
        the worker then rebuilds ownership from that same eviction and reclaims
        retired pages before mapping new ones.
        """
        if os.getenv("KVCACHED_NATIVE_RECLAIM", "1").strip() == "0":
            return 0
        coordinator = self.coordinator
        if coordinator is None:
            return 0
        with self.physical_lock:
            if self._prefetch_blocks:
                # Speculative pages are the cheapest first victim; they are
                # not owned by any request yet.
                self._release_prefetch(
                    list(self._prefetch_blocks), "pressure")
            pool = coordinator.block_pool
            avg = max(self.owners.layout.average_block_bytes(group)
                      for group in self.owners.layout.groups)
            wanted = ((need_bytes + avg - 1) // avg) if avg and need_bytes else 1
            wanted = max(1, wanted)
            # The free queue is ordered by LRU; walk it instead of block-id
            # order so pressure evicts the least recently reused prefixes.
            queue = pool.free_block_queue
            node = queue.fake_free_list_head.next_free_block
            tail = queue.fake_free_list_tail
            candidates = []
            order = 0
            while node is not None and node is not tail:
                if (not node.is_null and not node.ref_cnt
                        and (node.block_hash is not None
                             or node.block_id in pool.cached_block_hashes_by_block)):
                    candidates.append((node, order))
                order += 1
                node = node.next_free_block
            qos = os.getenv("KVCACHED_NATIVE_QOS", "1").strip() != "0"
            candidates.sort(
                key=lambda item: (
                    -self.block_priority.get(item[0].block_id, 0) if qos else 0,
                    item[1]))
            selected = [block for block, _ in candidates[:wanted]]
            if selected:
                pool.evict_blocks({block.block_id for block in selected})
            self.relief_events += 1
            print(
                "KVCACHED_NATIVE_RECLAIM "
                f"evicted={len(selected)} need={need_bytes} "
                f"priority={[self.block_priority.get(block.block_id, 0) for block in selected][:8]}",
                flush=True,
            )
            return len(selected)


def install_gate(arena, owners):
    global _GATE
    _GATE = NativeAdmissionGate(arena, owners)
    if _SCHEDULER_CONTEXT is not None:
        _GATE.bind_scheduler(*_SCHEDULER_CONTEXT)
    _GATE.start_idle_thread()
    _GATE.start_prefetch_thread()
    _GATE.start_watchdog()
    return _GATE


def install_scheduler_context(coordinator, scheduler=None):
    """Connect scheduler-side eviction to the worker-installed gate."""
    global _SCHEDULER_CONTEXT
    _SCHEDULER_CONTEXT = (coordinator, scheduler)
    if _GATE is not None:
        _GATE.bind_scheduler(coordinator, scheduler)


def reclaim_idle():
    """Release idle prefix backing if a worker gate is installed."""
    return _GATE.reclaim_idle() if _GATE is not None else 0


def note_progress():
    """Tell the stall watchdog that the engine loop completed an iteration."""
    gate = _GATE
    if gate is not None:
        gate.note_progress()


def _capacity_error(error):
    return isinstance(error, MemoryError) or is_capacity_rejection(error)


def _rollback_request(manager, request, marker):
    """Undo every CPU-side allocation made before a physical-capacity miss.

    ``allocate_new_computed_blocks`` touches prefix-cache blocks before
    ``allocate_new_blocks`` runs, and a multi-group allocation may already have
    committed earlier groups.  Returning ``None`` without freeing that state
    leaves ``num_cached_block`` set but no request blocks on the pool, which
    makes the next scheduling pass assert.  vLLM's own ``free`` is the only
    rollback that covers all groups and their bookkeeping.
    """
    try:
        manager.free(request)
    except Exception as error:
        print(f'{marker}_ROLLBACK_FAILED request={request.request_id} '
              f'error={str(error)[:200]}', flush=True)
        raise


def patch_allocate_slots():
    """Patch vLLM's KVCacheManager.allocate_slots once for native admission."""
    global _PATCHED
    with _LOCK:
        if _PATCHED:
            return
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        original = KVCacheManager.allocate_slots
        if getattr(original, "_kvcached_native_admission", False):
            _PATCHED = True
            return

        def allocate_slots(self, request, *args, **kwargs):
            gate = _GATE
            num_new_tokens = args[0] if args else kwargs.get(
                "num_new_tokens", 0)
            num_lookahead_tokens = args[3] if len(args) > 3 else kwargs.get(
                "num_lookahead_tokens", 0)
            num_new_tokens = int(num_new_tokens or 0)
            num_lookahead_tokens = int(num_lookahead_tokens or 0)
            # Only new requests reserve from free capacity.  A running request
            # already owns its blocks; checking free bytes again would reject
            # every decode step once the instance is full and preempt it
            # forever.  Exact block-pool admission still covers running growth.
            is_new_request = int(
                getattr(request, 'num_computed_tokens', 0) or 0) == 0
            if gate is not None:
                prompt_tokens = int(getattr(request, 'num_prompt_tokens', 0) or 0)
                computed_tokens = int(
                    getattr(request, 'num_computed_tokens', 0) or 0)
                if prompt_tokens > computed_tokens:
                    block_sizes = getattr(
                        gate.owners.layout, 'block_sizes', {}) or {896: 896}
                    block_size = max(block_sizes.values())
                    gate.note_prefill_blocks(
                        (prompt_tokens - computed_tokens + block_size - 1)
                        // block_size)

            def relieve(marker, need):
                if gate is None:
                    return 0
                try:
                    return gate.relieve_capacity(need)
                except Exception as error:
                    # Relief is best-effort: a quarantined map must not turn a
                    # scheduling miss into an engine failure.  The next step or
                    # worker restart still sees the original miss.
                    print(f"{marker}_RECLAIM_FAILED "
                          f"error={str(error)[:200]}", flush=True)
                    return 0

            if (gate is not None and is_new_request and os.getenv(
                    "KVCACHED_NATIVE_PREFLIGHT", "1").strip() != "0"):
                admitted, required, available = gate.preflight(
                    num_new_tokens, num_lookahead_tokens)
                if not admitted:
                    # First borrow unused shared budget by raising this
                    # instance's physical cap; this is a ceiling change, not a
                    # logical block-pool resize.  Only if the shared ledger has
                    # no free budget do we evict this instance's own cache.
                    if gate.try_raise_limit(required):
                        admitted, required, available = gate.preflight(
                            num_new_tokens, num_lookahead_tokens)
                if not admitted:
                    # No request state has been touched yet.  Reclaim cached
                    # pages before returning, then retry once: if enough came
                    # back, this request can be admitted in the same pass.
                    relieve("KVCACHED_NATIVE_PREFLIGHT_MISS", required)
                    admitted, required, available = gate.preflight(
                        num_new_tokens, num_lookahead_tokens)
                if not admitted:
                    gate.preflight_misses += 1
                    if gate.should_log_preflight(request.request_id):
                        print(
                            "KVCACHED_NATIVE_PREFLIGHT_MISS "
                            f"request={request.request_id} "
                            f"required={required} available={available}",
                            flush=True,
                        )
                    return None
            try:
                result = original(self, request, *args, **kwargs)
            except KVCachePoolExhausted:
                # The block pool reported a physical-capacity miss.  A new
                # request may already have touched prefix-cache blocks or
                # committed earlier groups, so free its partial bookkeeping.
                # A running request already owns its blocks: freeing them
                # would discard its KV history.  Its earlier groups may have
                # gained blocks this pass; they remain valid and the next
                # allocation only fills the missing groups.
                if is_new_request:
                    _rollback_request(self, request, "KVCACHED_NATIVE_POOL_MISS")
                print(
                    "KVCACHED_NATIVE_POOL_MISS "
                    f"request={request.request_id} "
                    f"running={not is_new_request}",
                    flush=True,
                )
                relieve("KVCACHED_NATIVE_POOL_MISS",
                        gate.estimate_bytes(num_new_tokens,
                                            num_lookahead_tokens)
                        if gate is not None else 0)
                return None
            if result is None or gate is None:
                return result
            try:
                gate.ensure_blocks(result)
            except Exception as error:
                if not _capacity_error(error):
                    raise
                # Return the logical blocks before reporting a scheduling miss.
                _rollback_request(self, request, "KVCACHED_NATIVE_ADMISSION_MISS")
                print(
                    "KVCACHED_NATIVE_ADMISSION_MISS "
                    f"request={request.request_id} error={str(error)[:200]}",
                    flush=True,
                )
                relieve("KVCACHED_NATIVE_ADMISSION_MISS",
                        gate.estimate_bytes(num_new_tokens,
                                            num_lookahead_tokens))
                return None
            gate.note_priority(request, result)
            new_computed = (args[2] if len(args) > 2
                            else kwargs.get("new_computed_blocks"))
            if new_computed is not None:
                gate.note_priority(request, new_computed)
            return result

        def timed_allocate_slots(self, request, *args, **kwargs):
            started = time.perf_counter()
            try:
                return allocate_slots(self, request, *args, **kwargs)
            finally:
                native_profile.record(
                    "admission.allocate_slots", time.perf_counter() - started)

        timed_allocate_slots._kvcached_native_admission = True
        KVCacheManager.allocate_slots = timed_allocate_slots
        _PATCHED = True


def patch_block_pool_admission():
    """Make native block allocation consult the physical gate first.

    ``SingleTypeKVCacheManager.allocate_new_blocks`` knows the cache group;
    set a context variable around it so the shared BlockPool.get_new_blocks
    can map exactly that group's pages.  A physical miss frees the polled
    blocks and raises KVCachePoolExhausted, which allocate_slots converts to
    vLLM's normal None scheduling miss.
    """
    global _POOL_PATCHED
    with _LOCK:
        if _POOL_PATCHED:
            return
        from vllm.v1.core.block_pool import BlockPool
        from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager

        original_get = BlockPool.get_new_blocks
        if getattr(original_get, "_kvcached_native_admission", False):
            _POOL_PATCHED = True
            return

        def get_new_blocks(self, num_blocks):
            gate = _GATE
            if gate is None:
                return original_get(self, num_blocks)
            with gate.physical_lock:
                blocks = original_get(self, num_blocks)
                gate.consume_prefetch(blocks)
                gate.forget_blocks(blocks)
                group_id = _GROUP.get()
                if group_id is not None and os.getenv(
                        "KVCACHED_NATIVE_DEFER_GROUP_MAP", "0").strip() != "0":
                    # Every group-context allocation is part of the
                    # KVCacheBlocks that allocate_slots maps once, after all
                    # groups are allocated.  Mapping here repeats the driver
                    # call up to four times per allocation for the same pages;
                    # a capacity miss is still reported by ensure_blocks and
                    # rolled back by the caller.
                    return blocks
                group_ids = (
                    [group_id] if group_id is not None
                    else list(gate.owners.layout.groups)
                )
                try:
                    for gid in group_ids:
                        gate.ensure_group_blocks(gid, blocks)
                except Exception as error:
                    if not _capacity_error(error):
                        raise
                    self.free_blocks(blocks)
                    raise KVCachePoolExhausted(
                        "native physical capacity miss during block allocation"
                    ) from error
                return blocks

        get_new_blocks._kvcached_native_admission = True
        BlockPool.get_new_blocks = get_new_blocks

        classes = [SingleTypeKVCacheManager]
        classes.extend(SingleTypeKVCacheManager.__subclasses__())
        for cls in classes:
            method = cls.__dict__.get("allocate_new_blocks")
            if method is None or getattr(
                    method, "_kvcached_native_group_ctx", False):
                continue

            def wrapped(self, *args, _method=method, **kwargs):
                token = _GROUP.set(self.kv_cache_group_id)
                try:
                    return _method(self, *args, **kwargs)
                finally:
                    _GROUP.reset(token)

            wrapped._kvcached_native_group_ctx = True
            setattr(cls, "allocate_new_blocks", wrapped)

        _POOL_PATCHED = True
