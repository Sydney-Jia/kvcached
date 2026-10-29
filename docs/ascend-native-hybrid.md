# Native DeepSeek cache residency experiment

This opt-in V2 worker is a development experiment, not an accepted replacement
for the existing Ascend adapter. Target sources are vLLM `c4df37dfcc7f` and
vLLM-Ascend `9fe0bbf380de`, with TP2/DP2/EP4, eager execution, no prefix retention,
no speculative decoding, no asynchronous scheduling, no KV connector or sleep.
The general kvcached autopatcher must be disabled.

The native allocator constructs one independent byte backing per cache slot.
Specialized long-cache, index, sliding-window and circular-state views alias
these backings using native offsets/strides. The experiment substitutes only
the backing allocation with kvcached VMM tensors; native reshape, attention and
logical block tables remain unchanged. It does not reinterpret these tensors
as ordinary MHA or MLA caches.

Build this path with `KVCACHED_BUILD_NPU_STORAGE=1`. It adds an optional
`_npu_storage` extension built for the installed **torch and torch_npu versions**;
unlike the core VMM extension, this bridge does not promise stable ABI across
versions. Native ACLNN cache operators inspect NPU storage format metadata.
Generic stable-ABI `from_blob` provides a pointer and shape but not an
`NPUStorageImpl` descriptor. The bridge constructs a zero-element native tensor,
attaches the existing VMM pointer with lifetime ownership, and initializes its
ND byte-storage descriptor before any native dtype/shape views are created.
It neither copies the cache nor changes its virtual address. A device acceptance
probe must check FP8 view format, pointer identity, live data preservation and
the physical free-memory delta; a successful generic tensor operation alone is
insufficient to validate native attention compatibility.

The Ascend budget key combines chip UUID with the runtime-resolved physical
device ID. On the tested A5, devices0/1 share one UUID and devices2/3 another;
UUID alone incorrectly merges independent memory budgets. A visibility-remapped
process must still resolve the same physical key. Processes also need the same
physical-growth lock directory in a shared IPC/filesystem namespace to obtain
coordinated admission; separate container `/dev/shm` directories do not provide
a common budget ledger. Mixed old/new key formats must not be treated as one
coordinated admission domain.

The worker retains all request-owned blocks, including unscheduled requests,
and transient zero/copy endpoints. A block interval pins every intersecting
2 MiB physical page. Block zero is always pinned. The union across cache groups
is conservative: a native slot can hold different cache types at different
block IDs. No page containing a live interval may be reclaimed. Device work is
synchronized before physical retirement. Newly mapped pages are initialized
without zeroing existing live pages.

Worker RPC `kvcached_native_control` accepts `status`, `elastic` and `held`.
`elastic` releases unowned pages and allows demand growth; `held` remaps the
original budget. The tensor addresses and logical block capacity stay fixed.
This experiment does **not yet** implement scheduler capacity feedback for a
failed physical grow, or a coordinated TP/DP rollback. Such a failure must stop
the trial rather than be presented as transparent resource sharing.

The separate `trim_allocator` diagnostic synchronizes device work and calls
`torch.npu.empty_cache()` to release unused **framework allocator** blocks.
It does not retire KV pages, evict live tensors or change the pool mode. Every
control reports driver free memory and Torch allocated/reserved memory before
and after the action. This distinction matters when temporary operator buffers
grow the framework cache and consume capacity returned by the KV allocator.
Matched experiments must apply this diagnostic to both held and elastic
conditions and report its memory delta separately. It is not automatic
cross-engine admission or a remedy for genuinely live-memory exhaustion.

CPU tests cover ownership and partial-page boundaries, cross-backing growth
rollback, transactional retirement and alias preservation. They do not prove
NPU kernel correctness. Hardware acceptance must separately record native,
held and elastic outputs, real driver memory, live-request transitions,
recipient use and latency. `tools/ascend_native_launch.py` produces a matched
configuration from a preserved deployment launcher and external machine
bindings. Its dev RPC binds localhost; it is not a production serving recipe.

## Acceptance observations (2026-10-30)

On Ascend 950DT with the above versions, short requests passed 9/9, held-pool
long/concurrent cases passed 10/10, and live shrink/restore/demand-growth cases
passed 11/11. Each rank's mapped backing went from 2,050 MiB to 8 MiB with unchanged
virtual addresses. Driver-free and Torch allocated/reserved counters were
recorded separately; unused framework cache trimming was applied to both
held and elastic conditions. Bulk KV transitions took seconds, not zero time.

A controlled pressure comparison used another Qwen3-0.6B engine on one card
and a fixed 4.125 GiB physical allocation. Eight identical requests completed in
9.59 s with held KV versus 4.76 s after KV reclamation; sampled running concurrency
rose from 3 to 8 and recipient peak HBM from 2,447 MiB to 3,679 MiB. Both sets of eight
outputs were correct. The experiment retained the allocator's 5% headroom;
it did not exhaust raw HBM. These are scoped observations, not a general
throughput guarantee or broad model-quality evaluation.

**Concurrent execution is not accepted as stable.** One full coactive cycle
passed, but a repeat hit MegaMoE cross-rank barrier timeouts. An independent
native DeepSeek run without this worker/VMM, without memory pressure or KV
retirement, reproduced the same barrier failure when running with Qwen. Only
the two serving processes appeared in that card's samples. This establishes
that the failure can occur without the DeepSeek KV adaptation; it does not
identify the exact scheduling/kernel root cause or exclude additional bugs.
Repair the native co-location baseline before claiming reliable simultaneous
execution. Independent container IPC namespaces and mixed budget-key versions
also mean this trial does not validate a common atomic admission ledger.
