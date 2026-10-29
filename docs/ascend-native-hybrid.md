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

CPU tests cover ownership and partial-page boundaries, cross-backing growth
rollback, transactional retirement and alias preservation. They do not prove
NPU kernel correctness. Hardware acceptance must separately record native,
held and elastic outputs, real driver memory, live-request transitions,
recipient use and latency. `tools/ascend_native_launch.py` produces a matched
configuration from a preserved deployment launcher and external machine
bindings. Its dev RPC binds localhost; it is not a production serving recipe.
