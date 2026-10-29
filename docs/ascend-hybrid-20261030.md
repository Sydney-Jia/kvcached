# Bounded Ascend hybrid experiment

This experiment preserves the original Ascend branch and installed wheel.
The candidate is on `experiment/ascend-hybrid-kv`. It is not a generic claim
that every MambaSpec or Ascend operator supports elastic storage.

## Changes to the storage contract

The tested Qwen GDN model has one full-attention group and three state groups.
The native layout uses separate dense state spans and aliases their allocation.
kvcached's unified pool instead stores one logical block in one uniform cell.
Before KV planning, the adapter normalizes padded spec sizes to that cell and
checks that every state fits, without changing state shape or dtype. In this
experiment the cell is 8 MiB, with 2048 tokens per attention block. Each group
still receives distinct logical block IDs from the shared pool.

Ascend's native tensor factory is also required. Generic stable `from_blob`
did not establish the NPU storage metadata needed by the convolution operator.
The isolated `ascend_tensor.cpp` translation unit constructs a torch_npu tensor
and transfers the Tensor-object handle to the stable wrapper. External physical
storage remains owned by FTensor. This part of the NPU wheel is version-coupled
to matching PyTorch/torch_npu headers and libraries; CUDA/HIP keep their stable
factory path.

## An operator can return the right output and still lose its state update

`tests/ascend/state_stride_probe.py` uses fully backed ordinary tensors, so its
result is independent of VMM, page retirement, and request scheduling. It
compares identical dense and strided states on the same native operators,
including outputs, updated state, a repeated native control, and explicit
compact-copy/writeback. The tested snapshot showed:

- KV scatter wrote both strided K/V views correctly.
- Causal convolution returned matching immediate output, but the strided
  original state was unchanged. The following decode step can therefore diverge.
- Recurrent GDN output and updated state differed for the strided view.
- Explicit compact-copy/writeback made both GDN cases match their native controls.

The recurrent operator in this snapshot requires a leading zero followed by
per-sequence lengths, not cumulative endpoints. Earlier diagnostic input using
the wrong encoding failed its native-repeat control and is not evidence about
stride support. Do not generalize these observations to other CANN versions or
operator implementations.

## Bounded active-state bridge

The opt-in adapter wraps only the worker's two GDN Python operator entry points.
Contiguous states pass through unchanged. Otherwise it gathers the active rows,
remaps indices into that compact temporary, invokes the native operator, and
copies the mutated state back in stream order. It does not materialize all
virtual rows: inactive rows can be unmapped. Null/padding slots remain protected
by the existing allocator's null-block lifetime.

This relies on non-speculative eager execution with distinct live state indices
per request. The model-family guard accepts only `qwen3_5_text`; prefix caching,
speculation, asynchronous/graph execution and multi-rank modes are outside this
candidate. Additional models must validate their state update and copy semantics.

The bridge adds copying, allocation/dispatch and potential synchronization costs.
For this model, one request's convolution plus recurrent state totals about
146.8 MiB across 48 layers. Gather plus writeback is about 293.6 MiB of copied
payload per decode step, before counting reads and writes separately. This is
a size calculation, not a bandwidth measurement or a performance guarantee.

## Acceptance

Keep tensor-construction, operator-layout and model-lifecycle tests separate.
Returning tokens is insufficient: compare complete outputs before and after
reclamation, mixed input lengths, borrowing by another engine, restoration,
real driver-memory pressure and both engines' latency. The native reference
must retain its verified native layout; record any intentional logical-block
size difference rather than silently treating layouts as identical.

The surrounding experiment workspace retains fixed-commit releases, complete
outputs, commands, driver samples, failure logs and process supervision. The
comparison report records which candidate and protocol passed each gate.

## Native admission headroom is a separate constraint

The tested Ascend snapshot's `AscendMambaManager.get_num_blocks_to_allocate`
adds one block whenever computed tokens exceed local prefix hits and new tokens
are scheduled. With an ordinary running decode and no prefix hits, that condition
also holds without an external KV connector. All three state groups therefore
report one extra block even while their live state still occupies the same rows.

An isolated quota staircase retained the same submitted request: prefill emitted
one token with 640 MiB physically backed, but decode reported three required blocks
against only one/two available at 768/896 MiB quotas. The next token arrived after
the quota reached 1024 MiB. That diagnostic preserved native admission behavior;
the protocol records the calibrated quota explicitly.

Real pressure then exposed the availability consequence: the borrower completed
all four requests correctly, but the donor timed out because the predictor
required extra capacity for already backed local decode. That run failed joint
acceptance; useful borrower progress alone was not counted as success.

The bounded candidate now binds the original base `MambaManager` predictor to
the inspected `AscendMambaManager` instances when hybrid mode is explicitly
enabled and KV connectors are absent. This removes only the external-load
addition, preserves actual new-block prediction/allocation/free behavior, and
does not mutate the manager class used by native or unrelated instances.
External connectors are rejected. A general native-framework fix should use
actual external-token provenance rather than computed-token count; it needs its
own connector tests and review. Hybrid asynchronous scheduling is also rejected.

## Apply live controls even when no new KV block is needed

An otherwise correct 128-token donor request completed before the generic
allocation/idle hooks applied its shrink. The large 2048-token logical block
required no new KV allocation during decode. Its control acknowledgement took
16.8 seconds, so that run failed the online-closure gate despite returning real
memory and producing the correct full output.

The Ascend integration now polls the existing runtime control managers before
each synchronous `EngineCore.step`. The previous step's future and output
processing have completed; native allocator synchronization still fences any
physical unmap. The hook changes no live-block ownership, does not run for
queued/asynchronous execution, and propagates failures before another batch can
be submitted. The generic idle watcher remains necessary when no step runs.
CPU checks cover operation ordering, disablement, queued/async exclusions,
idempotent installation and fail-closed behavior. Hardware acceptance must again
prove that the donor remains active when the borrower obtains added capacity.

## Completed bounded acceptance (2026-10-30)

Final Python candidate `04820ba` uses the native extension built at `1e919a0`;
the extension/build sources were checked unchanged. All final cases used the
same fixed candidate. The original wheel and native runtime were preserved.

With two independent 27B engines and about 256 MiB of real device memory left,
B ran four requests serially in 99.98 s while A retained its KV backing.
Reclaiming 512 MiB from A let B grow from 640 to 1152 MiB, run two requests
concurrently, and finish all four in 58.86 s. Both engines' complete outputs
matched the native reference in both arms. A completed in 35.27/34.61 s;
its maximum observed HTTP output gap was 325/319 ms (held/reclaimed).
These are fixed-load samples, not SLO distributions.

The separate ample-memory closure passed donor-live borrowing, restoration,
and two mixed-input replay rounds. The same final candidate passed the 0.6B
ordinary-attention closure. The native extension passed four NPU page-lifetime
checks; 98 related CPU tests passed. No foreign NPU process was detected in
the accepted measurement windows and all experiment-owned processes exited.

See [the compact evidence ledger](ascend-hybrid-20261030-evidence.json) for exact
run IDs, bounded configuration, latency, hashes and limits. Full token traces
and supervision logs remain in the experiment workspace. Failed donor-starvation
and delayed-control runs remain evidence, not successful acceptance.

This validates the explicitly guarded single-rank synchronous eager text path.
It does not establish graph, asynchronous, prefix-cache, speculative, connector,
multimodal or long-duration production support. The active-state bridge still
adds copies and needs a better layout/operator contract before broad adoption.
