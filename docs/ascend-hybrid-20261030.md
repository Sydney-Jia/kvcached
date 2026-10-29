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
