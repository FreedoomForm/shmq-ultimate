# SHMQ Unified Three-Level SM75 Architecture Specification

## Goal

Replace the current mixed-precision prefill split with one shared staged scheduler and epilogue, while retaining compile-time-specialized INT4, INT8, and FP16 arithmetic leaves. The implementation must preserve the exact 4/8/16 partition, output mapping, quantization arithmetic, quality model, benchmark conditions, and v271 rollback behavior.

## Non-goals

This work will not fuse different precision MMA instructions into one sequential instruction stream, remove channel partitions, change model weights, reduce computation, lower quality thresholds, change Qwen benchmark shapes, or claim speed from local execution. No candidate is accepted until the Kaggle T4 gates pass.

## Module seam

The deep internal module is `UnifiedPartitionLauncher`. Its interface accepts a validated partition plan, prepared activation, persistent precision storage, metadata, output buffer, and a CUDA stream topology. It returns only after all partition events have joined the caller stream. Callers do not know whether a precision leaf uses packed nibbles, signed bytes, or half values.

The implementation contains these internal adapters:

- `Int8StagedLeaf`: the existing v271 signed-INT8 CUTLASS runner, initially unchanged and used as the reference adapter.
- `Fp16StagedLeaf`: the existing FP16 WMMA path behind the same scheduler/epilogue interface; its arithmetic remains unchanged.
- `Int4PackedStagedLeaf`: a new native packed INT4 adapter using the persistent interleaved cache and a legal SM75 `u4*u4 m8n8k32` instruction path. It is guarded until every probe and large-M correctness contract passes.

## Shared execution stages

1. The Python module prepares or reuses immutable packed INT4, INT8, FP16, and metadata tensors using device/version/shape/stride signatures.
2. The backend validates the partition once and constructs a `PartitionPlan` with complete, disjoint indices and channel counts.
3. Activation quantization runs once for the integer leaves; FP16 uses the original activation.
4. A fork event is recorded on the caller stream.
5. Precision leaves launch on persistent auxiliary streams using the same tile planner and output contract. INT4 and INT8 wait on the fork; FP16 can remain on the caller stream or use a dedicated stream when the allocator/event contract proves safe.
6. Each leaf writes FP16 accumulations through the shared indexed scatter epilogue. No leaf writes another leaf’s output channels.
7. The caller stream waits on all leaf completion events before returning.

## Precision contracts

### INT4

The source storage remains two nibbles per byte and the persistent interleaved cache follows the original MixLLM K-order permutation. The native adapter must prove the mapping from packed bytes to logical uint4 fragments and exact zero-point semantics. It may use a widened logical k32 load and two internal k16 signed MMAs only if the fragment and metadata contracts are proven; otherwise it must use the legal native SM75 u4/u4 k32 leaf. No implicit use of the generic SM75 uint4 k16 iterator is allowed.

### INT8

The signed INT8 leaf remains the v271 reference path. Its m8n8k16 accumulator, scale application, zero behavior, and `32x128x64 / 32x32x64 / stage=2` core are unchanged in the first implementation step.

### FP16

The FP16 leaf retains the existing WMMA arithmetic and output conversion. The first unified implementation may share only scheduler and epilogue code; it must not insert integer conversion templates into the FP16 specialization.

## Memory contracts

For Qwen mixed rows=128, peak allocated bytes must be no greater than v271’s recorded **60,555,264 bytes** before native INT4 is enabled. The native INT4 path must report `expanded_int4_bytes=0` when it is actually selected and must not build an unused expanded cache on the hot path. Persistent packed, metadata, temporary, and peak allocation bytes must be reported separately.

## Quality and gate contracts

The exact Qwen/Qwen2.5-0.5B model remains the only quality model. Full-model quality and patched-vLLM execution are required production gates; unavailable environments are blockers. Operator correctness must cover mixed and pure precision partitions, rows=1/16/128, non-contiguous index order, and randomized reference outputs. Timing-integrity must remain within the existing accepted ratio bounds. The candidate must pass embedded contracts, T4 hardware, native correctness, decode GEMM/E2E, prefill GEMM/E2E, allocator checks, model quality, patched-vLLM, and timing-integrity.

## Rollout order

The shared planner/epilogue is introduced first with the existing INT8 and FP16 leaves. The next local suite must prove byte-for-byte output equivalence and no ABI regression. Native INT4 is then enabled only for a small probe shape, followed by mixed rows=1/16/128 reference tests. Only after local source and reference tests pass will one compressed Kaggle notebook be rebuilt and submitted when quota permits.

## Acceptance decision

Retain a candidate only if it reduces or preserves memory, preserves or improves quality, and preserves or improves speed against v271 under the identical T4 conditions. If the unified refactor adds overhead or changes any correctness result, restore the isolated v271 worktree state and keep the current safe baseline.
