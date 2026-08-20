# v230 native SM75 pair design

## Objective

Replace the upstream SM80 mixed-input operator with a legal SM75 implementation without changing the three-level ABI. The operation for an INT4 output row is

`sum_k A8[m,k] * (W4[n,k] - Z4[n,g]) * scale_A[g,m] * scale_W[g,n]`.

For each signed activation byte, use `A8 = A_low + 16*A_high`, where `A_low = A8 & 0xf` is unsigned and `A_high = A8 >> 4` is signed. Then compute two legal SM75 MMA products over the same staged K tile:

`P_low = MMA_u4_u4(A_low, W4)`

`P_high = MMA_s4_u4(A_high, W4)`

and reconstruct `P = P_low + 16*P_high - Z4*sum(A8)` before applying activation and weight scales. The exact identity is proven by `scripts/verify_v230_native_int4_reference.py`.

## Required implementation shape

The pair runner must stage one shared B tile and two shared A fragments per K tile, or stage a packed A tile that the warp operator expands into low/high fragments without a second global load. The warp-level operator must expose the same `FragmentC`, `MmaIterations`, `transform`, and call interface expected by `MQMmaPipelinedSm75`, while maintaining two internal `arch::Mma<GemmShape<8,8,32>, ...>` accumulators. The epilogue must apply the existing fine-grained metadata and indexed float scatter exactly once.

## Why this is not yet production code

A two-run wrapper around existing `Mma` objects would reload global/shared tiles and reproduce v204's poor performance. A scalar packed iterator would reproduce v201/v202. A type alias to the upstream `OpMultiplyAddMixedAndShuffledInputUpcast` would require the unsupported SM80 `16x8x32` internal operator. The fused pair needs a new C++ implementation and a real CUDA compile/correctness test. The sandbox has no `nvcc` or GPU, so implementing it here and sending it to Kaggle without compile proof would violate the no-intermediate-run and keep-only-safe-changes rules.

There is an additional integration seam that must be solved explicitly: the existing `MQMmaPipelinedSm75` calls `MmaTensorOpDequantizer::apply_zero` before the warp MMA for every non-`int8 x int8` operator. That is correct for the signed expanded path, but it is not correct for raw unsigned INT4 B fragments: subtracting a zero-point in a `uint4` fragment can underflow and the corrected value may not fit signed 4-bit range. The pair implementation must therefore bypass pre-MMA B mutation and apply the exact `-zero * sum(A8)` correction after the raw pair products, with per-output-row activation sums accumulated across all K tiles. This requires a small pipeline/epilogue extension, not only a new warp operator alias.

## Acceptance conditions

The design is not accepted until it compiles for `sm_75`, passes randomized CPU/reference identity, passes CUDA native correctness for mixed/pure INT4, passes timing-integrity, and improves mixed prefill without degrading decode or memory gates. If the first compile-capable environment is Kaggle, that environment can be used only after all static/reference/provenance checks for the complete implementation are finished.
