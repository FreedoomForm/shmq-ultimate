# v284 next-seam primary-source research

## Question

After v284 exposed the guarded fused pair kernel and measured a large-M regression, what source-grounded change can reduce its overhead without changing arithmetic, layout, partitioning, or benchmark conditions?

## Differential finding

| Source | Observed contract | SHMQ mismatch | Safe implication |
|---|---|---|---|
| NVIDIA CUTLASS GEMM API | The warp-level hierarchy iterates `warp_k`, then `mma_n`, then `mma_m`, issuing one MMA per `(m,n)` tile. The API describes warp-level operators as loading shared-memory fragments and traversing the warp tile with reusable fragments. [1] | `sm75_int4_pair_gemm_kernel` currently loops `row_tile` (M) outside `n_tile` (N). It loads the same B fragment once for every row tile even though B is unchanged across the four row subtiles. | Hoist each warp’s two B fragments out of the row-tile loop and reuse them for all four A row tiles. This removes repeated `wmma::load_matrix_sync` and fragment conversion without changing any operand values or output ownership. |
| Microsoft MixLLM | The original launcher sends a single packed interleaved B tensor to the INT4 CUTLASS testbed and relies on the CUTLASS warp/threadblock hierarchy for tile reuse. [2] | The direct pair kernel stages B once but re-reads it repeatedly from shared memory for each row tile. | Preserve the original packed tensor and existing persistent stream; optimize only the inner reuse order. |
| NVIDIA Turing documentation | Turing INT4 uses a legal `m8n8k32` operation; ldmatrix is warp-cooperative and shared-memory tiles are the intended source for warp MMA. [3] | The pair kernel already uses the legal `m8n8k32` WMMA load and pair instructions, but pays unnecessary repeated shared-memory loads. | B-fragment hoisting is a local, self-consistent optimization; it does not revive the rejected synthetic k32 adapter. |
| v284 T4 evidence | All explicit pair probes and native correctness passed, but Qwen mixed rows=128 E2E was only `0.166x` versus dense FP16 and pure INT4 rows=128 was `0.163x`; mixed prefill failed. | The kernel is correct but dominated by overhead. | Test the B-reuse optimization behind the same narrow aligned guard. If it fails performance or any gate, revert the whole production exposure immediately. |

## Chosen edit set for v285

1. In the direct pair kernel, load and convert the two B fragments once per warp and K chunk, before the row-tile loop.
2. Reuse those cached fragments across all four A row tiles; retain the existing low/high native `m8n8k32` arithmetic, zero correction, scale application, output mapping, and synchronization.
3. Re-enable the same guarded live dispatch only for complete aligned large-M shapes (`rows % 32 == 0`, `width % 128 == 0`, and at least 128 INT4 channels), with the expanded fallback for every other shape.
4. Add source/unit contracts proving B-fragment hoisting and guard symmetry.

This is the maximum compatible local change supported by the primary sources and v284 evidence. It is not a performance claim until the unchanged T4 gate passes, including full-model quality and vLLM production availability.

## References

[1]: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html "NVIDIA CUTLASS GEMM API"
[2]: https://raw.githubusercontent.com/microsoft/MixLLM/main/mixllm/kernels/mix_mma_multistage.cuh "Microsoft MixLLM mix_mma_multistage.cuh"
[3]: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html "NVIDIA CUTLASS Turing Tensor Core implicit GEMM"
