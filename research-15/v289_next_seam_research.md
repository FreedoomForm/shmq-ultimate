# v289 next-seam primary-source research

## Differential finding

Microsoft MixLLM’s original `LinearMixLLM` prepares `weight_int4` by applying a two-step interleave permutation before nibble packing, then passes that tensor directly to the mixed GEMM launcher [1]. Its launcher’s packed path consumes `matrix_B_interleaved` and separately supplies transposed scale/zero metadata [2]. NVIDIA CUTLASS documents that the Turing INT4 tensor-core operation is `m8n8k32` and that shared-memory layout/iterator contracts, not arithmetic alone, determine correctness [3].

SHMQ already implements the same two-step interleave permutation and caches its result as `_sm75_int4_interleaved`, but the generic `prepare_sm75_packed_tensors()` tuple currently returns raw `weight_int4` in slot 0. The live v2 path then always constructs the signed expanded INT4 copy. The existing C++ v3 ABI already has the correct slots for raw `weight_int4`, original-interleaved `weight_int4_interleaved`, expanded fallback, and cached transposed metadata; `begin_integer_prefill_overlap()` already routes non-fused INT4 to `run_cutlass_packed_int4_partition()` when the interleaved slot is nonempty.

## Safe experiment

Repair only the Python-side v3 handoff: prepare the original-equivalent interleaved INT4 tensor and transposed metadata, resolve the already registered `_three_level_linear_v3_unchecked` operator, and call it with the existing exact ABI. Keep v2 expanded dispatch as fallback if the v3 symbol or caches are unavailable. Do not alter the direct pair kernel, the CUTLASS core, quantization, model, benchmark settings, or output arithmetic.

The first gate must prove the packed path’s numerical correctness at small and large M, because a raw/interleaved mismatch can compile and still silently corrupt results. Only if correctness and timing-integrity pass may performance be considered. This is the first experiment that follows the original MixLLM packed-data contract rather than adapting the rejected synthetic pair arithmetic.

## References

[1]: https://raw.githubusercontent.com/microsoft/MixLLM/main/mixllm/nn/modules/linear.py
[2]: https://raw.githubusercontent.com/microsoft/MixLLM/main/mixllm/kernels/mix_mma_multistage.cuh
[3]: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html
