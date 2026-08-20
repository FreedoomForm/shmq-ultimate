# v201 deep research: direct packed INT4 consumption

## Scope
v200 proved that overlapping the existing INT4 and INT8 CUTLASS launches is safe and improves rows=128 from 0.2671x to 0.3072x, but it leaves an 8.6 MB expanded INT4 buffer and remains far below the prefill gate. v201 investigates whether the SM75 path can consume packed INT4 directly without changing quantization arithmetic, output quality, model, or benchmark conditions.

## Primary sources inspected

- Upstream MixLLM source in this repository: `external/MixLLM/mixllm/kernels/mix_mma_multistage.cuh` and `research-15/original_linear.py`.
- Current SM75 kernel: `external/MixLLM/mixllm/kernels/three_level_sm75.cu`.
- Current module storage: `external/MixLLM/mixllm/nn/modules/three_level_linear.py`.
- NVIDIA CUTLASS functionality documentation: https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/functionality.html
- NVIDIA CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- NVIDIA Turing architecture blog URL returned by search: https://developer.nvidia.com/blog/turing-architecture-in-depth/ (currently returns Not Found, so no claim is based on it).

## Findings

The upstream MixLLM source stores INT4 in a CUTLASS-specific interleaved layout at construction time. Its `interleave_uint4_for_cutlass` applies a 32-element permutation followed by an 8-element permutation and then packs nibbles. This is not the same as the current three-level module's sequential `_pack_uint4`; it cannot be copied into the active SM75 path without changing the serialized checkpoint contract or adding a validated conversion step.

The upstream SM80 launcher uses `OpMultiplyAddMixedAndShuffledInputUpcast` for INT4, while the current SM75 implementation uses signed INT8 WMMA for both INT4-after-expansion and INT8. The current custom SM75 CUTLASS helper is also an INT8 runner. Therefore, the mere presence of a packed INT4 tensor does not prove that SM75 can use it in the current CUTLASS ABI.

The official CUTLASS page describes independent TensorOp/WMMA device-level GEMM instances but does not establish that INT4 and INT8 operands can be mixed inside one instruction. The CUDA programming-guide page is a legacy redirect-sized documentation source and its automated extraction did not expose the sub-byte fragment details. The only safe claims for this iteration come from the checked-in upstream and vendor source, not from an unverified web snippet.

The current direct WMMA kernel already receives `weight_int4` and `zero_int4` in its ABI but intentionally does not read them for prefill: `expanded_int4` is used because SM75 lacks the upstream SM80 mixed INT4 instruction in this port. A correct expansion-free experiment must either (a) decode two nibbles per byte while filling the existing signed-INT8 B tile, or (b) use a verified SM75 sub-byte WMMA path and a compatible layout. Option (a) preserves the current instruction and numerical contract but may add integer unpack/zero-subtraction work to every B-tile load. Option (b) has high compile/correctness risk and should not be attempted without a concrete supported `wmma::experimental` or CUTLASS SM75 type.

## Safe v201 design choice

Use the existing v200 stream adapter and add a separate direct-packed INT4 path only for the rows>=32 large-M branch. The path should decode packed nibbles and subtract the per-group zero while staging each INT4 B tile into shared memory, then reuse the existing signed INT8 WMMA accumulation and scale application. Keep the current expanded path available as a fallback for unsupported shapes or if the direct path fails correctness. Add a source contract and a focused correctness seam before Kaggle. This changes one variable—INT4 materialization strategy—and leaves INT8, FP16, decode, benchmark settings, and model unchanged.

## Falsifiable prediction

If expansion is a dominant prefill cost, direct packed staging should reduce `expanded_int4_bytes` to zero and improve mixed rows>=32 E2E. If unpacking in every tile costs more than the avoided global expansion, correctness will pass but E2E will regress; in that case the direct path must be rejected while v200 stream overlap remains preserved.

## References

[1]: https://github.com/microsoft/MixLLM "Microsoft MixLLM source repository"
[2]: https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/functionality.html "NVIDIA CUTLASS Functionality documentation"
[3]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/ "NVIDIA CUDA C++ Programming Guide"
