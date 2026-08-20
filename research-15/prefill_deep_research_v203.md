# v203 deep research: vectorized unpacking and native SM75 INT4 mappings

## Evidence from v202
The v202 CUTLASS-pipelined packed iterator preserved the supported INT8 stage-2 core but performed scalar nibble decode inside `IteratorB::get()`. Kaggle T4 measured rows=128 at `0.0656x`, essentially identical to v201's standalone packed kernel and far worse than v200's expanded INT4 path. Therefore the next experiment must avoid per-element decode in the hot iterator.

## Primary-source findings

NVIDIA's CUDA integer intrinsic documentation defines `__byte_perm` as a device operation selecting four bytes from two 32-bit values and defines `__dp4a` as a four-way signed or unsigned INT8 dot product [1]. The existing decode kernel already demonstrates a four-nibble-at-a-time pattern: it loads two packed bytes, forms four signed INT8 weights, and invokes `__dp4a`. This confirms that v203 should use word/vector granularity, not one branch and one byte load per output element.

The vendored SM75 PTX wrappers expose native homogeneous INT4 Tensor Core instructions with `mma.sync.aligned.m8n8k32` for `s4*s4`, `u4*s4`, `s4*u4`, and `u4*u4` [2]. CUTLASS's SM75 default device configuration confirms a supported `int4b_t` by `uint4b_t` configuration with `8x8x32` instruction shape and two stages [3]. However, the current activation is signed INT8 and the checkpoint INT4 values are zero-point encoded. There is no supported SM75 mixed INT8-by-INT4 instruction in the vendored path, so simply changing the current runner's element types would be mathematically wrong.

A mathematically valid native route is to decompose each signed INT8 activation `a` into two signed 4-bit digits (`a = a_low + 16*a_high`) and compute two native `s4*u4` products against the packed unsigned INT4 code. The zero-point correction is then `-zero * sum(a)` per 128-element quantization group. This requires a custom SM75 4-bit threadblock core and a vectorized activation digit packer; it is substantially more invasive than v202 and must not be attempted as a blind geometry change. The current default CUTLASS SM75 core header has no `DefaultMmaCore` specialization for the 4-bit types, although `DefaultGemmConfiguration` documents the supported device shape. Thus the safe v203 candidate is limited to a separate, opt-in native-int4 runner or a vectorized expansion control experiment; the validated v200 INT8 path remains the fallback.

## Decision

The first v203 implementation will target vectorized INT4 expansion using aligned 32-bit packed loads and byte-level selection, preserving the existing expanded INT8 CUTLASS path and its ABI. This isolates conversion cost and avoids introducing a new unsupported SM75 threadblock core. If the vectorized expansion does not materially improve large-M prefill, the next research cycle will prototype the two-term native `s4*u4` decomposition with explicit correction and correctness gates before any production selection.

## References

[1]: https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html "CUDA integer intrinsic functions"
[2]: https://github.com/FreedoomForm/shmq-ultimate/blob/audit-v184-computer/shmq-ultimate/external/MixLLM/mixllm/kernels/cutlass/include/cutlass/arch/mma_sm75.h "Vendored SM75 MMA definitions"
[3]: https://github.com/FreedoomForm/shmq-ultimate/blob/audit-v184-computer/shmq-ultimate/external/MixLLM/mixllm/kernels/cutlass/include/cutlass/gemm/device/default_gemm_configuration.h "CUTLASS SM75 default INT4 configurations"
