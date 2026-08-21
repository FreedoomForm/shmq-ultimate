# v270 deep research: remove repeated activation divisions

## Upstream comparison

The original MixLLM quantizer assigns one warp to each 128-value activation group, loads the group as two vectorized `float2` values per lane, computes one reciprocal scale per warp, and multiplies the two `half2` fragments by that reciprocal before converting to signed int8 with inline PTX. The quantizer is launched on the current CUDA stream before the GEMM, so SHMQ's stream ordering is consistent with upstream; the upstream implementation does not fuse activation quantization with the staged GEMM.

SHMQ already matches the upstream warp/group decomposition and writes the same `[groups, rows]` scale layout, but its conversion loop performs `values[item] / scale` four times per lane. The current v263 path therefore pays four floating-point divisions for every four output bytes even though the scale is identical for the entire warp. This is a clear arithmetic-preserving optimization: compute one `inverse_scale = 1.0f / scale` per lane/warp and replace the four divisions with four multiplications. The max reduction, float-derived scale, clamp, round-to-nearest conversion, output packing, launch grid, and all kernel ABI contracts remain unchanged.

## Candidate and safety boundary

v270 will change only the activation quantizer's conversion arithmetic. It will not use the upstream half-precision reciprocal because that would alter the quantization boundary behavior more than necessary; it retains SHMQ's float scale and `__float2int_rn` conversion, replacing only repeated division by multiplication with a single reciprocal. Native correctness and model-quality gates must still pass on Kaggle; if the change fails those gates or does not improve E2E, it will be rejected.

References: original MixLLM `mixllm/kernels/kernels.cu`, quantize kernel lines 378-509; SHMQ `three_level_sm75.cu`, quantize kernel lines 247-290.
