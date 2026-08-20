# v257 deep research: resource-safe CUTLASS family

## v256 failure

The v256 Kaggle T4 run reached the benchmark but failed with `CUDA error: too many resources requested for launch` when the M=128,N=128 candidate was selected. That family requires `128/32 * 128/32 = 16` warps, or 512 threads per block, before accounting for the two-stage SM75 pipeline and accumulator/register footprint. It is rejected and must be removed from production selection.

## Official configuration comparison

The official Microsoft MixLLM `gemm_configs` table includes a `64x64` threadblock family with warp tiles `(16,64)`, `(32,32)`, `(32,64)`, `(64,16)`, `(64,32)`, and `(64,64)`. The existing SHMQ SM75 tuner has only `M=32,N=128`, `M=32,N=64`, and `M=128,N=64`. A `M=64,N=64,K=64` candidate with `WarpShape=32x32x64` uses four warps and the already validated SM75 stage-2 instruction `<8,8,16>`. It therefore retains the current legal operand and arithmetic path while avoiding the 512-thread resource failure.

This is a narrower, low-risk geometry experiment. It does not claim to reproduce the official SM80 stage-5/11 pipeline or its direct INT4 iterator. The direct packed INT4 path remains a separate research seam and is not mixed into v257.

## Safe v257 change

Remove the M=128,N=128 candidate and its cache value. Add only `M=64,N=64` as a fourth candidate, bumping the tuning ABI to 257. Keep N=128, N=64, and M=128,N=64. Preserve the v255 mixed dispatch selector `false` because v255 showed a large performance improvement relative to v254, even though timing integrity exposed a separate issue. The Kaggle gate must decide whether the lower-resource tile improves performance and whether timing integrity remains valid.

## Sources

1. Official Microsoft MixLLM repository: `https://github.com/microsoft/MixLLM`.
2. `mixllm/kernels/mix_mma_config.h`, official `64x64` configuration family.
3. Vendored SHMQ `default_mma_core_sm75.h`, legal stage-2 row-major-A/column-major-B core.
4. Kaggle v256 kernel version 251, which rejected `M=128,N=128` with launch resource exhaustion.
