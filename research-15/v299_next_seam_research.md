# v299 next-seam research: original 32x256 K64 family

## Differential finding

Microsoft MixLLM’s row-major configuration table contains `32x256x32x64`. SHMQ’s safe expanded tuner still has no N256 family. The rejected v297 `128x128` runner showed that 16-warp blocks exceed T4 launch resources, while v298 `64x128` was legal and correct but did not improve mixed prefill. A `32x256x64` threadblock with the current `32x32x64` warp and `8x8x16` instruction geometry has eight warps, matching the proven resource envelope of the existing N128 and M128N64 runners.

This candidate is materially different from simply copying arbitrary original configurations: it retains SHMQ’s existing plain row-major/column-major layouts, synchronous SM75 K64 pipeline, two stages, output scatter, and expanded INT4/INT8 arithmetic. It only adds N256 to the existing shape tuner and bumps the tuner ABI to invalidate the old four-family cache. The source-backed rationale is higher B-tile/channel reuse for wide partitions such as Qwen’s 2400 INT4 and 896 INT8 channels.

The candidate is admissible only if nvcc compiles it, T4 launches it, every smoke/Qwen partition passes correctness, timing-integrity remains true, and the mixed prefill speed gate improves or at least does not regress relative to the safe control. The packed v3 path remains disabled.

## References

[1]: https://raw.githubusercontent.com/microsoft/MixLLM/main/mixllm/kernels/mix_mma_config.h
[2]: https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/threadblock/default_mma_core_sm75.h
[3]: https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/transform/pitch_linear_thread_map.h
