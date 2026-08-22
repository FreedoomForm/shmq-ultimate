# v298 next-seam research: original 64x128 K64 family within the validated resource envelope

## Differential finding

The original Microsoft row-major table includes `64x128x64x64`, while SHMQ’s safe expanded tuner has only `32x128`, `32x64`, `128x64`, and `64x64` K64 families. The v297 experiment showed that `128x128` compiles but cannot launch on T4 because its 16-warp block exceeds launch resources. The existing SHMQ `128x64` runner already proves that an eight-warp K64 block is launchable under the current synchronous two-stage pipeline.

The next candidate is therefore only `CoreM64N128` with the same `WarpShape=32x32x64`, legal `8x8x16` SM75 INT8 instruction, K64, stages=2, plain row/column layouts, runner, output scatter, and tuner cache. The tuning ABI must be bumped to 298 so stale four-family cache entries cannot suppress the new candidate. It has the same eight-warp count as the existing `CoreM128N64` but exchanges M and N, matching an actual original configuration tuple. This is a bounded source-backed shape addition, not a generic copied configuration table.

The candidate must first compile and launch on T4, then pass full smoke/Qwen operator correctness and timing-integrity. The unchanged v299 expanded route remains the fallback for every unsupported shape and the packed v3 route remains disabled.

## References

[1]: https://raw.githubusercontent.com/microsoft/MixLLM/main/mixllm/kernels/mix_mma_config.h
[2]: https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/threadblock/default_mma_core_sm75.h
[3]: https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/transform/pitch_linear_thread_map.h
