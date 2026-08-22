# v297 next-seam research: narrow the expanded tuner after the v296 compile proof

## New evidence

The v296 T4 compile failed before execution in CUTLASS `PitchLinearWarpRakedThreadMap` with `Number of iterations must be non-zero`. The diagnostic identifies `CoreM256N64`: it instantiates a 512-thread block and a B thread map for a `64x64` tile with 16 elements per access, so the current plain-layout SM75 map has more aggregate access capacity than the tile and computes zero iterations. This is a concrete incompatibility, not a runtime-performance result.

The original Microsoft table contains `256x64`, but its original testbed/configuration system is not proof that this shape is legal under SHMQ’s fixed `WarpShape=32x32`, plain row/column layouts, and `blockDim.y=Core::WarpCount::kCount` wrapper. The v296 compile demonstrates that directly copying an original M/N tuple is insufficient.

The `128x128x64` addition remains a distinct candidate: with the same 32x32 warp shape it spans a 128x128 tile and its B tile has enough elements for the 512-thread map, while preserving the legal 8x8x16 instruction, K64, stages=2, and existing scatter ABI. Remove only `256x64`; do not change the arithmetic or gate criteria. The next T4 run must first prove that the remaining `128x128` alias compiles, then measure it against the unchanged v299 control.

## References

[1]: https://raw.githubusercontent.com/microsoft/MixLLM/main/mixllm/kernels/mix_mma_config.h
[2]: https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/transform/pitch_linear_thread_map.h
[3]: https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/threadblock/default_mma_core_sm75.h
