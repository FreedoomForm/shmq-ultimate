# v296 next-seam research: expand the safe expanded-path tuner with original-style M/N families

## Differential finding

Microsoft MixLLM’s row-major configuration table contains `128x128x64x64`, `256x64x64x64`, and `128x256x64x64` families in addition to the narrower `32x128`, `64x128`, and `128x64` families. SHMQ’s safe expanded SM75 path currently exposes only four runners: `32x128x64`, `32x64x64`, `128x64x64`, and `64x64x64`, all using the legal SM75 `8x8x16` INT8 instruction, K64, synchronous two-stage pipeline, and the same row-major scatter ABI.

The safe candidate is to add `128x128x64` and `256x64x64` runner aliases to the existing expanded INT8/expanded-INT4 dispatcher and let the existing per-device, per-shape event autotuner select them. This changes no arithmetic, tensor layout, quantization, partition, output mapping, or quality gate. It extends the search space only with core shapes already present in the original family and keeps K64, stages=2, and the existing legal SM75 specialization. The tuning ABI/cache version must be bumped so stale four-choice results cannot be reused.

`128x256` is deliberately excluded from this first safe expansion because it creates a 32-warp block with the current `blockDim.y = WarpCount::kCount`; while present upstream, it deserves a separate occupancy/shared-memory proof on T4. The two selected additions are the smallest original-compatible families that can reduce grid-M or improve N reuse without introducing the rejected packed adapter or stage-depth changes.

## Acceptance conditions

The candidate is admissible only if it compiles on SM75, passes all existing operator probes and mixed 4/8/16 correctness at rows 1/8/16/32/128, preserves timing-integrity, and the tuner’s selected result does not regress the measured gates. Any compile, correctness, memory, or performance failure requires rollback of the added aliases and ABI bump while retaining the research evidence.

## References

[1]: https://raw.githubusercontent.com/microsoft/MixLLM/main/mixllm/kernels/mix_mma_config.h
[2]: https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/threadblock/default_mma_core_sm75.h
[3]: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html
