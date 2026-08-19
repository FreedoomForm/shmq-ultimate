# SHMQ-Ultimate v192 Deep Research: Original MixLLM vs Current SM75 Prefill

## Scope and evidence

Kaggle version 192 is the corrected v191.1 bounded-discovery candidate. It completed on a Tesla T4, so the previous recursive model scan startup failure is resolved. The candidate remains `no_go`: native SM75 correctness, allocator, embedded contracts, and both decode gates passed, while mixed prefill failed at rows 16 and 128. The exact Qwen2.5-0.5B full-model gates were unavailable because the exact configuration fingerprint was not found at the two deterministic model paths; no substitute model was accepted.

| Scenario | Rows | SHMQ p50 (ms) | Dense FP16 p50 (ms) | Speedup vs dense | Ratio vs dense | Gate result |
|---|---:|---:|---:|---:|---:|---|
| Mixed 4/8/16 | 1 | 0.104992 | 0.110592 | 1.053337x | 0.949363 | decode pass |
| Mixed 4/8/16 | 16 | 0.401408 | 0.122816 | 0.305963x | 3.268369 | prefill fail |
| Mixed 4/8/16 | 128 | 0.837664 | 0.197392 | 0.235646x | 4.243657 | prefill fail |
| Pure INT4 | 128 | 0.239056 | 0.167824 | 0.702028x | 1.424445 | diagnostic only |
| Pure INT8 | 128 | 0.241072 | 0.167904 | 0.696490x | 1.435773 | diagnostic only |

## What the original MixLLM actually does

The original `mix_mma_multistage.cuh` uses a deep, staged CUTLASS-style module. Its public launcher is small, but the implementation selects separate row-major and column-major kernels, chooses threadblock/warp/instruction geometry through template parameters, and persists autotuned configurations keyed by `(M,N,K,precision partition)`. For the large column-major mixed path, its fallback is `gemm<5, 64, 128, 64, 32>`, which means five synchronous pipeline stages, a 64x128x64 threadblock tile, a 64x32x64 warp tile, and an INT8 instruction shape of 16x8x32. The original core also has distinct INT4 and INT8 MmaCore types and a custom mixed-input INT4 operator rather than expanding INT4 into an ordinary INT8 path.

The original launcher records an event on the caller stream, waits on it from dedicated INT4 and INT8 streams, runs the two partition GEMMs independently, records per-partition completion events, and makes the caller stream wait on both. This is a real concurrency and launch-decomposition difference, but the historical ledger already rejects blindly reintroducing parallel precision streams: v153/v154 were measured and remained far below the prefill gate. Therefore stream overlap is a secondary hypothesis, not the first candidate.

## What the current SM75 path does

The current `three_level_sm75.cu` dispatch has three materially different paths. Rows equal to one use the validated eight-warp DP4A decode kernel. Rows below 32 use the direct four-warp WMMA kernel for every precision partition. Rows at least 32 run INT4 and INT8 partitions through `run_cutlass_int_partition`, then launch a separate direct four-warp FP16 partition kernel. The CUTLASS helper is not the original large-M geometry: it uses a fixed `GemmShape<32,128,64>` threadblock, `GemmShape<32,32,64>` warp, `GemmShape<8,8,16>` instruction shape, and only two synchronous stages. It also uses a custom scatter epilogue that writes float output by indexed channel.

The Python host path correctly caches expanded INT4 weights and transposed CUTLASS scale/zero metadata for rows at least 32. It still launches the INT4 and INT8 helpers serially on the current stream, and the FP16 channels remain a separate direct WMMA launch. The v192 telemetry proves the scale/metadata cache is not the main issue: at rows=128 metadata is already cached and the pure INT4/INT8 paths still take about 0.239/0.241 ms versus dense FP16 0.168 ms. The current gap is therefore primarily kernel geometry, pipeline efficiency, and partition launch/epilogue cost rather than repeated Python materialization alone.

## Primary discrepancy

The strongest source-backed discrepancy is not an untested micro-optimization. It is that the current SM75 CUTLASS adapter uses a small two-stage 8x8x16-oriented core while the original large-M fallback uses a five-stage 16x8x32-oriented core with a 64x128 threadblock tile and 64x32 warp tile. NVIDIA's CUTLASS documentation describes threadblock and warp tile shapes as the mechanism for concurrency and data reuse, and explicitly states that software pipelining double-buffers shared-memory tiles and warp fragments to overlap memory movement with Tensor Core computation. NVIDIA's functionality table lists SM75 TensorOp support for signed INT8 16x8x32 and 8x8x16 instruction shapes, and lists corresponding 64x32x32 and 32x32x16 warp shapes. Thus the original geometry is architecturally valid on SM75, while the current fixed core is a conservative compatibility choice rather than an equivalence.

The current SM75 pipeline explicitly replaces `cp.async` with synchronous copies and CTA barriers, which is correct for SM75. That does not invalidate increasing the number of synchronous stages: the implementation's prologue, circular shared-memory indices, and `__syncthreads()`-based `gmem_wait()` are parameterized by `Base::kStages`. A stage-count/tile-shape candidate is therefore a plausible isolated experiment, but it must be compiled and checked for shared-memory and register pressure on T4.

## Ranked falsifiable hypotheses

1. **Current CUTLASS geometry is under-tiled and under-pipelined.** If this is the main cause, replacing only the fixed SM75 Core with an SM75-valid 64x128x64 / 64x32x64 / 16x8x32 / five-stage core should reduce pure INT4 and INT8 rows=128 latency while preserving the ABI and arithmetic. If it does not improve either pure path, revert immediately.

2. **Current mixed prefill is dominated by serial precision partition launches and indexed scatter epilogues.** If this is the main cause, a dedicated experiment that changes only stream scheduling or epilogue fusion should reduce mixed latency without materially changing pure-path latency. Historical v153/v154 make an unconditional stream reintroduction unsafe; it should not be the first implementation.

3. **The FP16 partition's direct WMMA kernel is disproportionately costly at large M.** If this is true, replacing only the FP16 partition with a dense FP16 GEMM adapter or a separately tuned FP16 CUTLASS kernel should reduce mixed latency while leaving INT4/INT8 and output scatter semantics intact. This must not alter the dense baseline and must preserve indexed output channels.

4. **Metadata/cache overhead still matters at rows=16.** The current rows=16 path does not use CUTLASS metadata, and v192 reports no metadata bytes. Any cache-only change cannot explain rows=16. This hypothesis is rejected for the next iteration.

## Safe next candidate

Implement one isolated Core-geometry experiment in `sm75_cutlass_testbed.h`, preserving the public runner interface, custom dequantization, indexed output scatter, all host dispatch, all precision arithmetic, and all benchmark settings. The first candidate should align the SM75 helper with the original large-M fallback: threadblock 64x128x64, warp 64x32x64, instruction 16x8x32, and five synchronous stages. Add a source contract asserting the intended SM75 geometry and a compile-time/runtime guard that reports the shared-memory footprint. Do not reintroduce streams, change the FP16 path, change quality thresholds, or change model discovery in this iteration.

Before the code change, add a test at the source seam that fails while the old Core remains. After the minimal change, run the local contracts and notebook freshness checks. Only a Kaggle T4 result passing all required gates can be retained; otherwise restore the previous source and keep the research record.

## References

[1]: https://docs.nvidia.com/cutlass/4.3.5/media/docs/cpp/efficient_gemm.html "NVIDIA CUTLASS Efficient GEMM in CUDA"
[2]: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/functionality.html "NVIDIA CUTLASS Functionality and SM75 TensorOp support"
[3]: https://developer.nvidia.com/blog/nvidia-turing-architecture-in-depth/ "NVIDIA Turing Architecture In-Depth"
[4]: https://github.com/NVIDIA/cutlass "NVIDIA CUTLASS source repository"
