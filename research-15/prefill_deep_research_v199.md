# v199 deep research: supported wide large-M SM75 core

## Finding from v196-v198

The v196 timing-integrity run proved the v188 v2 ABI is selected but still leaves mixed rows=128 at about `0.28x` of dense FP16. v197 could not compile because the vendored SM75 `DefaultMmaCore` specialization supports only stage 2. v198 compiled only after keeping the stage-2 core, but using `MQMmaPipelinedSm75` with three stages corrupted large-M outputs, so it is rejected.

## Original MixLLM comparison

The original MixLLM autotuner enumerates multiple large-M configurations, including `64x128` threadblocks with `32x32` warp tiles and `8x8x16` tensor instructions. The current helper only instantiates `32x128x64` and launches one runner for every large-M partition. The vendored SM75 header contains the required row-major/column-major int8 TensorOp `DefaultMmaCore` specialization for stage 2; the rejected v193 geometry was unsafe because it used a different instruction shape and unsupported five-stage core, not because a 64x128 stage-2 core is intrinsically invalid.

## Controlled v199 hypothesis

Keep the current v188-compatible `32x128x64` stage-2 runner for `rows < 64`. Add a second supported `64x128x64 / 32x32x64 / 8x8x16 / stage=2` runner and select it only for `rows >= 64`. This changes only threadblock M geometry and launch occupancy for the measured rows=128 case. It preserves the same synchronous SM75 pipeline, arithmetic, quantization, index/scatter semantics, v2/v3 ABI, telemetry, timing-integrity guard, quality checks, and exact Kaggle benchmark.

The candidate is accepted only if it compiles on T4 and passes native correctness, timing integrity, memory, decode, prefill, and all production contracts. Any corruption or regression rejects it.
