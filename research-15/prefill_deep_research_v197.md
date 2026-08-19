# v197 deep research: SM75 staged pipeline depth

## Question

Why does the restored v188/v2 path still achieve only about `0.28x` of dense FP16 for mixed prefill even after v196 proves that the v2 adapter is selected and timing is internally consistent?

## Original MixLLM comparison

The original `mix_mma_multistage.cuh` does not use a single fixed launch geometry. Its autotuned configuration table dispatches staged GEMM with `stage=5` or `stage=11` and searches many threadblock/warp shapes, including `32x128x64` and `64x128x64`. The current SM75 helper preserves only the `32x128x64` threadblock and `32x32x64` warp shape, but hard-codes both the `DefaultMmaCore` stage parameter and `MQMmaPipelinedSm75` stage parameter to `2`.

The current SM75 pipeline is synchronous, as required on Turing because `cp.async` is unavailable. Its main loop still double-buffers A/B and scale/zero metadata, so it is structurally valid, but two stages provide less global-memory latency hiding than the original staged design. A three-stage experiment is materially different from the rejected v193 geometry: it retains the known-compilable `32x128x64 / 32x32x64 / 8x8x16` SM75 core and changes only the pipeline depth. The v193 failure was caused by an unsupported `64x128x64` `DefaultMmaCore` specialization, not by stage count.

## Evidence and constraints

v196 measured mixed rows=128 at `0.280702x` end-to-end speedup versus dense FP16, with `timing_integrity=true`, while v188's historical replay reached approximately `0.30x` under the same gate notebook. Therefore the next optimization must target actual GEMM throughput and latency hiding, not restore an already-selected ABI. The benchmark must remain unchanged: same T4, Qwen-shaped `3584x3584`, same partitions, ten warmups, fifty timed iterations, CUDA events, exact FP16 baseline, and all quality gates.

The three-stage core must compile on the vendored SM75 CUTLASS specialization, use no `cp.async`, remain within T4 shared memory, and preserve the existing v2/v3 ABIs. If compilation, correctness, timing integrity, or any existing gate regresses, the candidate is rejected. The v197 experiment is therefore a single controlled change: parameterize the runner by `Stages` and instantiate the known v188 geometry with `Stages=3`.

## References

[1]: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.html "NVIDIA CUTLASS GEMM Performance Measurement Methodology Guidelines"
[2]: ../shmq-ultimate/external/MixLLM/mixllm/kernels/mix_mma_multistage.cuh "Original MixLLM staged launcher"
[3]: ../shmq-ultimate/external/MixLLM/mixllm/kernels/sm75_cutlass_testbed.h "Current SM75 CUTLASS helper"
