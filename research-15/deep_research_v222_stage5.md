# Deep research for v222: upstream-inspired stage-count experiment

## Original-versus-SMQ discrepancy

The upstream MixLLM launcher uses a fixed K=64 CUTLASS geometry but does not hard-code the smallest pipeline depth. Its autotuner evaluates stages `{5, 11}` and its production column-major fallback is `gemm<5, 64, 128, 64, 32>`. The upstream `LinearMixLLM::gemm` constructs `DefaultMmaCore` with `NumStages` and uses the same K=64 threadblock family for INT4 and INT8.

The clean v200 SM75 implementation uses `DefaultMmaCore<..., GemmShape<32,128,64>, GemmShape<32,32,64>, ..., OpMultiplyAddSaturate>` with `NumStages=2`, and `Int8Runner = Runner<Core,2>`. The SM75 pipeline explicitly uses synchronous shared-memory copies because Turing lacks Ampere `cp.async`, but its `MQMmaPipelinedSm75` template accepts a general `Stages` parameter and its shared storage is stage-parameterized. The production path therefore has a concrete, testable discrepancy: it uses the minimum two-stage staging depth while the original uses a five-stage fallback for the equivalent K=64 / 128-channel output family.

## Hypothesis and safety boundary

A five-stage SM75 runner may hide global-memory latency better for the large-M prefill partitions without changing arithmetic, quantization, partition mapping, benchmark settings, or output ABI. It must be a separate compile-time runner with a matching `DefaultMmaCore<..., 5>` and `Runner<Core5,5>`; mixing a 2-stage core policy with a 5-stage runner would violate the pipeline's shared-layout contract and is forbidden. The existing `Runner::smem_size()` and dynamic shared-memory attribute path already support stage-dependent storage.

The experiment will select the stage-5 runner only for the large-M CUTLASS prefill path, preserve v200's two auxiliary integer streams and caller-stream FP16 overlap, and keep the v200 two-stage runner as the fallback. Local source contracts will verify the matching Core5/Runner5 pair and the unchanged v200 event topology. The candidate is rejected unless all required correctness and timing-integrity gates pass and it improves the measured prefill result relative to the clean v200 baseline.
