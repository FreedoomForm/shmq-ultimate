# v226 research: legal SM75 tile-family and autotuning design

## Primary sources

1. [NVIDIA CUTLASS Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
2. [NVIDIA CUTLASS Autotuning with the DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/autotuning_gemm.html)
3. [Microsoft MixLLM repository](https://github.com/microsoft/MixLLM)
4. Local first-party upstream source: `/home/ubuntu/mixllm-upstream-research/mixllm/kernels/mix_mma_config.h` and `mix_mma_multistage.cuh`

## Findings

NVIDIA's CUTLASS documentation states that threadblock tiles should be tuned against the GEMM dimensions and hardware: larger tiles improve global-memory reuse, but partial tiles waste threads when M or N is small. It also describes software pipelining as a latency-hiding mechanism and emphasizes that accumulator/register usage limits occupancy. Therefore an SM75 tuner must be shape-aware and must reject configurations that are not legal for the vendored SM75 specialization or that have invalid resource requirements.

NVIDIA's autotuning guidance defines three required steps: define a valid search space, benchmark each candidate, and cache the winning configuration. It recommends warmups, multiple CUDA-event measurements, synchronization, and a cache key containing the relevant data type, layouts, and shape characteristics. This matches the original MixLLM launcher, which keys cached results by M/N/K and integer partition size, tests candidate configurations with CUDA events, stores the best stage/config pair, and falls back to a known configuration.

The upstream MixLLM repository confirms that the project includes both the kernel source and a vLLM patch, but the SM80 kernel configuration cannot be copied as-is to SM75. The local vendored `default_mma_core_sm75.h` contains TensorOp specializations only for `NumStages=2`; the previously attempted stage-5 alias failed to compile. The safe search space must therefore begin with explicitly instantiated SM75 `DefaultMmaCore` families using `NumStages=2` only.

## v226 design constraint

The first repair slice should create a legal SM75 candidate registry and a cache-key/proof seam, not blindly enable unmeasured tiles. Candidate selection must have a deterministic v200 fallback, be initialized before CUDA graph capture, and never silently select a candidate whose correctness/resource contract is absent. The final T4 run will be the first performance measurement under the user's one-run-only protocol; all other validation must be local/static before that run.
