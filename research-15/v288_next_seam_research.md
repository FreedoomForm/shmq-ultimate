# v288 next-seam primary-source research

## Differential finding

Microsoft MixLLM’s original row-major launcher searches stage counts `{5, 11}` and, for shapes outside its search range, uses a row-major `gemm_rm<5,64,64,32,32>` fallback. Its `DefaultMmaCore` still keeps the threadblock K tile at 64; the stage count is supplied separately to `MQMmaMultistage` [1].

The earlier SHMQ v222 attempt incorrectly transplanted the stage count into `DefaultMmaCore<..., NumStages=5>`. NVIDIA’s vendored SM75 header has no such specialization, so that attempt failed at compile time. The existing SHMQ `Runner<Core, Stages>` already separates the legal SM75 `Core` geometry from the custom synchronous pipeline’s `Stages` template. Therefore `Runner<Core,5>` is a distinct, source-supported experiment that does not request an unsupported SM75 core specialization.

NVIDIA CUTLASS explains that additional stages are useful only when they provide software-pipelined overlap; Turing’s synchronous SM75 path cannot use Ampere `cp.async`, so the expected benefit is uncertain and shared-memory/register occupancy is the explicit risk [2]. This makes a guarded, evidence-only experiment appropriate rather than a production assumption.

## Safe experiment

Add `Int8RunnerStage5 = Runner<Core,5>` using the unchanged `Core` (`32x128x64`, `32x32x64`, `8x8x16`, SM75 legal MMA). Select it only for expanded INT4/INT8 paths with rows >= 128 and only through the existing shape tuner; keep all other candidates and the v299 expanded correctness control unchanged. The stage-5 shared storage is allocated by the existing runner, and its dynamic shared-memory attribute path remains active.

No packed pair or synthetic adapter is enabled. The experiment preserves separate INT4/INT8 streams, exact metadata and arithmetic, output scatter, model, quality gates, benchmark settings, and fallback behavior. Retain only if compilation, correctness, timing integrity, memory, quality/vLLM, and prefill performance all pass.

## References

[1]: https://raw.githubusercontent.com/microsoft/MixLLM/main/mixllm/kernels/mix_mma_multistage.cuh
[2]: https://docs.nvidia.com/cutlass/4.3.5/media/docs/cpp/efficient_gemm.html
