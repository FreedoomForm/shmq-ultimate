# v300 next-seam research: cache per-kernel CUDA attributes

## Differential finding

Microsoft MixLLM constructs each `Testbed` once. Its constructor calls `set_shared_memory()`, which applies `cudaFuncSetAttribute` for the compiled kernel once, before repeated `run()` calls. SHMQ’s `Runner` has no persistent host object; its static `run()` currently recomputes the shared-memory size and calls `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` and `cudaFuncAttributePreferredSharedMemoryCarveout` on every partition invocation.

The v300 candidate caches those idempotent kernel attributes per runner specialization and CUDA device. The change removes repeated host-side attribute API calls from the hot path without changing the kernel, arithmetic, tensor layouts, stream topology, output mapping, partitions, or benchmark gates. A mutex-protected per-device set is used so the cache remains correct if multiple host threads or devices invoke the same runner. The attribute setup still occurs before the first launch on each device, and no launch is permitted to skip it.

This is intentionally narrower than changing the epilogue or scatter mapping. The original row-major epilogue uses a different output contract and cannot be substituted while preserving SHMQ’s indexed 4/8/16 scatter ABI. v300 therefore tests only the lifecycle/overhead difference that is directly evidenced by the original constructor organization.

## Acceptance conditions

The candidate must pass the full local suite, compile on SM75, preserve native correctness and timing integrity, and show no mixed-prefill E2E regression. Any failure or no measurable benefit requires rollback. Full-model Qwen quality and vLLM gates remain mandatory and are not relaxed.

## References

[1]: https://raw.githubusercontent.com/microsoft/MixLLM/main/mixllm/kernels/mma_multistage_testbed.h
[2]: https://raw.githubusercontent.com/microsoft/MixLLM/main/mixllm/kernels/mix_mma_multistage.cuh
[3]: https://raw.githubusercontent.com/NVIDIA/cuda-samples/master/Common/helper_cuda.h
