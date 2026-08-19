# SHMQ-Ultimate v194 research: restoring v188 performance without losing v192 safeguards

## New evidence from v193

Kaggle version 193 did not reach a benchmark. NVCC failed at `sm75_cutlass_testbed.h(119)` because the vendored `default_mma_core_sm75.h` only provides `DefaultMmaCore` TensorOp specializations for pipeline stage count `2`. The attempted v193 configuration used stage count `5`, so the type was incomplete. The failure is structural and explains why simply copying the original MixLLM template arguments is invalid on this SM75 port. The original source uses a different CUTLASS configuration/header context; the local SM75 adapter cannot accept that stage count without adding and validating an entire new core specialization.

The relevant local header specializes `DefaultMmaCore<..., arch::OpClassTensorOp, 2, Operator_>` for row-major A and column-major B, which is exactly the layout used by the SM75 helper. Therefore the supported compatibility baseline is stage count `2`; the v188 geometry (`32x128x64` threadblock, `32x32x64` warp, `8x8x16` instruction, stage `2`) is not merely conservative but is the only currently instantiated SM75 core contract in this adapter.

## Measured version matrix

| Version | Main performance-path change | Mixed rows=1 | Mixed rows=16 | Mixed rows=128 | Decision |
|---|---|---:|---:|---:|---|
| v188 | Activate staged SM75 CUTLASS for INT4/INT8 when rows>=32; direct WMMA FP16 partition remains | 1.1963x | 0.2884x | 0.2762x | NO-GO, but fastest rows=1/128 among the compared candidates |
| v189 | Add transposed metadata cache, v3 ABI, explicit model gates | not retained | regressed vs v188 | 0.2197x | NO-GO |
| v190 | Add unchecked v3 wrapper and bypass repeated partition validation | not retained | improved vs v189 | improved vs v189 but below v188 | NO-GO |
| v191 | Exact-Qwen discovery using unbounded rglob | no valid result; stalled | no valid result | no valid result | rejected due startup stall |
| v192 | Bounded deterministic discovery, preserving v190 path | 1.0533x | 0.3060x | 0.2356x | NO-GO; only valid corrected successor to v191 |
| v193 | Attempt original-style 5-stage / 64x128x64 SM75 core | no result | no result | no result | rejected at compile time |

v188 remains the performance reference for the measured staged CUTLASS path, but its gate and production safeguards are incomplete. v192 contains important correctness and observability improvements, yet its cached v3 path does not reproduce v188's mixed rows=128 timing. The safe restoration target is therefore the v188 dispatch and supported stage-2 core, not the unsupported original stage-5 geometry.

## Compatibility matrix for restoring v188 components

| v188 component | Evidence of speed | Compatibility with v192 safeguards | Restoration decision |
|---|---|---|---|
| Stage-2 SM75 CUTLASS core and `rows>=32` INT4/INT8 dispatch | v188 rows=128 mixed 0.2762x | Fully compatible; this is the supported local specialization | Restore/retain |
| Direct WMMA path for rows<32 | v188 rows=16 0.2884x; v192 0.3060x | Correctness-validated and still needed for small M | Retain; do not route rows=16 to unsupported CUTLASS |
| v188 v2 unchecked ABI for large prefill | v188 measured faster than later cached v3 path in the mixed scenario | ABI can be preserved as a performance adapter; Python partition validation remains before dispatch | Test as an isolated restoration |
| v189 metadata cache | Removes repeated host transposes and is required for honest memory telemetry | Safe but its use must be measured, not assumed faster | Keep cache and telemetry; compare v2/v3 execution separately |
| v190 unchecked v3 wrapper | Removes repeated partition sorting/validation | Safe for Python paths that already validate the partition | Keep public validated v3 and private unchecked entry points |
| v191 rglob discovery | Caused a Kaggle startup stall | Not compatible with bounded execution | Never restore |
| v191.1/v192 bounded model discovery | v192 completes reliably | Safe and required | Keep |
| v193 stage-5 geometry | No performance evidence; compilation failure | Incompatible with current vendored SM75 core | Revert completely |

## Five issue conclusions

The quantization and memory accounting issues are already instrumented, not speed-fixed: v192 reports activation quantization around `0.026–0.028 ms`, so it is not the dominant prefill cost, while expanded INT4 and peak allocations are now visible. Full-model Qwen quality remains an environment blocker because the exact fingerprint was not found; changing the model or accepting a substitute would violate the requirement. Runtime and vLLM contracts are source-checked, but the v192 Kaggle report still marks the actual apply path unavailable, so a real runtime pass remains outstanding.

The first safe repair is not a new kernel geometry. It is to restore the measured v188 execution choice at the existing supported seam: retain the v192 Python validation, caches, telemetry, model discovery, and public v3 operator, but add an explicitly named performance adapter that can route the already-validated large-prefill path through the v2 unchecked ABI. This changes no arithmetic, tensor layouts, benchmark settings, model, or quality criteria. It tests the concrete hypothesis that cached-v3 ABI handling or its metadata path is responsible for the mixed rows=128 regression. If it does not improve Kaggle T4 timing without gate regressions, revert the adapter.

## References

[1]: https://docs.nvidia.com/cutlass/4.3.5/media/docs/cpp/efficient_gemm.html "NVIDIA CUTLASS Efficient GEMM in CUDA"
[2]: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/functionality.html "NVIDIA CUTLASS Functionality and SM75 TensorOp support"
[3]: https://developer.nvidia.com/blog/nvidia-turing-architecture-in-depth/ "NVIDIA Turing Architecture In-Depth"


## Follow-up finding after v194

The v194 artifact reports `prefill_metadata_bytes=0` for mixed rows=128, proving that the v188-v2 adapter was selected and that the cached-v3 metadata allocation was not on the timed path. Its rows=128 mixed timing nevertheless remained `0.870432 ms`, close to v192 and far from the historical v188 `0.594176 ms`. Therefore the hypothesis “v3 metadata/ABI alone caused the regression” is falsified.

The exact v188 and v194 Python lifecycle are otherwise materially similar: both validate the partition, reuse the expanded INT4 cache after the first call, quantize activations outside the prequantized GEMM measurement, and time with CUDA events after ten warmups and fifty iterations. The current CUDA source adds v3 support and cached-metadata validation, but v194's v2 path passes undefined cache tensors, so the new metadata transpose is not executed. This leaves two live explanations: a kernel/resource difference caused by the added v3-capable translation unit or host-core checks, and ordinary T4 fixed-power/DVFS variation. NVIDIA's current CUTLASS measurement guidance warns that clocks can oscillate for seconds, small GEMMs have more run-to-run variation, and stable comparisons require separated warmup/profiling loops plus frequency monitoring [1]. The existing benchmark settings must not be changed for the production gate, so a fair next experiment is an exact v188 replay under the same current Kaggle conditions, not another speculative geometry change.

The next candidate is therefore a controlled historical replay: run the exact v188 source/notebook once more to establish whether its `0.276x` rows=128 result reproduces. If it does not, the prior v188 advantage was environmental noise and should not be restored. If it does, compare the compiled v188 and current v194 generated code/resource behavior before changing the kernel. No quality, model, or benchmark setting may be relaxed.

[1]: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.html "NVIDIA CUTLASS GEMM Performance Measurement Methodology Guidelines"
