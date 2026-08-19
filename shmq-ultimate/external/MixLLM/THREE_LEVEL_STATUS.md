# MixLLM three-level fork status

Branch: `mixllm-3level`

## Implemented contract

- Output-channel allocation across `{4, 8, 16}` using per-upgrade marginal loss.
- Versioned JSON allocation files with complete-partition validation.
- Packed `ThreeLevelLinear` state dict with INT4 asymmetric, INT8 symmetric and
  FP16 weights plus original output-channel indices.
- A correctness reference path and a composed CUDA ABI. The composed path runs
  INT4/INT8 through upstream MixLLM, FP16 through PyTorch GEMM, then scatters all
  results to original output positions.
- Lazy package import, so quantization metadata can be inspected without loading
  the compiled CUDA extension.
- Runtime capability gates. Upstream MixLLM is selected only on SM80+. T4 uses
  the reference backend by default; a separate SM75 correctness backend is now
  compiled and exercised on real T4 hardware, but is not marked production-ready.
- A config/partition contract pinned to vLLM 0.9.0 commit
  `5fbbfe9a4c13094ad72ed3d6b4ef208a7ddc0fd7`.
- A self-contained Kaggle reference notebook under
  `../mixllm_3level_kaggle/`.

## Kaggle reference gates

Kaggle kernel `freedomform/mixllm-3-level-reference-gate`, version 3,
completed successfully. The worker ran all 20 embedded contract tests and
reported `OK`. Its artifact is retained at
`../mixllm_3level_kaggle/output-v3/mixllm_3level_gate.json`.

The assigned GPU was Tesla P100, capability SM60. The report therefore has
`backend: reference`, `production_gpu_gate: false` and
`sm75_or_sm80_gate: false`. This is a successful reference/serialization
gate, not a T4 or SM75 kernel validation and not a model quality or throughput
benchmark.

The same private Kaggle kernel was then submitted with
`machine_shape: NvidiaTeslaT4`. The run completed on a Tesla T4 (SM75) with
PyTorch 2.10.0+cu128 and all 20 embedded tests passed. The downloaded evidence
is retained under `../mixllm_3level_kaggle/output-v4-retry/`:

- `mixllm_3level_gate.json` records `gpu_name: Tesla T4`, capability `[7, 5]`
  and `backend: reference` (schema v1 wording from that executed revision).
- `mixllm-3-level-reference-gate.log` records the environment, every test and
  the final `MIXLLM THREE-LEVEL REFERENCE GATE: PASS` marker.

This satisfies the T4 hardware/reference gate only. It proves that fallback,
allocation, packing and serialization contracts execute correctly in a T4
CUDA environment. It does not validate an SM75 native kernel. The notebook
builder now emits report schema v2, which separates `t4_hardware_gate`,
`reference_contract_gate` and `native_kernel_gate` to prevent that ambiguity.

The same kernel was updated to version 6 after the SM75 source was made
compatible with the installed PyTorch C++ API (`at::Tensor` and ATen headers).
The version-6 run completed on Tesla T4 (SM75): the CUDA extension compiled for
`compute_75/sm_75`, the two SM75 correctness tests passed, all reference and
contract tests passed, and the final marker was
`MIXLLM THREE-LEVEL REFERENCE GATE: PASS`. Evidence is retained under
`../mixllm_3level_kaggle/output-v6/`, including `mixllm_3level_gate.json`.
The measured report is intentionally explicit: `sm75_correctness_kernel_gate`
is `passed`, while `sm75_production_backend_ready` remains `false`.

Kernel version 7 was submitted on 2026-08-15 and completed on a Tesla T4.
The downloaded evidence is retained under `../mixllm_3level_kaggle/output-v7/`.
The run used Python 3.12.13, PyTorch 2.10.0+cu128 and capability SM75. NVCC
compiled `three_level_sm75.cu` with
`-gencode=arch=compute_75,code=sm_75`; all 24 embedded tests passed, including
mixed/empty precision partitions, random rows and widths, deterministic output,
global allocation, serialization and the pinned vLLM contract. Report schema 3
records:

- `reference_contract_gate: passed`
- `fake_quant_gate: passed`
- `sm75_correctness_kernel_gate: passed`
- `sm75_production_backend_ready: false`
- `native_three_level_gate: not_implemented`
- `vllm_plugin_gate: contract_only`
- `model_quality_gate: not_run`
- `throughput_gate: not_run`

The report's small-shape `throughput_scaffold` measures a dense FP16 PyTorch
matrix multiplication only. It is an environment/timing sanity check and must
not be reported as SM75 or MixLLM operator performance.

Kernel version 9 completed successfully on the same Tesla T4 after adding the
production-shaped runtime contracts and a real operator microbenchmark. Evidence
is retained under `../mixllm_3level_kaggle/output-v9/`. All 29 embedded tests
passed in 35.55 seconds, including CUDA Graph capture, non-default stream
dependency, mixed and empty partitions, random/odd row counts, deterministic
output, serialization, global allocation and tensor-parallel index remapping.

The SM75 operator matched its dequantized dense reference with maximum absolute
error between `3.44e-5` and `8.40e-5` for the measured shapes. It did not pass
the performance gate: median latency was `0.505-0.803 ms`, or `8.88x-12.20x`
the dense FP32 PyTorch reference for `M={1,8,32,128}`, `N=96`, `K=512`.
Consequently the report correctly records:

- `sm75_correctness_kernel_gate: passed`
- `sm75_graph_capture_gate: passed`
- `sm75_production_backend_ready: false`
- `selected_production_backend: reference`
- `throughput_gate: operator_microbenchmark_measured`

This closes the standalone T4 correctness gate, not the T4 production
performance gate. The current scalar SIMT SM75 implementation is a validated
reference/correctness backend and must not be presented as an accelerated
MixLLM backend.

Kernel version 10 completed successfully after replacing the one-thread-per-dot
implementation with a single-launch three-precision kernel. Each output dot
product is distributed across a 128-thread CUDA block and reduced in FP32. The
downloaded report is retained at
`../mixllm_3level_kaggle/output-v10/mixllm_3level_gate.json`; the complete Kaggle
log was also downloaded locally during validation.

All 29 tests passed on Tesla T4, including CUDA Graph capture and non-default
stream ordering. Maximum absolute error improved to `7.63e-6-2.29e-5`. Median
latency improved from version 9's `0.505-0.803 ms` to `0.225-0.336 ms`, but it is
still `4.26x-5.13x` slower than the measured dense FP32 PyTorch reference for
`M={1,8,32,128}`, `N=96`, `K=512`. The optimization is therefore retained as a
faster correctness backend, while `sm75_production_backend_ready` remains
`false` and runtime auto-selection remains on `reference` for T4.

Kernel version 31 completed on Tesla T4 after restoring the best measured
single-GEMM source and the full Qwen-shaped benchmark matrix. All 30 embedded
tests passed in 35.986 seconds, including CUDA Graph capture, non-default stream
ordering, native activation quantization, allocation/serialization and the
pinned vLLM contract. The downloaded report and log are retained under
`../mixllm_3level_kaggle/output-v31/`.

The repeated production-shaped measurements make the T4 decision unambiguous:

- Mixed 4/8/16 at six average weight bits, Qwen QKV shape `M=1, N=3584,
  K=3584`: `1.002x` GEMM-only speedup and `0.963x` end-to-end speedup versus
  dense FP16. The GEMM-only 5% gate passes; the complete operator gate fails.
- Pure INT4 at the same decode shape: `1.541x` GEMM-only and `1.467x`
  end-to-end. Pure INT8: `1.333x` and `1.294x`, respectively.
- Mixed prefill is slower than dense FP16: `0.167x` at `M=16` and `0.084x` at
  `M=128` end-to-end.
- Correctness still passes. Mixed-path maximum absolute error versus its
  dequantized reference is `0.0643` at `M=1`, `0.0951` at `M=16`, and `0.1025`
  at `M=128`.

The `4x` target is not a valid acceptance gate for a six-average-bit 4/8/16
mixture. Even an ideal weight-bandwidth-only model is bounded by `16/6 = 2.67x`;
activation traffic, scales, launches, synchronization and compute lower the
real ceiling. `4x` is the ideal byte-ratio ceiling only for pure INT4, not an
achievable promise for this mixed operator. The measured T4 backend therefore
remains a correctness/research backend and is not selected for production.

Kernel version 33 completed successfully after the final correctness hardening.
Evidence is retained under `../mixllm_3level_kaggle/output-1786887928/`.
All 49 embedded tests passed in 32.252 seconds on Tesla T4, including the native
CUDA extension, CUDA Graph and stream contracts, allocator/model-gate tests and
the FP16 decode arithmetic regression. The decode FP16 path now converts half
operands to FP32 before multiplication, avoiding premature FP16 product rounding.

The repeated performance result still rejects the backend for production:

- Mixed Qwen QKV decode: `1.066x` GEMM-only and `0.951x` end-to-end.
- Pure INT4 decode: `1.424x` GEMM-only and `1.278x` end-to-end.
- Pure INT8 decode: `1.320x` GEMM-only and `1.229x` end-to-end.
- Mixed prefill: `0.259x` at `M=16` and `0.146x` at `M=128` end-to-end.

The pinned vLLM patch now passes `git apply --check` against commit
`5fbbfe9a4c13094ad72ed3d6b4ef208a7ddc0fd7`. Its loader ordering, loaded-tensor
device placement and row-parallel group alignment checks were corrected. This
is an apply/contract gate only: fused-on-disk QKV/MLP checkpoints and an actual
vLLM inference smoke test remain unvalidated, and the adapter remains gated to
the reference backend.

## Explicitly not claimed

- A single physical SM75 GEMM kernel for the three precision paths exists and
  passes correctness, but full activation-quantization-plus-GEMM fusion and a
  production-performance result do not.
- The upstream CUTLASS implementation is SM80-specific and is not T4 compatible.
- There is no validated native vLLM plugin yet; only the pinned integration
  contract and manifest exist.
- The small Qwen2.5-0.5B equal-bit calibration gate improved loss by `0.01648`
  at 5.96 average bits. This is not a Qwen2.5-7B/Qwen3-8B perplexity or
  throughput claim.
- The local environment has no PyTorch, Transformers, NVCC or SM80+ GPU. The
  nested vLLM worktree is not populated, so neither an SM80 binary gate nor a
  pinned vLLM loader/apply smoke test can be executed from this checkout.

## Required gates

1. Keep SM75 runtime auto-selection on the reference backend. A production SM75
   attempt now requires a materially different implementation (for example,
   tuned CUTLASS/Tensor Core kernels plus fused activation handling), followed
   by the same Qwen-shaped end-to-end matrix. Incremental changes to the current
   kernel have not closed the measured gap.
2. Register the SM75 backend in runtime auto-selection only after it passes that
   performance gate; keep the current default as reference until then.
3. Register a native three-level op with the same packed ABI on SM80+ and compare
   it with the current composed path.
4. Implement the pinned vLLM 0.9.0 plugin only after native-op validation.
5. Benchmark FP16, upstream MixLLM, composed three-level, SM75 and fused
   three-level paths with identical shapes and synchronization.
6. Run model-level Qwen2.5-7B and parameterized Qwen3-8B quality/throughput
   gates; no such claim is made by the current contract notebook.

## Current completion boundary

Stages 1-3, the standalone three-path ABI, the optional single-physical-kernel
SM75 experiment, and the correctness portion of stage 5 have executable
evidence. CUDA Graph and stream-ordering contracts pass on real T4 hardware.
The current SM75 kernel also has repeated Qwen-shaped decode/prefill evidence,
and that evidence rejects it for production. The remaining production parts of
stages 4-7 are not complete: SM80+ validation has not run, the pinned vLLM
integration remains contract-only, and full-model Qwen2.5-7B/Qwen3-8B quality
and throughput gates have not run.
 
## v32 shared activation candidate 
The candidate stages each INT8 activation group once in the SM75 decode block for reuse across output-channel subwarps. Local source and notebook checks pass. The authenticated Kaggle T4 execution is still RUNNING; no speedup or quality claim is made until its terminal log is downloaded.
