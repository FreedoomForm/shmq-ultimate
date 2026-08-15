# SHMQ-Ultimate engineering audit

Audit date: 2026-08-15

## Verdict

The repository is an **experimental research prototype**, not a production
framework and not a system that can honestly be guaranteed to work “100%”. The
CPU/reference architecture is useful, but the requested final state — faithful
SHMQ input-channel partitioning, a native `{4,8,16}` tensor-core kernel, and
production vLLM inference on T4 — is not implemented or validated end to end.

No benchmark number, quality claim, or vLLM claim should be accepted until the
strict gates below pass on pinned hardware and software.

## Critical findings

### 1. SHMQ axis and the fused kernel axis differ

SHMQ ranks and permutes input channels (`Cin`, columns of a Linear weight). The
experimental CuPy kernel and MixLLM-style adapter partition output rows (`Cout`).
These layouts are not interchangeable. Reusing `Cin` cluster counts as `Cout`
counts silently changes the algorithm.

Mitigation now present:

- the output-row adapter rejects `cluster_axis="input"`;
- `SHMQKPartitionLinear` implements the faithful K/input-axis reference path;
- metadata marks that reference path as not vLLM/production-CUDA compatible.

Required production fix: implement a K-partitioned CUDA kernel (or redesign and
evaluate an explicitly output-channel algorithm as a different method).

### 2. The current CUDA kernel is not the requested tensor-core kernel

`inference/shmq_3level_kernel.py` uses a CuPy `RawKernel` with scalar CUDA-core
accumulation. It does not contain `mma.sync` and is not a native Turing INT4 /
INT8 / FP16 tensor-core implementation. One launch alone does not make it fast.

The generated Kaggle gate intentionally verifies only correctness of this
implemented scalar kernel and fails if PyTorch fallback is used.

### 3. “Modified MixLLM with three native levels” is not established

The adapter combines an FP16 path with INT8/INT4 paths and includes reference
packing logic. This is not evidence of a single native MixLLM tensor-core kernel
for all three levels. The repository must not describe the path as production
MixLLM unless the exact compiled extension, ABI, supported SM, shapes, and vLLM
version pass integration tests.

### 4. vLLM support is a prototype contract, not validated compatibility

A patch/config schema exists, but no completed test proves that a pinned vLLM
build can load a saved artifact, execute every replaced Linear layer with the
native kernel, generate tokens correctly, handle batching/KV cache/CUDA graphs,
and survive repeated serving. Fallback execution must be forbidden in this gate.

### 5. Model identity in the requested benchmark is ambiguous

The requested “Qwen 3 7B” must be replaced with an exact Hugging Face repository
ID and revision. Qwen model families commonly expose nearby sizes (for example
8B), and silently substituting Qwen2.5-7B or Qwen3-8B invalidates comparison.
Tokenizer revision, trust-remote-code policy, calibration corpus snapshot, and
evaluation harness revision also need pinning.

### 6. A single 16 GiB T4 is a severe quantization constraint

Full-model FP16 weights alone can consume most or all available VRAM for a
7B/8B model. Hessian capture, optimizer state, activations, temporary contiguous
copies, and packed weights add memory. A reliable T4 pipeline therefore needs
layer-wise streaming, CPU/NVMe offload, bounded calibration buffers, explicit
peak-memory assertions, and resume checkpoints. Runtime estimates from A100 are
not transferable to T4.

### 7. Combining methods is not automatically additive

SmoothQuant, AutoRound, GPTQ/OBS, SQC, ILP allocation, and permutation each
change the optimization problem or tensor layout. “More methods” does not imply
better quality. Their ordering requires ablation experiments. In particular,
post-permutation Hessians/codes/scales must remain in the same coordinate system,
and later conversion must consume those codes rather than requantizing weights.

## Corrections and hardening applied

- Preserved exact ISA cluster sizes through three-level permutation.
- Added a faithful K/input-axis packed reference module and converter.
- Preserved explicit input gathers for non-fusable projections such as
  `o_proj`/`down_proj`.
- Preserved Step-8 segment codes/scales during reference conversion.
- Added tests for axis rejection, INT4 representation conversion, three-level
  reference math, layout propagation, state-dict round trips, and conversion.
- Added packing, metadata-schema, finite-output, and contiguity checks.
- Added `require_cuda_kernel=True` so correctness gates fail instead of silently
  using PyTorch.
- Added a self-contained T4 Kaggle GPU-gate notebook generator.
- Rewrote README claims to distinguish references, prototypes, and validated
  behavior.
- Confirmed no Kaggle token is stored in the workspace.

## Verification performed in this environment

- `python -m compileall` over source, tests, and scripts: **PASS**.
- Kaggle strict GPU-gate notebook generation: **PASS**.
- Secret scan for the disclosed Kaggle token pattern: **0 matches**.
- Python unit tests: **NOT RUN** because the local Python environment has no
  `torch` or `pytest` installed.
- CUDA/T4, full-model quantization, quality, speed, and vLLM tests: **NOT RUN**.

The disclosed Kaggle API token was not used. Because it appeared in chat, it
must be revoked and replaced before any Kaggle automation.

## Required release gates

1. **CPU contract gate** — all tests pass in a locked environment.
2. **CUDA correctness gate** — T4/sm_75, CuPy/NVRTC kernel required, randomized
   shapes and edge partitions, reference comparison, sanitizer/race checks.
3. **No-fallback gate** — telemetry proves every target Linear uses the intended
   native backend; any fallback is a hard failure.
4. **Artifact round-trip gate** — save, reload in a clean process, identical
   metadata/layout, bounded numerical error.
5. **Model quality gate** — exact model/revision, perplexity and task metrics for
   FP16, original SHMQ reproduction, original MixLLM baseline, and this method.
6. **Performance gate** — prefill and decode separately, batch/sequence sweep,
   warmup, CUDA events, peak memory, p50/p95 latency and tokens/s.
7. **vLLM gate** — pinned vLLM commit, load/generate tests, batching, KV cache,
   CUDA graphs, tensor parallel behavior (or explicit rejection), and soak test.
8. **T4 quantization gate** — one-GPU layer-streaming run from start to saved
   artifact with checkpoint/resume and no OOM.
9. **Reproducibility gate** — lockfiles/container digest, source revisions,
   dataset hashes, seeds, generated report and raw benchmark JSON.

## Recommended implementation direction

Do not keep extending the output-row adapter and call it SHMQ. Use the current
K-partition module as the executable specification, then implement one
K-partitioned T4 backend behind the same contract. First optimize a correct
FP16-activation weight-only kernel. Add A8 only as a separately tested format.
Integrate with a pinned vLLM version after the standalone kernel and artifact
round trip pass. Benchmark before deciding whether a custom kernel, CUTLASS,
Marlin-derived code, or a maintained vLLM quantization backend is the best base.

Until all release gates pass, the honest status is **prototype / not production
ready**.