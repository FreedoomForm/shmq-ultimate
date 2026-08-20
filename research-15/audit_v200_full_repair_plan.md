# v200 full repair audit — original MixLLM versus SHMQ SM75

## Scope and freeze point

The repair base is the accepted v200 production implementation at commit `ffb63e1`, with the later v218 import-safe local test-harness repairs preserved separately. The current v224 worktree contains only a pending epilogue-hoist experiment in `sm75_cutlass_testbed.h`; it is not accepted and must not become the repair base. No intermediate Kaggle measurements will be run during the repair phase.

The upstream reference is the first-party MixLLM source under `/home/ubuntu/mixllm-upstream-research/mixllm/kernels`. The key control-flow source is `mix_mma_multistage.cuh`; its configuration family is `mix_mma_config.h`; its indexed column-major epilogue is `mma_multistage_testbed.h`.

## Audit matrix

| Audit item | Upstream reference | v200 state | Required local repair/proof |
|---|---|---|---|
| Legal tile family | `gemm_configs` / `gemm_configs_rm` enumerate many shape families and stage/config combinations. | One legal SM75 family: `32x128x64 / 32x32x64 / 8x8x16 / stage=2`. | Introduce only compile-supported SM75 families. Each family needs an independent operator/reference test, resource contract, and deterministic selection seam. Unsupported SM80 stage-5/11 or K=128 aliases are forbidden. |
| Autotuning | Upstream keys shape/partition, benchmarks candidates with CUDA events, saves `(stage, config_id)`, and reuses the result. | No production autotuner; manual heuristic selects the v188/v200 path. | Add a bounded first-run/offline tuner that can test only legal SM75 candidates, caches by exact shape/device/ABI key, never tunes during CUDA graph capture, and falls back deterministically to v200. |
| Stream lifecycle | Upstream separates integer branches and synchronizes through stream/event ordering. | v200 two auxiliary integer streams plus caller FP16 stream are preserved and passed correctness/timing. | Preserve event topology. Remove only redundant bookkeeping after a dependency proof; never remove `record_stream` or waits that protect tensors. |
| INT4 representation | Upstream SM80 consumes its prepared interleaved INT4 layout in the native mixed-input CUTLASS path. | SM75 expands packed INT4 to zero-subtracted signed INT8 for prefill; packed iterators and direct WMMA experiments were rejected for performance or legality. | Research a vectorized, pipelined, exact SM75 representation or a one-time persistent preparation path. It must not introduce scalar per-element decode in the hot kernel, change quantization, or increase peak memory without proof. |
| Metadata/dequantization | Upstream iterator family and layout are designed together with tile families. | v200 uses the measured v188 mixed path; cached-v2 and K=128 metadata experiments were rejected. SM75 pipeline is K=64/2-stage. | Establish an explicit metadata layout contract per candidate. Prove scale/zero advancement independently against CPU reference and graph-capture initialization rules. |
| Output mapping/epilogue | Upstream also uses indexed column-major output, but hoists index fragments per MMA column. | v200 performs repeated `indices[partition_channel]` lookup inside accumulator loops. v224 hoist is pending and not yet accepted. | Keep exact float output ABI and index semantics. Hoist/reuse index fragments or specialize epilogues only with source and reference tests. |
| Wrapper lifecycle | Upstream launcher has shape/config cache and a narrow execution path. | Python performs device/dtype/partition checks, contiguous decisions, stream recording, expansion-cache checks, and ABI dispatch. | Separate initialization-time invariants from per-call checks. Keep public safety checks and stream dependencies; remove only proven redundant work. |
| Quality and production validation | Original path is designed for model execution. | T4 gate reports native operator correctness, but full Qwen quality and patched-vLLM production are unavailable in the Kaggle environment. | Add/retain local independent model/reference checks and make missing environment explicit. Do not claim full-model quality until the environment actually runs it. |
| Benchmark integrity | Upstream tunes with CUDA events. | v200 gate has timing-integrity checks; mixed rows=128 prefill remains around `0.26–0.31x` speedup versus dense FP16, far from `2.6x`. | Keep identical benchmark settings and honest dense FP16 baseline. Every final gate must pass; no quality or workload reduction is permitted. |

## Existing evidence that constrains the repair

The K=128 route failed twice: v218/v219 produced rows=128 errors around `340–357` even after the double-consumption guard, so it is not a repair candidate without a new end-to-end design and proof. The upstream five-stage idea cannot be copied directly because the vendored SM75 `DefaultMmaCore` provides TensorOp specializations only for `NumStages=2`; v222 failed at compile time. The shared-memory carveout-only change was measured in v223 and failed timing-integrity, so it is rejected. The indexed scatter itself is not an upstream discrepancy; the safe difference is the repeated lookup pattern, which is the narrow v224 seam.

## Repair order

1. Restore and freeze v200 production code; retain only local test-import repairs and research records.
2. Build a legal-SM75 candidate registry and proof harness without enabling new candidates yet.
3. Implement bounded local autotuning over candidates that compile under the vendored SM75 headers; persist only exact device/shape/ABI keys and fall back to v200.
4. Repair or replace INT4 preparation only after an independent CPU byte/zero-point proof and allocation/lifetime proof.
5. Establish metadata and epilogue contracts, then remove only redundant wrapper work with stream-safety tests.
6. Run the complete local suite, CUDA-source contracts, compile/provenance checks, and any available GPU correctness checks. Do not submit intermediate notebooks.
7. Build one final notebook from a committed, fully validated tree and run Kaggle exactly once. If the final run reports an error, repair locally and use the authorized final rerun only after the repair is complete; otherwise report the gates and remaining environment limitations.
