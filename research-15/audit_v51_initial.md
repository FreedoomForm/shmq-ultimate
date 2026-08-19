# SHMQ-Ultimate v51 Audit, Initial Findings

## Scope and baseline

This audit targets the accepted v51 snapshot at `research-15/versions/v51_decode_expanded_cache_contract/`, not the later experimental active source. The source-of-truth comparison is the bundled upstream MixLLM implementation under `shmq-ultimate/external/MixLLM/mixllm/` plus the v51 snapshot and the unchanged Kaggle gate builder.

The current repository HEAD is `621fd2b` and the active source contains later experimental changes. The v51 host snapshot uses a rows==1 packed INT4 decode path, a rows>1 signed-INT8-expanded INT4 prefill path, cached INT4 expansion only for prefill, and a direct WMMA prefill kernel. The Kaggle gate benchmarks the Qwen-shaped mixed case at rows `(1, 16, 128)` and passes each decode/prefill end-to-end shape only when its p50 ratio versus dense FP16 is `<= 1.05`.

## Confirmed source-level findings before fixes

| ID | Area | Finding | Evidence | Risk |
|---|---|---|---|---|
| A1 | Active source drift | The active host backend removed v51's `rows == 1` early return from `_expanded_int4_for_prefill`. Decode now expands and caches the full signed INT4 matrix even though v51 decode consumes packed INT4 plus zero points. | Compare v51 `sm75_backend.before.py:141-144` with active `sm75_backend.py:172-177`. | Extra memory, extra host-side work, and decode regression. |
| A2 | Active source drift | The active CUDA decode kernel consumes `expanded_int4` instead of the packed INT4/zero-point ABI used by v51 and still accepts/loads an unused zero-point tensor. | v51 `three_level_sm75.before.cu:426-529` versus active decode region around lines 834-938. | More memory bandwidth and possible contract drift without a quality benefit. |
| A3 | Active source drift | The active dispatcher has an early mixed-INT branch that makes the later CUTLASS branch unreachable for all mixed INT prefill calls. | Active `three_level_sm75.cu:1125-1194`; the first `else if (n4 > 0 || n8 > 0)` captures the same cases as the later branch. | CUTLASS port is effectively dead for the intended mixed path; performance experiments may measure a different kernel than expected. |
| A4 | Active source drift | `_prefill_metadata()` constructs transposed metadata caches but is never called by the active Python dispatch. | Active `sm75_backend.py:206-224`, no call sites in the file. | Dead allocation/code path and evidence that the host/kernel metadata contract is unfinished. |
| A5 | Benchmark enforcement | The Kaggle builder embeds `test_sm75_backend.py` and `test_sm75_source.py` in the source manifest but executes only `test_three_level.py`. | `build_mixllm_3level_kaggle.py:161-167`. | SM75-specific regression contracts are shipped but not enforced in the official gate. |
| A6 | Benchmark interpretation | The gate threshold is parity with dense FP16 (`<=1.05` p50 ratio), not the separate long-term 2.6x target. | `build_mixllm_3level_kaggle.py:196-205`. | A passing gate does not prove the 2.6x goal; both claims must remain separate. |

## Upstream comparison observations

The upstream two-level MixLLM path uses a packed INT4 representation for its GEMM/decode ABI, a fused native activation quantizer, and padded scale storage for its CUDA layout. The v51 SM75 adaptation intentionally changes the activation-scale shape to exact `[groups, rows]` and changes INT4 prefill to a signed expanded cache because SM75 has no native mixed signed-INT8 by unsigned-INT4 instruction exposed through the chosen WMMA path. Those changes are not automatically defects, but they require explicit shape, stream, cache-lifetime, and quality tests.

The v51 module's `from_weight()` and `dequantize_weight()` provide an independent reference for the three precision partitions. The native path returns float32 accumulation and converts back to the input dtype only at the module boundary, which is appropriate for comparison but must be retained when assessing quality.

## Ranked hypotheses for the performance gap

1. **Prefill kernel tile under-utilization:** v51 assigns a 64-channel, 16-row WMMA tile shape through four warps. The gate includes rows 16 and 128, so the rows 16 case may waste tile capacity and pay global/CTA synchronization overhead relative to cuBLAS FP16.
2. **Per-call activation quantization launch:** every mixed prefill call launches a separate quantization kernel before the GEMM. This is a real end-to-end cost even when the GEMM itself is competitive.
3. **Signed INT4 expansion and cache lifetime:** prefill requires a full `[n4, K]` int8 expansion. The v51 cache avoids rebuilding it after warmup, but its memory footprint may reduce occupancy or interact with allocator synchronization.
4. **Metadata/output scatter overhead:** the native kernel scatters partition outputs through index tensors. This is required for channel order and cannot be removed without changing the contract, but it may dominate small-M cases.
5. **GEMM layout/occupancy mismatch:** the direct WMMA kernel is a correctness-first 16x16 tile implementation, while dense FP16 uses highly tuned cuBLAS kernels. Porting a CUTLASS path without fixing its metadata and dispatch contract may not improve the measured path.

## Required next steps

Build a red-capable local test loop first. Fix the deterministic missing imports and add explicit v51 regression assertions for the packed decode/no-expansion contract and the dead-dispatch condition. Then restore or isolate v51 from later experimental source before making performance changes. No CUDA change should be retained unless correctness, memory behavior, and all four Kaggle gates remain passing under the unchanged Qwen/Qwen2.5-0.5B T4 notebook.

Every performance iteration must compare the original MixLLM dataflow, the v51 snapshot, and the candidate implementation, and must be logged in `research-15/worklog.md`.

## Confirmed v51 measurement and accounting gaps

The recorded v51 Kaggle result is reproducible evidence that the mixed Qwen-shaped prefill path is the sole gate blocker. For the mixed `(2400 INT4, 896 INT8, 288 FP16)` Qwen-shaped case, the reported end-to-end ratios versus dense FP16 were approximately `0.827x` at rows 1, `3.514x` at rows 16, and `5.967x` at rows 128. The best accepted result came from the pure INT4 decode case at `1.221x` end-to-end speedup and `1.416x` GEMM-only speedup. Native correctness passed, but full-model Qwen quality was explicitly `not_run`.

The v51 benchmark helper reports only weight-bandwidth upper bounds from average bits. It does not include the signed INT4 prefill expansion cache, metadata tensors, indices, allocator overhead, activation buffers, or output buffers in a memory report. Its dense reference dequantizes the full weight outside the timed loop, which is appropriate for comparing operator latency but means memory results cannot be inferred from the reported timing artifact.

Historical iterations confirm that several post-v51 kernel variants were measured and rejected because they made prefill worse, including wide WMMA, barrier-free per-warp staging, and one-thread-per-output DP4A. Those variants must not be reintroduced merely because they look theoretically simpler. The next safe improvement should begin from v51 and target accounting, dispatch locality, launch overhead, or a properly integrated upstream-style staged layout, one variable at a time.

## Current audit decision

Treat v51 as the clean baseline for the next repair. Do not retain the later active experimental source except in the reversible audit backup. First add measurement and contract coverage that exposes memory and full-model quality status without changing the benchmark's workload. Only then modify the native implementation.
