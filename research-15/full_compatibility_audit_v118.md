# Full compatibility audit for v51 Frankenstein candidate

## Scope and acceptance rule

This audit reads every version entry in `research-15/worklog.md`. The target is not a faster isolated GEMM number. A speed method is admissible for integration only if its evidence shows no correctness loss and no regression in any required gate: `sm75_native_correctness`, `mixed_decode_gemm_performance`, `mixed_decode_end_to_end_performance`, and `mixed_prefill_end_to_end_performance`. For comparison against the current v51 baseline, the method must also not lower the validated best GEMM p50 of 1.4156x or best end-to-end p50 of 1.2210x. Contract/documentation/environment fixes are retained only when required for the v51 implementation, not counted as speed improvements.

## Version classification

| Version(s) | Evidence from worklog | Classification | Compatibility decision |
|---|---|---|---|
| v41 | Correctness passed; GEMM 1.6324x, E2E 1.1100x | Integration improvement but E2E regression | Keep only as already-required native dispatch; no new speed code |
| v42 | Correctness failed, max error 0.124847 | Incorrect | Exclude |
| v43 | Illegal address | Incorrect/runtime unsafe | Exclude |
| v44 | Correctness passed; GEMM 1.3434x, E2E 1.1365x | Slower than v51 | Exclude |
| v45 | Correctness passed; GEMM 1.6452x, E2E 1.1971x | GEMM-positive but E2E below v51 | Exclude as a new method; its compatible lineage is already embodied in v51 |
| v46 | Correctness passed; GEMM 1.5966x, E2E 1.1973x | GEMM-positive but E2E below v51 | Exclude as a new method; its compatible lineage is already embodied in v51 |
| v47 | Correctness passed; GEMM 1.4674x, E2E 1.1059x | E2E regression; shared staging overlaps the same decode reuse problem as v48 | Exclude |
| v48 | Correctness passed; GEMM 1.0860x, E2E 1.0530x | Regression and overlaps v47’s activation-reuse objective | Exclude |
| v49 | Correctness passed; GEMM 1.5737x, E2E 1.1589x | Both below v51 E2E and not a safe restriction-only addition | Exclude |
| v50 | Correctness passed; GEMM 1.4180x, E2E 1.1706x | Marginal GEMM increase but E2E regression; packed access overlaps v49/v45 packed-path risk | Exclude |
| v51 | Correctness passed; GEMM 1.4156x, E2E 1.2210x | Best validated end-to-end baseline; expanded INT4 rows=1 cache fixes v43 | Retain as base |
| v52 | Aborted; no valid measurement | No evidence | Exclude |
| v53 | Correctness passed; GEMM 1.4449x, E2E 1.2093x | E2E regression; launch-bounds objective overlaps v74 | Exclude |
| v54 | Correctness passed; GEMM 1.5597x, E2E 1.1625x | E2E regression; stale-read removal is already semantically exhausted | Exclude |
| v55 | Correctness passed; GEMM 1.3683x, E2E 1.1602x | Regression; `__ldg` idea overlaps v90 | Exclude |
| v56-v57 | Decode gates passed but prefill E2E failed | Prefill regression | Exclude |
| v58 | nvcc syntax failure | No valid evidence | Exclude |
| v59-v60 | Native correctness failed | Incorrect ABI/layout | Exclude |
| v61 | nvcc too many arguments | Compile failure | Exclude |
| v62-v64 | Correctness/decode passed but prefill E2E failed | Prefill regressions; v64 barrier change also conflicts with v51 synchronization safety | Exclude |
| v65-v66 | Aborted/compile failure | No valid evidence | Exclude |
| v67 | Audit-only instruction/layout observation | No measured speed improvement | Exclude from code integration |
| v68 | Decode gates passed but prefill E2E failed | Unroll directive did not preserve prefill | Exclude |
| v69 | Audit-only output-layout observation | No measured speed improvement | Exclude |
| v70 | Correctness/decode passed but prefill E2E failed | Reuse dispatch regression; overlaps v88 | Exclude |
| v71-v72 | Audit-only; cp.async path is SM80-specific | Not portable to SM75 | Exclude |
| v73 | Patch did not apply | No valid evidence | Exclude |
| v74-v76 | Correctness/decode passed but prefill E2E failed | Launch bounds, pointer hoist, and direct-global A each fail the prefill gate; objectives overlap v53/v75/v76 | Exclude |
| v77 | Correctness failed | Unsafe barrier removal | Exclude |
| v79-v81 | Audit/research-only or unsafe overlap pipeline proposals | No measured safe change | Exclude |
| v82 | Patch failed before validation | No valid evidence | Exclude |
| v83 | Correctness failed | Vector mapping incorrect | Exclude |
| v83-corrected | Correctness passed but prefill E2E failed badly | Correct but too slow | Exclude |
| v84-v85 | Research-only redesigns | Require full kernel/template rewrites, no evidence | Exclude |
| v87-v88 | Correctness/decode passed but prefill E2E failed | Pointer hoist/reuse dispatch regressions; v88 duplicates v70’s objective | Exclude |
| v89 | Correctness/decode passed but prefill E2E failed | Quantizer max-reduction change is not safe under full workload | Exclude |
| v90 | Correctness/decode passed but prefill E2E failed | `__ldg` overlaps v55 and does not fix bottleneck | Exclude |
| v92 | Correctness/decode passed but prefill E2E failed | Eight-warp prefill tile conflicts with v51’s four-warp resource balance | Exclude |
| v93 | Correctness/decode passed but prefill E2E failed | Unroll overlaps v68 and does not fix prefill | Exclude |
| v95 | Correctness/decode passed but prefill E2E failed | Predicate hoist is too small and not independent of the same runtime branch | Exclude |
| v100 | Compile failure | No valid evidence | Exclude |
| v100.1-v100.8 | Build/diagnostic/loader corrections; v100.8 eventually ran but prefill failed | Correctness/diagnostic work, not speed; packed INT4 prefill is slower | Exclude as speed methods |
| v100.9 | Inline PTX INT8 MMA pending/no accepted four-gate result | No valid evidence | Exclude until independently measured and all gates pass |
| v101 | Cumulative Frankenstein compiled and passed non-prefill gates but prefill E2E failed | Direct evidence that the historical safe-set combination is not a safe speed improvement | Exclude; retain clean v51 |
| v102 | Correctness failed | Invalid m8n32 mapping | Exclude |
| v102.1-v102.2 | Pending correction attempts before final measured result | No independent evidence | Exclude |
| v102.3-v102.4 | Correctness failed | Shared-B/A race or layout mismatch | Exclude |
| v102.5-v102.7 | Correctness passed but prefill E2E failed badly | m8n32 redesign remains incompatible with v51 prefill behavior | Exclude |
| v103-v110 | vLLM contract, native forward, manifest, and source-test fixes | Required integration/static correctness, not speed changes | Retain in project; no CUDA speed method to merge |
| v111 | vLLM absent in Kaggle environment | Environment failure | Do not classify as algorithm evidence |
| v112-v114 | Honest environment classification and generated assertion repair | Required gate honesty/integration fixes, no speed change | Retain in project; no CUDA speed method to merge |
| v115 | Full audit found no admissible post-v51 speed-only addition | Correct no-op conclusion | Retain v51 unchanged |
| v116 | `kDecodeChannelsPerWarp=4`; decode gates passed but prefill E2E failed; rows=1 GEMM 1.3639x and E2E 1.2796x, rows=16/128 severely regressed | Decode mapping is not compatible with complete workload | Exclude; v51 restored |
| v117 | Quantizer block 128→256 threads; decode gates passed but prefill E2E failed; rows=1 GEMM 1.2727x and E2E 1.2003x, rows=16/128 worsened | Quantizer launch-size change is not safe | Exclude; v51 restored |

## Conflict and redundancy analysis

The v45/v46 decode mapping, v47 shared activation staging, v48 shuffle reuse, v49 restriction hints, and v50 packed access all target the same decode-side activation/weight access bottleneck. They are not independent additive speedups. v45/v46 additionally change warp ownership; v47/v48 change activation reuse; v49 changes alias assumptions; and v50 changes packed access. Their measured E2E regressions and different memory/synchronization contracts make a Frankenstein merge unjustified. The v51 expanded INT4 cache is the only rows=1 INT4 representation that survived the illegal-address issue and is therefore the required base.

The prefill candidates v53, v56-v76, v83-corrected, v87-v95, and v102.x target overlapping small-tile, pointer-selection, staging, warp-count, and WMMA-layout costs. The repeated failure of the prefill E2E gate shows that these are not separable additive improvements. `cp.async`/multistage methods from the original MixLLM are SM80-specific and cannot be transplanted into SM75. The m8n32 series also demonstrates that a theoretically larger WMMA tile is not compatible without a full correctness-proven redesign.

## Selection result

Under the stated rule, the set of additional speed methods that are simultaneously non-regressive on all required gates and theoretically composable with v51 is **empty**. The only valid integrated candidate is therefore the **current v51 source plus the already-retained v103-v114 integration/audit fixes**, with no new CUDA optimization code. Adding any rejected method would violate the user’s keep-only-safe-changes rule. A fresh Kaggle run of this exact audited v51 candidate is still appropriate to revalidate the result after the complete audit, but it must be labeled a v51 revalidation, not a claimed improvement.
