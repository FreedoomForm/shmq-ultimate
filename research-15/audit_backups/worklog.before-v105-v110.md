# SHMQ-Ultimate Worklog

All performance evidence below is from the unchanged Kaggle T4 gate using Qwen/Qwen2.5-0.5B and the same dense FP16 baseline. No result is accepted unless correctness passes.

| Version | Differential finding / hypothesis | Correctness | Best GEMM p50 | Best end-to-end p50 | Decision |
|---|---|---:|---:|---:|---|
| v41 | Native vLLM patch bypassed SM75 ABI; replaced dense fallback with native dispatch. | passed | 1.6324x | 1.1100x | Keep integration fix; no-go performance |
| v42 | Original scale_act used M_even; naive padded stride/layout implementation. | failed (max_abs_error 0.124847) | invalid | invalid | Rejected and rolled back |
| v43 | Decode still unpacked packed INT4 nibbles; tried existing expanded INT4 cache for rows=1 decode. | gate did not complete (cudaErrorIllegalAddress) | invalid | invalid | Rejected and rolled back |
| v44 | Four 8-lane subwarps duplicated activation reads; tried one full warp per channel. | passed | 1.3434x | 1.1365x | Rejected; slower/no meaningful improvement |
| v45 | Same mapping audit; tried 16-lane subwarps as intermediate activation reuse. | passed | 1.6452x | 1.1971x | Current best correctness-passing baseline; still below 2.6x |
| v46 | v45 8-warps block may limit occupancy; tried 4 decode warps. | passed | 1.5966x | 1.1973x | Rejected; no meaningful improvement |
| v47 | v45 subwarps still duplicate global activations; tried dynamic shared activation staging. | passed | 1.4674x | 1.1059x | Rejected; regression; v45 restored |

## Current state

Current source is restored to v45: kDecodeWarps=8, kDecodeChannelsPerWarp=2. The best validated result is 1.6452x GEMM-only and 1.1971x end-to-end, with 	erminal_decision: no_go. The mandatory 2.6x target has not been reached. Continue with a new differential audit before any further change.

| v48 | v45 subwarps duplicate activation loads; register-level full-warp shuffle reuse. | passed after address fix | 1.0860x | 1.0530x | Rejected; regression; v45 restored |

| v49 | Original CUTLASS non-aliasing iterators vs raw decode pointers; added __restrict__. | passed | 1.5737x | 1.1589x | Rejected; below v45; v45 restored |
| v50 | Original packed/interleaved access vs two byte loads; used one aligned uint16 packed pair load. | passed | 1.4180x | 1.1706x | Rejected; below v45; v45 restored |

| v51 | v43 illegal-address root cause: rows=1 used empty expanded cache; initialized signed expanded INT4 cache and decoded from it. | passed | 1.4156x | 1.2210x | Current best end-to-end baseline; still no-go |

| v52 | Precision-tiled decode block mapping audit; static rewrite contaminated prefill scope before submission. | not measured | invalid | invalid | Aborted and v51 restored |

| v53 | Original static threadblock/resource configuration vs no decode launch bounds; added __launch_bounds__(256,2). | passed | 1.4449x | 1.2093x | Rejected; below v51; v51 restored |

| v54 | v51 expanded cache left a stale zero_int4 load; removed only that unused read. | passed | 1.5597x | 1.1625x | Rejected; below v51; v51 restored |

| v55 | Original read-only iterator audit vs v51 expanded INT4 ordinary loads; added __ldg only to expanded INT4 load. | passed | 1.3683x | 1.1602x | Rejected; below v51; v51 restored |

| v56 | Original-first audit identified CUTLASS multistage/interleaved-layout/padded-scale mismatches; scoped precision-tiled decode mapping was measured. | native correctness passed; mixed prefill e2e gate failed | decode GEMM/e2e gates passed; prefill e2e failed | no_go | Rejected; restored v51 |

| v57 | Original-first audit: original @torch.compile activation quantization versus our custom native quantizer; replaced only activation dispatch. | passed | mixed decode GEMM passed; decode/prefill e2e failed | no_go | Rejected; restored v51 |

| v58 | Original-first audit: original separate INT4/INT8 MmaCore paths versus v51 runtime precision ternaries; attempted macro-expanded compile-time inner loop | not executed; nvcc syntax failure | no measurement | rejected | Restored v51 |

| v59 | Original-first audit: original padded scale_act M_even; added backend padding, explicit prefill stride, and sliced reference reconstruction | native correctness failed | no admissible performance result | no_go | Rejected; restored v51 |

| v60 | Original-first audit: completed padded scale contract across both prefill kernels with explicit scale_rows | native correctness failed (same max_abs_error pattern) | no admissible performance result; decode GEMM passed | no_go | Rejected; restored v51 |

| v61 | Original-first audit of actual ops.quantize padded scale_act; attempted decode scale_rows ABI correction | not executed; nvcc too many arguments at decode launch | no measurement | rejected | Restored v51 |

| v62 | Original-first audit of actual ops.quantize padded M_even; corrected atomic scale_rows ABI for decode and both prefill kernels | passed | mixed decode GEMM passed; decode/prefill e2e failed | no_go | Rejected; restored v51 |

| v63 | Original-first audit: original direct packed-buffer path vs repeated wrapper contiguous conversion; cached 9 packed tensors by id/version | passed | decode GEMM/e2e passed; prefill e2e failed | no_go | Rejected; restored v51 |

| v64 | Original-first audit: original stage-level pipeline vs first SM75 INT path full CTA barrier; changed one post-MMA barrier to __syncwarp | passed | decode GEMM/e2e passed; prefill e2e failed | no_go | Rejected; restored v51 |

| v65 | Original-first audit: separate original MmaCore_INT4/INT8 vs current runtime branches; attempted pointer hoist was statically contaminated across FP16/INT scopes | aborted before build | no measurement | rejected | Restored v51 |

| v66 | Original-first audit: separate INT4/INT8 MmaCore vs runtime precision branch; scoped pointer hoist in first prefill kernel | not executed; nvcc expected statement at line 223 | no measurement | rejected | Restored v51 |

| v67 | Original-first audit: INT4 K interleave [0,4,1,5,2,6,3,7] and original InstructionShape<16,8,32> vs current WMMA 16x16x16 row-major path | audit-only; no code change | not measured | not_applicable | Kept v51 baseline |

| v68 | Original-first audit: compile-time multistage/K iterations vs first INT prefill loop; added one scoped #pragma unroll | passed | decode GEMM/e2e passed; prefill e2e failed | no_go | Rejected; restored v51 |

| v69 | Original-first audit: original column-oriented output plus transpose vs current direct row-oriented stores | audit-only; no code change | not measured | not_applicable | Kept v51 baseline |

| v70 | Original-first audit: original CTA tile reuse vs current always-simple prefill dispatch; routed rows>1 to existing 8-warp reuse kernel | passed | decode GEMM/e2e passed; prefill e2e failed | no_go | Rejected; restored v51 |

| v71 | Original-first audit: original kStages/cp_async circular pipeline vs current synchronous single-pass prefill loop | audit-only; no code change | not measured | not_applicable | Kept v51 baseline |

| v72 | Original-first architecture guard: original cp.async multistage header requires Sm80; direct pipeline port is invalid for SM75 | audit-only; no code change | not measured | not_applicable | Kept v51 baseline |
| v73 | Precision-plane warp parallelism based on original separate INT4/INT8 streams | Patch did not apply; no measurement | No source change; v51 preserved | Implementation attempt stalled; no correctness/performance evidence |
| v74 | Add launch_bounds to active four-warp prefill kernel | Local validation passed; Kaggle T4 v89: correctness passed, decode GEMM/e2e passed, prefill e2e failed | Reverted completely; v51 preserved | Occupancy annotation alone did not pass the prefill gate |
| v75 | Hoist precision-selected INT4/INT8 pointers out of the hot loop | Local validation passed; Kaggle T4 v90: correctness passed, decode GEMM/e2e passed, prefill e2e failed | Reverted completely; v51 preserved | Removing repeated runtime pointer ternaries was insufficient |
| v76 | Direct global WMMA load for full-row INT activation tiles | Local validation passed; Kaggle T4 v91: correctness passed, decode GEMM/e2e passed, prefill e2e failed | Reverted completely; v51 preserved | Removing full-row INT A staging alone did not solve the prefill bottleneck |
| v77 | Direct global INT A load plus conditional full-tile CTA barrier bypass | Kaggle T4 v92: correctness failed; decode gates passed; prefill e2e failed | Reverted completely; v51 preserved | Barrier removal is unsafe even for full tiles |
| v79 | Two-slot software B-tail double-buffer design audit | No code change; no measurement | Rejected as unsafe; v51 preserved | A local B-slot tweak cannot prove overlap or shared-A synchronization on SM75 |
| v80 | Deep-research-informed M-stratified prefill dispatch | No code change; no Kaggle measurement | Rejected before implementation | vLLM evidence supports shape stratification, but current kernels cannot encode safe M=128 geometry without repeating failed v70 reuse routing |
| v81 | Deep research of native SM75 PTX INT8/INT4 microkernel references | No code change; no Kaggle measurement | Research-only; baseline preserved | Existing references do not prove a safe ABI-compatible PTX replacement; exact gate extraction blocked by remote quoting |
| v82 | Deep-research-informed vectorized 4-byte activation A staging | Patch failed before validation; no Kaggle measurement | Reverted to v51 | Remote InsertRange failure occurred; before-snapshot restored source |
| v83 | Aligned 4-byte vectorized full-row activation A staging | Local validation passed; Kaggle T4 correctness failed; no_go | Reverted | Vector load mapping was not numerically equivalent to the WMMA tile despite alignment proof |
| v83-corrected | Exact INT8/INT4 branch vectorized A staging after forensic repair | Native correctness passed; prefill e2e failed badly; no_go | Reverted | Correct target fixed correctness, but 4-byte vectors cover only 64 of 128 lanes for a 16x16 int8 tile; further improvement requires a larger tile/pipeline redesign |
| v84 | Deep-research explicit 8-warp 64x32 tile redesign | No code change; no Kaggle measurement | Rejected before implementation | Existing reuse kernel is structurally 32x64; safe 64x32 conversion requires an atomic full-kernel rewrite, not local edits |
| v85 | Deep-research compile-time precision specialization | No code change; no Kaggle measurement | Rejected before implementation | Current monolithic grid mixes precision CTAs via blockIdx.x; specialization requires separate launches or a full template-body refactor |


## v87 — invariant scale/index pointer hoist (rejected: prefill e2e regression)

Deep research identified fixed small-tile staging and synchronization overhead as the mismatch. v87 hoisted only precision-selected scale and index pointers; WMMA ownership, arithmetic, barriers, benchmark settings, and quality checks were unchanged.

Kaggle T4 version 98 compiled successfully. sm75_native_correctness, mixed_decode_gemm_performance, and mixed_decode_end_to_end_performance passed; mixed_prefill_end_to_end_performance failed. Gate decision: no_go. The change was reverted to v51; no performance claim is accepted.



## v88 existing reuse dispatch rejected

Deep research supported larger warp tiles for reuse but required bounded M. v88 routed only rows >=32 through the existing 8-warp 2x4 reuse kernel and kept the original 4-warp path below 32. No arithmetic, quality, or benchmark settings changed.

Kaggle T4 version 99 compiled; native correctness, mixed decode GEMM, and mixed decode end-to-end passed; mixed prefill end-to-end failed. Gate decision: no_go. Reverted to clean v51.


## v89 corrected — quantizer max-reduction micro-optimization (rejected: prefill e2e regression)

Deep research found that activation quantization uses half2 loads and warp shuffles already; the cache idea was unsafe, so v89 tested only direct fmax updates while retaining the values array required by packing. Initial v100 and v101 submissions exposed a compile bug because values was removed too early; v101 log identified values undefined at line 68. The corrected v102 source retained values and changed only the max reduction.

Kaggle v102 compiled and passed native correctness, mixed decode GEMM, and mixed decode end-to-end, but mixed prefill end-to-end failed. Gate decision: no_go. Reverted to clean v51; no v89 performance claim is accepted.


## v90 — SM75 read-only activation load hint (rejected: prefill e2e regression)

Fresh Marlin research identified activation reuse and L2/register reuse as transferable principles, while excluding Ampere-only cp.async. v90 changed exactly one INT prefill activation load to __ldg and left geometry, arithmetic, barriers, weights, scales, quality, and benchmark settings unchanged.

Kaggle T4 version 103 compiled and passed native correctness, mixed decode GEMM, and mixed decode end-to-end, but mixed prefill end-to-end failed. Gate decision: no_go. Reverted to clean v51.


## v92 — full 8-warp 128-channel active prefill tile (rejected: prefill e2e regression)

Fresh Marlin and NVIDIA research supported wide N tiles and eight warps, but warned about register pressure and synchronization. The experiment changed only kPrefillWarps from 4 to 8; kPrefillChannels and the host launch width propagated to 128 while WMMA arithmetic, masks, and ABI remained unchanged.

Kaggle T4 version 104 compiled and passed native correctness, mixed decode GEMM, and mixed decode end-to-end, but mixed prefill end-to-end failed. Qwen mixed rows=16 remained approximately 0.??x and rows=128 remained approximately 0.??x versus dense; the exact artifact is output-1787041018. Gate decision: no_go. Reverted to clean v51.


## v93 — INT prefill group-K unroll directive (rejected: prefill e2e regression)

Fresh Stream-K research showed that persistent scheduling addresses wave quantization but would require a new reduction path, so v93 tested only a static group-K unroll directive in the active 4-warp INT prefill loop. No loads, barriers, WMMA arithmetic, ABI, quality, or benchmark settings changed.

Kaggle T4 version 105 compiled and passed native correctness, mixed decode GEMM, and mixed decode end-to-end, but mixed prefill end-to-end failed. Qwen mixed rows=128 remained about 0.??x versus dense in output-1787041329. Gate decision: no_go. Reverted to clean v51.


## v95 — INT4 predicate hoist (rejected: prefill e2e regression)

Fresh fusion and occupancy research found no redundant activation or weight load that could be removed safely. v95 introduced only const bool is_int4 = (precision == 4) and replaced the repeated INT4 predicate in the active INT prefill path; no arithmetic, WMMA operands, barriers, masks, ABI, quality, or benchmark settings changed.

Kaggle T4 version 106 passed native correctness, mixed decode GEMM, and mixed decode end-to-end, but mixed prefill end-to-end failed. Exact artifact: output-1787041958. Gate decision: no_go. Reverted to clean v51.

## v100 — packed INT4 prefill (compile failure)
Research found a bandwidth mismatch between packed INT4 and the expanded INT8 cache. v100 added packed_int4 and zero_int4 pointers and on-the-fly nibble unpacking without changing model, arithmetic, quality, or benchmark settings. Local embedding validation passed. Kaggle version 107 reached the CUDA build but failed with RuntimeError: Error building extension; no correctness or performance result is accepted. Source audit found two consecutive } else { lines after the new full-tile INT4 branch.

## v100.1 — duplicate control-flow terminator correction (pending Kaggle)
The only change from v100 is deletion of the one duplicate } else { line. Notebook rebuilt and validated with 19 embedded files; Kaggle T4 measurement is required before any keep/revert decision.

## v100.2 — kernel launch ABI correction (pending Kaggle)
Deep research of PyTorch custom CUDA operators and nvcc confirmed that raw pointer arguments must match the __global__ declaration in order. Source audit found v100 had inserted packed and zero pointers into expand_int4_sm75_kernel, duplicating its existing arguments, while the prefill launch lacked them. The single correction restored the expansion launch and added the two pointers only to three_level_tensorcore_kernel. Notebook rebuilt and validated with 19 embedded files; no model, quality, arithmetic, or benchmark settings changed.

## v100.3 — WMMA const-cast correction (pending Kaggle)
Deep source comparison against clean v51 showed the new full-tile INT4 WMMA load used reinterpret_cast<signed char*> while the accepted v51 full-tile overload uses reinterpret_cast<const signed char*>. The only change is restoring that exact const cast; launch ABI, packed nibble unpacking, arithmetic, model, quality, and benchmark settings are unchanged. Notebook rebuilt and validated with 19 embedded files.

## v100.4 — diagnostic traceback capture (pending Kaggle)
Deep research found the Kaggle artifact truncates nvcc stderr after RuntimeError: Error building extension. The only change is a try/except around torch.utils.cpp_extension.load that writes traceback.format_exc() to /kaggle/working/sm75_build_traceback.txt before re-raising. CUDA source, model, quality, arithmetic, and benchmark settings are unchanged. Notebook rebuilt and validated with 19 embedded files.

## v100.5 — captured ninja diagnostic (pending Kaggle)
The v100.4 traceback confirmed PyTorch preserves only a generic Error building extension and omits nvcc stderr. The diagnostic-only change now reruns ninja -v in the first discovered torch-extensions build directory and writes captured stdout/stderr to sm75_ninja_diagnostic.txt. CUDA source, model, quality, arithmetic, and benchmark settings are unchanged.

## v100.6 — restore missing expansion-launch close (pending Kaggle)
Captured ninja diagnostics reported expected an expression, expected a ), and too few arguments at line 770. Source audit found expanded_int4.data_ptr<int8_t>(), was followed directly by C10_CUDA_KERNEL_LAUNCH_CHECK(); in expand_int4_sm75_kernel. The only correction adds the missing );. Notebook rebuilt and validated with 19 embedded files; no model, quality, arithmetic, or benchmark settings changed.

## v100.7 — restore expansion channels and width arguments (pending Kaggle)
Version 114 reported too few arguments at the expansion launch. Source comparison confirmed expand_int4_sm75_kernel requires packed, zero, expanded, channels, and width; v100.6 supplied only the first three. The only correction restored weight_int4.size(0) and input_fp16.size(1). The notebook was rebuilt and validated with 19 embedded files; no model, quality, arithmetic, or benchmark settings changed.

## v100.8 — restore loader state assignment (pending Kaggle)
Version 115 compiled but benchmark_sm75_backend failed because _LOADED remained false. Source audit showed diagnostic instrumentation had removed the original _LOADED = True after torch extension load. The only correction restored that assignment inside the successful try block. Notebook rebuilt and validated with 19 embedded files; no CUDA algorithm, model, quality, or benchmark setting changed.

## v100.8 result — packed INT4 path executes but prefill rejected
Kaggle T4 version 116 compiled and ran. t4_hardware, native correctness, mixed decode GEMM, mixed decode end-to-end, embedded contract tests, fixed allocator, and auto allocator passed; full-model Qwen quality/throughput were not run by this microbenchmark gate. Terminal decision: no_go because mixed prefill end-to-end failed. Qwen QKV mixed rows=1: e2e 1.1999x, GEMM 1.2726x; rows=16: e2e 0.2349x, GEMM 0.1512x; rows=128: e2e 0.0808x, GEMM 0.0576x. The packed on-the-fly hypothesis is therefore not a safe improvement and is not promoted over v51. No model, quality, arithmetic, or benchmark setting was changed.

## v100.9 — inline PTX INT8 MMA (pending Kaggle)
Deep research from NVIDIA/CUTLASS sources identified high-level WMMA INT8 byte-permutation overhead on Turing and recommends direct mma/inline PTX for peak performance. The isolated change replaces only the INT8/INT4 prefill wmma::mma_sync call with a direct wmma.mma.sync.aligned.row.col.m16n16k16.s32.s8.s8.s32 inline-PTX helper using the existing fragment registers; FP16 path, loads, tiling, accumulator semantics, model, quality, and benchmark settings are unchanged. Notebook rebuilt and validated with 19 embedded files. Pending identical Kaggle T4 gates.

## v101 Frankenstein safe set (v51 cumulative)
Compatibility audit found no independently validated post-v51 change that improves performance without a required-gate regression. Retained cumulative set: v41 integration, v45/v46 decode lineage as finalized in v51, and v51 expanded INT4 cache initialization. Excluded all changes with correctness, compile, decode/prefill regression, no measurement, or pending status. Restored clean v51 CUDA source from v86 rollback snapshot and rebuilt a separate notebook.

## v101 Frankenstein result (Kaggle version 118 / output-1787050809)
The v51-compatible cumulative Frankenstein compiled and passed native correctness, decode GEMM/e2e, T4, allocator, and embedded-contract gates, but mixed_prefill_end_to_end_performance failed. Qwen/Qwen2.5-0.5B rows=1: e2e 1.19846x, GEMM 1.26953x; rows=16: e2e 0.28739x, GEMM 0.28522x; rows=128: e2e 0.16562x, GEMM 0.16603x. Decision: reject as a new improvement; v51 remains the best validated baseline. The result confirms that combining all safe v51-lineage changes does not unlock the 2.6x target.

## v102 Frankenstein m8n32k16 INT8/INT4 prefill tile (pending Kaggle)
Deep research identified high-level Turing INT8 WMMA overhead and small tile utilization as the concrete mismatch. Single change: remap only the integer prefill path to documented WMMA m8n32k16, splitting each 64-channel CTA into two 32-channel warps and two 8-row warp pairs; FP16, K loop, quantization, scales, indices, model, and gates unchanged. Local notebook rebuilt and validated with 19 embedded files. Pending identical Kaggle T4 measurement.
v102 corrected source audit: documented m8n32k16 integer fragments, warp row split, 32-channel split, shared-B leading dimension 32, and 8x32 accumulator/output mapping all verified; clean v51 restored before patch. Notebook rebuilt and check passed with 19 embedded files.

## v102 Frankenstein m8n32k16 result (Kaggle version 119 / output-1787051874)
The documented m8n32k16 remap compiled and passed decode GEMM/e2e, allocator, contract, and T4 gates, but native correctness failed with large error and mixed prefill failed. rows=1 e2e 1.18161x, GEMM 1.26779x; rows=128 e2e 0.13154x, GEMM 0.13257x. The speed signal is not valid because correctness failed. Diagnosis pending: m8n32 A/B fragment layout or shared-memory leading dimension/mapping mismatch. Decision: do not retain v102.

## v102.1 Frankenstein m8n32k16 shared-B layout correction (pending Kaggle)
Deep research confirmed v102 correctness failure came from using ldm=32 for a shared B tile physically stored as 32 columns of 16 K values; column-major ldm must be 16. Single change: restore only shared-B WMMA load ldm from 32 to kTile. Notebook rebuilt and check passed with 19 embedded files.

## v102.2 Frankenstein m8n32k16 output indexing correction (pending Kaggle)
Single correctness fix after v102.1 source audit: changed only post-store output loop from 16x16 bounds and kTile row/channel indexing to 8x32 bounds with /32 and %32; retained shared-B ldm=kTile. Notebook rebuilt and check passed with 19 embedded files.

## v102.3 result (Kaggle version 122 / output-1787052769)
- Hypothesis: forcing all full-channel INT8/INT4 B tiles through explicit shared-memory col-major packing would eliminate the m8n32 direct-global B layout mismatch.
- Change: disabled both direct global m8n32 B loads; used shared b_int8 staging with ldm=kTile; no benchmark/model/quality changes.
- Gates: sm75_native_correctness=failed; mixed_decode_gemm_performance=passed; mixed_decode_end_to_end_performance=passed; mixed_prefill_end_to_end_performance=failed.
- Results: rows=1 e2e=1.1972x, GEMM=1.2699x, max_error=0.0643; rows=16 e2e=0.1914x, GEMM=0.1909x, max_error=357.6736.
- Decision: rejected; retained v51 as last all-gates-passing baseline. Residual error remains in m8n32 prefill.

## v102.4 result (Kaggle version 123 / output-1787053175)
- Hypothesis: m8n32 warps raced on one shared activation panel; provide four independent 8x16 A panels indexed by warp/2.
- Change: a_int8[4][8*kTile], write/load a_int8[warp/2]; all other m8n32, B staging, quantization, model, and benchmark settings unchanged.
- Gates: sm75_native_correctness=failed; mixed_decode_gemm_performance=passed; mixed_decode_end_to_end_performance=passed; mixed_prefill_end_to_end_performance=failed.
- Results: rows=1 e2e=1.1979x, GEMM=1.2691x, max_error=0.0643; rows=16 e2e=0.1913x, GEMM=0.1921x, max_error=595.6472; rows=128 e2e=0.0992x, GEMM=0.0999x, max_error=508.3988.
- Decision: rejected; the race fix did not restore correctness and worsened the m8n32 prefill result.

## v102.5 result (Kaggle version 124 / output-1787053473)
- Hypothesis: after per-row A panels, v102.4 still used global threadIdx.x offsets; pair-local offsets would initialize each 8x16 panel completely.
- Change: pair-local A fill start=(warp%2)*32+lane and stride=64; repaired newline-only script artifact before build. No other kernel or benchmark changes.
- Gates: sm75_native_correctness=passed; mixed_decode_gemm_performance=passed; mixed_decode_end_to_end_performance=passed; mixed_prefill_end_to_end_performance=failed.
- Results: rows=1 e2e=1.1839x, GEMM=1.2731x, max_error=0.0643; rows=16 e2e=0.1971x, GEMM=0.1972x, max_error=0.0951; rows=128 e2e=0.1028x, GEMM=0.0560x, max_error=0.1025.
- Decision: not retained as a passing baseline; correctness is recovered, but prefill is far below v51 and FP16.

## v102.6 result (Kaggle version 125 / output-1787053687)
- Hypothesis: v102.2 direct-global-B failure was confounded by bad A staging; re-enable direct B only after v102.5 corrected A panels.
- Change: restored the two full-channel direct global INT8/INT4 WMMA B loads; preserved pair-local A staging, m8n32 mapping, quantization, and benchmarks.
- Gates: sm75_native_correctness=passed; mixed_decode_gemm_performance=passed; mixed_decode_end_to_end_performance=passed; mixed_prefill_end_to_end_performance=failed.
- Results: rows=1 e2e=1.0329x, GEMM=1.2706x, max_error=0.0643; rows=16 e2e=0.2737x, GEMM=0.2524x, max_error=0.0951; rows=128 e2e=0.1471x, GEMM=0.0875x, max_error=0.1025.
- Decision: rejected as a full-gate version; direct B is correct and improves prefill versus forced shared B, but remains far below v51/FP16.

## v102.7 result (Kaggle version 126 / output-1787053943)
- Hypothesis: shared activation staging was the remaining prefill bottleneck; direct global A for full 8-row tiles would remove per-K-step shared traffic/barriers while keeping partial-row fallback.
- Change: full_rows uses direct row-major global A load with ldm=width; partial rows retain pair-local shared A; direct full-channel B remains enabled.
- Gates: sm75_native_correctness=passed; mixed_decode_gemm_performance=passed; mixed_decode_end_to_end_performance=passed; mixed_prefill_end_to_end_performance=failed.
- Results: rows=1 e2e=1.2082x, GEMM=1.2735x, max_error=0.0643; rows=16 e2e=0.2815x, GEMM=0.2355x, max_error=0.0951; rows=128 e2e=0.1317x, GEMM=0.1000x, max_error=0.1025.
- Decision: rejected as a full-gate version; correctness remains valid, but prefill is still far below the target and v51.

## v103 — vLLM SM75 native contract correction (static validation only)
Hypothesis: the external vLLM patch was internally inconsistent and would fail at runtime before reaching the native backend. The patch advertised capability 70, gated to reference/auto, then called the native SM75 operator, and used unquoted cache attribute names.
Changes: changed get_min_capability from 70 to 75; changed the accepted backend set to auto/sm75 with an SM75-specific error; quoted _sm75_int4_expanded and _sm75_fp16_placeholders. No CUDA source, quantization arithmetic, benchmark settings, or model code changed.
Validation: extracted patched Python compiled with py_compile; all four replacement postconditions passed; CUDA SHA remained D10FD597508DA10521A0530AB7AD62398B7EC2974B722625613E397C195A1F0C.
Performance: not measured because this was a non-CUDA serving-contract correction; the next notebook must embed the audited sources before any T4 performance claim. Decision: retain as a contract-only fix pending pinned vLLM execution test.

## v104 — block-uniform m8n32 A barrier correction (Kaggle version 127 / output-1787056456)
Deep research found a CUDA block-barrier convergence defect: full_rows differs by warp pair on a partial final tile, but __syncthreads was inside only the partial-row branch. The single change moved the barrier outside the branch; no B layout, arithmetic, tile shape, benchmark, or quality settings changed.
Result: compiled and executed on Tesla T4. sm75_native_correctness=passed; mixed_decode_gemm_performance=passed; mixed_decode_end_to_end_performance=passed; mixed_prefill_end_to_end_performance=failed. Rows=1 end-to-end was about 1.20x versus dense; rows=128 prefill end-to-end remained below 1x (about 0.36x).
Decision: reject as a full-gate performance version; retain the synchronization correction as a correctness prerequisite for the next isolated prefill hypothesis.
