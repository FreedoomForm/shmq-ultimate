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

## v105 â€” native ThreeLevelLinear SM75 forward path
Date: 2026-08-18. Change: `ThreeLevelLinear.forward()` now selects the native `mixllm.sm75_backend.three_level_linear` path only for CUDA capability `(7, 5)`, restores leading dimensions, and applies bias after the operator; other devices retain the reference fallback. Validation: static source inspection and Python compilation passed. Kaggle: not run because the audit was still open.

## v106 â€” vLLM contract tests aligned with native SM75 execution
Date: 2026-08-18. Change: replaced stale reference-only expectations in the vLLM contract tests with assertions for `three_level_linear`, native SM75 backend selection, and the accepted `backend=auto|sm75` contract. Validation: all 11 local vLLM contract tests passed. Kaggle: not run.

## v107 â€” vLLM patch format corrected
Date: 2026-08-18. Change: corrected patch hunk counts and restored the final newline. Validation: `git apply --check` passed against the pinned vLLM commit `5fbbfe9a4c13094ad72ed3d6b4ef208a7ddc0fd7`. Kaggle: not run.

## v108 â€” runtime SM75 guard and contract assertion
Date: 2026-08-18. Change: vLLM `apply()` now verifies `torch.cuda.get_device_capability(x.device) == (7, 5)` before loading the native backend; the test asserts the guard is present. Validation: Python compilation, 11 local vLLM tests, and `git apply --check` passed. Kaggle: not run.

## v109 â€” manifest contract reconciliation
Date: 2026-08-18. Change: rewrote `THREE_LEVEL_MANIFEST.md` to document native SM75-only execution, direct output-channel checkpoint layout, tensor-parallel remapping, separate activation/INT4-expansion launches, and honest benchmark boundaries. Validation: stale reference-only manifest claim removed; a disposable patch-application check still passed. Kaggle: not run.

## v110 â€” CUDA source-contract test repair and invariant coverage
Date: 2026-08-18. Change: replaced the malformed source-contract test with a clean test module covering the native kernel count, m8n32 fragment shapes, per-panel A staging, warp ownership, output indexing, block-uniform `__syncthreads()`, and native `ThreeLevelLinear.forward()`. Validation: Python compilation and all 4 source-contract tests passed. Kaggle: not run.

## v111 â€” real vLLM apply-path gate attempt
Date: 2026-08-18. Change: embedded the pinned vLLM patch and added a real `MixLLMThreeLevelLinearMethod.apply()` smoke path to the Kaggle notebook. Result: rejected as a gate version because Kaggle version 128 stopped with `ModuleNotFoundError: No module named 'vllm'`; the failure was an environment/setup defect, not a CUDA correctness result. No performance claim retained.

## v112 â€” honest offline vLLM gate behavior
Date: 2026-08-18. Change: retained embedded patch-contract verification and changed the runtime smoke to record `unavailable_environment` when vLLM is absent, instead of converting an environment absence into a false kernel failure. Any actual vLLM import/apply error remains `failed`; real apply execution is reported as `passed` only when it executes. Validation: 15 local contract tests, Python compilation, patch applicability, notebook build, and provenance check passed. Kaggle: corrected run pending.

## v113 â€” classify incompatible preinstalled vLLM honestly
Date: 2026-08-18. Change: vLLM `0.27.1` in the Kaggle image is not the pinned vLLM `0.9.0` commit targeted by patch `0002`; the runtime gate now records this as `unavailable_environment` while preserving embedded patch-contract checks. It does not apply a 0.9.0 patch to an incompatible 0.27.1 package. Validation: builder compilation, notebook build, and provenance check passed. Kaggle: version 130 was pushed but remained queued beyond the 900-second watchdog; no benchmark result was attributed to v113.

## v114 â€” remove stale generated vLLM assertion from repaired v129 gate
Date: 2026-08-18. Change: the v113 notebook still contained a literal generated `\\nassert gates['vllm_apply_path'] == 'passed'` in its final cell, so `unavailable_environment` was converted back into an AssertionError. Removed only that stale generated assertion; retained patch-contract verification and honest unavailable-environment reporting. Validation: 15 local contract tests, Python compilation, patch applicability, notebook rebuild, provenance check, and stale-assertion scan passed. Kaggle: version 131 pushed; it remained queued during the polling window, so no T4 benchmark result is attributed to v114 yet.

## v115 - v51-safe-set audit: no admissible speed-only addition
Deep research and historical gate comparison found no post-v51 change that improves both GEMM and end-to-end while preserving all required gates. v45/v46 improved GEMM but regressed E2E versus v51; v47-v50 regressed; v87-v95 and v100-v104 had prefill regressions or contract-only outcomes; v105-v114 were integration/documentation fixes, not speed improvements. No performance source change applied. Candidate remains v51. No Kaggle run needed for this no-op candidate.

## v116 - v51 plus isolated v45 four-channel decode mapping
Restored the exact v51 CUDA kernel and retained its expanded INT4 rows=1 cache. Applied only v45 kDecodeChannelsPerWarp=4 to the decode path; did not port v47 shared staging, v48 shuffle reuse, v49 restrict pointers, or any m8n32 prefill changes. Local py_compile, 14 contract tests, patch applicability, notebook rebuild, and 21-file provenance check passed. Kaggle measurement pending.

## v116 result - REJECTED, restored v51
Kaggle version 133 on Tesla T4 completed with execution passed. Gates: sm75_native_correctness=passed, mixed_decode_gemm_performance=passed, mixed_decode_end_to_end_performance=passed, mixed_prefill_end_to_end_performance=failed; t4_production=failed; terminal_decision=no_go. For qwen_qkv_mixed_4_8_16, speedups versus dense FP16 were rows=1: GEMM 1.363901x and E2E 1.279556x; rows=16: GEMM 0.254375x and E2E 0.257640x; rows=128: GEMM 0.150291x and E2E 0.149877x. v116 therefore did not improve the v51 GEMM baseline of 1.4156x and caused severe larger-row regressions despite its rows=1 E2E point. The exact v51 CUDA snapshot was restored and the source contract expectation reverted to kDecodeChannelsPerWarp=2. No quality or benchmark settings were changed.

## v117 - v51 with isolated activation-quantizer 8-warp launch
Deep research found v51 maps one warp to each 128-element activation group and launches the quantizer with 128 threads (4 warps). v117 changes only that launch constant to 256 threads (8 warps), preserving the exact per-group arithmetic, output layout, scales, weights, model, benchmark settings, and v51 decode/prefill kernels. Hypothesis: fewer blocks and more concurrent groups reduce quantization launch/scheduling overhead, especially for rows > 1. Snapshot saved as v117_v51_before_quant_threads.cu. Measurement pending.

## v117 result - REJECTED, restored v51
Kaggle version 134 on Tesla T4 completed with execution passed. Gates: sm75_native_correctness=passed, mixed_decode_gemm_performance=passed, mixed_decode_end_to_end_performance=passed, mixed_prefill_end_to_end_performance=failed; t4_production=failed; terminal_decision=no_go. For qwen_qkv_mixed_4_8_16, speedups versus dense FP16 were rows=1: GEMM 1.272727x and E2E 1.200268x; rows=16: GEMM 0.283217x and E2E 0.285522x; rows=128: GEMM 0.098802x and E2E 0.173662x. Increasing the activation quantizer from 128 to 256 threads did not improve v51 and made the larger-row behavior worse. Exact v51 CUDA restored; no quality or benchmark changes.

## v118 - complete compatibility audit and exact v51 revalidation candidate
Read every version entry in the worklog and reconciled speed, correctness, prefill/decode gates, source overlap, and v51 compatibility. The strict admissible speed-method set is empty: every post-v51 speed candidate either regressed a required gate, failed correctness/compile, was audit-only/pending, or duplicated/conflicted with an already-tested method. v45/v46/v47/v48/v49/v50 are overlapping decode access strategies; prefill candidates v53-v95 and v102.x are overlapping resource/layout/staging strategies with repeated prefill failures; SM80 cp.async paths are not portable to SM75. No rejected CUDA method is added. Current CUDA SHA equals the clean v51 snapshot. v118 is a fresh Kaggle revalidation of v51 plus the retained v103-v114 integration/contract fixes; it is not claimed as a speed improvement.

## v118 result - exact-v51 revalidation after full compatibility audit
Kaggle version 135 completed on Tesla T4 with normalized embedded CUDA SHA matching the current v51 source (the raw Windows SHA differs only because the builder normalizes CRLF to LF). Gates: sm75_native_correctness=passed, mixed_decode_gemm_performance=passed, mixed_decode_end_to_end_performance=passed, mixed_prefill_end_to_end_performance=failed; t4_production=failed; terminal_decision=no_go; vLLM apply path=unavailable_environment for the incompatible Kaggle vLLM environment. Qwen mixed rows=1: GEMM 1.255814x, E2E 1.158954x; rows=16: GEMM 0.263636x, E2E 0.267995x; rows=128: GEMM 0.151838x, E2E 0.151902x; max_abs_error remained within the gate threshold. No new CUDA method was added. This is a revalidation result, not a claimed improvement; the previously recorded v51 best remains the project comparison baseline, while the current rerun is conservatively classified no_go under the unchanged four-gate policy.

## v119 — Complete SM75 CUTLASS-style synchronous prefill port (final Kaggle candidate, pending one measurement)

- **Date:** 2026-08-19
- **Baseline:** exact restored v51 CUDA path and ABI, with previously accepted vLLM/SM75 integration fixes retained.
- **Deep research:** audited the original MixLLM CUTLASS `DefaultMmaCore`, SM80 custom `MQMmaMultistage`, scale/zero iterators, dequantizer, and CUTLASS SM75 `mma_sm75.h`. The port uses the verified Turing `m8n8k16` signed INT8 Tensor Core instruction, a `64x64x64` threadblock tile, `32x32x64` warp tile, and a two-stage synchronous shared-memory pipeline. No SM80 `cp.async` instruction is used.
- **Implementation:** added `mq_mma_pipelined_sm75.h` with guarded synchronous A/B/scale/zero staging, block-uniform barriers, and one K-tile mainloop iteration per stage; added `sm75_cutlass_testbed.h` with a row-major float output epilogue; routed prefill INT4 and INT8 partitions through the new path while retaining the v51 FP16 path and v51 fallback for unsupported K alignment.
- **ABI protection:** packed INT4 is still expanded by the existing v51 cache; the CUTLASS path consumes the same exact expanded signed-INT8 values and the same activation/weight scale semantics. Channel-major project metadata is explicitly transposed into the CUTLASS group-major iterator layout before dispatch.
- **Notebook:** builder now embeds the new custom headers and the vendored CUTLASS include tree (849 files total) and passes deterministic provenance validation.
- **Local validation:** Python compilation passed; SM75 source contract passed; vLLM contract tests passed; runtime-capability contract passed; static audit confirmed no `cp_async`, `Sm80`, or `sm80` token in the new SM75 pipeline; notebook rebuild and `--check` passed. Local CUDA compilation is unavailable because `nvcc` is not installed on the Windows workspace. The broad test discovery also reports only expected torch-dependent import errors in the Windows environment; the valid torch-free contract tests pass.
- **Kaggle policy:** no intermediate Kaggle measurements were run. One final Kaggle T4 run is permitted only after this completed pre-Kaggle audit. The candidate is **pending final T4 measurement**; keep only if all four required gates pass.
- **Decision:** pending final Kaggle result; v51 rollback snapshot saved at `research-15/versions/v51_before_cutlass_port.cu`.

---

### v119 notebook packaging correction

The initial v119 notebook embedded the full CUTLASS include tree (849 files, approximately 29 MB), which caused the Kaggle upload connection to abort with Windows socket error 10053 before the new source was accepted. A deterministic include-closure scan found the complete dependency closure: 139 CUTLASS/custom headers, approximately 3.12 MB. The builder now embeds 166 total files and produces a 3,451,695-byte notebook; provenance validation passes. A read-only fetch of the latest completed Kaggle artifact confirmed it still contained the prior exact-v51 source (`HAS_CUTLASS_INCLUDE=False`, `HAS_RUNNER=False`), so no CUTLASS benchmark was accidentally created. The final CUTLASS submission remains pending and has not yet produced a T4 measurement.

---

## v119 result — Kaggle version 136 rejected at compile stage

The final v119 submission reached Kaggle T4 and downloaded the complete CUTLASS source closure, but CUDA compilation failed before any correctness or performance gate. The exact error was that CUTLASS SM75 `RegularTileIterator` has no `set_iteration_index()` member; the copied SM80-style staging code called that method on shared-memory tile iterators in the main copy path, prologue, and clear-last-stage path. No speed or quality claim is accepted from v136, and no benchmark result was recorded.

## v120 — SM75 shared-staging iterator API correction (final candidate after compile fix)

The correction replaces only the custom pipeline’s shared-memory writer iterator types in `sm75_cutlass_testbed.h`: `Core::SmemIteratorA/B` (`RegularTileIterator`) are replaced with CUTLASS `RegularTileAccessIterator` instantiations using the same SM75 tensor-op layouts and thread maps. This preserves the grouped `set_iteration_index()` staging semantics while matching the actual CUTLASS API; global predicated iterators, arithmetic, tile shapes, scales, zero semantics, model, quality checks, and benchmark settings are unchanged. All seven valid torch-free local contracts pass (4 SM75 source, 6 vLLM, 7 runtime-capability tests); the compact 28-file notebook rebuild and provenance check pass. The vendor archive was regenerated after the correction. One final Kaggle T4 measurement remains necessary; keep only if all four required gates pass.

---


## v137 / v121 pointer-ABI audit

Version 137 reached Kaggle T4 compilation but failed before any gate. The first errors were in `sm75_cutlass_testbed.h`: CUTLASS `PredicatedTileAccessIterator` has a non-const `Pointer` ABI, while the kernel passed `const ElementA*` and `const ElementB*`. This is an API compatibility failure only; no correctness or performance claim is valid.

For v121, changed only the CUTLASS testbed kernel parameters to `ElementA*` and `ElementB*`, matching the vendored CUTLASS iterator ABI. Values, scales, zero semantics, model, quality checks, benchmark settings, and the v51 decode path are unchanged. Regenerated the 139-file vendor archive, rebuilt the 28-file notebook, and reran 4 SM75 source, 6 vLLM, and 7 runtime-capability contract tests; all passed. Notebook provenance check passed. v121 is ready for the next Kaggle T4 compilation. Keep only if all four required gates pass; otherwise restore v51.

**Deep-research finding:** the original MixLLM/CUTLASS global iterator path uses mutable element pointers as a staging ABI even for read-only operands. The safe fix is therefore limited to pointer cv-qualification and does not alter computation.


## v121 Kaggle result — compiled, then failed in asynchronous runtime setup

The corrected v121 candidate compiled successfully on Kaggle T4 for `compute_75` / `sm_75` and passed the embedded 18-test contract suite. It failed before the four gates during the first benchmark case. The call stack showed `actual = three_level_linear_prequantized(...)` followed by `quantized_reference_prequantized(...)`; the latter failed at the independent FP32 `torch.nn.functional.linear` with `CUBLAS_STATUS_EXECUTION_FAILED`. No gate, quality, or speed result is admissible. Because CUDA launches are asynchronous, the later CUBLAS call may only be reporting an earlier error from the new kernel.

## v122 — non-timed CUDA fault-localization diagnostic

Deep research of NVIDIA CUDA error-reporting guidance and CUTLASS synchronization requirements indicates that an asynchronous custom-kernel fault can surface in a later CUBLAS call. Added one `torch.cuda.synchronize()` immediately after `actual` and before the independent reference GEMM in the benchmark setup path only. This is outside every timed measurement loop and does not change model computation, quality checks, benchmark settings, or comparison baselines. The purpose is to attribute the failure to the CUTLASS kernel if it is already pending. Python syntax, 4 SM75 source contracts, 6 vLLM contracts, 7 runtime-capability contracts, notebook rebuild, and provenance validation all passed. v122 is pending one Kaggle T4 diagnostic run; no performance claim is made.

**Decision:** v121 rejected; v51 remains the admissible baseline until v122 completes all four gates.


## v122 Kaggle result — synchronization localized an illegal access to the CUTLASS kernel

Version 122 compiled successfully on Kaggle T4 and passed the embedded 18-test contract suite. The added non-timed `torch.cuda.synchronize()` immediately after `actual = three_level_linear_prequantized(...)` raised `CUDA error: an illegal memory access was encountered`. This proves the prior CUBLAS failure was a delayed report of an error from the new CUTLASS kernel, not a reference-GEMM or benchmark-setting problem. No gate or speed result is admissible.

## v123 — restore original per-group staging counts

Deep comparison against the original MixLLM `mq_mma_multistage.h` found a concrete semantic divergence in the SM75 synchronous port. The original `copy_tiles_and_advance()` copies `Detail::kAccessesPerGroupA` and `Detail::kAccessesPerGroupB` in each warp-MMA group. The SM75 port incorrectly looped over the full `AsyncCopyIterationsPerStageA/B` for every group, while also resetting the iterator to the group start. This over-issued shared/global staging work and could advance access iterators beyond their intended group boundary, consistent with the localized illegal access.

Changed only those two loop bounds back to the original per-group counts. The v122 synchronization diagnostic remains in the non-timed benchmark setup so any remaining asynchronous fault will be attributed directly. No arithmetic, tensor values, quality check, model, or benchmark timing configuration changed. Python compilation, 4 SM75 source contracts, 6 vLLM contracts, 7 runtime-capability contracts, notebook rebuild, and provenance validation passed. v123 is pending a Kaggle T4 run; keep only if all four required gates pass.

**Decision:** v122 rejected; v51 remains the admissible baseline pending v123.


## v123 Kaggle result — illegal access persists after per-group copy repair

Version 123 compiled successfully on Kaggle T4 and passed the 18 embedded tests, but the non-timed synchronization after the combined dispatch still reported `CUDA error: an illegal memory access was encountered`. The v123 staging-loop correction is retained as a logically necessary original-compatible repair, but v123 is rejected because the runtime fault remains and no gate result is admissible.

## v124 — partition-boundary fault localization

Deep research and source comparison narrowed the remaining fault to the integer CUTLASS path rather than the FP32 reference. Added a temporary `cudaStreamSynchronize(stream)` immediately after each `Int8Runner::run()` inside `run_cutlass_int_partition()`, before the next integer partition launches. This is diagnostic only, outside the timed measurement loops, and will be removed before accepting any performance result. It preserves all arithmetic and benchmark settings. Local Python compilation, 4 SM75 source contracts, 6 vLLM contracts, 7 runtime-capability contracts, notebook rebuild, and provenance validation passed. v124 is pending Kaggle T4 localization; no speed claim is made.

**Decision:** v123 rejected; v51 remains the admissible baseline pending v124.


## v124 Kaggle result — fault remains inside the integer operator call

Version 124 compiled and passed the 18 embedded tests. The partition-boundary synchronization did not produce an admissible result; the notebook still failed during `three_level_linear_prequantized` with an asynchronous illegal-memory-access error. Because the exception was raised from the extension call, the output did not unambiguously identify whether the first INT4 or second INT8 partition failed. No gate or performance result is valid.

## v125 — host completion markers for INT4/INT8 localization

Deep research of the original CUTLASS runner and CUDA asynchronous error behavior indicates that a host-side completion marker after a successful partition synchronization is the least invasive way to distinguish the two integer launches. Added a temporary `CUTLASS_PARTITION_SYNC_OK channels=...` print after the existing per-partition `cudaStreamSynchronize`; if the marker appears, that partition completed without the pending illegal access, and if it does not, that partition is the first failure. This instrumentation is diagnostic only and will be removed before any performance measurement. Local 4 SM75 source, 6 vLLM, and 7 runtime-capability contracts pass; notebook rebuild and provenance check pass. v125 is pending Kaggle localization.

**Decision:** v124 rejected; v51 remains the admissible baseline pending localization.


## v125 Kaggle result — first failure was before any prefill partition marker

Version 125 compiled and passed the 18 embedded tests. No `CUTLASS_PARTITION_SYNC_OK` marker appeared, but the failing benchmark begins with `rows=1`, which uses the existing v51 decode branch and never enters the new CUTLASS prefill partitions. Therefore v125 localized the first failing operation to the rows=1 operator call or an earlier native quantization launch, not to the prefill partition markers. No gate or speed result is admissible.

## v126 — rows=1 decode-boundary localization

Deep research of the dispatch code showed that the first smoke benchmark row is rows=1 and follows the validated v51 decode path. Added a temporary stream synchronization and `DECODE_SYNC_OK rows=1` marker immediately after the rows=1 decode launch, before the generic launch check. This is diagnostic-only and does not alter timed execution or model computation in an accepted candidate. The earlier partition synchronizations and markers remain temporarily in place. All local torch-free contracts and notebook provenance validation passed. v126 is pending one Kaggle localization run.

**Decision:** v125 rejected; v51 remains the admissible baseline pending v126.


## v126 Kaggle result — rows=1 decode path is clean

Version 144 (v126) compiled successfully and passed the 18 embedded tests. The log contains repeated `DECODE_SYNC_OK rows=1` markers, proving the rows=1 v51 decode launches complete without a pending CUDA fault. The run then failed on the first rows>1 prefill invocation before any partition completion marker. Therefore the remaining illegal access is in the new prefill/CUTLASS path. No gate or performance result is admissible.

## v127 — CUTLASS prelaunch marker

Deep research of the dispatch sequence now isolates the fault to the first rows>1 integer prefill call. Added a temporary `CUTLASS_PARTITION_BEGIN channels=...` marker immediately before each CUTLASS runner launch, retaining the post-synchronize completion marker. This will identify whether the first INT4 launch (`channels=64` in the smoke case) fails before synchronization. The diagnostic is outside accepted timing and will be removed before any performance run. Local contracts, notebook rebuild, and provenance validation passed. v127 is pending one Kaggle localization run.

**Decision:** v126 rejected; v51 remains the admissible baseline pending v127.


## v127 Kaggle result — first INT4 CUTLASS launch fails

Version 145 (v127) compiled and passed the 18 embedded tests. The log showed every rows=1 decode synchronization completing, then `CUTLASS_PARTITION_BEGIN channels=64` with no `CUTLASS_PARTITION_SYNC_OK` marker. This conclusively localized the first illegal access to the initial INT4 prefill CUTLASS launch. No quality or performance result is admissible.

## v128 — canonical SM75 shared-iterator AdvanceRank correction

Deep comparison against CUTLASS `default_mma_core_sm75.h` identified a concrete shared-memory iterator orientation error. The canonical SM75 specialization uses A `AdvanceRank=0` and B `AdvanceRank=1`; the port had these reversed. Corrected only the two `RegularTileAccessIterator` aliases in `sm75_cutlass_testbed.h` to A=0/B=1. The earlier per-group copy-count repair remains. Temporary synchronizations and markers are still diagnostic-only and will be removed before any accepted performance run. Local 4 SM75 source, 6 vLLM, and 7 runtime-capability contracts, notebook rebuild, and provenance validation passed. v128 is pending Kaggle T4 correctness localization.

**Decision:** v127 rejected; v51 remains the admissible baseline pending v128.


## v128 Kaggle result — AdvanceRank correction insufficient

Version 146 (v128) compiled, all rows=1 decode synchronizations passed, but the first `CUTLASS_PARTITION_BEGIN channels=64` still ended with `CUDA error: an illegal memory access was encountered`. The AdvanceRank reversal was a real canonical mismatch but not the only defect. No benchmark result is admissible.

## v129 — guard skipped warp-MMA copy groups before iterator setup

Deep audit of the original `mq_mma_multistage.h` and current access-iterator implementation found that `copy_tiles_and_advance()` computes `kAccessesPerGroupA/B=1` while `kWarpGemmIterations=4` and each iterator has only two thread-map iterations. The old port placed `set_iteration_index(group_start_A/B)` outside the validity guard, so groups 2 and 3 programmed out-of-range shared/global iterator indices even though those groups had no valid global copy. The repair wraps iterator setup and the copy loop in `if (group_start_X < AsyncCopyIterationsPerStageX)`, preserving valid groups and skipping invalid groups without touching iterator state. Canonical shared iterator AdvanceRank remains A=0/B=1. Local validation and Kaggle correctness remain pending.

**Decision:** v128 rejected; v51 remains the admissible baseline pending v129.


## v129 Kaggle result — first INT4 launch still illegal

Version 147 (v129) compiled and passed the rows=1 decode localization, but the first INT4 CUTLASS partition still raised an illegal memory access. No performance or quality result is admissible.

## v130 — predicate metadata shared-memory stores

Deep comparison with the original `mq_mma_multistage.h` showed that the synchronous SM75 metadata copy incorrectly used the operand `sync_copy()` helper, which writes a zero destination even when a scale/zero iterator is invalid. Fine-grained metadata has only one valid thread-row; invalid thread rows must issue no store because their shared destinations are outside the staged metadata buffer. Changed only `copy_scales_and_advance()` to perform stores conditionally on `iterator.valid()`, preserving operand A/B zero-fill semantics and the original scale/zero predication.

**Decision:** v129 rejected; v51 remains the admissible baseline pending v130 correctness validation.


## v130 Kaggle result — metadata predication removed one direct fault but correctness still fails

Version 148 (v130) compiled and executed the full benchmark collection, but the gate report was `t4_production: failed`, with `sm75_native_correctness: failed`, all mixed performance gates failed, and terminal decision `no_go`. The report contains measured timing rows but they are not admissible because correctness failed; several rows have NaN errors after the asynchronous CUDA fault. The repair therefore remains rejected. The direct metadata invalid-store bug was real but additional correctness divergence remains in the SM75 pipeline or fragment/scatter path.

Observed validated baseline remains v51: 1.4156x best GEMM and 1.2210x best E2E, all four required gates passing.


## v131 — restore original column-major accumulator contract

The original MixLLM testbed requires `Mma::LayoutC == ColumnMajor` for its converted accumulator indexing and scatter path. The SM75 port had selected `RowMajor` while copying that same mapping. Changed only the `LayoutC` type alias in `sm75_cutlass_testbed.h` to `cutlass::layout::ColumnMajor`; the destination tensor remains explicitly row-major through the existing `ptr_C[global_row * ldc + channel]` indexing. This is a correctness-contract alignment, not a benchmark-setting change.


## v131 Kaggle result — column-major accumulator alignment insufficient

Version 149 compiled and passed partition stream synchronization, but native correctness still failed: the smoke prefill rows=8 error remained approximately 69.65 and later rows produced NaN after the asynchronous fault. All production performance gates therefore failed and no timing result is admissible. The column-major `LayoutC` alignment matched the original fragment contract but did not solve the remaining error.

## v132 — restore original dynamic shared-memory launch configuration

The original MixLLM runner explicitly calls `cudaFuncSetAttribute` for `cudaFuncAttributeMaxDynamicSharedMemorySize` and `cudaFuncAttributePreferredSharedMemoryCarveout` before launching when shared storage reaches 48 KiB. The SM75 runner launched with the dynamic byte count but never configured these attributes. Added the same conditional configuration, preserving the existing grid, block, model, and benchmark settings. This is a launch-correctness fix, not a performance or quality shortcut.


## v132 Kaggle result — dynamic shared-memory attributes did not restore correctness

Version 150 compiled and completed the benchmark notebook. The launch now uses the original-style dynamic shared-memory limit and carveout configuration, but `sm75_native_correctness` remained failed and all three mixed performance gates remained failed. The measured rows=1 timings are not admissible because correctness failed; rows=16/128 still contain NaN errors after the CUTLASS path. The repair is rejected and v51 remains the only accepted baseline.


## v133 — duplicate quantization metadata across the two 64-K stages

The SM75 pipeline stages 64 K elements, while all three-level metadata tensors use a 128-element group. The previous port copied metadata only on every other 64-K iteration and left the second stage uninitialized. v133 copies the current valid scale/activation-scale/zero into every stage; it advances the global metadata iterators only after the odd half, preserving one metadata value per 128 K group while making both pipeline stages valid. Invalid metadata lanes remain predicated and do not write shared memory.


## v134 — restore original CUTLASS INT4 B interleave

Deep audit of the original MixLLM found `interleave_uint4_for_cutlass()` in `nn/modules/linear.py` and `test/test_kernel.py`. The original CUTLASS path does not consume raw `[channels,K]` INT4 values: it applies a two-step K-column permutation and then repacks. The SM75 path was feeding a raw signed zero-subtracted expansion, which explains the large prefill errors despite valid memory accesses. v134 adds a cached CUTLASS-only signed INT4 tensor by applying the identical composed K permutation to the expanded cache. Decode continues to use the raw signed cache, and the fallback WMMA prefill path remains unchanged. A new tensor is threaded through the SM75 CUDA ABI with shape/dtype/device checks; no benchmark inputs, quality thresholds, or model settings were changed. Python syntax, torch-free source/vLLM/runtime contracts, notebook rebuild, and provenance check passed. The optional GPU backend unit test cannot run on the Windows workspace because its Python environment has no PyTorch; Kaggle remains the authoritative GPU validation.


## v135 — restore original per-thread accumulator lane offset

Deep comparison of `sm75_cutlass_testbed.h` against both original MixLLM CUTLASS testbeds found that the port copied the accumulator fragment indexing but omitted the original per-thread lane offset. The original computes `quad = threadIdx.x >> 2`, `lane_in_quad = threadIdx.x & 3`, then adds `quad` to the output row origin and `lane_in_quad * kElementsPerAccess` to the output channel origin before scattering. Without this term, all 32 threads in a warp target overlapping output coordinates and overwrite one another, explaining large prefill errors while the separate v51 decode path remains correct. v135 adds exactly this lane offset; no accumulator arithmetic, metadata, quantization, model, quality, or benchmark setting changed. The rejected v134 interleaved INT4 ABI was fully reverted to the original 12-tensor ABI and raw signed-int8 CUTLASS operand. Local Python compilation, 4 SM75 source contracts, 6 vLLM contracts, 7 runtime-capability contracts, notebook rebuild, and provenance validation passed. Kaggle T4 version 154 is required; keep only if all four gates pass.

**Decision:** pending Kaggle T4 result; v51 remains the accepted baseline.


## v135 result — lane offset improved but did not pass correctness

Kaggle version 154 compiled and executed on Tesla T4. Gates: `mixed_decode_gemm_performance=passed`, `mixed_decode_end_to_end_performance=passed`, `mixed_prefill_end_to_end_performance=failed`, `sm75_native_correctness=failed`, so `t4_production=failed` and `terminal_decision=no_go`. The lane-aware scatter reduced mixed rows=8 max error from approximately 69 to 30.93 and pure INT4/INT8 prefill errors to approximately 38–41, confirming that the missing original per-thread lane offset was a real bug but not the final one. No timing result is admissible and v135 is rejected; v51 remains the safe baseline.

The next deep audit identified the current SM75 pipeline’s `pipe_state.tmp_accum_.fill(1262485504)` immediately before `mma.sync` accumulation into the same fragment. CUTLASS SM75 is explicitly `S32 = S8*S8 + S32`, and the warp wrapper passes the existing accumulator as C, so this is not a neutral initialization. The next isolated candidate will replace only this sentinel fill with `clear()` in the SM75 pipeline, with no quantization, model, quality, or benchmark changes. Kaggle validation is required.


## v136 — zero-initialize the SM75 post-scaled temporary accumulator

Deep research of the original CUTLASS SM75 `mma.sync.aligned.m8n8k16...s32.s8.s8.s32` implementation confirmed that the instruction computes `D = A*B + C`. The SM75 pipeline was filling `tmp_accum_` with the nonzero sentinel `1262485504` before passing it as both D and C, which is not a mathematically neutral initialization. v136 changes only this line to `pipe_state.tmp_accum_.clear()`. The v135 original lane-offset correction remains; the rejected v134 interleaved ABI remains reverted. No quantization, scale/zero, model, quality, or benchmark setting changed. Local 4 SM75 source contracts, 6 vLLM contracts, 7 runtime-capability contracts, notebook rebuild, and provenance validation passed. Kaggle T4 measurement is required; keep only if all four gates pass.

**Decision:** pending Kaggle T4 result; v51 remains the accepted baseline.


## v136 result — rejected zero-initialization experiment

Kaggle version 155 compiled and ran on T4, but `sm75_native_correctness` failed with NaN errors from rows=16 onward and `mixed_prefill_end_to_end_performance` failed. Decode GEMM/E2E passed. No timing result is admissible. The change from the original `tmp_accum_.fill(1262485504)` sentinel to `clear()` is rejected and will be reverted; v51 remains the safe production baseline while the CUTLASS audit continues.


## v137 — account for SM75 vertical serpentine accumulator visitation

Deep source comparison of CUTLASS `MmaTensorOp` found `kVerticalVisit=true` for SM75. For odd `mma_n`, CUTLASS stores the accumulator at fragment slot `m_serpentine + mma_n * MmaIterations::kRow`, where `m_serpentine = MmaIterations::kRow - 1 - mma_m`, while the SM75 port’s custom scatter used `mma_m` directly. v137 will change only the SM75 scatter to use the architecture-correct serpentine fragment index and corresponding logical row mapping, retaining the original lane offset, raw signed-int8 operand ABI, quantization, model, quality checks, and benchmark settings. Local contracts and Kaggle T4 validation are required.


## v137 result — serpentine output remap did not change the residual error

Kaggle version 156 compiled and ran on T4. Results remained effectively identical to v135: `sm75_native_correctness=failed`, `mixed_prefill_end_to_end_performance=failed`, while both decode performance gates passed; representative errors were smoke rows=8 approximately 30.93, pure INT4 rows=16 approximately 40.01, and pure INT8 rows=16 approximately 39.19. No timing result is admissible and v137 is rejected.

Deep audit of the original `mq_mma_tensor_op_dequantizer.h` against CUTLASS SM75 `kVerticalVisit=true` now identifies the likely residual: `apply_scale_accum_act` indexes activation scales by the physical fragment row slot (`n/2`) but does not reverse that slot for odd N MMA iterations. v138 will change only activation-scale indexing for SM75 vertical serpentine order; all operand, weight-scale, quantization, model, quality, and benchmark settings remain unchanged. Kaggle T4 validation is required.


## v138 result — activation-scale serpentine correction rejected

Kaggle version 157 compiled, but `sm75_native_correctness` and `mixed_prefill_end_to_end_performance` failed. The dequantizer change worsened errors: smoke rows=8 approximately 62.98, pure INT4 rows=128 approximately 70.94, and pure INT8 rows=128 approximately 73.75, with NaN at some rows=16. Decode GEMM/E2E passed. No timing result is admissible; the dequantizer was restored to the original mapping.

The refined v139 hypothesis is that SM75 serpentine order affects only the physical fragment slot used for `start`, while the logical output row must continue using `mma_m`. v139 retains the lane offset and changes only that distinction; all scaling, operands, model, quality, and benchmark settings remain unchanged.


## v139 result — serpentine physical-fragment scatter rejected

Kaggle version 158 compiled and ran on T4, but `sm75_native_correctness` and `mixed_prefill_end_to_end_performance` failed. Using the serpentine index only for the physical accumulator fragment slot made large-row errors much worse: mixed rows=128 approximately 417.9, pure INT4 approximately 389.2, and pure INT8 approximately 368.6; rows=16 produced NaN. Decode GEMM/E2E passed, and no timing result is admissible. v139 is rejected. The next source state restores the original `mma_m` fragment-slot scatter formula while retaining only the original per-thread lane offset, which was the only scatter edit that measurably reduced errors.


## v140 — execute the final preloaded 64-K SM75 tile

Deep comparison with the original SM80 loop found that its two `mac_loop_iter()` calls per outer iteration allow the second call to run after the counter reaches zero. The SM75 port changed to one call per outer iteration but retained `for (; gemm_k_iterations > 0;)`; because prologue already decrements the initial tile count, this skips the final 64-K tile. For Qwen K=3584, 56 tiles are required but the port computes only 55, explaining finite similar INT4/INT8 prefill errors and decode correctness. v140 changes only the SM75 loop condition to `gemm_k_iterations >= 0`; all other semantics and benchmark settings remain unchanged. Local contracts and Kaggle T4 validation are required.


## v140 result — native correctness passed, mixed prefill performance failed

Kaggle version 159 passed `sm75_native_correctness` and both decode performance gates. Prefill arithmetic is now near exact: mixed rows=16/128 errors approximately 0.095/0.103, pure INT4 approximately 0.000107/0.000183, and pure INT8 approximately 0.000107/0.000198. However, `mixed_prefill_end_to_end_performance` failed: mixed rows=16/128 speedups versus dense were approximately 0.182/0.192, with GEMM speedups approximately 0.184/0.194. Under the mandatory all-four-gates rule v140 is not kept as production baseline; v51 remains the accepted best baseline. The final-K-tile condition is retained as a correctness prerequisite while the CUTLASS prefill performance path is audited.


## v141 — audit CUTLASS prefill overhead against the validated v51 WMMA path

v140 restored native correctness but mixed prefill was approximately 5x slower than dense. Source comparison found that the current rows>1 dispatch launches `run_cutlass_int_partition()` independently for INT4 and INT8, repeating full activation staging and synchronization, whereas the v51 path launches one precision-tiled `three_level_tensorcore_kernel` over all planes. v141 will route rows>1 through the unchanged v51 WMMA prefill launch as an honest fallback measurement; it does not remove computations, lower quality, alter the model, or change benchmark settings. The corrected CUTLASS implementation remains in source for subsequent optimization, but no CUTLASS speed claim is made unless all four gates pass.


## v141 result — WMMA fallback remained below the mixed prefill gate

Kaggle version 160 passed native correctness and both decode performance gates, but mixed prefill E2E failed. Mixed rows=16/128 GEMM speedups were approximately 0.282/0.166 and E2E speedups approximately 0.285/0.170; pure INT4/INT8 prefill rows=16/128 also remained below dense. v141 is rejected as a production improvement and v51 remains the accepted baseline.

Deep comparison with original MixLLM found that the original SM80 loop batches two `mac_loop_iter()` calls and performs one dequantization/float-conversion pass on the combined temporary accumulator. The SM75 port currently performs one 64-K call plus scaling/conversion for every tile. v142 will batch two synchronous calls per outer iteration, guard the second call for odd tile counts, and retain every K tile and all arithmetic. This is an overhead-only optimization with no quality or benchmark changes.


## v143 — widen the SM75 CUTLASS N tile to 128

Deep comparison with original MixLLM’s configuration table found that the current SM75 runner uses 64x64x64 while original tested configurations include 64x128 and 128x128 threadblocks. v143 changes only the CUTLASS ThreadblockShape from 64x64x64 to 64x128x64, retaining WarpShape 32x32x64, instruction m8n8k16, two synchronous stages, all arithmetic/metadata/output semantics, the model, quality, and benchmark settings. The wider tile should reuse each activation tile across twice as many output channels and halve the N-grid. Local contracts and Kaggle T4 validation are required.


## v143 result — inert CUTLASS geometry change, no valid performance evidence

Kaggle version 162 passed native correctness and decode gates but failed mixed prefill performance, with mixed rows=16/128 speedups approximately 0.288/0.173. The embedded `three_level_sm75.cu` hash matched v142 because v141 had switched the active rows>1 path back to WMMA; therefore the 64x128 Core edit was not exercised. v143 is rejected and provides no CUTLASS geometry evidence. Revert the unused 64x128 edit and stale test expectation, then restore the v140 CUTLASS dispatch before the next real wider-N measurement.


## v144 result — active 64x64 CUTLASS path still failed mixed prefill

Kaggle version 163 exercised the restored CUTLASS integer dispatch with 64x64x64 Core and two-call batching. Native correctness and both decode gates passed, but mixed prefill failed: rows=16 GEMM/E2E speedups were approximately 0.151/0.150 and rows=128 approximately 0.311/0.306. Pure INT4/INT8 prefill also remained below dense. v144 is rejected; v51 remains the accepted baseline. v143’s wider-N change was inert because WMMA fallback was active, so v145 will measure the 64x128x64 Core with CUTLASS dispatch active.


## v145 result — wider active CUTLASS tile improved pure integer rows=128 but failed mixed prefill

Kaggle version 164 exercised 64x128x64 CUTLASS geometry. Native correctness and decode gates passed; mixed prefill rows=16/128 speedups were approximately 0.164/0.163 and 0.332/0.329. Pure INT4 rows=16/128 reached approximately 0.804/0.663 GEMM speedup and pure INT8 approximately 0.826/0.680, an improvement over 64x64 but still below the required mixed prefill gate. v145 is rejected.

Deep audit found `run_cutlass_int_partition()` creates transposed contiguous scale/zero tensors on every invocation, while stable expanded INT4 state is already cached per module. v146 will cache the prefill metadata layout per layer and invalidate it with device/state changes, while retaining original metadata layout for decode. This removes only repeated host allocation/copy overhead and preserves all arithmetic, quality, model, and benchmark settings.


## v146 result — cached metadata remained below the prefill gate

Kaggle version 165 passed native correctness and decode gates after caching transposed prefill metadata while preserving decode layout. Mixed prefill GEMM/E2E speedups were approximately 0.169/0.168 at rows=16 and 0.181/0.358 at rows=128; pure INT4 rows=16/128 were approximately 0.891/0.711 GEMM and pure INT8 approximately 0.864/0.657. The host-cache optimization is rejected as a production improvement because the dominant cost is inside the synchronous CUTLASS kernel.

Next deep audit: compare every CTA barrier and guarded global-to-shared copy in the SM75 pipeline with original MixLLM’s cp.async mainloop. Any removal must be justified by a complete producer/consumer synchronization proof; no quality, computation, model, or benchmark shortcut is allowed.


## v147 hypothesis — remove one provably redundant synchronous barrier

The SM75 mainloop performs guarded global-to-shared copies, then executes an explicit `__syncthreads()` and immediately calls `gmem_wait()`, which is another `__syncthreads()` on SM75. Since SM75 `sync_copy` is ordinary synchronous assignment rather than cp.async, the first barrier already protects the same producer/consumer boundary. v147 removes only the explicit barrier and keeps `gmem_wait()` as the single barrier. No computation, quantization, output mapping, quality, model, or benchmark setting changes.


## v147 / Kaggle version 166 result — single-barrier optimization is correct but not sufficient

Kaggle version 166 completed. The redundant-barrier removal passed native correctness, both decode performance gates, embedded contracts, and T4 hardware checks, but mixed prefill performance still failed. Mixed rows=16 measured approximately 0.149 GEMM / 0.148 E2E speedup; rows=128 measured approximately 0.340 GEMM / 0.335 E2E speedup. The synchronization reduction is therefore rejected as a production candidate; v51 remains the accepted baseline. It is retained only as a documented safe micro-optimization if a future candidate independently passes all four gates.


## v148 hypothesis — remove inherited end-of-mainloop barrier

Original MixLLM drains SM80 cp.async operations at the end of `gemm_iters()` with fence, wait, and CTA barrier. The SM75 port uses synchronous guarded copies, and all stage boundaries already use `gmem_wait()` barriers. After the final MAC, the kernel reads only register accumulators for the epilogue, so the final standalone `__syncthreads()` has no remaining producer/consumer role. v148 removes only this final barrier, preserving all computation and one required barrier at every stage boundary.


## v148 / Kaggle version 167 result — end-barrier removal preserved correctness but did not pass prefill

Kaggle version 167 completed with native correctness, decode GEMM/E2E, embedded contracts, and T4 checks passed. Mixed prefill remained failed: rows=16 measured approximately 0.172 GEMM / 0.171 E2E speedup and rows=128 approximately 0.336 GEMM / 0.332 E2E. The final inherited end-of-mainloop barrier was therefore not the dominant bottleneck. v148 is rejected under the all-four-gates rule; v51 remains the accepted baseline.


## v149 hypothesis — reduce SM75 CUTLASS M tile from 64 to 32

Version 167/v148 preserved correctness but failed mixed prefill, so the final barrier was not dominant. Deep comparison found the active CUTLASS runner uses an 8-warp 64x128x64 CTA even for rows=16, where three quarters of the M tile are inactive. v149 tests a 32x128x64 threadblock with the same 32x32x64 warp tile, m8n8k16 instruction, two stages, metadata, output, model, quality, and benchmark settings. This reduces the CTA to 4 warps and should lower idle-M work and barrier overhead without reducing mathematical computation for valid rows. Local compilation and Kaggle correctness are mandatory.


## v149 / Kaggle version 168 result — 32x128 geometry preserved correctness but failed prefill

Kaggle version 168 passed native correctness, both decode gates, embedded contracts, and T4 checks. Mixed prefill remained no-go: rows=16 measured approximately 0.181 GEMM / 0.180 E2E speedup and rows=128 approximately 0.298 GEMM / 0.295 E2E. The smaller M tile did not recover the required performance; v149 is rejected and v51 remains the accepted baseline.


## v150 hypothesis — restore monolithic balanced WMMA reuse prefill

Kaggle version 168/v149 passed correctness and decode but failed mixed prefill; smaller M geometry was insufficient. Deep comparison found the active current branch launches separate INT4, INT8, and optional FP16 kernels, while the preserved project-local `three_level_tensorcore_reuse_kernel` loads a 2x4 WMMA tile and handles precision planes in one monolithic dispatch. v150 restores that exact kernel and uses it for rows>1, restoring original `[channels, groups]` metadata indexing. No arithmetic, quantization, model, quality, or benchmark settings change. Local contracts and Kaggle T4 validation are required.


## v150 / Kaggle version 170 result — monolithic WMMA reuse is correct but slower

The corrected v150 submission compiled after disabling the duplicate preserved definition and fixing the launch namespace typo. Kaggle version 170 passed native correctness, both decode gates, embedded contracts, and T4 checks. However, the monolithic 8-warp WMMA reuse kernel was substantially slower: mixed rows=16 measured approximately 0.105 GEMM / 0.177 E2E speedup, and rows=128 approximately 0.090 GEMM / 0.095 E2E. Mixed prefill failed decisively. v150 is rejected; the active CUTLASS branch remains preferable to this reuse control.


## v151 hypothesis — combine INT4 and INT8 CUTLASS partitions into one integer launch

Version 170/v150 proved the monolithic WMMA reuse control correct but too slow. Deep launch audit found the active CUTLASS path runs two complete integer kernels with identical SM75 INT8 MMA pipelines, differing only in B, scales, and output indices. v151 will construct cached combined signed-INT8 B, combined scale, and combined output-index tensors for INT4 followed by INT8 channels and invoke one CUTLASS integer kernel over `n4+n8`. It removes one kernel launch and one duplicated prologue/epilogue while preserving every integer GEMM operation and quantization value; FP16 remains separate. All local contracts and Kaggle gates are required.


## v151 / Kaggle 171 — combined INT4+INT8 CUTLASS launch (NO-GO)
The v151 candidate combined the expanded signed-INT8 INT4 partition and the native INT8 partition into one `run_cutlass_int_partition` launch for rows > 1, while retaining a separate FP16 launch and the validated decode path. The active CUTLASS Core was restored to `GemmShape<64, 128, 64>` before submission. Local source, vLLM contract, and runtime-capability tests passed, and the Kaggle artifact confirmed SM75 native correctness and both mixed decode gates.

Kaggle T4 version 171 completed successfully with `gate_status.execution=passed`, but production was `no_go`: `mixed_prefill_end_to_end_performance=failed`, while `sm75_native_correctness`, `mixed_decode_gemm_performance`, and `mixed_decode_end_to_end_performance` passed. In the Qwen QKV mixed 4/8/16 scenario, the measured end-to-end speedup versus dense FP16 was 1.1149x for rows=1, 0.17666x for rows=16, and 0.27338x for rows=128. The corresponding GEMM speedups were 1.1966x, 0.18211x, and 0.28403x. Thus the combined launch did not solve the prefill bottleneck and remains rejected under the all-four-gates rule.

The report used the required Tesla T4 / SM75 environment and unchanged Qwen/Qwen2.5-0.5B gate settings. No quality reduction, benchmark alteration, or model substitution was used. The combined-launch and related geometry changes must not be retained as a validated baseline; the next iteration requires a deep comparison against the original MixLLM prefill path and the exact gate implementation before another Kaggle submission.


## v152 research — original MixLLM comparison and prefill gate audit
The exact Kaggle prefill gate is not defined in `model_gate.py`; it is in the generated gate notebook. It constructs `qwen_qkv_mixed_4_8_16` with `in_features=out_features=3584`, partition counts `(2400, 896, 288)`, and rows `(1, 16, 128)`. The gate passes only when every mixed row with `rows > 1` has `end_to_end_p50_ratio_vs_dense <= 1.05`, i.e. the quantized operator must be at least as fast as dense FP16 within the 5% tolerance. v151 therefore failed for both rows=16 and rows=128, not because of a hidden threshold above 1.0x.

The original MixLLM implementation (`mix_mma_multistage.cuh`) uses two independent CUDA streams and two independent CUTLASS GEMM launches: INT4 and INT8 are forked from the caller stream with an event, run concurrently, and are joined with two completion events. It also autotunes across many geometries, explicitly including `16x128`, `32x128`, and `64x128` threadblock families, and uses SM80 `cp.async` multistage staging. The current v151 path instead concatenates INT4 and INT8 into one temporary tensor, launches one 64x128x64 synchronous SM75 CUTLASS CTA, and then launches FP16 on the same stream. This removes original launch-level concurrency and forces small-M work into a 64-row CTA, leaving most lanes inactive at rows=16. The SM75 pipeline has only synchronous global-to-shared copies with a CTA barrier at stage boundaries because SM75 lacks `cp.async`; this explains why the one combined launch is still 5.49x slower than FP16 GEMM at rows=16 and 3.52x slower at rows=128.

The next change is intentionally isolated: add a 16x128x64 SM75 CUTLASS core for the rows=16 regime, selected only when `rows <= 16`, while preserving the existing 64x128x64 core for larger prefill, the validated decode kernel, all tensor dtypes and metadata, exact benchmark rows, and the current correctness checks. This is evidence-backed by the original MixLLM configuration catalog and directly targets the 75% M-tile under-utilization observed at rows=16. If it fails any gate, it will be reverted.


## v152 implementation — small-M 16x128x64 CUTLASS dispatch (Kaggle pending)
Implemented the isolated research hypothesis without changing benchmark settings, quantization, tensor dtypes, model, or decode code. Added `SmallMCore = GemmShape<16, 128, 64>` with `WarpShape<16, 32, 64>` and the same SM75 `m8n8k16` instruction and two-stage synchronous pipeline. `run_cutlass_int_partition` selects `SmallMInt8Runner` only for `rows <= 16`; the existing validated `64x128x64` runner remains selected for rows > 16. The source contract now checks both geometries and the small-M dispatch symbol.

Local py_compile, SM75 source contracts (4 tests), vLLM three-level contracts (6 tests), runtime capability contracts (7 tests), notebook rebuild, and notebook integrity check all passed. Kaggle T4 measurement is required; this candidate will be retained only if all four gates pass.


## v152 result — compile failure (REJECTED)
Kaggle version 172 did not reach the gates. NVCC rejected the `16x128x64` / `16x32x64` SM75 `DefaultMmaCore` during `RegularTileIterator` instantiation with `static assertion failed: Number of iterations must be non-zero` for the A tile map (`PitchLinearShape<64,16>`, 128 threads, 4x8 arrangement, 16 elements/access). This is a real SM75 iterator-geometry constraint: the original SM80 catalog’s 16x128 option is not directly portable to the current SM75 core. No performance or quality result is valid. The candidate is rejected and will be reverted before the next hypothesis.


## v153 hypothesis — restore original MixLLM partition concurrency
After v152’s direct 16x128 port failed SM75 iterator constraints, the next isolated change restores a key original MixLLM property without changing the math: INT4 and INT8 prefill partitions are launched separately on two pooled CUDA streams, forked from the caller stream by an event and joined back by completion events. The temporary `at::cat` allocations and combined metadata ABI are removed. Each partition keeps its own signed-INT8 weights, scales, zero-point tensor, and output indices; the FP16 partition remains on the caller stream. This directly tests whether v151’s serialization and concatenation erased the original launch-level parallelism. The existing 64x128x64 SM75 core, synchronous pipeline, decode path, model, quality, and benchmark settings are otherwise unchanged. Kaggle validation is required and any gate failure rejects the change.


## v153 result — parallel launch exposed asynchronous metadata lifetime bug (REJECTED)
Kaggle version 173 reached execution but failed `sm75_native_correctness` and `mixed_prefill_end_to_end_performance`. In the mixed QKV case rows=16, the output had `max_abs_error=219.8911`; rows=128 happened to report `max_abs_error=0.10254`, showing nondeterministic corruption rather than a stable arithmetic mismatch. The parallel launches created temporary transposed `scale` and `zero` tensors inside `run_cutlass_int_partition`, then returned before the auxiliary streams finished. The PyTorch allocator only automatically protects tensors on the current stream; the new streams were not registered with the allocator, so their metadata storage could be recycled while CUTLASS was still reading it. The original MixLLM avoids this by keeping persistent stream-owned launcher state.

Deep research confirmed the supported fix: PyTorch exposes `c10::cuda::CUDACachingAllocator::recordStream(const DataPtr&, CUDAStream)`, and `c10::cuda::getStreamFromExternal(cudaStream_t, DeviceIndex)` wraps the raw auxiliary stream handle. v154 will record each temporary transposed metadata tensor against the actual CUTLASS stream before launch. This changes lifetime bookkeeping only; it does not alter arithmetic, quantization, benchmark settings, or the model. v153 remains rejected.


## v154 result — allocator-protected parallel launches (NO-GO)
Kaggle version 174 confirmed the lifetime diagnosis: `sm75_native_correctness` passed again, and the large rows=16 corruption disappeared (`max_abs_error=0.09509`). Decode GEMM and decode end-to-end also passed. However, mixed prefill remained far slower than dense FP16: mixed QKV end-to-end speedup was 0.20294x at rows=16 and 0.29165x at rows=128, with GEMM speedups 0.20437x and 0.30162x. Therefore the original-style stream concurrency is correct but insufficient, and the all-four-gates rule rejects v154. The v154 source must be reverted rather than retained as a partial improvement.


## v155 hypothesis — direct monolithic WMMA prefill instead of synchronous CUTLASS
The original MixLLM’s advantage comes from SM80 `cp.async` multistage CUTLASS, which cannot be transferred directly to SM75. v154 proved that restoring launch concurrency fixes correctness but not the fundamental cost: the synchronous SM75 CUTLASS pipeline still takes 4.89x dense GEMM time at rows=16 and 3.32x at rows=128. The current source already contains a monolithic SM75 WMMA kernel that uses signed-INT8 Tensor Core MMA for INT4/INT8 partitions, FP16 WMMA for FP16 channels, per-group scales, and indexed output writes without the CUTLASS stage pipeline. v155 will route all rows>1 prefill through that existing kernel, eliminating the synchronous CUTLASS metadata/stage barriers while preserving the exact quantized arithmetic and all partition computations. Decode and the fallback reuse kernel remain unchanged. This is a performance hypothesis only and will be rejected unless all four gates pass.


## v155 implementation — direct WMMA prefill (Kaggle pending)
The mixed-prefill dispatch now launches the existing four-warp `three_level_tensorcore_kernel` for all rows>1 cases with any INT4 or INT8 channels. It uses the exact existing signed-INT8 INT4 expansion, INT8 weights, per-group scales, FP16 partition, and output-index mapping in one kernel launch; the CUTLASS helper remains compiled but is no longer selected by the mixed prefill dispatch. The unsupported empty-integer case still uses the existing FP16 path, and the non-CUTLASS fallback remains unchanged. Local py_compile, SM75 source contracts, vLLM contracts, runtime capability tests, notebook rebuild, and notebook integrity check all passed. Kaggle T4 validation is required.


## v155 result — direct WMMA prefill (NO-GO)
Kaggle version 175 passed native correctness and both decode gates, but mixed prefill still failed. In the mixed QKV case, direct WMMA improved rows=16 end-to-end speedup from v151’s 0.1767x to 0.2677x, but rows=128 fell to 0.1563x versus v151’s 0.2734x. The direct kernel removes CUTLASS stage overhead for small M but launches many 16x64 channel tiles for large M, so its block-count and repeated K-group barriers dominate at rows=128. The candidate is rejected under the all-four-gates rule.

## v156 hypothesis — row-stratified hybrid
The three T4 measurements now provide a non-guessing dispatch choice: direct WMMA is better for rows=16 (0.2677x) than the allocator-safe parallel CUTLASS path (0.2029x), while parallel CUTLASS is better for rows=128 (0.2916x) than direct WMMA (0.1563x). v156 will combine these two already-tested, correctness-preserving paths by selecting direct WMMA only for rows <= 16 and the allocator-protected original-style parallel CUTLASS launches for rows > 16. No kernel arithmetic or benchmark setting changes. This tests the exact gate’s two regimes with the best measured path for each; failure of any gate rejects the combination.


## v156 implementation — row-stratified hybrid (Kaggle pending)
Implemented the evidence-backed hybrid dispatch. For rows <= 16 and any integer partition, the operator uses the direct four-warp WMMA kernel. For larger rows with width divisible by 64 and any integer partition, it uses the allocator-protected original-style parallel INT4/INT8 CUTLASS launches plus the existing FP16 WMMA launch. The fallback reuse kernel remains unchanged. Local py_compile, SM75 source contracts, vLLM contracts, runtime capability tests, notebook rebuild, and integrity check all passed. Kaggle T4 validation is required; v156 will be kept only if all four gates pass.


## v156 result — row-stratified hybrid (NO-GO)
Kaggle version 176 passed native correctness and both decode gates, but mixed prefill still failed. The mixed QKV end-to-end speedups were 0.28798x at rows=16 and 0.36697x at rows=128. Compared with v155, the hybrid improved both tested prefill rows, but remained below the required 1.0x gate threshold by a wide margin. No change is retained as a validated baseline.

## v157 hypothesis — combine row stratification with 32x128 CUTLASS geometry
The original MixLLM catalog explicitly includes both 32x128 and 64x128 column-major families. The current hybrid’s rows=128 CUTLASS path still uses a 64-row CTA, while the earlier 32x128 experiment was measured only in a different combined-launch context. v157 will keep the already-measured direct WMMA path for rows<=16, retain allocator-protected partition concurrency for rows>16, and change only the CUTLASS Core to `GemmShape<32,128,64>` with the existing `32x32x64` warp tile. This reduces inactive M work and shared staging for the 128-row gate while preserving the same SM75 instruction, metadata, arithmetic, and output ABI. Any gate failure rejects it.


## v157 implementation — 32x128 CUTLASS geometry (Kaggle pending)
Changed only the SM75 CUTLASS threadblock shape from `64x128x64` to `32x128x64`; the warp tile, instruction tile, two-stage synchronous pipeline, row-stratified dispatch, allocator stream recording, and all numerical paths remain unchanged. Local py_compile, SM75 source contracts, vLLM contracts, runtime capability tests, notebook rebuild, and integrity check all passed. Kaggle T4 validation is required.


## v158 hypothesis — widen the direct WMMA integer tile to 128 channels
v157 confirmed that shrinking the CUTLASS M tile did not solve prefill; the hybrid remained at only 0.272x mixed rows=128. Mandatory research of NVIDIA Turing and CUTLASS sources found that SM75’s true INT4 Tensor Core instruction is `m8n8k32` with S4/U4 operands, while the current quantization contract supplies int8 activations and asymmetric zero-subtracted INT4 values. A naive native S4/U4 replacement would therefore change the activation ABI or require an exact correction identity and is not safe to introduce speculatively. The original path’s apparent INT4 storage is transformed through its dequantizer and does not provide an immediately reusable S8xU4 operator.

The safe next hypothesis stays within the validated signed-INT8 arithmetic: replace the existing 16x64 integer WMMA tiling with a 16x128 direct kernel. Each of four warps reuses one activation tile across two 16x16 channel subtiles, reducing CTA count and activation staging while preserving every computation, scale, zero-subtracted expansion, and indexed output. This is a performance-only change and will be rejected unless all four Kaggle gates pass.

## v158 implementation — wide direct WMMA integer kernel (Kaggle pending)
Added `three_level_tensorcore_int_n128_kernel` and routed mixed/pure-integer prefill through it, with the existing FP16 WMMA kernel launched separately for the FP16 partition. The previous CUTLASS branches remain as fallback code but are no longer selected when integer partitions are present. Added a source contract for the new active symbol. The kernel uses four warps, two 16x16 subtiles per warp, shared activation/weight tiles, exact signed-INT8 MMA, existing per-group scale multiplication, and existing output-index mapping. Local validation is next.


## v158 result — wide direct WMMA integer kernel (NO-GO)
Kaggle version 178 passed native correctness and both decode gates, but the wide kernel was substantially slower: mixed QKV GEMM speedups were only 0.0695x at rows=16 and 0.0614x at rows=128, compared with 0.2635x and 0.1545x for v155. The extra per-warp register/shared-memory state for two subtiles caused severe occupancy or spilling; simply doubling the N tile is not safe or useful. v158 is rejected and its path must not be retained.

The next deep comparison will focus on global weight-load layout and occupancy rather than another tile-size guess. The current direct WMMA kernel uses `wmma::load_matrix_sync` directly from row-major `[channels, K]` weights interpreted as column-major B, whereas the original MixLLM stages/interleaves B through CUTLASS iterators and shared-memory TensorOp layouts. The measured v155/v158 contrast indicates that a smaller four-warp WMMA tile is more stable than duplicating two subtiles per warp. The next candidate should therefore improve B staging or launch shape without adding large per-thread fragment arrays.


## v159 hypothesis — remove CTA-wide barriers from the stable 16x64 WMMA path
The original MixLLM uses persistent independent streams and a multistage shared-memory pipeline, while the current direct WMMA path performs `__syncthreads()` after every K sub-tile even though each warp owns a disjoint A/B tile. v158 showed that widening the tile increases register/shared-memory pressure and is harmful. v159 returns to the stable 16x64 decomposition but assigns each warp private shared A/B/accumulator storage and uses only `__syncwarp()`. The kernel still performs the same signed-INT8 Tensor Core operations, all 128 K values per group, all per-group scales, and the same partitioned output writes. This is an isolated synchronization/occupancy optimization; it must pass all gates to be retained.

## v159 implementation — barrier-free direct WMMA integer kernel (Kaggle pending)
Added `three_level_tensorcore_local_kernel` and routed integer prefill through it with the original 16x64 channel grid. The previous wide kernel remains compiled but is no longer selected. Added a source contract for the active local kernel. No decode, quantization, metadata, benchmark, or quality path was changed. Local validation is next.


## v159 result — barrier-free per-warp WMMA (NO-GO)
Kaggle version 179 passed native correctness and both decode gates, but mixed QKV prefill slowed further: end-to-end speedup was 0.1687x at rows=16 and 0.0596x at rows=128, while v155’s stable direct WMMA was 0.2677x and 0.1563x. The per-warp shared-memory arrays did not replace the cost of global WMMA loads; they increased shared-memory traffic and reduced effective throughput. v159 is rejected.

## v160 hypothesis — DP4A prefill for the integer partitions
The original SM80 path relies on Tensor Core multistage staging, but its performance advantage cannot be transferred directly to SM75 because Turing lacks `cp.async` and the current synchronous WMMA/CUTLASS paths are slower than FP16 for the exact gate shapes. The validated decode path already uses `__dp4a` with the same signed INT8 activation and weight ABI, and it is faster than FP16 at rows=1. v160 will isolate a DP4A integer-prefill kernel: one thread computes one `(row, channel)` output over all 128-element groups, using the exact expanded signed-INT4 or signed-INT8 bytes, exact per-group activation/weight scales, and existing output indices. This removes WMMA fragment/shared-memory/barrier overhead but performs no fewer arithmetic operations and does not alter quality or the benchmark. The FP16 partition remains on its existing WMMA kernel. It will be measured on Kaggle and retained only if all four gates pass.


## v160 implementation — DP4A integer prefill (Kaggle pending)
Added `three_level_dp4a_prefill_kernel`, with one thread per output element and 32 signed `__dp4a` operations per 128-element group. The kernel selects the exact expanded INT4 or existing INT8 weight row, applies the exact activation and weight scales for each group, and writes through the existing output-index arrays. The FP16 partition remains on the existing WMMA kernel. Local py_compile, source contracts, vLLM contracts, runtime capability tests, notebook rebuild, and integrity check all passed.


## v160 result — DP4A integer prefill (NO-GO)
Kaggle version 180 passed native correctness and both decode gates but was dramatically slower in prefill: mixed QKV end-to-end speedup was 0.0593x at rows=16 and 0.0112x at rows=128; GEMM speedup was 0.0290x and 0.0112x. The one-thread-per-output DP4A design loses too much parallel dot-product throughput for the large 3584x3584 shapes. It is rejected and will not remain active.


## v161 restoration — stable direct WMMA baseline
After v160’s DP4A failure, restored the v155 dispatch semantics: all non-decode mixed prefill partitions use the existing four-warp 16x16 signed-INT8/FP16 WMMA kernel in one launch. The rejected wide, barrier-free, and DP4A kernels remain non-dispatched experimental source only and do not affect runtime behavior. No quantization, model, benchmark, or quality behavior was changed. Revalidation is required because the source now contains the accumulated experiments.

## v182 — v51 audit telemetry and focused regression enforcement

Deep research compared the accepted v51 snapshots with the bundled original MixLLM implementation and the later active experimental source. The v51 result remains the correctness-passing baseline: best GEMM-only speedup 1.4156x and best end-to-end speedup 1.2210x on the same T4 gate, with mixed prefill still failing. The detailed result shows quantization at roughly 0.027–0.029 ms while mixed prefill GEMM costs roughly 0.49 ms at rows=16 and 1.00 ms at rows=128, so CPU fallback is not the cause.

Restored the active SM75 host/CUDA sources to the clean v51 snapshots, preserving the later source only under `research-15/audit_tmp/active_before_v51_repair/` for comparison. Added `audit_v51_initial.md` and `qwen_quality_sources.md` with source-backed findings. Added `test_v51_audit_contract.py` covering the packed decode/no-expansion v51 contract, focused Kaggle regression coverage, and memory telemetry requirements. Added honest per-shape peak allocated-memory and persistent-storage telemetry to `benchmark_sm75_backend` without changing timed workloads, arithmetic, model, or benchmark shapes. Updated the Kaggle builder to embed and execute the focused unit/contract suites instead of only `test_three_level.py`.

Local dependency-light vLLM/runtime and audit-contract tests pass. CUDA-dependent suites cannot run in this sandbox because PyTorch is not installed; Kaggle T4 remains the authoritative measurement environment. This iteration is not a performance claim and is retained only if the unchanged T4 correctness/decode/prefill gates continue to pass, with the new contract suite passing as well.

## v183 — restore embedded CUTLASS include extraction after v182 compile failure

The v182 Kaggle submission failed before tests and benchmarks because `sm75_cutlass_testbed.h` included `cutlass/array.h`, but the notebook did not unpack the embedded `cutlass_sm75_vendor.b64` archive or pass its include root to nvcc. This was a real missing build component exposed by restoring the v51 source-contract CUTLASS port.

The backend loader now validates the embedded archive, decodes it as ZIP, checks every member for path traversal, extracts it under the pinned MixLLM package root, verifies `cutlass/include/cutlass/array.h`, and passes `extra_include_paths=[cutlass/include]` to `torch.utils.cpp_extension.load`. No kernel dispatch, arithmetic, model, quality target, or benchmark setting changed. The v182 memory telemetry and focused regression suite remain included. Added a regression assertion for the vendor extraction contract.

Local source, runtime, vLLM, audit-contract, compileall, notebook rebuild, and notebook freshness checks pass. Kaggle T4 validation is required; v183 is not accepted unless the expanded tests and all required performance/correctness gates execute successfully.

## v184 — correct CUTLASS helper ABI and portable audit contracts

Deep diagnosis of the v183 Kaggle log found that embedded CUTLASS extraction succeeded, but nvcc rejected the optional helper because `shmq_cutlass_sm75::Int8Runner::run` takes mutable Tensor-handle references while `run_cutlass_int_partition` passed const references. Changed the helper's Tensor handles to pass by value, preserving the same storage and non-dispatched runtime behavior. The expanded audit test also assumed the sandbox repository root and failed in Kaggle; it now resolves the builder from either the sandbox root or the embedded package working directory. The CUTLASS source contract now normalizes whitespace before checking shape and pipeline markers.

No native dispatch, arithmetic, model, quality, memory, or benchmark workload was changed. Local source, runtime, vLLM, audit-contract, compileall, notebook rebuild, and freshness checks pass. v184 requires a fresh Kaggle T4 run; previous v183 is rejected because the extension did not compile.

## v183 result — computer-network rerun of the pinned v183 artifact (NO-GO)

Per user request, the existing Kaggle version `183` was queried and rerun/monitored through the connected computer rather than the sandbox network. It reached `KernelWorkerStatus.ERROR` after executing on Tesla T4 / SM75. The downloaded report is preserved under `shmq-ultimate/mixllm_3level_kaggle/latest-output-v183-computer/`.

The run did compile and measure the native operator. Embedded contract tests failed because the v183 notebook still contained the older exact CUTLASS source-contract expectation and the audit test resolved `/scripts/build_mixllm_3level_kaggle.py` instead of the embedded repository path. Native correctness passed; mixed decode GEMM and end-to-end gates passed; mixed prefill end-to-end failed. Representative mixed QKV results were 1.1991x decode end-to-end speedup at rows=1, 0.2876x at rows=16, and 0.0978x at rows=128. Mixed rows=16/128 GEMM speedups were 0.2852x/0.0975x. Quantization remained about 0.026 ms, confirming it is not the main prefill bottleneck.

Memory telemetry showed mixed QKV packed storage of 9,842,560 bytes versus 25,690,112 bytes dense weights, but the warmed INT4 expansion adds 8,601,600 bytes for prefill; peak allocated memory at mixed rows=128 was 62,401,536 bytes end-to-end versus 60,461,056 bytes for dense FP16. Full-model Qwen quality and throughput remained `not_run`. v183 is rejected and must not be treated as a passing baseline.

## v184 result — corrected ABI still exposed vendor-overwrite defect (NO-GO)

v184 was submitted and executed through the connected computer. T4 hardware, native correctness, decode GEMM, decode end-to-end, fixed/auto allocator, and native benchmark setup passed, but embedded contract tests and mixed prefill end-to-end failed. The exact traceback showed `test_sm75_source.py` reading `GemmShape<64,64,64>` after the extension load, although the embedded source hash had verified the current `GemmShape<32,128,64>` header before execution. Root cause: `load_sm75_backend()` extracted the vendor ZIP directly into `source.parents[2]`; the ZIP contains a historical `mixllm/kernels/sm75_cutlass_testbed.h`, which overwrote the current project header after the source manifest check. The audit contract also raised `StopIteration` in Kaggle because the sandbox-only builder path does not exist there.

## v185 candidate — isolate vendor ZIP and make embedded audit path optional

Changed vendor extraction to a private `.cutlass_vendor_staging` directory and copy only `mixllm/kernels/cutlass` into the vendor root, preserving the authoritative project `sm75_cutlass_testbed.h` and custom `cutlass_extension` sources. Added a cleanup `finally` and retained path-traversal checks. Updated the audit contract to skip builder-specific assertions when the Kaggle package has no builder, while preserving them locally. Local 21-test contract suite, Python compileall, notebook rebuild, freshness check, and diff check pass. v185 requires a fresh computer-network Kaggle run and is not accepted until all required gates pass.

## v185 result — vendor isolation fixed contracts, prefill remains the sole blocker (NO-GO)

v185 was rebuilt and submitted from the connected computer, then executed on Tesla T4 / SM75. The isolated vendor staging fix worked: `embedded_contract_tests=passed`, the T4 gate execution completed, native correctness passed, allocator/import checks passed, and both mixed decode gates passed. The production decision remains `no_go` solely because mixed prefill end-to-end performance failed; full-model Qwen quality and throughput remain `not_run`.

Mixed QKV p50 speedups versus the identical FP16 baseline were 1.0438x decode E2E at rows=1, 0.2661x at rows=16, and 0.1539x at rows=128. Mixed QKV GEMM speedups were 1.3257x, 0.2623x, and 0.1511x at rows=1/16/128. Thus v185 is a valid, complete T4 measurement but not an accepted production baseline. Compared with v184, rows=128 prefill improved materially (0.0978x to 0.1539x) but remains far below the 1.05 ratio gate; rows=16 regressed (0.2876x to 0.2661x). Quantization remained about 0.029-0.031 ms, so it is not the dominant prefill bottleneck. No quality-reduction or benchmark-setting change was made.

The v185 gate report, benchmark JSON, source manifest, and Kaggle log are preserved under `shmq-ultimate/mixllm_3level_kaggle/latest-output-v185-computer/`.

## v186 hypothesis — eight-warp one-subtile SM75 prefill candidate

Deep comparison with the original MixLLM showed that its advantage is a family of M/N/warp configurations rather than a fixed tile; the current direct WMMA path uses one fixed 16x64 channel CTA. Historical v70/v88/v150 measurements rejected the existing 8-warp 2x4 reuse kernel, and v158 rejected two 16x16 subtiles per warp because of register/shared-memory pressure. v186 therefore keeps one 16x16 WMMA subtile per warp, templates the stable kernel over the number of warps, and launches an explicit 8-warp/128-channel variant. It preserves the signed-INT8 arithmetic, FP16 partition, per-group scales, output indices, quantization, benchmark scenarios, and decode dispatch. Local 22 contract/runtime/vLLM tests, compileall, notebook rebuild, freshness, and diff checks pass. Kaggle T4 validation is required; the candidate is not accepted before all gates pass.

## v186 result — eight-warp one-subtile candidate improves rows=16 but remains NO-GO

v186 compiled and executed on Tesla T4 / SM75 with all embedded contracts passing. Native correctness, allocator/import checks, and both mixed decode gates passed. The candidate improved mixed QKV rows=16 end-to-end speedup from v185's 0.2661x to 0.3102x and rows=128 end-to-end from 0.1539x to 0.1883x, but mixed prefill remained far below the required gate. Mixed GEMM was 0.3116x at rows=16 and 0.1044x at rows=128; the rows=128 GEMM result regressed versus v185's 0.1511x. The candidate is rejected and must not replace the accepted path. Peak memory for mixed rows=128 remained 62,401,536 bytes end-to-end versus 60,461,056 bytes dense; no quality or benchmark-setting changes were made. Artifacts are preserved under `shmq-ultimate/mixllm_3level_kaggle/latest-output-v186-computer/`.

## v187 hypothesis — shape-stratified dispatch from v185/v186 evidence

v186's eight-warp one-subtile kernel improved rows=16 mixed GEMM from 0.2623x to 0.3116x but regressed rows=128 from 0.1511x to 0.1044x. v187 will re-use the exact v186 kernel only for rows==16 and retain the v185 four-warp path for rows>16. This is a dispatch-only combination of two measured, correctness-passing kernels; no arithmetic, metadata, model, quality, memory policy, or benchmark workload changes are planned. Kaggle T4 validation is required.

## v187 implementation — shape-stratified dispatch

Restored the measured v186 eight-warp kernel without its unconditional dispatch, and changed only the host launch selection: rows==16 uses the 8-warp/128-channel kernel; all other prefill rows use the v185 4-warp/64-channel kernel. Added source-contract coverage for both branches and corrected the audit test to stop at any `} else` boundary. Local 23 contract/runtime/vLLM tests, compileall, notebook rebuild, freshness, and diff checks pass. Kaggle T4 validation is required; no production claim is made yet.

## v187 Kaggle T4 result — NO-GO

Kaggle v187 completed on Tesla T4 with native correctness, decode GEMM, and decode end-to-end gates passing, but mixed prefill end-to-end failed. For the real Qwen QKV mixed 4/8/16 scenario, rows=1 was 1.2023x, rows=16 was 0.3104x (3.2216x latency ratio), and rows=128 was 0.0995x (10.0525x latency ratio) versus the identical torch FP16 baseline. The rows==16 eight-warp dispatch did not generalize: rows=16 improved versus v185's 0.2661x, but rows=128 regressed versus v185's 0.1539x. Gate decision: no_go; preserve artifacts under `shmq-ultimate/mixllm_3level_kaggle/latest-output-v187-computer` and revert v187 as a production candidate.

## v188 hypothesis and implementation — activate staged CUTLASS only for rows>=32

Deep research compared the original MixLLM launcher (`mix_mma_multistage.cuh`) with v187. The original uses iterator-based multistage shared-memory Tensor Core GEMMs, many shape families, and separate precision streams; v187 still used direct global WMMA loads with only two fixed channel geometries. Historical measurements reject another simple tile widening, private shared-memory WMMA, DP4A, and a small-M SM75 CUTLASS core. The existing 32x128x64 SM75 CUTLASS helper is compiled but dormant in the active v187 dispatch. v188 restores the v185 direct-WMMA path for rows<32 and activates the existing signed-INT8 CUTLASS helper for INT4/INT8 partitions only when rows>=32; FP16 remains a separate direct-WMMA launch on the caller stream. No arithmetic, quantization, model, quality, or benchmark setting changes. NVIDIA Turing/CUDA primary documentation was consulted and saved in `research-15/prefill_deep_research_v188.md`.

Local 22 contract/runtime/vLLM tests, compileall, notebook rebuild, freshness, and diff checks pass. Kaggle T4 validation is required; v188 is not accepted before all four gates pass.

## v189 — v188 audit fixes: CUTLASS metadata cache and explicit quality/production gates

The complete v188 audit is documented in `research-15/v188_full_audit_ru.md`. Deep comparison with the original MixLLM found that v188's large-M staged CUTLASS path is useful but was still transposing scale/zero metadata on every prefill launch; `_sm75_prefill_metadata` existed in the module object but was unused. The audit also found that the Kaggle notebook left the exact Qwen/Qwen2.5-0.5B full-model quality and throughput paths at `not_run`, and that production readiness did not require those results or actual patched-vLLM execution.

v189 implements only safe, independently justified fixes: a state/device/version-aware persistent transposed metadata cache, a new v3 CUDA operator ABI that consumes cached metadata while preserving v2/legacy wrappers, persistent metadata-byte telemetry, exact Qwen2.5-0.5B model-source attachment and bounded full-model quality/throughput reporting, and separate `operator_production` versus `model_vllm_production` decisions. Missing model/vLLM environments are now explicit blockers, never silent passes. No model, arithmetic, benchmark shape, partition budget, or quality comparison is reduced.

Sandbox source/audit contracts (11 tests), notebook rebuild/freshness, Python compilation, and diff checks pass. The CUDA extension and full Qwen gate require connected Kaggle T4 validation.

## v190 hypothesis — remove repeated v3 partition validation from cached large-prefill path

Kaggle v189 proved the metadata cache is active (`prefill_metadata_bytes` reported) and preserved correctness, but mixed rows=128 regressed from v188 `0.2762x` to `0.2197x` E2E speedup. Deep comparison with the original MixLLM and the SHMQ call chain found that v189's new public v3 wrapper calls `validate_partition()` on every large-prefill invocation, while Python already caches the exact partition signature before CUDA execution and the original hot path does not sort/allocate index validation tensors per GEMM. v190 adds an unchecked v3 operator wrapper, routes only the already-validated Python path to it, preserves the public validated v3 API for external callers, and corrects the Kaggle `model_sources` handle to the documented Qwen2.5-0.5B variation form.

Prediction: native correctness and decode remain unchanged; rows>=32 CUTLASS latency should improve if repeated validation caused the v189 regression. If T4 does not confirm improvement, the metadata-cache path will not be retained as a performance claim. No arithmetic, model, quality threshold, partition budget, or benchmark setting changed. Sandbox notebook/source/audit contracts (11), py_compile, freshness, and diff checks pass.

## v190 model-source correction

Kaggle rejected the first `model_sources` form because it required the full `{owner}/{model}/{framework}/{instance}/{version}` handle. The candidate now uses `qwen-lm/qwen2.5/Transformers/0.5b/1`, preserving the exact Qwen2.5-0.5B base model.


## v191 — exact Qwen model discovery and rejected parallel-stream reintroduction

v190 T4 preserved correctness/decode and improved large-M CUTLASS versus v189, but mixed prefill remained no-go. A fresh original-versus-current audit found the tempting original parallel precision-stream design, then checked the historical ledger: v153/v154 already tested it, with allocator-safe v154 still only `0.20294x` rows=16 and `0.29165x` rows=128 E2E, so it is explicitly not reintroduced. The new v191 red stream contract was removed before implementation.

The remaining safe v191 fix improves the full-model gate’s environment handling: instead of assuming one `/kaggle/input/qwen2.5/...` path, the notebook now discovers `config.json` files under `/kaggle/input` and accepts only the exact Qwen2.5-0.5B architecture fingerprint (`qwen2`, hidden size 896, 24 layers, vocab 151936, intermediate 4864, 14 attention heads). Missing input remains an explicit unavailable blocker; no substitute model is accepted. Notebook rebuild, freshness, 11 sandbox source/audit tests, py_compile, and diff checks pass. Kaggle T4 validation is required.


## v191.1 — bounded exact-Qwen discovery repair

The v191 Kaggle job remained `RUNNING` for the entire bounded polling window and produced no gate report or intermediate log. The local poll was stopped; the remote kernel was not deleted because Kaggle CLI exposes no non-destructive stop command. Diagnosis identified a high-risk startup regression in v191: `Path('/kaggle/input').rglob('config.json')` recursively scanned the entire Kaggle input tree. v191.1 removes that unbounded scan and checks only the deterministic model mount paths `/kaggle/input/qwen2.5/transformers/0.5b/1` and `/kaggle/input/qwen2-5/transformers/0.5b/1`, still requiring the exact Qwen2.5-0.5B config fingerprint. Missing input remains an explicit unavailable status; no substitute model is accepted.

The audit contract now forbids `rglob('config.json')` and requires bounded paths. Notebook rebuild/freshness, 24 local source/runtime/vLLM/audit tests, py_compile, and diff checks pass. v191 remains unvalidated; Kaggle T4 validation of this corrected candidate is required.

## v192 — bounded exact-Qwen discovery candidate: Kaggle T4 result
The corrected v191.1 notebook (Kaggle version 192) completed successfully instead of stalling, confirming that deterministic model-path discovery removed the startup hang. The execution, embedded-contract, allocator, SM75 correctness, and decode gates passed. The candidate is still NO-GO because the mixed prefill gate failed by a wide margin: on the mixed Qwen QKV microbenchmark, rows=16 measured 0.401408 ms versus dense FP16 0.122816 ms (0.305963x speedup, 3.268369x ratio), and rows=128 measured 0.837664 ms versus dense FP16 0.197392 ms (0.235646x speedup, 4.243657x ratio). The pure INT4 path reached 0.558277x at rows=16 and 0.702028x at rows=128; pure INT8 reached 0.547025x and 0.696490x, showing the large-M CUTLASS path is functional but remains slower than cuBLAS FP16. Mixed decode continued to pass, with rows=1 mixed end-to-end speedup 1.053337x. Native correctness passed.

The full Qwen2.5-0.5B quality and throughput gates were unavailable_environment: no exact model fingerprint was found under the two deterministic input paths. This is an environment/provenance blocker, not evidence of model quality, and no substitute model was accepted. The terminal decision was no_go; no v192 change is retained as a performance success. Artifact: research-15/kaggle-v192-output/mixllm_3level_gate.json.

Deep-research implication for the next iteration: v192 does not support further tuning of the current direct-WMMA/CUTLASS dispatch as a sufficient solution. Before code changes, compare the original staged/autotuned MixLLM dataflow and the current SM75 host path specifically for repeated materialization, launch decomposition, tile shape, and dense FP16 subpartition handling. Any candidate must preserve all gates and be measured on Kaggle T4.

## v193 — SM75 CUTLASS geometry aligned with original large-M fallback (pending Kaggle)
Deep research compared the original `mix_mma_multistage.cuh` large-M fallback (`gemm<5,64,128,64,32>`, instruction shape 16x8x32, five stages) with the current SM75 adapter (`32x128x64`, `32x32x64`, `8x8x16`, two stages). NVIDIA's CUTLASS documentation identifies threadblock/warp tile reuse and software pipelining as the relevant performance mechanisms, and lists the 16x8x32 signed-INT8 TensorOp shape as SM75-supported. The v193 candidate changes only `sm75_cutlass_testbed.h`: the helper now uses 64x128x64 threadblock shape, 64x32x64 warp shape, 16x8x32 instruction shape, and five synchronous stages. The public runner, custom dequantization, indexed scatter epilogue, host dispatch, FP16 path, model, arithmetic, quality gates, and benchmark settings are unchanged. A source-contract test was written red first against the old core, then passed after the minimal change. Twenty-five non-PyTorch source/runtime/vLLM/audit contracts, Python compilation, and diff checks pass locally. The complete local discovery includes nine tests that cannot import because the sandbox lacks PyTorch; the Kaggle T4 environment remains the authoritative CUDA validation environment. Kaggle measurement is required; retain only if all required gates pass.

## v193 result — SM75 geometry candidate rejected at CUDA compilation
Kaggle version 193 was submitted successfully and reached the CUDA extension build on a Tesla T4, but produced no benchmark or gate report. NVCC rejected the researched geometry because the vendored CUTLASS `DefaultMmaCore` has no matching specialization for `GemmShape<64,128,64>`, `GemmShape<64,32,64>`, `GemmShape<16,8,32>`, SM75, and five stages; the log reports an incomplete `DefaultMmaCore` type at `sm75_cutlass_testbed.h(119)`. This is a compile-compatibility failure, not a performance result. The v193 geometry change is rejected and must not replace the measured v188/v192 path. The next iteration must restore a supported SM75 core geometry while preserving v188's measured dispatch and v192's contracts/telemetry/model safeguards.

## v194 — restore measured v188 mixed-large-prefill ABI on supported SM75 core (pending Kaggle)
Deep research of the v193 compiler failure found that the vendored SM75 `DefaultMmaCore` only specializes TensorOp stage count 2; the attempted original-style five-stage geometry is not a valid specialization in this project and is rejected. The v194 candidate restores the measured v188-compatible core (`32x128x64`, `32x32x64`, `8x8x16`, stage 2) and adds one isolated Python performance adapter: only the already partition-validated mixed 4/8/16 path with rows>=32 routes through the existing v2 unchecked ABI used by v188. Pure INT4, pure INT8, pure FP16, rows<32, public validated v3, metadata cache, telemetry, bounded model discovery, and all benchmark/quality settings remain unchanged. This tests whether v189/v190/v192 cached-v3 metadata/ABI handling caused the v188 mixed rows=128 regression. Twenty-six local source/runtime/vLLM/audit contracts, Python compilation, and diff checks pass. Kaggle T4 measurement is required; retain only if all gates pass.

## v194 result — v188 v2 adapter did not recover v188 speed; NO-GO
Kaggle version 194 compiled and completed on Tesla T4. The isolated v188 mixed-large-prefill adapter was actually selected: mixed rows=128 reported `prefill_metadata_bytes=0`, confirming it bypassed the cached-v3 metadata path. Nevertheless, mixed end-to-end timing was `0.870432 ms` at rows=128 (`0.225838x` speedup) versus v192 `0.837664 ms` (`0.235646x`) and v188 `0.594176 ms` (`0.276228x`); rows=16 was `0.301567x` versus v192 `0.305963x`. The v2 ABI restoration therefore did not reproduce v188's performance. Decode end-to-end also failed this run at `1.091619x` ratio (1.053337x in v192), while native correctness and decode GEMM passed. Full Qwen quality/throughput and patched vLLM remained unavailable because the exact model was absent. Terminal decision: no_go. The v194 adapter is not retained as a performance success; the next audit must compare the exact v188 CUDA/backend implementation and benchmark lifecycle, not assume ABI choice alone explains the historical result.

## v195 — exact v188 historical replay control (pending Kaggle)
Deep research after v194 falsified the hypothesis that the v3 ABI alone caused the v188 speed advantage: v194 selected the v2 unchecked path (`prefill_metadata_bytes=0`) but remained at `0.225838x` for mixed rows=128. NVIDIA's benchmark guidance also identifies fixed-power clock settling and small-GEMM run-to-run variation as confounders. v195 is a control experiment, not a production change: it rebuilds the same Kaggle notebook from the exact v188 committed CUDA, SM75 helper, backend, and source contracts under the current Kaggle environment. Benchmark settings, model requirement, quality thresholds, and gate logic remain unchanged. If v195 does not reproduce v188, the historical advantage was environmental variance; if it does, the next audit will compare compiled resource behavior before changing code.

## v195 result — exact v188 replay reproduced fast e2e but exposed timing-integrity failure
The exact v188 replay completed on Tesla T4. Mixed end-to-end reproduced the historical fast range: rows=16 `0.285325x`, rows=128 `0.298549x`, and rows=1 `1.196461x`; native correctness and all decode gates passed. However, the same rows=128 case reported `sm75_gemm=1.091584 ms` while `sm75_end_to_end=0.547984 ms`, even though end-to-end includes activation quantization plus the same prequantized operation. This physically inconsistent ordering indicates fixed-power clock/event measurement instability or a benchmark lifecycle artifact, not a safe optimization proof. v195 remains NO-GO because mixed prefill failed and model/vLLM gates were unavailable. The v188 fast e2e number must not be restored blindly; the next repair must add timing-integrity observability/guarding and compare repeated same-path timings before accepting a performance claim.

## v196 — timing-integrity guard and honest gate enforcement (pending Kaggle)
The v195 exact-v188 replay reproduced fast E2E but showed an impossible component ordering: rows=128 GEMM p50 was `1.091584 ms` while E2E p50 was `0.547984 ms`. Deep research against NVIDIA CUTLASS measurement guidance identifies clock settling and small-GEMM run-to-run variation as relevant confounders, but the benchmark must not silently accept physically inconsistent timings. v196 adds explicit `timing_integrity_ratio = gemm_p50_ms / end_to_end_p50_ms`, records `timing_integrity` per shape, marks the native launch-check boundary in the CUDA source, and makes timing integrity a mandatory operator gate. No benchmark settings, model, arithmetic, precision path, or quality threshold was changed. The v188 ABI adapter and supported stage-2 core remain present for measurement, but no result is accepted unless timing-integrity and all existing gates pass. Twenty-seven local source/runtime/vLLM/audit contracts, Python compilation, notebook compilation/freshness, and diff checks pass. Kaggle T4 validation is required.

## v196 result — timing integrity passed; prefill still NO-GO
Kaggle version 196 compiled and completed on Tesla T4. The timing-integrity guard passed for all mixed shapes, including rows=128 ratio `0.992736`; the previous v195 impossible ordering was not reproduced in this run. Native correctness, decode GEMM, decode end-to-end, allocator, embedded contracts, and T4 hardware gates passed. Mixed prefill remained failed: rows=16 end-to-end `0.488224 ms` vs dense `0.139152 ms` (`0.285017x`), rows=128 `0.583680 ms` vs dense `0.163840 ms` (`0.280702x`). The v188 v2 adapter is selected (`prefill_metadata_bytes=0`) and improves over v192/v194 for rows=128, but it still does not meet the required `>=0.95x` speedup. Exact Qwen full-model quality/throughput and patched vLLM remained unavailable because no exact model candidate was mounted. Terminal decision: no_go. v196 is retained only as an honest timing baseline, not as a production success.

## v197 — supported SM75 three-stage pipeline depth experiment (pending Kaggle)
Deep research compared the original MixLLM autotuned staged launcher with the current SM75 helper. The original dispatch supports stage 5/11 configurations, while the SM75 port hard-coded both `DefaultMmaCore` and `MQMmaPipelinedSm75` to two stages. v197 preserves the known-compilable v188 geometry (`32x128x64` threadblock, `32x32x64` warp, `8x8x16` instruction) and changes only the pipeline depth to three stages by parameterizing the runner. It does not use the rejected unsupported `64x128x64` geometry, does not enable cp.async, and does not change benchmark settings, arithmetic, model, quality gates, or v2/v3 ABIs. Twenty-eight local source/audit/runtime/vLLM contracts, Python compilation, notebook compilation/freshness, and diff checks pass. Kaggle T4 compilation, correctness, timing integrity, and performance are pending.

## v198 — supported stage-2 core with separate three-stage pipeline (pending Kaggle)
v197 failed CUDA compilation because the vendored SM75 `DefaultMmaCore` specialization exists only for stage 2 and therefore provides no stage-3 `SmemLayoutA`/`SmemLayoutB`. Deep research confirms that v198 must preserve the supported v188 core stage parameter (`DefaultMmaCore[...,2,...]`) while testing only `MQMmaPipelinedSm75[...,Stages=3]` through the parameterized runner. This keeps the same `32x128x64 / 32x32x64 / 8x8x16` geometry, synchronous SM75 copies, arithmetic, ABI, benchmark settings, and all quality/memory/timing gates. Twenty-nine local contracts, Python compilation, notebook rebuild/freshness, and diff checks pass. Kaggle T4 compilation and runtime correctness/performance are pending.

## v198 result — three-stage pipeline compiled but failed correctness
Kaggle version 198 compiled and ran on Tesla T4. Timing integrity passed and the supported stage-2 `DefaultMmaCore` plus separate `MQMmaPipelinedSm75` stage-3 experiment was selected. However, native correctness failed at large M: mixed rows=128 `max_abs_error=37.6042`, pure INT4 rows=128 `37.9927`, and pure INT8 rows=128 `41.0668`; the stage-3 pipeline's shared-memory/metadata lifecycle is not functionally correct. Mixed prefill remained slow: rows=16 `0.2840x`, rows=128 `0.2843x`. Terminal decision: no_go. v198 is rejected and must not be used as a performance baseline.

## v199 — supported 64x128 stage-2 wide-core dispatch (pending Kaggle)
Deep research found that original MixLLM autotuning includes `64x128` large-M configurations while the SM75 port used only `32x128`. v199 keeps the supported v188-compatible `32x128x64 / 32x32x64 / 8x8x16 / stage=2` runner for rows<64 and adds a second supported `64x128x64 / 32x32x64 / 8x8x16 / stage=2` runner for rows>=64. This explicitly avoids the rejected v193 five-stage/16x8x32 core and the correctness-failing v198 three-stage pipeline. It changes only large-M threadblock geometry and host dispatch; arithmetic, quantization, ABI, benchmark settings, telemetry, timing-integrity, quality, and memory gates remain unchanged. Thirty local contracts, Python compilation, notebook freshness, and diff checks pass. Kaggle T4 compilation, correctness, and performance are pending.

## v199 result — wide 64x128 stage-2 core compiled but was rejected
Kaggle version 199 compiled and native correctness passed. The rows>=64 64x128 stage-2 runner did not restore v188 speed: mixed rows=128 measured `0.2671x` E2E speedup versus FP16, below v196's `0.2807x`. Mixed rows=16 remained on the narrow path but timing-integrity failed with ratio `1.2554`; the prefill gate failed. Decode, native correctness, embedded contracts, and T4 hardware gates passed, but timing-integrity, operator-production, vLLM production, and mixed-prefill gates did not. Terminal decision: no_go. The wide-core dispatch is rejected.

## v200 candidate — upstream-style concurrent integer prefill streams
Deep research completed before code changes in `research-15/prefill_deep_research_v200.md`. The upstream MixLLM source does not fuse INT4 and INT8 into one instruction; it uses two auxiliary streams and event joins to overlap its two staged integer GEMMs. The current SM75 large-M path serialized INT4 CUTLASS, INT8 CUTLASS, and FP16 WMMA after materializing the INT4 expansion cache. v200 therefore reverts the rejected v199 `WideCore`/`WideInt8Runner` dispatch and preserves the validated narrow `32x128x64` stage-2 core, while adding a private per-device `IntegerPrefillStreams` adapter. For rows>=32, INT4 and INT8 CUTLASS launches wait on a caller-stream fork event and run concurrently on persistent nonblocking streams; the existing FP16 WMMA kernel stays on the caller stream and the caller stream joins both completion events before return. All integer input/weight/metadata/index/output tensors are allocator-recorded on their auxiliary streams. Fallback metadata transposes are prepared before the fork event. No benchmark settings, model, arithmetic precision, packed ABI, or quality thresholds were changed.

Local validation before Kaggle submission: `test_sm75_source.py` 11/11 passed; `test_vllm_three_level.py` 6/6 passed; `test_runtime_capability.py` 7/7 passed; `test_v51_audit_contract.py` 6/6 passed; notebook build and freshness check passed; `git diff --check` passed. Kaggle T4 validation is still required; this candidate is not accepted until every gate including timing-integrity and mixed-prefill passes.

## v200 result — integer stream overlap compiled and passed correctness, but prefill target remains unmet
Kaggle version 200 compiled on Tesla T4. Native correctness, mixed decode GEMM, mixed decode E2E, embedded contracts, T4 hardware, and timing-integrity all passed. The upstream-style concurrent INT4/INT8 auxiliary-stream adapter improved mixed rows=128 to `0.3072x` E2E speedup versus FP16 (`0.53864 ms` vs `0.16549 ms`), compared with v199's `0.2671x`, but remains far below the required `0.95x` gate and the `2.6x` project target. Mixed rows=16 measured `0.2829x` and also failed prefill. The stream overlap is therefore a real safe improvement over v199 but not sufficient; it is not a production GO. Expanded INT4 memory remains `8,601,600` bytes at rows=128, confirming that launch overlap did not remove the main expansion cost. Terminal decision: no_go. Preserve the stream adapter as a safe improvement and target direct packed-INT4 consumption or another expansion-elimination design next.
