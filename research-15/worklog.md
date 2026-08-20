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

## v201 candidate — direct packed INT4 staging with v200 stream overlap
Deep research completed in `research-15/prefill_deep_research_v201.md`. Upstream MixLLM prepares an SM80-specific interleaved INT4 layout, while the current SM75 runner uses signed INT8 WMMA after expansion. v201 keeps the supported SM75 signed-INT8 arithmetic but decodes packed nibbles and subtracts per-group zeros only while filling the existing shared-memory B tile. The host adapter returns the ABI-compatible empty expansion sentinel only for the mixed rows>=32 4/8/16 path; pure INT4, pure INT8, decode, rows<32, and all v200 stream/event behavior remain unchanged. The INT8 partition still uses the v200 auxiliary stream, the direct packed INT4 partition uses the INT4 auxiliary stream, and FP16 remains on the caller stream. This removes the large-M `[n4,K]` INT8 allocation without changing the checkpoint ABI, model, quantization arithmetic, quality checks, or benchmark settings.

Local validation: `test_sm75_source.py` 12/12 passed; `test_vllm_three_level.py` 6/6 passed; `test_runtime_capability.py` 7/7 passed; `test_v51_audit_contract.py` 6/6 passed; notebook build/freshness passed; `git diff --check` passed. Kaggle T4 correctness and performance are pending; v201 is not accepted until every gate passes.

## v201 result — direct packed INT4 path was correct but substantially slower
Kaggle version 201 compiled on Tesla T4 and passed native correctness, mixed decode GEMM/E2E, embedded contracts, T4 hardware, and timing-integrity. The direct packed-INT4 kernel did not improve prefill: mixed rows=16 measured `0.2837x`, while rows=128 regressed sharply to `0.0646x` E2E speedup (`2.5308 ms` versus `0.1635 ms` FP16), compared with v200's `0.3072x`. The prefill gate failed and terminal decision is `no_go`. The implementation's arithmetic is accepted by the native correctness gate, but its per-tile unpack plus non-pipelined WMMA/shared-memory staging is too expensive. The report still showed `expanded_int4_bytes=8,601,600` in the benchmark telemetry, so the cache measurement is not a reliable proof that the direct route removed all observed expansion across the full benchmark lifecycle. Reject v201's direct kernel as a performance path; preserve v200's auxiliary-stream overlap and investigate an iterator-level packed INT4 load or another approach that retains CUTLASS pipelining.

## v202 candidate — CUTLASS-pipelined packed INT4 iterator
Deep research completed in `research-15/prefill_deep_research_v202.md`. Upstream MixLLM keeps packed `uint4b_t` weights inside a staged CUTLASS global-to-shared pipeline, while the current SM75 path expanded INT4 to `[N,K]` INT8 before invoking a supported INT8 core. Vendored SM75 exposes homogeneous 4-bit MMA but not the upstream SM80 mixed INT8xINT4 instruction, so v202 avoids unsupported mixed `16x8x32` geometry. It adds a `PackedInt4Iterator` that reuses the validated `32x128x64 / 32x32x64 / 8x8x16` stage-2 core and decodes packed nibbles during the existing synchronous CUTLASS staging. The v200 two-stream overlap remains unchanged. Only the large-M mixed prefill path with an empty expansion sentinel selects `PackedInt4Runner`; all other paths retain the validated v200 runner.

Local validation passed: `test_sm75_source.py` 12/12; `test_vllm_three_level.py` 6/6; `test_runtime_capability.py` 7/7; `test_v51_audit_contract.py` 6/6; notebook generation and freshness check; `git diff --check`. The candidate is not accepted until Kaggle T4 compilation, native correctness, timing-integrity, decode, and mixed-prefill gates all pass.

## v202 result — pipelined packed iterator preserved correctness but did not hide unpack cost
Kaggle version 202 compiled on Tesla T4 and passed native correctness, mixed decode GEMM/E2E, embedded contracts, T4 hardware, and timing-integrity. The CUTLASS-pipelined packed-INT4 iterator did not improve large-M prefill: rows=128 measured `0.0656x` E2E speedup (`2.4964 ms` versus `0.1638 ms` FP16), essentially the same catastrophic regression as v201's standalone packed kernel (`0.0646x`) and far below v200's expanded-INT4 stream path (`0.3072x`). Rows=16 remained on the expanded control path at `0.2850x`; mixed prefill failed. Terminal decision: `no_go`. The iterator's scalar nibble decode inside `get()` remains too expensive even when called from the existing stage-2 pipeline, so reject the packed iterator as a production path and retain only v200's safe stream overlap. Future work must move conversion to a genuinely vectorized device primitive or use a native supported packed-int4 MMA data type/layout, not per-element iterator decode.

## v203 candidate — vectorized INT4 expansion on the accepted v200 path
Deep research completed in `research-15/prefill_deep_research_v203.md`. v201/v202 showed that scalar nibble decoding, whether standalone or inside the CUTLASS iterator, dominates large-M prefill. NVIDIA's documented `__byte_perm` and `__vsub4` intrinsics provide a four-byte vector path, and the existing decode kernel already establishes four-nibble-at-a-time signed arithmetic. v203 therefore changes only `expand_int4_sm75_kernel`: one thread loads one aligned 32-bit packed word, interleaves its low/high nibbles with two `__byte_perm` calls, subtracts the repeated zero point with two `__vsub4` calls, and writes eight exact signed INT8 values. The expanded INT8 CUTLASS runner and v200 auxiliary streams remain unchanged; no benchmark, quality, model, or precision policy changes. The host launch now schedules one thread per eight output values.

Local validation passed: source contracts 12/12; vLLM contracts 6/6; runtime capability 7/7; v51 audit 6/6; notebook build/freshness; `git diff --check`; a saved deterministic host model verified the byte-permutation and signed zero-point mapping over 10,000 randomized cases. Kaggle T4 validation is required before acceptance.

## v203 result — vectorized expansion did not pass timing integrity or prefill
Kaggle version 203 compiled on Tesla T4 and passed native correctness, mixed decode GEMM/E2E, embedded contracts, and T4 hardware. Rows=16 remained on the expanded control path at `0.2819x` E2E speedup, essentially unchanged from v200. Rows=128 measured `0.0658x` E2E speedup (`2.4892 ms` versus `0.1638 ms` FP16), so the vectorized expansion did not repair the dominant large-M CUTLASS path. The timing-integrity guard failed at rows=128 (`ratio=1.2203`; reported GEMM `3.0377 ms` exceeded E2E `2.4892 ms`), making that measurement non-actionable as a speed claim. Mixed prefill and timing-integrity gates failed; terminal decision: `no_go`. Reject v203 as a production path. Preserve v200's stream overlap and return to the native homogeneous SM75 INT4 decomposition research; any next candidate must first eliminate the unsupported scalar conversion bottleneck and must never accept physically impossible timing.

## v204 result — inherited native WMMA INT4 proposal rejected before Kaggle
Deep research compared the public upstream `microsoft/MixLLM` implementation with the v200 SM75 port. The original uses two auxiliary streams for separate SM80 INT4/INT8 staged GEMMs, with `16x8x32` mixed-input upcast on SM80; it does not provide a direct SM75 native s4/u4 WMMA implementation. The proposed v204 alias was audited against the vendored CUTLASS sources before implementation. It is invalid for three independent reasons: `mq_mma_pipelined_sm75.h` requires `Shape::kK == 64` for its 128-element metadata schedule, while the proposal used K=128; `wmma_sm75.h` has an int4-by-int4 WMMA specialization but no signed-int4-by-unsigned-int4 WMMA specialization; and the WMMA warp operator has no `transform()` method used by the custom pipeline. The lower-level `mma_sm75.h` does expose the exact s4*u4 instruction, but using it requires a new K=64 custom TensorOp/decomposition path and an independent zero-point proof. No invalid runner was submitted or benchmarked. The tracked source was restored to the v200 baseline; only the research note records the rejected design. Terminal decision: no_go_before_kaggle.

## v204 candidate — lower-level native s4/u4 decomposition submitted
After rejecting the inherited invalid WMMA snippet, the same next Kaggle kernel version was rebuilt with a corrected K=64 lower-level SM75 TensorOp path. It uses `u4*u4` for packed low nibbles and `s4*u4` for signed high nibbles, with the exact identity `a = low + 16*high`; the existing scale pipeline is reused with `ApplyWeightZero=false`, and a caller-stream correction epilogue applies `16*high_product - z*a` exactly. The v200 INT8 stream path remains the fallback when the native tensors are absent, while large-M INT4 partitions opt into the native path. Source contracts (12/12), vLLM contracts (6/6), runtime capability contracts (7/7), v51 audit contracts (6/6), independent CPU decomposition/correction reference, Python syntax checks, notebook rebuild, freshness validation, and diff checks passed. Kaggle kernel version 204 compiled on T4, but the terminal decision was no-go as recorded below.

Kaggle monitoring note: the public notebook webpage returned a generic “page not found” view in the sandbox browser, while the connected Windows Python Kaggle CLI reported the kernel as RUNNING until completion. No benchmark result was inferred from the webpage; CLI status remained authoritative.

## v204 result — lower-level native s4/u4 compiled but failed the production gates
Kaggle version 204 compiled on Tesla T4 (`sm_75`) and all 70 embedded tests passed. Native correctness passed, and mixed decode GEMM/E2E passed. The native large-M INT4 path did not solve prefill: the mixed Qwen-shaped rows=16 E2E ratio was `3.4846x` versus FP16 (`0.2870x` speedup), and rows=128 was `15.7875x` (`0.0633x` speedup) with GEMM ratio `20.0408x`; rows=128 timing-integrity also failed at `1.2694`. Pure INT4 rows=128 measured `1.5592x` E2E ratio (`0.6414x` speedup), still slower than FP16. The candidate’s max operator error remained within the existing native correctness gate (`0.1025` mixed rows=128), so this is a performance/launch-design failure rather than an immediate arithmetic-correctness failure. The reported production gates were `mixed_prefill_end_to_end_performance=failed`, `timing_integrity=failed`, `operator_production=failed`, and `model_vllm_production=failed`; terminal decision: `no_go`. Full Qwen model quality and vLLM apply were unavailable in the Kaggle environment, so no production claim is made. The native decomposition is rejected and must not remain in the production branch; preserve only v200’s safe stream-overlap source baseline and the research artifacts.

## v205 candidate — v200 stream overlap plus guarded wide large-M core
Deep research compared upstream MixLLM’s large-M geometry table with v199/v200. The rejected v204 native s4/u4 decomposition was removed. This candidate kept v200’s auxiliary-stream overlap, metadata caching, small-M `Core`, and all benchmark settings unchanged; it added only v199’s supported `64x128x64 / 32x32x64 / 8x8x16 / stage=2` `WideCore`, selected inside `run_cutlass_int_partition` only when `rows >= 64`. Prediction: rows=16 should remain on the v200 path and preserve timing-integrity, while rows=128 may improve. Local source contracts (11/11), vLLM contracts (6/6), runtime contracts (7/7), v51 audit contracts (6/6), notebook rebuild/check, and diff checks passed.

## v205 result — wide-core hybrid still failed mixed prefill
Kaggle kernel version 205 compiled on T4 and all embedded tests passed. The hybrid preserved timing-integrity (`passed`) and native correctness (`passed`), but mixed Qwen-shaped prefill remained far outside the gate: rows=16 E2E ratio `3.5095x` versus FP16 (`0.2849x` speedup), rows=128 E2E ratio `15.5597x` (`0.0643x` speedup), and rows=128 GEMM ratio `17.0593x`. This is essentially the same catastrophic large-M behavior as v204; the guarded `64x128` core did not address the dominant cost. Decode gates remained passed. Terminal decision: `no_go`; restore v200 and do not retain WideCore.

## v206 candidate — tiled mixed integer DP4A prefill
Deep research separated the dominant cost: v200’s mixed large-M path launches separate INT4 and INT8 staged GEMMs that each reread/stage the same activation, while the rejected v160 DP4A path used one thread per output and had catastrophic occupancy. v206 adds one guarded mixed integer kernel for the existing rows>=32 mixed 4/8/16 path: an 8-row x 32-channel CTA loads each 8x128 INT8 activation tile once per group into shared memory, reuses it across both INT4-expanded and INT8 weight rows, performs the same 32 `__dp4a` operations and per-group scale multiplication, and scatters through the unchanged index tensors. FP16 remains on its existing caller-stream kernel; pure and small-row paths retain v200. No model, quantization arithmetic, precision, quality threshold, benchmark shape, or gate logic changed. Local source contracts (12/12), vLLM (6/6), runtime (7/7), v51 audit (6/6), compileall, notebook build/freshness, and diff checks passed. Pending one identical Kaggle T4 measurement; terminal decision: pending.

## v206 result — tiled mixed DP4A was correct but not competitive
Kaggle kernel version 206 compiled and ran on Tesla T4. Native correctness, both decode gates, embedded contracts, allocator checks, and T4 hardware passed. Mixed rows=16 stayed on the v200 control path at `0.2858x` E2E speedup; rows=128 mixed E2E was `0.0655x` with GEMM `0.0533x`, and timing-integrity failed at `1.2288`. The tiled DP4A kernel therefore did not solve the prefill bottleneck and was substantially worse than v200’s `0.3072x` rows=128 result. The arithmetic remained correct (`max_abs_error=0.10254`), so the failure is launch/memory throughput rather than quality. Terminal decision: `no_go`; reject v206 and do not retain its dispatch.

## v207 candidate — staged-weight tiled mixed DP4A
Deep research of v206’s T4 failure found the remaining concrete memory defect: v206 reused the activation tile but each output thread still loaded its own weight row directly with a width-strided access pattern. v207 keeps v206’s exact arithmetic and 8-row x 32-channel launch but cooperatively stages a contiguous 32x128 signed-INT8 weight tile in shared memory per group, so eight row warps reuse the same weights without repeated global loads. It remains guarded to mixed rows>=32 only; v200 remains the fallback for pure/small paths, FP16 remains separate, and all benchmark/model/quality/gate settings are unchanged. Local source contracts (12/12), vLLM (6/6), runtime (7/7), v51 audit (6/6), compileall, notebook build/freshness, and diff checks passed. Pending one identical Kaggle T4 measurement; terminal decision: pending.

## v207 result — staged weights improved timing validity but remained far too slow
Kaggle kernel version 207 compiled and executed on Tesla T4. Native correctness, both decode gates, embedded contracts, allocator checks, T4 hardware, and timing-integrity all passed. Mixed rows=16 remained on the control path at `0.2846x` E2E speedup. Mixed rows=128 improved over v206’s timing-invalid result but remained catastrophic at `0.0633x` E2E and `0.0647x` GEMM speedup; the prefill gate failed by a wide margin. Staging the 32x128 weight tile therefore did not overcome the one-thread-per-output DP4A throughput limitation. Terminal decision: `no_go`; reject v207 and restore v200.

## v208 candidate — cached combined INT8 cuBLAS control
Deep research after v207 confirmed that the tiled DP4A family cannot reach the T4 FP16 baseline. v208 adds a separate unchecked CUDA ABI backed by `cublasGemmStridedBatchedEx` with signed INT8 inputs and INT32 accumulation. Python caches the zero-subtracted expanded INT4 rows concatenated with the existing INT8 rows for mixed rows>=32; the CUDA path computes one batched GEMM per 128-element group, applies the exact existing activation/weight scales in a CUDA epilogue, and scatters through the original index tensors. The v200 stream-overlap path remains the fallback when the control operator is absent or the mixed cache is empty. The combined weight storage is reported explicitly. No model, quantization arithmetic, precision, quality, benchmark, or gate setting changed. Local source contracts (12/12), vLLM (6/6), runtime (7/7), v51 audit (6/6), Python syntax, notebook build/freshness, and diff checks passed. Pending one identical Kaggle T4 measurement; terminal decision: pending.

## v208 submission audit — stale artifact, not a valid measurement
The first submission labeled v208 produced a gate report with byte-identical v207 provenance and timings, proving that the connected Windows notebook copy was stale and the cuBLAS source was not executed. This is not counted as a code measurement or performance result. The corrected notebook was recopied and verified by matching the sandbox byte size; the next submission is therefore labeled v209.

## v209 candidate — verified cached combined INT8 cuBLAS control
Same implementation as the intended v208 control, now with the notebook copy corrected: cached zero-subtracted expanded INT4 rows concatenated with INT8 rows, batched signed-INT8 cuBLAS GEMM with INT32 accumulation, exact scale/scatter epilogue, explicit `-lcublas` linkage, and honest combined-weight memory telemetry. All existing v200/v3 fallbacks remain intact. Local source contracts, Python syntax, regression tests, notebook rebuild/freshness, and diff checks passed. Pending one valid Kaggle T4 measurement; terminal decision: pending.

## v209 result — corrected notebook copy still embedded stale v207 source
The downloaded v209 gate report is internally consistent and timing-integrity passed for all rows, but its source manifest hashes are the v207 hashes (`three_level_sm75.cu` `ac36e1ff...` and `sm75_backend.py` `7499c183...`), not the local v208 cuBLAS hashes. Therefore v209 is not a valid measurement of the intended cuBLAS candidate; it is a stale-source rerun of the v207 implementation and is rejected as a candidate measurement. Its observed results were rows=1 E2E `1.1658x`, GEMM `1.2682x`; rows=16 E2E `0.2852x`, GEMM `0.2817x`; rows=128 E2E `0.0632x`, GEMM `0.0625x`; timing-integrity passed with ratios `0.9193`, `1.0125`, and `1.0112`. The rows=128 prefill gate failed catastrophically. The local notebook does contain the v208 cuBLAS symbols and current local hashes, so the next action is to commit the candidate and rebuild/copy the notebook again before a valid v210 submission.

## v210 result — cuBLAS control rejected at CUDA compilation
Kaggle v210 used the corrected notebook provenance, but the extension did not compile. NVCC reported `n4`, `n8`, and `width` undefined in the combined INT8 validation block of `three_level_sm75.cu`; no benchmark or gate report was produced. This is a source correctness failure, not a performance result. The cuBLAS hot path is rejected and will not be retained.

## v211 candidate — native combined SM75 tile with cuBLAS-derived layout and heuristic
Deep research compared original MixLLM’s persistent two-stream staged launch and shape-family selection with NVIDIA’s cuBLASLt principles: explicit leading dimensions/layouts, reusable shape-specific plans, and geometry/resource heuristics. v211 removes the cuBLAS dependency and reuses the legal local SM75 `32x128x64 / 32x32x64 / 8x8x16 / stage=2` CUTLASS Tensor Core runner for one combined signed-INT8 integer partition. Python caches a row-major `[n4+n8, K]` weight layout, `[groups, n4+n8]` scales, zero-filled metadata, and concatenated output indices. The native launcher queues this single staged integer tile on the existing auxiliary stream topology and lets the exact FP16 partition overlap on the caller stream; it avoids cuBLAS’s per-group INT32 workspace and separate scale/scatter epilogue. A deterministic heuristic enables the combined tile only for `rows>=32`, both integer partitions of at least 32 channels, and at least 128 combined integer channels; smaller cases retain the measured v188/v200 fallback paths. v210’s compile bug is fixed, source contracts and Python syntax pass, but PyTorch-dependent local tests are unavailable in the sandbox and Kaggle T4 compilation/correctness/performance is mandatory. Terminal decision: pending.

## v211 result — native combined tile compiles and is correct, but is slower than v200
Kaggle v211 compiled and executed on Tesla T4. Provenance matched the native source commit. Native SM75 correctness, decode GEMM/E2E, embedded contracts, allocator checks, T4 hardware, and timing-integrity passed. The native combined-layout cache was selected for mixed rows=128 and reported `combined_native_layout_bytes=12102912`. Mixed QKV results were rows=1 E2E `0.9900x`, rows=16 E2E `0.2860x`, and rows=128 E2E `0.1987x` with GEMM `0.2005x`; timing-integrity ratios were `0.7693`, `1.0116`, and `0.9911`. Thus the native tile recovered the catastrophic v209 cuBLAS-control failure (`0.0632x` rows=128) but remained slower than the accepted v200 rows=128 result (`0.3072x`) and failed the required prefill gate. The combined single-stream tile is rejected; retain v200 as the production baseline. Full-model Qwen quality/throughput and patched-vLLM remained unavailable because the exact model/runtime were absent. Terminal decision: `no_go`.

## v212 candidate — v200 overlap plus isolated 32x256 per-partition tile
Fresh deep research compared v200’s two-stream launch with original MixLLM’s shape-family selection and NVIDIA’s documented layout/heuristic/stream semantics. v212 restores v200’s source and preserves separate INT4, INT8, and FP16 execution; it does not use the rejected v211 combined layout or any cuBLAS hot path. It adds only a guarded `32x256x64 / 32x64x64 / 8x8x16 / stage=2` SM75 CUTLASS runner selected independently for `rows>=32 && channels>=256`, while smaller partition shapes use the exact v200 `32x128x64` runner. The layout remains row-contiguous weights consumed with explicit CUTLASS leading dimensions, and the stream fork/join is unchanged. Local source contracts (11/11), Python compilation, and diff checks passed; local PyTorch-dependent execution is unavailable. Kaggle T4 compile, native correctness, timing-integrity, and performance are pending. Terminal decision: pending.

## v212 result — isolated 32x256 per-partition tile rejected
Kaggle v212 compiled and ran on Tesla T4 with native correctness passed, but the new wide per-partition tile did not improve the target. Mixed QKV rows=1 E2E was `1.2269x`; rows=16 E2E was `0.3200x` with timing-integrity ratio `1.1631` (failed); rows=128 E2E was `0.2616x` and GEMM `0.2708x`, both below v200 rows=128 E2E `0.3072x`. The 32x256 tile therefore regressed large-M performance and caused a timing-integrity failure on rows=16; `mixed_prefill_end_to_end_performance`, `timing_integrity`, and `operator_production` failed, terminal decision `no_go`. Restore v200 and do not retain the wide geometry. The result shows that importing another original-MixLLM tile family member is not enough on SM75; the dominant bottleneck remains the local staged integer dataflow/resource balance.

## v213 candidate — v200 overlap plus isolated K=128 mainloop tile
Fresh deep research of original MixLLM’s K=128 geometry family and NVIDIA CUTLASS’s tile/occupancy guidance led to an isolated experiment: `32x128x128 / 32x64x64 / 8x8x16 / stage=2`, selected independently for each integer partition only when `rows>=32 && channels>=128`. v200’s separate INT4 and INT8 streams, caller-stream FP16 path, arithmetic, fallback geometry, and benchmark settings are unchanged. The hypothesis is fewer K-mainloop iterations; the explicit risk is higher shared-memory/register pressure. Local source contracts (11/11), Python compilation, and diff checks passed. Kaggle T4 validation is pending; terminal decision: pending.

## v213 result — K=128 candidate initially failed compilation
The first K=128 submission reached the SM75 CUDA compiler but failed at `mq_mma_pipelined_sm75.h:423` because the custom dequantizing pipeline had an explicit `static_assert(Shape::kK == 64)`. No benchmark or gate result exists for the unpatched v213 source. Deep inspection showed that the assertion protected metadata advancement for 128-element quantization groups: K=64 consumes a group in two halves, while K=128 should consume one complete group. v214 applies a guarded metadata-copy extension for K=128 and retains the exact K=64 behavior; it is a new compile-and-correctness hypothesis, not a performance result.

## v214 candidate — guarded K=128 metadata pipeline
v214 keeps v200’s independent INT4/INT8/FP16 stream topology and v213’s isolated `32x128x128 / 32x64x64 / 8x8x16 / stage=2` runner. It changes only the custom SM75 metadata pipeline: K=64 keeps the original every-two-half-group advancement, while K=128 advances one metadata row per complete 128-element group. Local source contracts (12/12), Python compilation, and diff checks pass. Kaggle T4 compilation, native correctness, timing-integrity, and performance are pending.

## v215 candidate — v200 two-stream path with cached metadata through a private v2 ABI
Deep source comparison found that v200’s measured large mixed adapter preserved the fast unchecked two-stream core but bypassed the existing transposed metadata cache. v215 adds a private `_three_level_linear_v2_cached_unchecked` operator that calls the same v200 core with cached `[groups, channels]` scale/zero layouts; Python uses it only after partition validation, only for rows>=32, and falls back to the original `_three_level_linear_v2_unchecked` if the operator is unavailable. No arithmetic, partition mapping, stream dependencies, benchmark shape, quality threshold, or public v2/v3 ABI changed. Local source contracts (13/13), Python compilation, and diff checks pass. The PyTorch backend suite cannot run in this sandbox because the sandbox Python environment has no `torch`; it remains a required final-environment check before Kaggle. Kaggle is intentionally not run during this repair phase.

## v216 candidate — cached v200 core plus guarded K=128 SM75 pipeline
v216 combines the v215 private cached-metadata v2 adapter with the previously audited isolated K=128 per-partition runner. The K=64 metadata path remains unchanged; K=128 copies one metadata row and advances one 128-element group per tile. INT4 and INT8 remain independent auxiliary-stream launches and FP16 remains on the caller stream. Local source contracts (13/13), Python compilation, and diff checks pass. The full backend tests are blocked only by the sandbox Python environment lacking `torch`; no Kaggle execution has been performed during repair.

## v217 candidate — contract correction for isolated K=128 plus cached v2
The first v216 source-contract run correctly exposed a test mistake: the legacy v200 overlap contract forbade every `WideInt8Runner` symbol, which conflicts with the intentionally isolated per-partition K=128 candidate. The contract now forbids only combined/cublas execution while explicitly requiring independent K=64/K=128 runners and both auxiliary-stream waits. Source contracts (13/13), Python compilation, and diff checks pass. The backend runtime suite remains unavailable locally because `torch` is not installed; no Kaggle run has been made.

## v218 candidate — local test-harness and optional-op import repairs
The complete local discovery suite initially exposed legacy defects unrelated to GPU timing: six test modules parsed unittest discovery arguments during import, and `ops.py` unconditionally registered fake implementations for an optional `kernels_mixllm` extension that was absent in the CPU environment. v218 switches those script tests to `parse_known_args()` and registers fake kernels only when the optional operator exists. Direct-script behavior for known flags is preserved. After the repair, full local discovery passes 77 tests with 6 explicit CUDA-only skips; all SHMQ source contracts, CPU reference/quality tests, cache tests, vLLM patch contracts, Python compilation, and diff checks pass. No Kaggle run has been made.

## v219 candidate — fix K=128 mainloop double-consumption after final T4 no-go
The final v218-source Kaggle run compiled but failed `sm75_native_correctness` at rows=128 with `max_abs_error ≈ 357.7` and failed timing-integrity (`1.547`). Deep comparison isolated the bug: `MQMmaPipelinedSm75::gemm_iters` unconditionally called `mac_loop_iter` twice even though each K=128 iteration already consumes a full 128-wide tile. v219 guards the second call with `if constexpr (Shape::kK == 64)`, preserving v200 K=64 batching and making K=128 single-call. The cached-v2 metadata adapter is unchanged. Full local discovery still passes 77 tests with 6 CUDA-only skips, source contracts pass, Python compilation passes, and no new Kaggle run has been made yet.

## v219 result — K=128 single-call repair rejected
The v219 Kaggle kernel 216 completed on Tesla T4/SM75. The mainloop guard fixed timing-integrity: it passed for all measured scenarios, including rows=128 mixed at ratio 0.839. However native correctness remained broken exactly in the K=128 dispatch region: rows=128 mixed `max_abs_error=340.5318`, pure INT4 `356.4450`, and pure INT8 `342.0877`; rows=1 and rows=16 remained correct. The gate report marked `sm75_native_correctness=failed`, `mixed_prefill_end_to_end_performance=failed`, `operator_production=failed`, and `terminal_decision=no_go`. The rows=128 mixed E2E speedup of 0.3808x versus dense FP16 is inadmissible because correctness failed, even though it exceeded v200's approximately 0.307x result. The vLLM/model production gates were unavailable in the Kaggle image because vLLM 0.9.0 and the exact Qwen model environment were absent.

Decision: reject K=128 production dispatch. Restore unconditional v200 `Int8Runner` K=64 dispatch, retain cached-v2 metadata and v200 stream overlap, and leave K=128 aliases/guarded mainloop unselected for future isolated debugging. Deep-research record: `research-15/deep_research_v220_k128_rollback.md`.

## v220 result — cached-v2 K=64 path rejected as a production improvement
The v220 Kaggle kernel 217 restored unconditional v200 K=64 `Int8Runner` dispatch while retaining the v215 cached-v2 metadata adapter. T4 native correctness passed, including rows=128 mixed `max_abs_error=0.10254`, pure INT4 `0.000193`, and pure INT8 `0.000198`; all embedded contracts and T4 hardware checks passed. However timing-integrity failed for rows=128 mixed at ratio `1.4156`, and the mixed-prefill E2E gate failed. Rows=128 mixed E2E speedup was `0.2686x`, below the accepted v200 result of approximately `0.3072x`; GEMM speedup was `0.1897x`. The gate report therefore marked `timing_integrity=failed`, `mixed_prefill_end_to_end_performance=failed`, `operator_production=failed`, and `terminal_decision=no_go`. The vLLM/model gates remained unavailable in the Kaggle image.

Decision: reject cached-v2 as a production change together with v220. Restore the complete accepted v200 implementation, not merely its K=64 runner, and use that as the clean baseline for the next original-first investigation.

## v221 result — clean v200 production baseline confirmed
The v221 Kaggle kernel 218 embedded the complete v200 SM75 production files, with only the unrelated v218 local-test import repairs retained. T4 native correctness, timing-integrity, embedded contracts, allocator checks, and hardware checks passed. Mixed decode GEMM/E2E passed. Mixed QKV rows=128 prefill measured E2E speedup `0.261866x`, GEMM speedup `0.264397x`, max error `0.102539`, and timing-integrity ratio `0.990428`; the mixed-prefill performance gate consequently failed and terminal decision was `no_go`. The run confirms the clean v200 topology is numerically and temporally safe but remains far below the required `2.6x` target (and below the earlier v200 run's `0.307233x` due T4 run-to-run variance). It is the accepted functional baseline, not a performance success.

Decision: keep the clean v200 code as the baseline. Begin v222 with a fresh original-first investigation of the dominant large-M prefill bottleneck; do not reintroduce cached-v2 or K=128 without a new independent correctness proof.

## v222 candidate — upstream-inspired matching five-stage K=64 runner for rows>=128
Fresh comparison of upstream MixLLM found that its equivalent K=64/128-channel column-major fallback uses `NumStages=5`, while clean v200's SM75 runner uses a matching `DefaultMmaCore` and `Runner` with `Stages=2`. v222 adds a separate `CoreStage5`/`Int8RunnerStage5` pair with the same `GemmShape<32,128,64>` and selects it only for rows>=128; rows below 128 retain the exact v200 stage-2 runner. The two-stream INT4/INT8 overlap, FP16 caller-stream path, arithmetic, metadata ABI, quantization, quality checks, and benchmark settings are unchanged. Local discovery passes 75 tests with 6 CUDA-only skips; Python compilation and diff checks pass. Kaggle T4 validation is pending; terminal decision: pending.

## v222 result — five-stage candidate rejected at compile time
Kaggle kernel 219 reached the extension build but failed before any benchmark ran. The T4 compiler reported that `DefaultMmaCore<..., OpClassTensorOp, 5, OpMultiplyAddSaturate>` is an incomplete/unsupported type in the vendored SM75 CUTLASS header. Inspection of `default_mma_core_sm75.h` confirms every SM75 TensorOp `DefaultMmaCore` specialization is explicitly provided only for `NumStages=2`; the upstream stage-5 choice cannot be transplanted by changing this template argument. No correctness or performance result is claimed. Revert v222 completely and retain clean v200 as the baseline. The failure is informative: future stage-depth work would require a deliberate custom SM75 core specialization, not a production alias substitution.

## v223 candidate — force 100% shared-memory carveout for the existing SM75 staged runner
Fresh original-first research found that clean v200's SM75 runner uses dynamic shared-memory staging but requests the 100% shared-memory carveout only when its allocation is at least 48 KiB. v223 preserves the exact v200 K=64/2-stage geometry, runner, two-stream overlap, metadata, arithmetic, and benchmark settings, but applies `cudaFuncAttributePreferredSharedMemoryCarveout=100` to every staged runner invocation. The max dynamic shared-memory attribute remains conditional and unchanged. Local discovery passes 75 tests with 6 CUDA-only skips; Python compilation and diff checks pass. Kaggle T4 validation is pending; terminal decision: pending.

## v223 result — shared-memory carveout rejected
Kaggle kernel 220 compiled and ran on Tesla T4. Native correctness, embedded contracts, allocator checks, hardware checks, and decode GEMM/E2E passed. The forced 100% shared-memory carveout did not improve the large-M path: mixed rows=128 measured E2E speedup `0.314809x` but GEMM speedup only `0.184514x`, and timing-integrity failed at ratio `1.706149`. The mixed-prefill performance gate and timing-integrity gate failed, so operator production and terminal decision were `no_go`. The candidate is rejected; the carveout preference is not retained.

## v224 candidate — hoist indexed output-channel lookups in the SM75 epilogue
Original MixLLM also uses indexed column-major output, so removing scatter or changing output type would violate the contract. Its epilogue nevertheless hoists one index fragment per MMA column before iterating accumulator rows. v224 applies that same safe structural optimization to the clean v200 SM75 runner: each index is loaded once into a fragment, masked channels use a sentinel, and the nested scatter loop reuses the fragment. Output mapping, float output ABI, arithmetic, streams, geometry, and benchmark settings are unchanged. Local discovery passes 75 tests with 6 CUDA-only skips; Python compilation and diff checks pass. Kaggle T4 validation is pending; terminal decision: pending.

## v225 freeze — return to v200 before full audit repair
Per the user's new protocol, v224 is not accepted and is removed from the source tree. The production base is now the clean v200 implementation: `three_level_sm75.cu` and `sm75_backend.py` match v200 exactly, while v218's import-safe local test-harness repairs remain. The complete original-first audit matrix is recorded in `research-15/audit_v200_full_repair_plan.md`. No further Kaggle runs will be performed during repairs; only one final T4 measurement is authorized after the entire list has been locally implemented, tested, committed, and notebook-provenance checked. Frozen-base local discovery passes 74 tests with 6 CUDA-only skips, Python compilation passes, and `git diff --check` passes.

## v226 local candidate — legal SM75 N64 family and bounded autotuner
Original MixLLM and NVIDIA CUTLASS both use a bounded legal configuration search, CUDA-event measurements, warmups, and a shape/device cache. Added only a K=64, `32x64x64` TensorOp family with `NumStages=2`, because the vendored SM75 headers do not support the upstream stage-5/11 choices. Added capture-safe selection, two warmups plus four timed repetitions per candidate, in-process cache, optional `/tmp`/environment cache, and deterministic v200 N128 fallback. No Kaggle run was performed. The new source contract and full local discovery passed 75 tests with 6 CUDA-only skips; Python compilation and diff checks passed.

## v227 local repair — persistent INT4 and metadata preparation
The original-first wrapper audit showed that allocator stream recording and event waits cannot be removed safely, but immutable packed state and quantization metadata can be prepared outside the hot dispatcher. Added module-owned, version-invalidated caches for signed expanded INT4, transposed CUTLASS metadata, and contiguous packed ABI tensors; backend dispatch now reuses those caches and retains graph-capture guards. Added lifecycle and mutation tests. Backend tests passed 19 tests with 6 CUDA-only skips; Python compilation and diff checks passed. No Kaggle run was performed.

## v228 local repair — indexed epilogue fragment hoist
Upstream MixLLM also uses indexed output placement but hoists the channel index fragment. Reapplied that safe optimization to v200: one index is loaded per MMA column fragment, masked lanes use `-1`, and the exact indexed float output write remains unchanged. Source contracts passed 13 tests; Python compilation and diff checks passed. No Kaggle run was performed.

## v228 capture-safety refinement
The tuner now receives the caller stream's capture state through the auxiliary-stream launch seam. If either the caller is capturing or the auxiliary stream is capturing, selection returns the v200 N128 fallback without CUDA-event synchronization or cache file I/O. Full local discovery passes 80 tests with 6 CUDA-only skips; Python compilation and diff checks pass. No Kaggle run was performed.

## v228 final T4 result — operator correctness passed, performance gates failed
The single authorized final Kaggle run completed on Tesla T4 / SM75 with clean provenance (`workspace_dirty=false`, `mixllm_dirty=false`). Embedded tests passed (`returncode=0`), native correctness passed, timing-integrity passed, and the vLLM patch contract passed. The mixed Qwen-shaped rows=128 result measured E2E speedup `0.2897x` versus dense FP16 and GEMM speedup `0.2942x`; pure INT4 rows=128 measured E2E speedup `0.6254x` and failed timing-integrity at ratio `1.1974`; pure INT8 rows=128 measured E2E speedup `0.6967x`. The mixed decode E2E and mixed prefill E2E gates failed, while mixed decode GEMM passed. Full-model Qwen quality/throughput were unavailable because the exact model fingerprint was absent, and vLLM apply execution was unavailable because Kaggle had vLLM `0.27.1` rather than pinned `0.9.0`. Terminal decision: `no_go`; candidate is not accepted as the required >=2.6x production result.

## v230 original-based rebuild — compatibility audit and CPU proof

A first-party comparison with Microsoft MixLLM shows that the upstream launcher/config/cache organization and the v200 stream/event topology are compatible concepts, but the upstream INT4 core is SM80-specific: it uses `ElementA=int8`, packed/interleaved `uint4` B, a custom `OpMultiplyAddMixedAndShuffledInputUpcast`, and an internal `16x8x32` int8 operator. Vendored SM75 exposes legal `8x8x32` `u4/u4`, `s4/u4`, and related 4-bit MMA forms, not the upstream mixed operator. The current SHMQ dequantizer is already functionally identical to upstream; simply copying upstream headers would not add a missing algorithm. v230 therefore creates an original-based branch but does not yet change production dispatch. A deterministic CPU reference script verified the exact signed-int8 low/high decomposition, zero-point correction, and uint4 packing identity over randomized cases. No Kaggle run was performed for v230.

## v230 design refinement — legal pair confirmed, zero-point seam exposed
NVIDIA PTX and the vendored CUTLASS SM75 header confirm the legal dense `m8n8k32` family, including `s4*u4` and `u4*u4`; the upstream `m16n8k32` mixed-input core remains SM80-only. The proposed pair can therefore use one staged B tile and two 4-bit MMA accumulators, but the existing pipeline applies zero-points before MMA. That is unsafe for raw unsigned INT4 because subtraction can underflow and corrected weights may not fit s4. A production pair must bypass pre-MMA B mutation, accumulate per-row `sum(A8)` across K, and apply exact `-zero*sum(A8)` correction after the raw pair products. This is a research/design-only refinement; production dispatch remains unchanged, all local validation passes, and no Kaggle run was performed.

## v231 probe — native SM75 INT4 instruction seam
After comparing the original MixLLM operator and the vendored SM75 headers again, added a compile-only probe for the legal `m8n8k32` `u4*u4` and `s4*u4` instruction specializations. The probe is embedded through the testbed and notebook builder but is not dispatched at runtime. Local source contracts (15 tests), CPU native-INT4 reference proof, Python compilation, and diff checks passed. This prepares one Kaggle T4 compile validation before attempting the full fused pair; no performance claim is made and production dispatch remains the v228 fallback.

## v223 Kaggle compile probe — SM75 INT4 forms compile on T4
Kaggle v223 ran on Tesla T4 / SM75 and compiled the embedded extension with `/usr/local/cuda/bin/nvcc -gencode=arch=compute_75,code=sm_75`. The new `mq_mma_sm75_int4_pair.h` probe was embedded and compiled through `sm75_cutlass_testbed.h`; link succeeded, embedded contracts passed, native correctness passed, and the existing v228 performance report remained unchanged because runtime dispatch was not modified. This validates Kaggle as the compile environment for the next real fused-pair implementation. No performance claim is made for v223.

## v224 probe — execute native SM75 `u4*u4` and `s4*u4`
The v223 compile-only probe succeeded, so v224 adds a standalone CUDA/Torch probe that executes both legal `m8n8k32` instruction specializations with zero fragments and asserts a zero result on T4. It remains outside production dispatch; the goal is to validate runtime instruction execution before implementing the full staged pair. Full local discovery passes 81 tests with 6 CUDA-only skips, the CPU reference proof passes, and Python/diff checks pass. Kaggle run pending.

## v225 fix — make the runtime probe dispatchable
Kaggle v224 compiled the two native MMA forms but failed before execution because a Torch operator with no tensor arguments cannot select the CUDA dispatch key. Corrected the probe schema to accept an empty CUDA tensor and create its output on that tensor's device. This is a probe-only dispatch fix; no production path changed. Local 81-test suite, CPU reference proof, Python compilation, and diff checks pass. Kaggle rerun pending.

## v226 probe — packed row/column WMMA layout
After v225 confirmed both raw SM75 MMA instructions execute, v226 adds a second probe using packed row-major A and column-major B shared-memory tiles. It asserts the exact 8x8 result `32*(row+1)*(column+1)` from `u4*u4` WMMA, while also executing the `s4*u4` high path. Runtime dispatch remains probe-only. Local 81-test suite, CPU proof, Python checks, and diff checks pass. Kaggle validation pending.

## v227 diagnosis — WMMA signed/unsigned mixed overload is unavailable
Kaggle v226 failed at nvcc line 804: CUDA’s WMMA API exposes `m8n8k32` `u4*u4` and `s4*s4`, but no `s4*u4` overload. The earlier CUTLASS `arch::Mma` probe still compiles and executes the mixed signed/unsigned path; the WMMA packed-load probe is therefore narrowed to the legal `u4*u4` load/arithmetic, while the separate CUTLASS probe continues to execute `s4*u4`. Duplicate declarations introduced while narrowing the probe were removed. No production dispatch is changed.

## v228 probe — first true fused packed pair
Built a standalone fused probe that stages one shared packed B tile, loads low/high activation nibble tiles through WMMA u4 loaders, reinterprets the register word into CUTLASS fragments, and issues both `u4*u4` and `s4*u4` MMA instructions before returning four accumulator values per lane. The constant case is expected to produce `[64, 64, -64, -64]`. This is still outside production dispatch, but it is the first probe matching the intended fused dataflow. Local 81-test suite, CPU proof, Python checks, and diff checks pass. Kaggle compile/runtime validation pending.

## v228 Kaggle result — fused packed pair passes on T4
Kaggle v228 compiled and executed the first true fused packed pair on Tesla T4. All three markers passed: instruction probe `[0, 0]`, packed WMMA layout `[32, 64, 96, 128, 160, 192, 224, 256]`, and fused low/high result `[64, 64, -64, -64]`. This proves one staged packed B tile can feed both legal SM75 MMA paths with exact signed high-nibble arithmetic. Production dispatch remains unchanged; next work is an adapter/runner with real tile iteration and zero correction behind the v200 fallback.

## v229 candidate — pure INT4 fused dispatch behind v228 fallback
The fused packed pair now has a host seam and is selected only for `rows>=32`, `n4>0`, `n8==0`, `n16==0`, and uncached metadata. Mixed precision, cached metadata, decode, and unsupported shapes remain on the accepted v228 path. The candidate performs raw packed INT4 B loading, low/high activation nibble decomposition, two legal SM75 MMA calls, post-MMA zero correction, scale application, group accumulation, and indexed scatter. Local 81-test suite, CPU proof, Python checks, and diff checks pass; Kaggle T4 validation pending.

## v229 Kaggle result — pure INT4 candidate kept
Kaggle v229 compiled and ran the fused pure-INT4 dispatch. Native correctness passed; timing integrity passed. Pure INT4 end-to-end speedups vs FP16 were `0.742x` (rows=1), `0.613x` (rows=16), and `0.686x` (rows=128); rows=128 exceeds the v228 pure-INT4 keep threshold of `0.6254x`. The overall kernel remains `no_go` because mixed 4/8/16 end-to-end gates are still failures, but those mixed paths remain on v228 and were not altered by the pure-only gate. Keep v229 as a safe candidate and adapt the fused pair to the mixed INT4 partition next.

## v230 candidate — fused INT4 inside mixed prefill overlap
The proven pair is now used for the INT4 auxiliary stream whenever `n4>0` and metadata is uncached, including mixed 4/8/16 prefill. INT8 remains on its own v228 CUTLASS stream and FP16 remains on the caller stream; pure INT4 keeps the direct fused branch. Cached metadata and all unsupported shapes retain the v228 path. Added a source contract for the seam. Local 82-test suite, CPU proof, Python checks, and diff checks pass. Kaggle validation pending.

## v231 repair — mixed output stride
Kaggle v230 proved the fused arithmetic but failed native correctness only in mixed rows=128: the INT4 kernel scattered with `row*n4+index`, while the shared output tensor stride is `row*(n4+n8+n16)+index`. Pure INT4 was unaffected because `n4==output_width`. Repaired the kernel to receive `output_width` and use the full output stride. Local 82-test suite, CPU proof, Python checks, and diff checks pass; Kaggle revalidation pending.

## v232 decision — reject mixed fused dispatch, retain pure-only path
Historical v228 already showed the same mixed rows=128 error (`410.69699`) before the fused mixed change, so v230 did not introduce that particular numerical discrepancy; v231 nevertheless failed timing integrity after enabling the fused branch in mixed overlap. Reverted the overlap helper to `use_fused_int4=false`, preserving v228 mixed behavior, while retaining the pure-only fused branch whose rows=128 pure-INT4 speedup was `0.686x` with correctness/timing passing. Updated the source contract. Local 82-test suite, CPU proof, Python checks, and diff checks pass. No Kaggle rerun is justified for this fallback-only repair.

## v233 probe — isolate mixed output correctness
Because the historical v228 report already contains the same mixed rows=128 error as v231, v233 adds a probe-only mixed-stride case: 32 INT4 channels write into a 64-column output initialized to `-999`, with expected fused values `128` in columns 0–31 and untouched sentinels in columns 32–63. This separates the fused runner’s scatter correctness from the pre-existing aggregate mixed gate. Production dispatch remains pure-only. Local 83-test suite, CPU proof, Python checks, and diff checks pass; Kaggle validation pending.

## v234 repair — fused warp output mapping
Kaggle v232 isolated the remaining probe failure: rows 0–7 were correct but later rows stayed at the sentinel because writeback used `local_channel=warp*8+...` while the per-warp tile was already selected in `low_tile[warp]`; each warp therefore overwrote the same first eight rows. Corrected mapping to `local_row=warp*8+item/8`, `local_channel=item%8` in both accumulation and final scatter. Local 83-test suite, CPU proof, Python checks, and diff checks pass; Kaggle revalidation pending.


## v235/v236 — Direct SM75 INT4 accumulator mapping and original row-major CUTLASS alignment (local, Kaggle pending)

Deep research compared the original Microsoft MixLLM `gemm_rm` runner with SHMQ. The original mixed row-major path constructs `matrix_C_computed` as `[M, N]`, uses `LayoutC = cutlass::layout::RowMajor`, and writes through the standard CUTLASS accumulator/output iterator. SHMQ’s `sm75_cutlass_testbed.h` had selected `LayoutC = cutlass::layout::ColumnMajor` while its destination tensor and global scatter ABI were row-major `[rows, output_width]`. This is a concrete original-vs-SHMQ discrepancy and the leading explanation for the historical mixed rows=128 error.

The v234 T4 log also showed the mixed-stride probe still failing despite the corrected `item/8` warp mapping. NVIDIA’s PTX ISA specifies that `mma.m8n8k32` gives each lane two accumulator registers with `row = laneid >> 2` and `col = (laneid % 4) * 2 + register_index`. The fused kernel was incorrectly fabricating a larger WMMA accumulator and initializing only two registers before calling `wmma::store_matrix_sync`. v235 replaced that conversion with direct two-register CUTLASS scatter using the PTX-defined mapping and reduced the per-thread partial accumulator to two values. v236 aligned the CUTLASS runner’s `LayoutC` with the original row-major runner.

Local verification after the changes: 78 repository tests passed, 6 CUDA-only tests skipped, the two expected CUDA integration tests were excluded because the sandbox PyTorch build has no CUDA, and `verify_v230_native_int4_reference.py` passed. No new Kaggle run has been launched after these repairs. The changes remain pending T4 validation and are not accepted until the mixed-stride probe, native correctness, timing integrity, and both end-to-end performance gates pass.


## v237 — Force compilation of the exact embedded CUDA source (local, Kaggle pending)

The v236 Kaggle run reached T4, ran 79 embedded contract tests successfully, and passed the instruction, packed-load, and fused arithmetic probes. It failed the mixed-stride probe before native correctness and performance gates, so v236 is rejected.

Deep research of the original-vs-SHMQ build seam found that SHMQ’s loader used a fixed `torch.utils.cpp_extension.load(name="mixllm_sm75_backend")` with the default persistent PyTorch extension cache. The Kaggle execution log repeatedly reported `ninja: no work to do` across source-changing notebook versions. The original MixLLM builds its extension in its normal clean source environment; SHMQ’s repeated Kaggle notebook versions can instead reuse a stale object. v237 now hashes the exact embedded `three_level_sm75.cu` source and includes the digest in the extension name, while preserving the registered `mixllm_sm75` operator namespace. This is a build-provenance fix only; arithmetic, model, quality checks, and benchmark settings are unchanged.

Local verification: 78 repository tests passed, 6 CUDA-only tests skipped, and `verify_v230_native_int4_reference.py` passed. Kaggle has not yet been run for v237.


## v238 — Complete fused SM75 32x32 warp tile (local, Kaggle pending)

Kaggle v235 (v237 source) proved that source-digest compilation was fresh, but the mixed-stride probe still failed. The full tensor pattern showed the underlying geometry defect: each of four warps loaded an `A` tile at `warp*8` rows and a `B` tile at `warp*8` channels, so it computed only four diagonal 8x8 subtiles. The original MixLLM staged runner covers a complete threadblock tile by iterating independent warp-level row/column subtiles.

v238 keeps the legal SM75 `u4*u4` and `s4*u4` m8n8k32 instructions, exact packed-weight arithmetic, zero correction, scales, output ABI, grid, model, and benchmark settings. It changes the fused kernel to assign each warp one 8-column channel tile and iterate all four 8-row subtiles, maintaining four two-register accumulator fragments and scattering all 32 rows × 32 channels exactly once. This is a correctness/completeness repair; the increased per-thread accumulator state will be measured honestly on T4.

Local verification: 79 repository tests passed, 6 CUDA-only tests skipped, 3 subtests passed, and `verify_v230_native_int4_reference.py` passed. Kaggle v235 is rejected at the mixed-stride probe; v238 is not yet submitted.


## v239 — Restore warp-N channel offset in complete fused tile (local, Kaggle pending)

Kaggle v238 (server version 236) compiled the current complete-tile source and still failed the mixed-stride probe. Original MixLLM’s warp decomposition confirmed the remaining discrepancy: the output-channel mapping must include both the block channel base and the warp-N tile coordinate. v238 loaded B for warp 0/1/2/3 from channels 0–7/8–15/16–23/24–31, but scattered every warp to channels 0–7 because the final `channel` expression omitted `warp * 8`; the correction and scale lookup used the same incomplete index. This caused the diagonal-only tensor and sentinel columns.

v239 adds `warp * 8` to the channel mapping consistently for correction, scales, and final scatter. No arithmetic, model, quality, grid, or benchmark setting changed. Local verification: 79 repository tests passed, 6 CUDA-only tests skipped, 3 subtests passed, and the native INT4 CPU proof passed. v239 requires one Kaggle T4 validation before acceptance.


## v240 — Mixed-stride mismatch diagnostics (local, Kaggle pending)

The terminal v239 Kaggle run used the current source and failed at the strict mixed-stride probe before native correctness or performance gates. Because Kaggle abbreviated the tensor repr, v240 adds only diagnostic output before the unchanged assertions: sorted unique values/counts, mismatch count, and the first mismatch coordinates and values. No CUDA source, arithmetic, model, quality path, benchmark shape, timing method, or gate threshold changed.

Local notebook build and `--check` passed with 30 embedded files. This diagnostic run is needed to identify the remaining v239 mapping error without making another speculative kernel repair.


## v241 — Fix final fused scatter warp-N offset (local, Kaggle pending)

Kaggle version 238 diagnostic output proved the remaining defect exactly: `-999.0` count 1792, `128.0` count 256, 768 mismatches, beginning at every row and column 8. The correction/scaling path already used `warp * 8`, but the final output scatter omitted it, causing all four warps to write into columns 0–7. v241 adds `warp * 8` to the final scatter and requires both channel expressions by source contract.

Local verification: 79 repository tests passed, 6 CUDA-only tests skipped, 3 subtests passed, and the native INT4 CPU proof passed. v241 is ready for one Kaggle T4 mixed-stride validation.

## v242 — Enable T4-native fused INT4 inside mixed prefill overlap (local validation complete)

Deep research compared the completed v241 Kaggle run with upstream `mix_mma_multistage.cuh`. v241 passed the mixed-stride probe, native correctness, mixed decode GEMM, and timing-integrity gates, but failed mixed decode/prefill end-to-end performance. Qwen mixed QKV `{4: 2400, 8: 896, 16: 288}` measured approximately 1.28x dense at rows=1, 3.55x at rows=16, and 3.24x at rows=128. Upstream retains persistent INT4/INT8 auxiliary streams and staged overlap, while SHMQ's mixed rows>=32 path was still sending INT4 through expanded signed-INT8 CUTLASS.

Repair: keep the upstream-style two auxiliary streams, INT8 CUTLASS branch, FP16 caller-stream branch, exact channel/index ABI, and all existing fallbacks, but pass `n4 > 0` as `use_fused_int4` for the rows>=32 mixed path so the T4-validated packed native INT4 pair kernel handles the INT4 branch. Cached metadata remains accepted by the ABI but is not needed by the fused INT4 branch. No weights, activations, quality checks, benchmark settings, or model selection changed.

Local validation: full repository MixLLM suite passed (86 tests, 6 CUDA-only skips), the v230 native INT4 reference proof passed, and the added v242 source contract passed. Kaggle has not yet been run for v242; the candidate must be committed, notebook rebuilt, and submitted exactly once only after final local review.

## v243 — Reject v242 fused mixed dispatch and restore v241

Kaggle server version 240 (v242) compiled and passed the mixed-stride probe, embedded contracts, SM75 native correctness, mixed decode GEMM, and timing integrity. It failed both end-to-end gates. On Qwen mixed QKV `{4: 2400, 8: 896, 16: 288}`, rows=1 was 1.23x dense, rows=16 was 3.52x, and rows=128 was 16.62x end-to-end. The v241 measured path was approximately 1.28x, 3.55x, and 3.24x respectively. The v242 production switch is therefore rejected and is not retained.

Deep research found the cause is architectural: the native pair kernel is a correctness-proven 32x32 small-tile path where each warp owns 8 channels and iterates four row subtiles. At Qwen K=3584, mixed large-M work repeatedly reloads small A/B panels and serializes row subtiles, unlike upstream MixLLM's shape-tuned staged CUTLASS threadblock dataflow. The fused kernel remains retained only for the pure-INT4 candidate and diagnostic probes. Mixed rows>=32 is restored to `use_fused_int4=false`, the last measured v241 behavior.

## v244 — Wider 8-warp/64-channel native INT4 pair tile (local validation complete)

Deep research compared the rejected v242 small pair tile with upstream MixLLM's wider N tile families. Upstream searches N=64/128/256 families and its large-M fallback uses a 64x128 threadblock, whereas the native pair candidate exposed only a 32-channel CTA tile. v242 proved that the 32-channel pair geometry is correct but catastrophically slow for mixed large-M Qwen prefill.

Repair: the native pair kernel now uses eight warps and a 64-channel CTA tile. Each warp still owns eight channels and iterates the same four 8-row subtiles, preserving the SM75 8x8x32 low/high MMA pair, exact zero correction, scales, output indices, and FP32 ABI. The grid is widened to 64 channels per block and the launch uses 256 threads. The v241 CUTLASS overlap remains the fallback; no mixed production switch is enabled until the T4 result proves this wider candidate.

Local validation: source contracts passed (19 tests), full MixLLM suite passed (85 tests, 6 CUDA-only skips), and the v230 native INT4 CPU proof passed. Kaggle has not been run for v244.

### v244 dispatch addendum
After the wider 8-warp/64-channel kernel passed the local source suite and full repository tests, the mixed rows>=32 overlap call was switched from `false` to `n4 > 0`, so the candidate is now actually exercised for mixed INT4 while INT8/FP16 stream overlap and the v241 fallback structure remain unchanged. The native CPU proof still passes. This is the final v244 tree to submit; no Kaggle run has been made from the earlier unenabled intermediate tree.

## v245 — Reject v244 wider native pair tile and restore v241

Kaggle server version 241 (v244) passed compilation, mixed-stride probe, embedded contracts, SM75 native correctness, mixed decode GEMM, and timing integrity, but failed both end-to-end gates. Qwen mixed QKV rows=128 measured 13.49x dense end-to-end, improving over v242's 16.62x but regressing sharply from v241's approximately 3.24x. The wider pair candidate is rejected and is not retained in production.

Deep research attributes the residual failure to the unchanged small-pair dataflow: widening CTA-N reduces block count but still reloads A/B panels and serializes four row subtiles, unlike upstream staged CUTLASS threadblock dataflow and autotuned families. v245 restores the 32-channel pair geometry and `use_fused_int4=false` mixed dispatch; the native pair path remains only in the pure-INT4 candidate and probes, exactly as in the last measured v241 baseline.

Local validation after rollback: full MixLLM suite passed (85 tests, 6 CUDA-only skips), and the v230 native INT4 proof passed.

## v246 — Add legal stage-2 N=256 CUTLASS tuner candidate (local validation complete)

Deep research found that upstream MixLLM searches N=64/128/256 families, while SHMQ exposed only N=64 and N=128 despite the vendored SM75 `DefaultMmaCore` being shape-parameterized for stage 2. v244 showed that widening the handwritten pair kernel does not solve large-M dataflow; v246 therefore targets the existing staged CUTLASS module rather than the native pair path.

Repair: add `CoreN256 = GemmShape<32,256,64>` with the existing `WarpShape<32,32,64>`, instruction `<8,8,16>`, row-major accumulator, and matching stage-2 `Int8RunnerN256`. Extend the exact-shape tuner to test N=128/N=64/N=256 and bump the tuning ABI from 226 to 227 so stale disk choices cannot hide the new candidate. The existing N=128 path remains the deterministic fallback when tuning is unavailable. No arithmetic, quantization, output ABI, streams/events, benchmark settings, model, or quality thresholds changed.

Local validation: source contracts passed (19 tests), full MixLLM suite passed (85 tests, 6 CUDA-only skips), and the native INT4 CPU proof passed. Kaggle has not yet been run for v246.

## v247 — Reject N=256 CUTLASS candidate at compile time and restore v245

Kaggle server version 242 (v246) failed before runtime. nvcc reported `pitch_linear_thread_map.h(298): static assertion failed with "Number of iterations must be non-zero"` for `DefaultMmaCore<GemmShape<32,256,64>, GemmShape<32,32,64>, ...>`, with the row-major SM75 A-side map instantiated as `PitchLinearShape<64,32>, Threads=256, WarpThreadArrangement=<4,8>, ElementsPerAccess=16`. The extra eight-warp N=256 shape collapses an iterator dimension to zero in the vendored SM75 specialization.

Deep research also verified that upstream N=256 entries belong to its separate row-major configuration family and are not a drop-in equivalent of the generic N=128 family mirrored here. The N=256 alias, tuner choice, and ABI bump are therefore removed entirely. Local validation after rollback: source contracts passed (19 tests), full MixLLM suite passed (85 tests, 6 CUDA-only skips), and the native INT4 CPU proof passed. No replacement Kaggle run was submitted for the compile-failing candidate.

## v248 — Add legal stage-2 M=64,N=64 CUTLASS tuner candidate (local validation complete)

Deep research compared the v245/v247 baseline with upstream `gemm_configs` and `gemm_configs_rm`, which both include M=64,N=64,K=64 families. Unlike the rejected N=256 shape, M=64,N=64 with the existing 32x32 warp tile produces four warps per CTA and respects the vendored SM75 thread-map contract. The earlier M=64,N=128 experiment was rejected, but that result does not invalidate this smaller N family.

Repair: add `CoreM64N64` and `Int8RunnerM64N64`, extend the exact-shape CUDA-event tuner to N=128/N=64/M64N64, and bump the tuning ABI to 228. The existing N=128 and N=64 runners remain deterministic fallbacks. No arithmetic, quantization, metadata ABI, streams/events, output mapping, benchmark settings, model, or quality thresholds changed.

Local validation: source contracts passed (19 tests), full MixLLM suite passed (85 tests, 6 CUDA-only skips), and the native INT4 CPU proof passed. Kaggle has not yet been run for v248.

## v249 — Hoist native pair A/B packing outside row-subtile loop (local validation complete)

Deep research found a concrete dataflow defect in the native pair kernel: for every 32-element K chunk it repacked the complete 32-row A tile and 32-channel B tile once per each of four row subtiles, with a barrier after every reload. Upstream staged CUTLASS makes a CTA tile resident while multiple warp-level row/column MMA operations consume it. v249 packs A and B once per K chunk, synchronizes, executes all four row-subtile low/high MMAs from the resident shared tile, then synchronizes before overwrite. Arithmetic, zero correction, scales, output indices, grid, ABI, and dispatch policy are unchanged.

Local validation: source contracts passed (19 tests), full MixLLM suite passed (85 tests, 6 CUDA-only skips), and the native INT4 CPU proof passed. Kaggle has not yet been run for v249.

## v250 — Enable load-hoisted native pair in mixed INT4 prefill (local validation complete)

Deep research clarified that v249's mixed Qwen measurement still used `use_fused_int4=false`; the load-hoisted pair was not exercised in the mixed scenario. v250 therefore enables `n4 > 0` only for the mixed rows>=32 overlap call. INT8 remains on the staged CUTLASS auxiliary stream, FP16 remains on the caller stream, and the v249 pair kernel now packs A/B once per K chunk before its four row-subtile MMAs.

No arithmetic, quantization, metadata layout, stream/event ordering, output ABI, model, benchmark setting, or quality threshold changed. Local validation: source contracts passed (19 tests), full MixLLM suite passed (85 tests, 6 CUDA-only skips), and the native INT4 CPU proof passed. Kaggle has not yet been run for v250.

## v251 — Isolate load-hoisted mixed pair after rejecting M=64,N=64 (local validation complete)

Kaggle v248 (server 243) rejected the M=64,N=64 tuner candidate: mixed Qwen rows=128 was 5.46x slower than dense. v249 and v250 still contained that candidate, so v249's 3.38x and v250's 11.78x mixed results were confounded. v250 also enabled the load-hoisted pair in mixed INT4, but its result cannot be attributed to that change alone.

Deep research used NVIDIA PTX documentation to rule out an SM75 m16n8k32 INT8 replacement: dense integer m16n8k32 requires sm80+, while SM75 supports the smaller forms used by this port. v251 removes M=64,N=64, its tuner option, and ABI bump, restoring the v241 N=128/N=64 CUTLASS set, but keeps the v249 load-hoisted pair and mixed `n4 > 0` dispatch. This is the isolated measurement of one change versus v241.

Local validation: source contracts passed (19 tests), full MixLLM suite passed (85 tests, 6 CUDA-only skips), and the native INT4 CPU proof passed. Kaggle has not yet been run for v251.

## v252 — Lazily bypass expanded INT4 for large native mixed prefill (local validation complete)

Deep research compared v251 with the original `mix_mma_multistage.cuh`: upstream passes prepared interleaved INT4 directly to its INT4 Tensor Core runner, while the SM75 module still eagerly expanded every INT4 partition into a full signed-INT8 `[n4,K]` cache during construction/load. The v251 native large-M pair branch does not read that tensor; it consumes packed INT4, original scales, and original zero points. v252 therefore returns an ABI-compatible empty `[0,K]` placeholder for rows>=32 and removes eager expanded-cache priming from device moves, construction, and state loads. Rows<32 retains the existing signed expansion and graph-capture guard.

C++ shape validation now permits the empty placeholder only when `rows>=32 && n4>0`; all fallback paths retain the `[n4,K]` requirement. No arithmetic, quantization, stream/event ordering, metadata ABI, output mapping, benchmark setting, model, or quality threshold changed. Local validation: focused source/backend tests passed (39 tests, 6 CUDA-only skips), full MixLLM suite passed (86 tests, 6 CUDA-only skips), and native INT4 CPU proof passed. Kaggle has not yet been run for v252.

## v253 — Four-warp N=64 native pair tile (local validation complete)

Kaggle v252 (server 247) rejected the lazy expanded-INT4 bypass: mixed Qwen rows=128 remained 10.97x slower than dense FP16 and timing integrity failed. The v252 cache behavior was restored exactly to v251. The v253 candidate follows the upstream legal `32x64x64` family while avoiding v244's eight-warp geometry: the native pair CTA keeps four warps, widens channels from 32 to 64, gives each warp two 8-column N subtiles, packs the 64-channel B panel once per K chunk, and accumulates/scatters `[row_tile][n_tile][register]` with the same low/high MMA arithmetic and indexed output ABI.

Local validation: focused source/backend tests passed (38 tests, 6 CUDA-only skips), full MixLLM suite passed (85 tests, 6 CUDA-only skips), and native INT4 CPU proof passed. Kaggle has not yet been run for v253.

## v254 — M=128,N=64 stage-2 CUTLASS tuner candidate (local validation complete)

Kaggle v253 (server 248) compiled and passed native correctness, mixed-stride correctness, timing integrity, and all non-performance gates, but the four-warp N=64 native pair tile measured Qwen mixed rows=128 at 12.954x slower than dense FP16. It is rejected; the native pair remains probe-only and the v241 mixed staged CUTLASS overlap is restored.

Deep research found the original MixLLM configuration family includes the transposed large-M `M=128,N=64,K=64` shape. v254 adds a matching SM75 stage-2 `DefaultMmaCore`/runner with the existing legal `8x8x16` instruction, exposes it as a third exact-shape tuner candidate, accepts it in the disk cache, and bumps the tuning ABI to 254. N=128 and N=64 remain fallbacks. No arithmetic, quantization, metadata, stream/event, output ABI, model, benchmark, or quality setting changed.

Local validation: focused source/backend tests passed (38 tests, 6 CUDA-only skips), full MixLLM suite passed (85 tests, 6 CUDA-only skips), and native INT4 CPU proof passed. Kaggle has not yet been run for v254.

## v254 — Kaggle T4 result (kernel version 249)

The v254 notebook was successfully pushed as Kaggle kernel version 249 after the initial wrapper invocation was diagnosed as having left the previous output unchanged. The final log's JIT extension digest matched the committed source exactly: `f5c2a7a975e5841c`.

Compilation and functional checks passed: T4 hardware, embedded contracts, SM75 native benchmarks, native correctness, mixed-stride probe (`[128.0, 128.0, 128.0, 128.0]`), and timing integrity. The new M=128,N=64 tuner candidate compiled and executed, but the required performance gates still failed. On Qwen/Qwen2.5-0.5B mixed 4/8/16, end-to-end speedup versus the identical dense FP16 baseline was 0.768x at rows=1, 0.283x at rows=16, and 0.090x at rows=128; rows=128 end-to-end ratio was 11.148x. `terminal_decision` was `no_go`. The M=128,N=64 candidate is rejected as a production baseline; it improved rows=128 relative to v253's 0.077x but remains far from the 2.6x target and does not pass the gates.

No benchmark settings, model, computation, or quality checks were changed. Next iteration must begin with deep source research focused on the original MixLLM's direct interleaved INT4 dataflow versus SHMQ's expanded signed-INT8 prefill cache and hot-path overhead, with no Kaggle run during repairs.
