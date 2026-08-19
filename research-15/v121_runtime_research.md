# v121 Runtime Failure Research

## Evidence from Kaggle T4

Version 138 (v121) compiled the CUDA extension successfully for `compute_75` / `sm_75`. The compiler emitted only unused-variable warnings. The run passed the embedded 18-test contract suite, then failed during the first benchmark case while constructing `operator_reference` in `quantized_reference_prequantized`. The terminal exception was `RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED` at `torch.nn.functional.linear(activation, weight)` in `sm75_backend.py:134`. Therefore, no correctness or performance gate result is admissible for v121.

The call order is significant: `actual = three_level_linear_prequantized(...)` executes the new SM75 kernel; immediately afterward `operator_reference = quantized_reference_prequantized(...)` launches the FP32 reference GEMM; only later does the benchmark time the operator and dense FP16 baseline. Because CUDA launches are asynchronous, the CUBLAS call may be reporting an earlier illegal access or failed launch from the custom kernel rather than being the root cause.

## Local/source audit

`quantized_reference_prequantized` reconstructs the prequantized activation to FP32, obtains the dense FP32 weight, and performs a mathematically independent FP32 per-partition reference GEMM. This is the same quality/reference contract used before the CUTLASS port and is not a benchmark shortcut. The new v121 CUDA kernel is invoked immediately before this reference call, making a delayed custom-kernel error the primary hypothesis.

The SM75 pipeline uses synchronous global-to-shared copies and block-wide `__syncthreads()` because Turing has no `cp.async`. CUTLASS documentation states that shared-memory writes must be completed and visible to consumers before consumption, and its pipeline examples explicitly pair producer completion with consumer waits. NVIDIA CUDA debugging guidance states that asynchronous kernel faults are often reported by a later CUDA API call and recommends host synchronization or `CUDA_LAUNCH_BLOCKING=1` to localize the offending operation.

A first diagnostic change is therefore warranted: synchronize immediately after `actual` and before the independent reference GEMM, only in the non-timed correctness/reference setup path. This does not change model computation, quality checks, benchmark settings, or timed measurements. It will reveal whether the v121 CUTLASS kernel itself produces a CUDA error. If the synchronization passes, investigate reference tensor lifetime/dtype/layout or the next operation; if it fails, repair the kernel's memory or synchronization issue.

## External references

[1] NVIDIA Developer Forums, "Help catching an illegal memory access," https://forums.developer.nvidia.com/t/help-catching-an-illegal-memory-access/310635

[2] NVIDIA CUTLASS documentation, "Synchronization primitives," https://docs.nvidia.com/cutlass/latest/media/docs/cpp/pipeline.html

[3] PyTorch CUDA notes, https://docs.pytorch.org/docs/2.13/notes/cuda.html

## Safety decision

Do not report speedup from v121. Do not keep v121. Proceed with a diagnostic-only synchronization change and require all four Kaggle gates after the next candidate; restore v51 if any required gate fails.


## v123 result and follow-up audit

The per-group loop repair did not remove the fault: v123 compiled, passed the 18 embedded tests, and still raised `CUDA error: an illegal memory access was encountered` at the non-timed synchronize immediately after the combined integer/fp16 dispatch. Thus the over-issued per-group copy count was a real original-vs-port divergence but not the only fault.

The original `mma_multistage_testbed.h` also confirms two separate invariants: the output scatter is lane-aware, and the runner configures dynamic shared memory when storage exceeds 48 KiB. The current SM75 shared storage is approximately 17 KiB for the 64x64x64, two-stage shape, so the >48 KiB attribute is not the current explanation. The current output scatter's simplified indexing is bounded by the accumulator fragment and guarded by global row/channel checks, so it is a quality/mapping concern rather than an obvious illegal access.

The next safe diagnostic is to synchronize after each CUTLASS integer partition launch separately, first after INT4 and then after INT8, without placing synchronization inside timed loops. This will identify which partition and which tensor layout path triggers the fault. The original pipeline uses the same iterator construction and tensor shapes; the remaining likely differences are partition-specific null zero metadata, stage/iterator state, or the SM75 output mapping.


## v125 localization outcome

No `CUTLASS_PARTITION_SYNC_OK` marker appeared before the error, and the traceback now points to the combined operator call before the Python-side post-call synchronization. The first benchmark case is `smoke_mixed` with row counts beginning at 1. For `rows == 1`, `three_level_linear_v2_core` takes the validated v51 decode branch and never invokes `run_cutlass_int_partition`; therefore the partition markers cannot appear for the failing first row. This changes the immediate hypothesis: the failure may be in the rows=1 decode launch/check or in an earlier native activation-quantization launch, even though the new CUTLASS path is present in the same compilation unit.

The next diagnostic must place a direct stream synchronization and completion marker immediately after the rows=1 decode launch, before the generic `C10_CUDA_KERNEL_LAUNCH_CHECK()`. If that marker is absent, the failure is in the decode path; if it appears, the failure is in the subsequent rows>1 CUTLASS path. This remains diagnostic-only and no timing result is admissible.


## v127 localization and decisive iterator audit

Version 145 compiled and passed the 18 embedded tests. Its log shows all rows=1 decode synchronizations succeeding, then `CUTLASS_PARTITION_BEGIN channels=64` with no corresponding completion marker. The first INT4 prefill CUTLASS launch is therefore the first failing operation.

Deep comparison against the canonical `default_mma_core_sm75.h` found a concrete porting error in `sm75_cutlass_testbed.h`. The canonical SM75 specialization defines the shared-memory iterator A as `RegularTileIterator<MatrixShape<M,K>, SmemLayoutA, 0, IteratorThreadMapA>` and iterator B as `RegularTileIterator<MatrixShape<K,N>, SmemLayoutB, 1, IteratorThreadMapB>`. The port's replacement aliases used `AdvanceRank=1` for A and `AdvanceRank=0` for B—the two values were reversed. Since AdvanceRank selects the pitch-linear coordinate that the crosswise shared iterator advances, this changes shared-memory pointer arithmetic and is consistent with an immediate illegal access in the first INT4 launch.

Next change: switch only the two SM75 `RegularTileAccessIterator` aliases to the canonical A=0/B=1 AdvanceRank values. Keep the per-group staging repair. Diagnostic synchronizations/markers remain only until the corrected kernel proves clean; they are removed before any accepted benchmark.


## v129 result and metadata-write diagnosis

Version 147 (v129) still failed at the first INT4 `CUTLASS_PARTITION_BEGIN channels=64` after compilation and successful rows=1 decode synchronization. The guarded A/B staging copy groups did not change the failure point.

The next concrete divergence is in the synchronous replacement of `cp.async` for scale/zero metadata. `FineGrainedScaleZeroIterator<Shape<1,N>>` is constructed with the full block thread id (128 threads) but only thread-row 0 is valid; thread rows 1–3 intentionally report `iterator.valid()==false`. The original `cp.async` path predicates the store and performs no write when invalid. The SM75 `sync_copy()` helper instead writes a cleared zero to `dst` even when `valid==false`. For metadata, those invalid threads' shared destinations are outside the two-stage `[Stages,N]` scale buffer, so this is a direct out-of-bounds shared-memory write during the first INT4 launch. The safe repair is to make metadata copies no-op when invalid while retaining zero-fill behavior for A/B operand copies.


## v130 follow-up audit: accumulator layout discrepancy

The original `mma_multistage_testbed.h` epilogue path statically requires `Mma::LayoutC == cutlass::layout::ColumnMajor` and scatters the converted accumulator using the original `accum_m/accum_n` mapping. The SM75 testbed instead hard-codes `LayoutC = cutlass::layout::RowMajor` while retaining that same fragment-index/scatter formula. The output pointer itself is manually row-major and can remain so; `LayoutC` controls the warp accumulator fragment contract. This is a correctness-relevant original-vs-port mismatch and is the next isolated candidate to test. It is not accepted as a performance change unless all four gates pass.


## v132 result and two-stage metadata cadence diagnosis

Version 150 compiled and ran but remained a no-go; the original-style dynamic shared-memory attributes did not change correctness. The current two-stage SM75 pipeline has `Shape::kK=64` while quantization metadata changes every 128 K elements. The prologue fills only stage 0. At the first mainloop iteration, `copy_scales_and_advance()` sees the odd 64-element half-group and advances the global metadata iterator without writing stage 1; the next read-stage transition then makes the dequantizer read uninitialized stage-1 scales/zeros. For a two-stage synchronous pipeline, the odd half-group must duplicate the current scale/zero into the next metadata stage, then advance the global metadata iterator to the next 128-element group. The proposed repair preserves the quantization cadence and fills each stage with the correct unchanged metadata.


## v134 interleave correction — important scope distinction

The original MixLLM `interleave_uint4_for_cutlass()` is applied before **uint4 repacking** and is part of the packed-INT4 B operand contract. The SM75 port's CUTLASS testbed does not use an INT4 element type or packed nibble MMA; it uses `ElementB=int8_t` and the Turing `m8n8k16` INT8 instruction. Applying the original packed-INT4 K permutation to an already expanded signed-int8 matrix is therefore not justified by the original code and can itself scramble the logical K order. v134 remained incorrect, so the interleaved-operand experiment is rejected; the safe SM75 operand is the raw signed zero-subtracted int8 expansion, as used by the validated WMMA prefill path. Revert the extra interleaved tensor ABI and continue the audit on the remaining SM75 pipeline/scatter semantics.


## v135 audit — accumulator scatter lane-offset discrepancy

The original `mma_multistage_testbed.h` computes a per-thread lane offset before scattering the accumulator fragment:

```cpp
int quad = (threadIdx.x >> 2);
int lane_in_quad = (threadIdx.x & 3);
cutlass::MatrixCoord lane_offset(quad, lane_in_quad * kElementsPerAccess);
offset = tile_offset + lane_offset;
```

The SM75 port computes the warp tile origin (`row_tile`, `channel_tile`) but omits this lane offset entirely. Every thread in the warp then writes to the same warp-tile origin plus the MMA iteration offset, meaning 32 threads all scatter to the same output rows/columns and overwrite each other. This is the leading explanation for large multi-row CUTLASS errors while decode (which uses a different scatter path) remains correct.

The fix: in `sm75_cutlass_testbed.h` kernel, compute `quad = (threadIdx.x >> 2)` and `lane_in_quad = (threadIdx.x & 3)`, then add `quad` to `local_row` and `lane_in_quad * kElementsPerAccess` to `local_channel` in the scatter loop. This matches the original epilogue exactly without changing any other part of the pipeline.


## v135 Kaggle result — lane offset was necessary but insufficient

Kaggle T4 version 154 compiled and ran. The lane-aware scatter reduced the mixed rows=8 error from approximately 69 to 30.93, pure INT4 rows=16/128 errors to approximately 40.01/37.99, and pure INT8 rows=16/128 errors to approximately 39.19/41.07. This confirms the original lane-offset discrepancy was real, but `sm75_native_correctness` still failed; the candidate is rejected and no performance result is admissible. Decode GEMM and E2E gates passed, while mixed prefill E2E failed.

The next audit found another exact original-vs-port difference in the post-scaled accumulator loop: both the original SM80 `mq_mma_multistage.h` and the current SM75 port execute `pipe_state.tmp_accum_.fill(1262485504)` before an MMA accumulation into `tmp_accum_`. The CUTLASS SM75 instruction is explicitly `S32 = S8*S8 + S32`, and `MmaTensorOp::operator()` passes the existing fragment as C, so this nonzero fill is not a neutral initialization. The original code calls two `mac_loop_iter()` operations per outer loop and retains the same unusual fill, while the SM75 port intentionally calls one 64-K iteration per loop. This must be treated as a correctness-sensitive porting decision, not a performance tweak. The next isolated experiment should replace only the SM75 sentinel fill with `clear()` so each 64-K temporary accumulator starts at zero before scaled accumulation; this preserves the quantization, scales, model, and benchmark settings and is required by the S32 MMA contract.

External cross-checks also confirm that CUTLASS accumulator fragment element-to-thread mapping is architecture-specific and not guaranteed by WMMA APIs; the local original CUTLASS epilogue remains the authoritative mapping reference. The vLLM T4 INT8 kernel work shows shape-stratified SM75 performance is realistic, but it does not alter this correctness diagnosis.


## v136 Kaggle result — zero initialization made the path less stable

Kaggle T4 version 155 compiled, but `sm75_native_correctness` failed with NaN errors beginning in rows=16 prefill; mixed prefill E2E also failed. Decode GEMM/E2E remained passing. The nonzero sentinel experiment is rejected and provides no admissible performance result. Because replacing the original sentinel with `clear()` worsened the runtime behavior rather than yielding a finite corrected result, restore the exact original sentinel for the next isolated audit and do not infer a speed or quality benefit from v136.

The remaining investigation returns to the original-versus-SM75 port contract around the 64-K synchronous loop, metadata stage addressing, and the architecture-specific accumulator/dequantizer ordering. The lane-offset correction is retained only as a diagnosticly supported prerequisite, not as an accepted production change.


## v137 audit — SM75 vertical-visit serpentine mapping

The CUTLASS `MmaTensorOp` source sets `kVerticalVisit = true` for compute capability below 8.0. In that branch, for odd `mma_n`, it computes `m_serpentine = MmaIterations::kRow - 1 - mma_m` and stores the instruction result at fragment slot `m_serpentine + mma_n * MmaIterations::kRow`; it also selects the A fragment using `m_serpentine`. The SM75 port’s custom epilogue scatter copied the original SM80 formula and uses `mma_m` directly for both fragment `start` and logical output row, so its mapping is valid for the original SM80 vertical-visit setting but not for SM75’s odd-N serpentine order.

The next isolated correction is to derive `m_fragment = (mma_n & 1) ? (MmaIterations::kRow - 1 - mma_m) : mma_m` in the SM75 scatter, use `m_fragment` for the accumulator fragment start, and use the corresponding logical row position for the output. This is a direct architecture-specific CUTLASS contract fix, independent of quantization and benchmark settings. If correctness remains imperfect afterward, the same serpentine ordering must be reflected in the dequantizer’s activation-scale application.


## v137 Kaggle result and v138 dequantizer audit

Kaggle version 156 produced the same finite errors as v135 (`smoke` rows=8 about 30.93; pure INT4/INT8 rows=16 about 40/39), with decode gates passing and native correctness/prefill E2E failing. The serpentine scatter edit was effectively a no-op for the observed output because the residual mismatch is not repaired by changing the final fragment-slot row mapping alone.

A more precise fragment audit identifies the remaining mismatch in `apply_scale_accum_act`. For the SM75 `32x32` warp tile and `m8n8k16` instruction, each thread has four row instruction fragments. CUTLASS's `kVerticalVisit=true` stores the fragment for odd column MMA iterations at the reversed physical row slot `m_serpentine`. The current dequantizer applies activation scales using `scale_frag[n/2]` based on the physical fragment slot, regardless of the odd-column serpentine reversal. Therefore, for odd `mma_n`, a fragment physically at row slot 0 contains logical row slot 3, but receives activation scale 0 instead of scale 3; this affects both INT4 and INT8 and explains the similar residual errors after the lane-aware scatter fix.

The next isolated fix is to change only `apply_scale_accum_act`: when `MmaOperator::kVerticalVisit` is true, map the inner physical row slot `n/2` through `(m / kColsPerMmaPerThread & 1) ? (MmaIterations::kRow - 1 - n/2) : n/2` before indexing `scale_frag`. The SM80 path remains unchanged under `if constexpr (!kVerticalVisit)`. Weight-scale mapping, quantization, operands, model, and benchmark settings remain unchanged.


## v138 Kaggle result — activation-scale reversal worsened correctness

Kaggle version 157 compiled and ran, but the activation-scale serpentine edit made the prefill result worse: smoke rows=8 error rose to approximately 62.98, pure INT4 rows=128 to approximately 70.94, and pure INT8 rows=128 to approximately 73.75; rows=16 also produced NaN in the gate report. Decode gates passed, while native correctness and prefill E2E failed. The dequantizer edit is rejected and restored to the original `scale_frag[n/2]` mapping.

The v137 source audit also exposed a narrower correction: CUTLASS vertical serpentine order changes the **physical accumulator fragment slot**, so `fragment_m` must be used only to compute `start`; the logical output row remains `mma_m`. v139 will test that exact physical-slot-only change. This avoids conflating fragment storage order with the logical row coordinate and leaves all scaling and operands unchanged.


## v139 Kaggle result — physical-slot-only serpentine scatter also rejected

Kaggle version 158 compiled and ran on T4. Using the serpentine index only for the physical fragment `start` still failed native correctness and prefill E2E, and it substantially worsened large-row errors: mixed rows=128 reached approximately 417.9, pure INT4 rows=128 approximately 389.2, and pure INT8 rows=128 approximately 368.6; rows=16 produced NaN. Decode gates passed. Therefore the whole serpentine scatter experiment is rejected: restore the original `mma_m` fragment-slot indexing in the custom SM75 scatter while retaining only the empirically beneficial lane offset. The CUTLASS source audit shows that the low-level `MmaTensorOp` visitor order and the custom epilogue’s `FragmentC` indexing are not interchangeable assumptions; the original MixLLM epilogue formula should remain authoritative for the port’s selected accumulator contract.

The residual finite error after the lane offset (about 30–41) is therefore not solved by final-scatter serpentine remapping. The next investigation must target the synchronous SM75 mainloop’s operand/stage semantics or the exact accumulator/dequantizer contract, not another output permutation guess.


## v140 audit — single-iteration SM75 loop skipped the final K tile

The original SM80 mainloop calls `mac_loop_iter()` twice per outer loop. Its prologue decrements `gemm_k_iterations` once, then each `mac_loop_iter()` decrements it again; the outer condition is evaluated only once per pair, so the final call still runs when the counter reaches zero. The SM75 port correctly changed to one `mac_loop_iter()` per outer loop because it has no cp.async pair, but retained `for (; gemm_k_iterations > 0;)`. Since prologue already changes an initial K-tile count `T` to `T-1`, this condition executes only `T-1` compute iterations and permanently omits the final 64-K tile. For K=3584 (`T=56`), the CUTLASS path computes only 55 tiles; the validated WMMA reference computes all 56. This explains why the error is finite and similar across INT4 and INT8 while rows=1 decode remains correct.

The isolated v140 fix changes only the SM75 loop condition to `gemm_k_iterations >= 0`, preserving one synchronous 64-K mac per loop while executing the final preloaded/read stage. The original bias sentinel, lane offset, raw signed-int8 operands, metadata cadence, scales, model, and benchmark settings remain unchanged.


## v140 Kaggle result — correctness restored; prefill performance still fails

Kaggle version 159 is the first CUTLASS-port candidate after the lane-offset audit to pass `sm75_native_correctness`. Representative prefill errors are now finite and near zero: mixed rows=16 0.0951 and rows=128 0.1025; pure INT4 rows=16 0.0001068 and rows=128 0.0001831; pure INT8 rows=16 0.0001068 and rows=128 0.0001984. Decode correctness/performance gates also pass. However, `mixed_prefill_end_to_end_performance` fails badly: mixed rows=16/128 speedups versus dense are only about 0.182/0.192, and mixed GEMM speedups about 0.184/0.194. The candidate is therefore not production-safe under the four-gate rule and yields no accepted baseline, although the final-K-tile correction is retained as a correctness prerequisite for further performance work.

This result isolates the remaining problem to performance, not arithmetic correctness. The SM75 synchronous CUTLASS path spends approximately 5x dense FP16 time for multi-row prefill despite using the correct full-K computation. The next deep research must compare the current synchronous copy/mainloop and launch structure with original MixLLM’s prefill path and the validated v51 SM75 WMMA path, focusing on unnecessary per-partition launches, shared-memory staging overhead, and whether the CUTLASS testbed is being used for shapes where the established WMMA kernel is faster. No benchmark settings or computations may be reduced.


## v141 performance audit — v140 replaced one validated WMMA prefill launch with two slow CUTLASS launches

The current v140 dispatch for rows>1 calls `run_cutlass_int_partition()` once for INT4 and once for INT8, then optionally launches the FP16 WMMA kernel. Each CUTLASS call stages the full activation matrix independently, runs a 64x64x64 synchronous SM75 pipeline, and performs its own metadata copies and barriers. Thus a mixed layer executes two independent integer GEMM pipelines and repeats the activation staging work. The v51 rollback path instead launches the existing `three_level_tensorcore_kernel` once with a precision-tiled grid; each CTA loads its own required activation tile and directly executes the validated SM75 WMMA path. This is an exact source-level dispatch difference, not a benchmark or quality shortcut.

The next isolated v141 experiment will route rows>1 back to the unchanged v51 WMMA prefill kernel while retaining the corrected final-K CUTLASS code in the source for continued audit. This is an honest fallback test: it restores the previously validated arithmetic/launch path and removes the new CUTLASS overhead without reducing work, changing quantization, changing the model, or changing the benchmark. If all four gates pass, it is the safe production state while CUTLASS performance is separately optimized; if it fails, the cause lies in the baseline snapshot or gate assumptions rather than v140’s CUTLASS overhead.


## v141 Kaggle result — WMMA fallback is correct but still fails the prefill gate

Kaggle version 160 passed `sm75_native_correctness` and both decode performance gates, but `mixed_prefill_end_to_end_performance` still failed. Mixed rows=16/128 GEMM speedups were approximately 0.282/0.166 and E2E speedups approximately 0.285/0.170; pure INT4/INT8 rows=16/128 were also below dense. The fallback did not reproduce the expected v51 prefill performance, so it is not an accepted production state. The v140 final-K correction remains necessary and correct.

The original MixLLM SM80 pipeline performs two `mac_loop_iter()` calls per outer loop and applies weight/activation scaling once to the combined temporary accumulator. The current SM75 port performs one 64-K `mac_loop_iter()` followed immediately by two scale passes and a float conversion per tile. For K=3584 this doubles the scale/convert/fragment-add overhead relative to the original two-call batching. The next v142 candidate will preserve the exact 56 K-tile arithmetic while batching two synchronous `mac_loop_iter()` calls per outer iteration and applying scales once to their combined accumulator. The second call is guarded after the first so odd tile counts remain correct; no computations are removed and no benchmark/model settings change.


## v143 performance hypothesis — widen the CUTLASS N tile from 64 to 128

The current SM75 CUTLASS runner is fixed at ThreadblockShape 64x64x64 with WarpShape 32x32x64, so a Qwen 3584-channel partition launches 56 N-tiles. The original MixLLM configuration table explicitly includes 64x128 and 128x128 threadblock configurations, reflecting the key optimization for GEMM: reuse each activation tile across more output channels and reduce CTA-level staging/barrier overhead. A 64x128x64 SM75 port is structurally compatible with the existing m8n8k16 instruction: it uses 2 warp rows x 4 warp columns (8 warps), keeps the same K tile and two synchronous stages, and halves the N-grid from 56 to 28 CTAs per integer partition.

v143 will change only the CUTLASS Core threadblock N dimension from 64 to 128 while keeping WarpShape 32x32x64, instruction m8n8k16, raw signed-int8 operands, metadata indexing, accumulator conversion, quantization, output mapping, model, quality, and benchmark settings unchanged. This is an honest wider-tile performance test; if compilation or correctness fails, it will be reverted.


## v143 Kaggle result — invalid as a CUTLASS geometry measurement

Kaggle version 162 again passed native correctness and decode gates but failed mixed prefill performance, with mixed rows=16/128 speedups approximately 0.288/0.173. Source provenance shows `three_level_sm75.cu` was unchanged from v142: v141’s honest fallback had replaced the rows>1 CUTLASS dispatch with the WMMA kernel, so changing the unused CUTLASS Core from 64x64 to 64x128 could not affect the measured path. v143 is therefore not evidence for or against wider CUTLASS geometry and is not kept. The geometry and stale test expectation are reverted before the next real CUTLASS experiment.

The next candidate must first restore the v140 CUTLASS dispatch, then test the 64x128 Core so the geometry change is actually exercised. This is the direct original-versus-current discrepancy: the source contains a new CUTLASS runner, but the active dispatch was intentionally switched back to v51 WMMA while isolating overhead. No performance conclusion may be drawn from v143’s inert Core edit.


## v144 Kaggle result — real CUTLASS path correct but still far below prefill performance gate

Kaggle version 163 exercised the restored CUTLASS integer path with the 64x64x64 Core and corrected two-call batching. Native correctness and decode gates passed. Mixed prefill remained a no-go: rows=16 GEMM/E2E speedups were approximately 0.151/0.150, and rows=128 approximately 0.311/0.306. Pure INT4/INT8 rows=16/128 also remained below dense, although the wider effective workload showed a substantial improvement over v140 at rows=128. The candidate is rejected under the all-four-gates rule.

The next measured geometry must be the original-informed 64x128x64 Core while retaining the active CUTLASS dispatch; v143 had changed this Core while the WMMA fallback was active and was therefore inert. The only expected code difference in v145 is the real wider-N Core plus its source-contract update.


## v145 Kaggle result — active 64x128 CUTLASS geometry improves rows=128 but still fails production

Kaggle version 164 exercised the 64x128x64 Core on the active CUTLASS path. Native correctness and decode gates passed. Mixed prefill remained below the required gate: rows=16 GEMM/E2E speedups were approximately 0.164/0.163 and rows=128 approximately 0.332/0.329. Pure INT4 rows=16/128 reached about 0.804/0.663 GEMM speedup; pure INT8 reached about 0.826/0.680. The wider tile is a real improvement over v144 for pure integer prefill but not enough for mixed production, so v145 is rejected.

Deep backend comparison found another exact mismatch with the original MixLLM runtime contract: `run_cutlass_int_partition()` transposes and allocates contiguous copies of every scale/zero matrix on every operator invocation. The module already caches expanded INT4 weights but does not cache these stable per-layer metadata layouts. The next v146 candidate will cache transposed `[groups, channels]` metadata for prefill, invalidate it on device/state changes, and keep the original `[channels, groups]` layout for rows=1 decode. CUDA validation will branch on rows so decode and fallback semantics remain unchanged. This removes only repeated host-side allocation/copy overhead; all GEMM arithmetic, quantization, model, quality checks, and benchmark settings remain unchanged.


## External references used in the CUTLASS layout audit

The CUTLASS layout audit cross-checked NVIDIA’s row-major/column-major mapping definitions in `cutlass/layout/matrix.h` and the CUTLASS discussion confirming that tensor-core int8 GEMM commonly uses row-major A multiplied by column-major B. Sources: [NVIDIA CUTLASS matrix layouts](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/layout/matrix.h) and [CUTLASS issue #1533 on int8 row/column layouts](https://github.com/NVIDIA/cutlass/issues/1533). These references support the local source conclusion that the SM75 path must consume a logical column-major B operand and that the packed-INT4 K permutation cannot be applied to the expanded signed-int8 matrix.


## v146 Kaggle result — cached metadata did not pass prefill and did not produce a reliable improvement

Kaggle version 165 passed native correctness and decode gates after introducing cached `[groups, channels]` scale/zero tensors for multi-row prefill while preserving original decode layout. Mixed prefill still failed: rows=16 GEMM/E2E speedups were approximately 0.169/0.168 and rows=128 approximately 0.181/0.358. Pure INT4 rows=16/128 reached approximately 0.891/0.711 GEMM speedup and pure INT8 approximately 0.864/0.657. The change removes only repeated host transposes but cannot address the dominant CUTLASS kernel overhead; it is rejected as a production improvement, though its correctness-preserving cache lifecycle can remain if no later change conflicts.

The next audit returns to the original-vs-port synchronous mainloop. The SM75 kernel has no cp.async, so every K tile performs guarded global-to-shared copies, CTA barriers, shared-memory MMA loads, and metadata operations serially. The original SM80 path overlaps copies with MMA through cp.async and uses larger configured tiles. The next candidate must target a demonstrably redundant barrier or copy in `mq_mma_pipelined_sm75.h`/`sm75_cutlass_testbed.h`; no barrier may be removed without proving that all dependent warps have completed both A and B shared-memory writes.


## v147 deep audit — one of two consecutive SM75 CTA barriers is provably redundant

In `mq_mma_pipelined_sm75.h`, the final grouped global-to-shared copies for each 64-K tile finish at lines 723–727. The code then executes `__syncthreads()` at lines 729–730 and immediately calls `gmem_wait()`, whose SM75 implementation is exactly another `__syncthreads()` at lines 623–628. Because SM75 has no asynchronous copy operation and `sync_copy` is ordinary guarded assignment, the first barrier already establishes the required producer/consumer ordering; the second barrier has no intervening writes or reads and cannot add correctness. The original SM80 `gmem_wait()` exists to wait for cp.async completion, but that semantic is not needed twice in the synchronous SM75 port.

v147 removes only the explicit barrier before `gmem_wait()` and leaves `gmem_wait()` as the single CTA barrier. All shared-memory writes, MMA operations, K iterations, metadata operations, scales, zero points, output mapping, model, quality, and benchmark settings remain unchanged. This is a synchronization reduction with a complete local proof, unlike speculative barrier removal elsewhere.


## v148 deep audit — final gemm_iters barrier is inherited async-drain overhead

The original MixLLM `gemm_iters()` ends with `cp_async_fence()`, `cp_async_wait<0>()`, and `__syncthreads()` to drain outstanding SM80 asynchronous global-to-shared copies before returning. The SM75 port has replaced all copy operations with synchronous guarded assignments and already performs the required CTA barrier inside `gmem_wait()` at each stage boundary. After the final `mac_loop_iter`, no asynchronous copy remains in flight, and the kernel immediately consumes register accumulators in the epilogue; no thread reads another thread’s shared-memory state.

Therefore the final standalone `__syncthreads()` at the end of SM75 `gemm_iters()` is not required for SM75 correctness. v148 will remove only this inherited end-of-mainloop barrier. The preceding per-stage `gmem_wait()` barrier remains untouched, as do all MMA, metadata, scale, zero, and output operations. This is a synchronization-only optimization with a direct original-versus-port proof.


## v148 Kaggle result and v149 geometry hypothesis

Kaggle version 167 confirmed that removing the inherited final `__syncthreads()` preserved native correctness and decode gates but did not solve prefill: mixed rows=16/128 were approximately 0.172/0.171 and 0.336/0.332 GEMM/E2E speedup. The final barrier was not the dominant cost.

A full source comparison with the accepted v51 snapshot found that the preserved `three_level_tensorcore_reuse_kernel` definition is not actually launched in the v51 snapshot; therefore it cannot be claimed as the validated v51 dispatch without provenance evidence. The active CUTLASS path currently uses ThreadblockShape 64x128x64 and WarpShape 32x32x64, which is an 8-warp CTA. For rows=16, only one quarter of the M tile is populated, but all 8 warps still execute the CUTLASS mainloop and barriers. The next principled geometry test is ThreadblockShape 32x128x64 with the same 32x32x64 warp tile: one M warp by four N warps, 4 total warps, no arithmetic reduction, no quality change, and the same K/instruction shape. It should reduce idle-M warp and barrier overhead for rows=16 while remaining a valid exact tile for rows=32/128. Compile-time contracts and Kaggle correctness will decide whether the accumulator scatter supports this geometry.


## v149 Kaggle result and v150 monolithic reuse-kernel hypothesis

Kaggle version 168 passed native correctness and both decode gates, but the 32x128x64 geometry still failed mixed prefill: rows=16/128 were approximately 0.180/0.295 E2E speedup. Reducing inactive M warps did not address the dominant cost.

The next deep source comparison found a more consequential architectural mismatch. The active rows>1 branch launches two independent CUTLASS integer kernels (`run_cutlass_int_partition` once for INT4 and once for INT8), followed by a separate FP16 WMMA launch when needed. The v51 source tree preserves a balanced `three_level_tensorcore_reuse_kernel` that uses an 8-warp 2x4 WMMA tile, loads a 32x16 activation panel and 16x64 weight panel once per K step, and handles INT4, INT8, and FP16 through one monolithic kernel dispatch. Although the rollback snapshot did not launch that definition, its implementation is an exact project-local candidate and directly satisfies the original three-level parallel-kernel architecture.

v150 will restore that kernel into the current source and make it the rows>1 dispatch for a controlled benchmark. The backend/core metadata ABI will be restored to the original `[channels, groups]` layout because the reuse kernel indexes scales as `scale[c * groups + group]`; the rejected prefill-transpose cache is removed from the active path. The experiment changes no arithmetic, quantization, model, quality, or benchmark settings; it changes only kernel fusion and tile reuse. Correctness and all four Kaggle gates decide whether it can replace v51.


## v150 Kaggle result and v151 combined-integer-launch hypothesis

Kaggle version 170 validated the restored monolithic reuse kernel: native correctness and both decode gates passed, but mixed prefill became much slower than CUTLASS. Rows=16 measured approximately 0.105 GEMM / 0.177 E2E speedup and rows=128 approximately 0.090/0.095. The reuse implementation is rejected as a performance path.

The next original-versus-current discrepancy is launch granularity. The current CUTLASS path launches one complete CUTLASS kernel for INT4 and another for INT8, then may launch a separate FP16 kernel. The two integer kernels use identical SM75 INT8 MMA code and differ only in B/scale/index storage. A safe optimization is to cache a combined signed-INT8 B matrix, combined scale metadata, and combined output-index vector for the two integer partitions, then invoke the same CUTLASS kernel once over `n4+n8` channels. This does not reduce any GEMM work or alter quantization; it removes one kernel launch and one repeated iterator/prologue/epilogue sequence. FP16 remains a separate typed path. If the combined cache adds more overhead or fails any gate, it will be removed.
