# v252 deep research: stop materializing expanded INT4 for native mixed large prefill

## v251 evidence

Kaggle server version 246 (v251) compiled and passed the mixed-stride probe, embedded contracts, SM75 native correctness, and timing integrity, but mixed Qwen rows=128 was 11.547x slower than dense FP16. The v251 production branch used the native packed INT4 pair kernel for the INT4 partition, while INT8 remained staged CUTLASS and FP16 remained on the caller stream.

## Upstream comparison

The original MixLLM launcher passes `matrix_B_interleaved` directly to its INT4 CUTLASS testbed and does not construct a signed expanded INT4 matrix. In the current SM75 module, `prepare_sm75_prefill_cache()` always unpacks every INT4 weight and subtracts zeros into a full `[n4,K]` signed-INT8 tensor during `from_weight`/state loading. `sm75_backend.three_level_linear_prequantized()` also requests that cache for every prefill row count. The v251 native mixed INT4 branch does not read this tensor: `three_level_linear_v2_core()` selects `begin_integer_prefill_overlap(..., use_fused_int4=n4>0)`, and the fused branch calls `run_int4_pair_partition()` with the original packed `weight_int4`, original `scale_int4`, and original `zero_int4`. The only remaining use of `expanded_int4` in that rows>=32 mixed branch is as an ABI argument to the separate FP16 kernel; that kernel is launched only for `n16` channels and does not access INT4 rows.

## Candidate and safety boundary

For rows>=32 with any INT4 partition, pass an empty `[0,K]` INT8 placeholder instead of eagerly expanding INT4. Relax the C++ shape validation only for this exact native-pair condition (`rows>=32 && n4>0`); retain the existing `[n4,K]` requirement for rows<32 or any future non-pair path. Keep expanded INT4 preparation for rows<32, where direct WMMA fallback still consumes it. Preserve all quantization arithmetic, packed weights, zero/scales, streams/events, output ABI, model, benchmark settings, and quality gates. Add local Python/C++ source contracts for the conditional placeholder and run full tests before any T4 measurement.
