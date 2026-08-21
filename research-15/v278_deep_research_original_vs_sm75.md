# v278 deep research: repair the SM75 packed-B iterator policy

## v277 compiler finding

The compressed v277 bundle reached the T4 compiler. The new native runner failed before any gate because the generic SM75 `MmaTensorOpMultiplicandTileIterator` was instantiated with `Element_=cutlass::uint4b_t` and an internal `m8n8k16` instruction shape. CUTLASS computes `kElementsPerAccess = 128 / 4 = 32`; with `InstructionShape::kContiguous = 16`, the generic policy computes `LdsmShapeContiguous = 16 / 32 = 0`, producing division-by-zero and non-constant LDSM shapes. The subsequent `ldsm` call cannot match the invalid shape.

This is a structural iterator mismatch, not a bad packed-weight cache. The original MixLLM SM80 mixed specialization explicitly uses a widened logical load shape (`GemmShape<16,8,32>` for the 4-bit path) while its instruction policy is defined for the mixed Tensor Core operator. That widened load shape is what gives the iterator enough contiguous 4-bit elements for legal `ldmatrix` operations.

## v278 design derived from both implementations

SM75 has only legal integer `m8n8k16` s8*s8 MMA for the internal upcast route, so the Arch policy must remain k16. The custom warp operator will decouple **load shape** from **MMA instruction shape**: its A and B iterators will load `kK=32` logical elements per pipeline iteration, producing two k16 operand fragments; its operator will convert the full fragments and issue two legal k16 MMAs for each M/N subtile. This mirrors the original mixed path's widened 4-bit load geometry while respecting SM75's legal instruction forms.

For B, the existing original-compatible interleaved packed `uint4` storage remains unchanged and the existing `mq_numeric_conversion.h` specialization converts `Array<uint4b_t,N>` to `Array<int8_t,N>` for `N` divisible by eight. No global signed-INT8 expansion is used by the native large-M branch. For A, the existing SHMQ activation layout and exact quantization contract remain unchanged. The pipeline, metadata, zero correction, output mapping, events, benchmark settings, model, and quality gates remain unchanged.

## Acceptance and rollback

v278 must compile on T4, pass all embedded correctness/contracts and timing-integrity gates, and improve the rows=128 mixed prefill path relative to v271. If the widened iterator/operator does not compile or produces incorrect output, reject it and restore the exact v271 baseline. No result is accepted without the unchanged four production gates and timing-integrity pass.
