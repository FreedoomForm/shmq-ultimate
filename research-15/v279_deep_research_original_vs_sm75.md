# v279 deep research: widen dequantizer metadata handling with the k32 adapter

## Evidence from the v278 T4 compiler

Kaggle version 274 compiled the new v278 source digest and no longer emitted the v277 generic SM75 uint4 LDSM-policy failure. The only compilation error was the existing dequantizer assertion in `mq_mma_tensor_op_dequantizer.h:229`:

```cpp
MmaOperandB::kElements * MmaOperator::MmaIterations::kColumn
  == MmaOperator::TransformedFragmentB::kElements
```

The v278 adapter intentionally makes `TransformedFragmentB` twice as large: a k32 logical load is converted to two consecutive k16 B operand groups, while `MmaOperandB` still describes one legal SM75 k16 instruction. The assertion is therefore a stale single-k16 assumption, not evidence that the widened iterator is invalid.

## Original MixLLM versus the SM75 adapter

The original mixed-input CUTLASS dequantizer applies one zero point per output-channel fragment. Its `zero_frag` contains one value for each `MmaIterations::kColumn` B fragment, and the zero is subtracted from every B operand element associated with that output channel. Zero points are metadata for the quantization group and do not vary across K subgroups within the threadblock K tile. Therefore, when the SM75 adapter splits one k32 load into two k16 MMAs, the exact original arithmetic is to apply the same zero vector to both consecutive B halves.

The v279 implementation will generalize the static assertion to require that the transformed B fragment is an integer multiple of the original one-k16 B-fragment footprint. It will compute the compile-time number of k16 halves and repeat the existing zero-application loop at an offset of `half * MmaIterations::kColumn`. The packed four-zero-point SIMD path will retain its `vsub4` operations; only the outer half loop is added. No zero values, scales, weights, activation quantization, output mapping, launch geometry, or benchmark settings change.

## Acceptance criteria

v279 must pass local contracts and compile on the Tesla T4. If it compiles, correctness must prove that both k16 halves receive the same metadata and that the full four-gate contract plus timing-integrity pass. A compile pass alone is not acceptance. If correctness fails, reject v279 and return to the safe v271 baseline before the next research iteration.
