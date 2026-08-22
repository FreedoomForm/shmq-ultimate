# v296 deep research: SM75 packed fragment contract

## Evidence compared

The preserved original MixLLM path uses `MQMmaMixedInputTensorOp` with the same `FragmentConverter<int8_t, uint4b_t, N>` implementation used by SHMQ. CUTLASS `Array<uint4b_t, N>` is subbyte-packed: eight logical uint4 values occupy four physical bytes, and the converter expands each four-byte word into eight int8 register values by taking low and high nibbles. SHMQ’s `FragmentConverter` is the same implementation, not a divergent nibble order.

The original packed SM80 adapter uses an instruction shape `16x8x32`. SHMQ cannot instantiate that instruction on SM75, so v280+ presents a widened k32 fragment to a custom adapter and executes two legal `m8n8k16 s8*s8` MMAs. The custom adapter’s static counts are internally consistent: each widened B fragment contains 64 logical uint4 values and expands to 64 int8 values; each widened A fragment contains 64 int8 values and is split into two 32-value halves. However, static element counts do not prove that the two halves correspond to the original k16 operand register groups.

The SM75 `MmaTensorOpMultiplicandTileIterator` selected for row-major congruous B is the generic `TensorOpMultiplicandCongruous<4,64>` specialization. Its lane mapping and LDSM sequence are written for a concrete instruction shape and use `Shape::kStrided / InstructionShape::kStrided` groups. SHMQ changes the adapter’s instruction shape from the original k32 contract to k16 while passing a widened k32 iterator shape. That is the highest-confidence remaining source of the unchanged rows>=32 corruption. The output scatter formula matches CUTLASS’s row-major accumulator iterator formula, so output mapping is less likely to be the primary fault than the widened iterator/fragment split.

## Next action

Do not change benchmark conditions, arithmetic quality, or the packed memory permutation. Add a Colab-only diagnostic/probe or a minimal alternative adapter that tests the two k16 halves against the proven SM75 `u4*u4` pair decomposition. Accept a production repair only if all large-M smoke, Qwen-shaped, pure INT4, timing-integrity, and existing regression checks pass. Until then v271 remains the safe baseline and v294/v295 remain rejected.

## Colab artifact audit

The current runner was still pinned to commit prefix `7be8661` (v291) and cloned the committed full gate notebook, while the helper script generated a v293 operator artifact. That would silently validate stale source. The v296 validation path therefore updates the runner’s immutable commit pin and workspace labels to the v296 commit, and updates the helper’s versioned source/output names. The gate cells, model, benchmark rows, thresholds, and operator-only quality skip remain unchanged.

## Stronger iterator mismatch

The direct type comparison found a concrete divergence: original `MQMmaMixedInputTensorOp` constructs `IteratorA` with `MatrixShape<ArchMmaOperator::Shape::kM, ArchMmaOperator::Shape::kK>`, which is `<16,32>` for the original SM80 `m16n8k32` operator. SHMQ’s SM75 adapter instead used `<8,32>` because its internal legal MMA is `m8n8k16`. The widened fragment still has the same total element count, so static assertions do not detect this difference, but the row-major crosswise iterator’s lane/LDSM mapping is instruction-shape dependent. The next candidate changes only SHMQ’s A iterator shape back to the original widened `<16,32>` contract while retaining the legal SM75 `m8n8k16` arithmetic and the two-half accumulation.
