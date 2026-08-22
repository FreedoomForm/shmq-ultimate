# v296 deep research: SM75 packed fragment contract

## Evidence compared

The preserved original MixLLM path uses `MQMmaMixedInputTensorOp` with the same `FragmentConverter<int8_t, uint4b_t, N>` implementation used by SHMQ. CUTLASS `Array<uint4b_t, N>` is subbyte-packed: eight logical uint4 values occupy four physical bytes, and the converter expands each four-byte word into eight int8 register values by taking low and high nibbles. SHMQ’s `FragmentConverter` is the same implementation, not a divergent nibble order.

The original packed SM80 adapter uses an instruction shape `16x8x32`. SHMQ cannot instantiate that instruction on SM75, so v280+ presents a widened k32 fragment to a custom adapter and executes two legal `m8n8k16 s8*s8` MMAs. The custom adapter’s static counts are internally consistent: each widened B fragment contains 64 logical uint4 values and expands to 64 int8 values; each widened A fragment contains 64 int8 values and is split into two 32-value halves. However, static element counts do not prove that the two halves correspond to the original k16 operand register groups.

The SM75 `MmaTensorOpMultiplicandTileIterator` selected for row-major congruous B is the generic `TensorOpMultiplicandCongruous<4,64>` specialization. Its lane mapping and LDSM sequence are written for a concrete instruction shape and use `Shape::kStrided / InstructionShape::kStrided` groups. SHMQ changes the adapter’s instruction shape from the original k32 contract to k16 while passing a widened k32 iterator shape. That is the highest-confidence remaining source of the unchanged rows>=32 corruption. The output scatter formula matches CUTLASS’s row-major accumulator iterator formula, so output mapping is less likely to be the primary fault than the widened iterator/fragment split.

## Next action

Do not change benchmark conditions, arithmetic quality, or the packed memory permutation. Add a Colab-only diagnostic/probe or a minimal alternative adapter that tests the two k16 halves against the proven SM75 `u4*u4` pair decomposition. Accept a production repair only if all large-M smoke, Qwen-shaped, pure INT4, timing-integrity, and existing regression checks pass. Until then v271 remains the safe baseline and v294/v295 remain rejected.

## Colab artifact audit

The current runner was still pinned to commit prefix `7be8661` (v291) and cloned the committed full gate notebook, while the helper script generated a v293 operator artifact. That would silently validate stale source. The v296 validation path therefore updates the runner’s immutable commit pin and workspace labels to the v296 commit, and updates the helper’s versioned source/output names. The gate cells, model, benchmark rows, thresholds, and operator-only quality skip remain unchanged.

## Local Colab upload fallback

The GitHub CLI credential helper is currently invalid in the sandbox, so a GitHub push cannot be used as the transport for this validation attempt. The official Colab CLI exposes `upload` and `exec -f` for an existing session. The runner is therefore made transport-independent: when `SHMQ_NOTEBOOK_PATH` is set, it skips the remote Git clone/pin and executes the explicitly uploaded v296 operator notebook, while retaining the same dependency installation, T4/SM75 assertion, notebook cells, benchmark inputs, thresholds, timing-integrity logic, and operator-only quality semantics. This changes only artifact transport and cannot change the measured kernel.

## Stronger iterator mismatch

The direct type comparison found a concrete divergence: original `MQMmaMixedInputTensorOp` constructs `IteratorA` with `MatrixShape<ArchMmaOperator::Shape::kM, ArchMmaOperator::Shape::kK>`, which is `<16,32>` for the original SM80 `m16n8k32` operator. SHMQ’s SM75 adapter instead used `<8,32>` because its internal legal MMA is `m8n8k16`. The widened fragment still has the same total element count, so static assertions do not detect this difference, but the row-major crosswise iterator’s lane/LDSM mapping is instruction-shape dependent. The next candidate changes only SHMQ’s A iterator shape back to the original widened `<16,32>` contract while retaining the legal SM75 `m8n8k16` arithmetic and the two-half accumulation.

## v297 vertical-visit mismatch

A second direct comparison found a concrete SM75 architecture contract that the custom adapter omitted. CUTLASS `MmaTensorOp` sets `kVerticalVisit = true` for `__CUDA_ARCH__ < 800`, and its SM75 operator visits `n` outer / `m` inner with `m_serpentine`, indexing row-major accumulators as `n + m_serpentine * MmaIterations::kColumn`. SHMQ’s custom adapter copied the SM80/nonvertical `m` outer / `n_serpentine` order unconditionally. The adapter’s two k16 calls must remain, but their logical MMA visitation and A/B register indices should follow CUTLASS’s vertical SM75 rule. This is the next isolated correctness repair; the generic SM75 transform also confirms that A should not receive an SM80-only shuffle when the SM75 iterator already supplies the vertical TensorOp register layout.


## External source cross-checks for v298 research

- NVIDIA CUTLASS main’s `include/cutlass/arch/mma_sm75.h` publishes the SM75 signed INT8 Tensor Core ABI as `m8n8k16`, with `FragmentA = Array<int8_t, 4>`, `FragmentB = Array<int8_t, 4>`, and `FragmentC = Array<int, 2>`, and emits `mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32`. Source: https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/mma_sm75.h
- NVIDIA’s PTX ISA documentation is the authoritative instruction reference, but its current HTML extraction did not expose a detailed m8n8k16 fragment table in the retrieved section. Source: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- The repository’s original MixLLM adapter still uses the SM80-style widened `m16n8k32` fragment contract and two 16-byte A halves, while the SM75 legal instruction exposes only four A bytes and four B bytes per lane. Therefore a correct SM75 emulation cannot be justified solely by pointer reinterpretation; the lane-wise mapping between the widened shared-memory fragment and each legal k16 instruction must be proven or replaced by a direct SM75-native dataflow.


## v298 path localization

- The current v3 entrypoint proves the failure is path-specific. Rows 1 use the decode kernel and rows 2–31 use the expanded INT8 CUTLASS path; these shapes pass. Mixed rows>=32 enter `run_unified_prefill`, where INT4 is sent to `run_cutlass_packed_int4_partition` and INT8 is sent to the ordinary signed-INT8 runner. Mixed rows=128 therefore isolates the packed B adapter. Pure INT4 rows>=32 are a separate `run_int4_pair_partition` WMMA kernel and must be audited independently; their large-M error is not evidence that the packed CUTLASS adapter’s visitation is wrong.
- The original MixLLM `linear.py` constructs `weight_int4` by applying the two-stage K permutation and nibble packing once, then passes that tensor directly to the CUTLASS runner with global `LayoutB=ColumnMajor`. SHMQ’s Python cache reproduces the same permutation exactly, and its unit test compares byte-for-byte. The remaining packed-path discrepancy is therefore between the cached bytes and the SM75 shared-memory/warp fragment interpretation, not the high-level quantization or metadata orientation.
- The original SM80 `m16n8k32` MMA exposes `FragmentA=16` signed bytes, `FragmentB=8` signed bytes, and `FragmentC=4` integers. The legal SM75 `m8n8k16` instruction exposes half-sized operand fragments and two accumulator integers. A naive reinterpretation/split of the widened SM80 fragment is not source-proven to preserve lane ownership. The next candidate should use a direct SM75-native shared-memory arrangement or an explicitly proven lane remap, rather than another loop-order change.


## v299 diagnostic control

The v298 B-group repair reduced but did not eliminate mixed rows=128 error (`818.6` to `740.1`), confirming that the packed adapter’s register grouping is one mismatch but not the whole contract. Pure INT4 rows=128 also remains corrupt through the separate fused WMMA pair, while pure INT8 rows=128 passes; this separates the remaining problems into packed INT4 paths rather than the shared stream/event or INT8 CUTLASS pipeline.

Before attempting another native fragment mapping, v299 will disable only the cached native-v3 selection in the Python dispatcher and force the existing expanded-INT4 plus staged signed-INT8 CUTLASS path for rows>=32. This is the original-safe control: v271 used the expanded INT4 representation and its large-M correctness was accepted. It changes no arithmetic, tolerances, model, benchmark, or quality settings. The diagnostic result will establish whether the current gate’s large-M error is entirely below the packed adapter boundary. If correctness returns, the control will be rejected as a performance candidate if it fails the prefill gate, then the native repair will target on-the-fly unpacking into the already-correct signed-INT8 SM75 dataflow rather than guessing another opaque fragment permutation.


### v300 refinement before implementation

A second direct comparison found a more immediate inconsistency than the synthetic-policy hypothesis: `MmaTensorOpDequantizer::apply_zero()` indexes the transformed B fragment as `[k_group][n]`, i.e. half-major (`operand_frag_ptr[k_group * MmaIterations::kColumn + n]`). v298 changed only `operator()` to consume B as n-major (`[n][k_group]`) but left zero-point application half-major. Because Qwen’s zero points vary by output channel, this explains the residual large-M corruption and is a stronger isolated repair target. v300 will therefore preserve the pipeline’s established dequantizer ABI: reorder the converted n-major B fragment to half-major in `transform()`, then use the original `ptr_B[n]` / `ptr_B[kColumn+n]` MMA indexing. The synthetic k16 policy remains a separate later hypothesis and will not be changed in the same candidate.
