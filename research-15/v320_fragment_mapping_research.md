# v320 SM75 INT4 Fragment-Mapping Research

## Question
Why does the production-layout Crosswise warp iterator compile and load logical values, but expose alternating zero slots and produce four pair-MMA accumulators of 32 rather than the earlier naïve expectation of 64?

## Evidence from the T4 gate

Kaggle version 318 (the v320 source probe) passed the instruction and native decomposition probes, then failed only at the expanded iterator assertion. The complete first-lane payload was:

`A_u4 = [1, 0, 1, 0, 1, 0, 1, 0]`

`A_s4 = [1, 0, 1, 0, 1, 0, 1, 0]`

`B_u4 = [2, 2, 2, 2, 2, 2, 2, 2]`

`C_pair = [32, 32, 32, 32]`

All 32 lanes had the same payload. Therefore v320 did not validate the naïve all-one logical fragment contract; it exposed a deterministic storage/fragment mismatch. It also shows that the pair MMA consumes the values actually presented by the register fragments, not an assumed dense eight-nibble sequence.

## Local CUTLASS source findings

The vendored CUTLASS SM75 iterator implementation defines the Crosswise wrappers as dimension-reversing wrappers over the generic `TensorOpMultiplicandCongruous<sizeof_bits<Element>, 64>` implementation. The RowMajor and ColumnMajor Crosswise wrappers preserve the base iterator's LDSM pointer arithmetic and map matrix coordinates to PitchLinear coordinates. The iterator's `Fragment` is `Array<Element, Shape::kContiguous * InstructionShape::kStrided / 32>`; with the current matrix shapes this is eight logical subbyte elements per lane.

The iterator's load implementation reinterprets the subbyte fragment as `Array<unsigned, Policy::LdsmShape::kCount>` and invokes `ldsm<ColumnMajor, ...>`. The SM75 `ldsm<ColumnMajor, 4>` wrapper emits `ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16`, which returns four 32-bit registers. CUTLASS `Array<T,N,false>` packs logical subbyte values into storage items, while the LDSM instruction itself moves 16-bit matrix words. Consequently, observing `[1,0,1,0,...]` is consistent with a register-level 16-bit-word layout being read through a dense 4-bit logical accessor; it is not proof that the Crosswise backing-store fill matches the MMA operand register contract.

The primary CUTLASS sources also make clear that warp fragments are intentionally lane-local and non-contiguous. The official tile-iterator documentation describes a fragment as each thread's part of a tile, not a continuous matrix slice. NVIDIA's CUTLASS issue #1635 explicitly records that the per-thread `ldmatrix` fragment is not a continuous 16x16 matrix fragment and that these are internal ISA details. Thus the four accumulator slots cannot be assigned matrix coordinates by scalar intuition; a full warp-level reconstruction is required.

## Original MixLLM architectural implication

The original MixLLM/CUTLASS organization relies on a complete threadblock dataflow: a legal shared-memory layout feeds warp iterators, warp fragments are passed directly into the MMA operator, and an epilogue reassembles the warp-owned accumulator fragments. It does not treat the first element of a fragment as a scalar matrix result. The current SHMQ research probe is intentionally below that level, so the next proof must compare a full warp output to a reference matrix after reconstructing the documented CUTLASS fragment ownership.

## References

1. NVIDIA, *CUTLASS Tile Iterator Concepts*: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/tile_iterator_concept.html
2. NVIDIA, *PTX ISA 9.3*: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
3. NVIDIA CUTLASS issue #1635, *ldmatrix instruction*: https://github.com/NVIDIA/cutlass/issues/1635
4. Vendored CUTLASS SM75 iterator: `shmq-ultimate/external/MixLLM/mixllm/kernels/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator.h`
5. Vendored CUTLASS SM75 LDSM wrappers: `shmq-ultimate/external/MixLLM/mixllm/kernels/cutlass/include/cutlass/arch/memory_sm75.h`
6. Vendored CUTLASS subbyte Array: `shmq-ultimate/external/MixLLM/mixllm/kernels/cutlass/include/cutlass/array_subbyte.h`
7. SHMQ T4 evidence: `research-15/kaggle-v320-log-only/mixllm-3-level-real-t4-gate.log`

## Decision

Retain v320 only as diagnostic evidence; do not relax its expected values to call it a pass. Do not enable native production dispatch. The next experiment should create an explicit per-lane raw-register dump and a warp-level output reconstruction/reference proof, preferably reusing CUTLASS's own warp-level MMA/epilogue mapping rather than inventing a scalar interpretation.
