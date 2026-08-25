# v338 source-backed design for the next native SM75 proof

## Established contract

The vendored CUTLASS `MmaTensorOpAccumulatorTileIterator` row-major implementation is explicit about its ownership model. In `shmq-ultimate/external/MixLLM/mixllm/kernels/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator.h:3258-3263`, an accumulator tile is treated as replicated 8-by-8 tiles, with each quad mapped to one output row and each lane mapped to one quarter of that row. The constructor at lines 3289-3300 applies `quad = lane_id >> 2` and `lane_in_quad = lane_id & 3`, with a column offset of `lane_in_quad * (InstructionShape::kN / 4)`. The store implementation at lines 3413-3449 writes those lane-owned values back to their logical matrix coordinates. The positive v338 result validates this mapping for the manually combined native SM75 pair-MMA fragment.

The native pair aliases in `mq_mma_sm75_int4_pair.h:19-38` are the legal instruction contracts: `m8n8k32`, U4/U4 for the low plane, and S4/U4 for the signed high plane. Each exposes eight operand elements and two accumulator elements (`:40-53`). The physical Crosswise U16 aliases at `:91-105` deliberately use `uint16_t` storage because the SM75 Crosswise path transfers b16 words; the probe then repacks the low nibble of each loaded word into the native four-byte operand register. The positive v338 T4 result validates this loader/repack seam for uniform values.

## Next bounded experiment

Extend the existing research-only three-case full-matrix kernel to four case warps in one block. Cases 0–2 remain unchanged and preserve the exact uniform expectations 64, 1024, and -960. Case 3 uses nonuniform logical matrices packed into the WMMA-compatible shared tiles: A low nibbles use `((row * 3 + k * 5 + 1) % 16)`, A high raw nibbles use `((row * 5 + k * 3 + 7) % 16)` and are interpreted as signed int4 (`raw >= 8` becomes `raw - 16`), and B uses `((k * 7 + col * 3 + 2) % 16)`. A is stored row-major with 32 logical elements per row, and B is stored column-major with 32 logical elements per column, matching the existing `wmma::load_matrix_sync(..., 32)` calls.

The reference for case 3 is computed independently as `C[row,col] = sum_k (A_low[row,k] + 16 * A_high_signed[row,k]) * B[k,col]` over 32 K elements. The gate must compare the complete returned 8-by-8 matrix against this reference, not only a lane-local fragment or a selected row. The edit must not touch production dispatch, v299 expanded-INT4 control, indexed scatter, model quality, memory accounting, timing logic, or any expected gate threshold.

## Architectural implication

If the four-case T4 proof passes, the evidence will establish that the physical Crosswise b16 loader/repack and the native two-MMA accumulator/store seam work for varied signs, zeros, and values—not merely repeated uniform values. It will still not prove block-level K tiling, tails, quantization scale/zero correction, indexed scatter, or production performance. A failure must be classified from the full matrix payload and the exact reference; expectations must not be adapted to the observed output.

## Primary sources

1. Vendored CUTLASS accumulator ownership and store mapping: `shmq-ultimate/external/MixLLM/mixllm/kernels/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator.h`, lines 3258–3449.
2. Vendored SM75 native U4/U4 and S4/U4 aliases and fragment sizes: `shmq-ultimate/external/MixLLM/mixllm/kernels/cutlass_extension/mq_mma_sm75_int4_pair.h`, lines 19–53.
3. Vendored SM75 physical Crosswise U16 iterator aliases: the same header, lines 66–105.
4. Current positive T4 evidence: `research-15/worklog.md`, v338 entry and `research-15/kaggle-v338-log-only/mixllm-3-level-real-t4-gate.log` in the connected checkout.
