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

## Follow-up physical-loader proof

The WMMA-loaded nonuniform proof validates native accumulator ownership but does not by itself prove that varied physical Crosswise U16 words arrive in the same native operand slots. The next proof therefore uses the existing positive `NativeWarpU16AIterator`, `NativeWarpS16AIterator`, and `NativeWarpU16BIterator` directly, repacks all eight loaded values per lane exactly as the positive probe does, invokes the two native MMAs, and stores the complete 8x8 result through the same CUTLASS accumulator iterator.

The physical test uses one warp and the already-established full Crosswise backing extents A=`<64,128>` and B=`<128,64>`. Logical A rows 0–7 and B columns 0–7 are filled with the same deterministic nonuniform formulas as the WMMA reference. Because vendored CUTLASS's `add_tile_offset` for Crosswise advances by the instruction's strided dimension and physical layout factor, the first iterator load is the logical origin tile; the full matrix reference is therefore formed over A[0:8,0:32] and B[0:32,0:8]. No guessed lane-to-matrix mapping is used: the accumulator iterator remains the authoritative store mapping.

Historical compile risk is explicit. v327/v328 showed that separate accumulator-iterator instantiations in two kernels could terminate nvcc with code 255. The direct physical-loader full-matrix proof is consequently bounded and may need to share the existing iterator instantiation or be reduced if the compiler reproduces that failure. A compile failure will remain negative evidence, and no production behavior will be changed.

## Multi-K extension

The next bounded proof will use K=64 represented by two consecutive native m8n8k32 operations. CUTLASS’s Crosswise RowMajor and ColumnMajor wrappers delegate `operator++` to the underlying warp iterator; the source documents that this is the iterator’s advance dimension, and the wrapper’s `add_tile_offset` swaps row/column coordinates into the pitch-linear base. The proof will therefore call `++u16a`, `++s16a`, and `++u16b` between loads rather than inventing byte offsets. A and B shared backing storage will be initialized through the same Crosswise `Layout::operator()` mapping for logical K values 0–63, with unused extent zeroed.

Each K tile is loaded, repacked, and accumulated into the same cleared native `FragmentC` pair. The final combined fragment is stored once through the CUTLASS accumulator iterator and checked against an independent CPU int64 reference over all 64 K terms. A positive result would validate iterator advancement and multi-instruction accumulation in addition to the already positive K=32 origin-tile seam; a failure will identify whether the defect is the Crosswise increment, physical backing extent, fragment repack, or accumulation.

## M/N tile-offset extension

After the K64 pass, the remaining iterator geometry seam is a non-origin M/N tile. A separate one-warp proof will initialize A rows 0–15 and B columns 0–15 for K=32, then use CUTLASS `add_tile_offset({1,0})` on the RowMajor A iterator to select the next 8-row tile and `add_tile_offset({0,1})` on the ColumnMajor B iterator to select the next 8-column tile. Four independent 8x8 outputs (A0/B0, A0/B1, A1/B0, A1/B1) will be stored in separate matrix slices, avoiding any unproven global 16x16 store stride while still testing the iterator’s logical M/N offset mapping. Each tile will be compared against an exact CPU reference for its corresponding row and column ranges.

This proof remains separate from production and from the K64 proof. It is specifically intended to distinguish a correct K advance from a correct M/N tile advance and to exercise the same physical Crosswise U16 backing layouts under nonuniform signed/unsigned values.

## Four-warp block-level extension

The four one-warp M/N cases now pass, but production uses blocks and shared-memory barriers. The next bounded harness will launch four warps in one 128-thread block, initialize the same physical Crosswise U16 A/B tiles cooperatively with `__syncthreads()`, and assign warp IDs to the four M/N combinations. Each warp will construct its own iterator from the common shared backing storage, apply the source-backed A row or B column tile offset, execute the native pair MMA, and store into a disjoint 8x8 output slice. The exact CPU reference remains independent. This tests co-resident warp ownership, barrier visibility, shared-memory layout reuse, and output-slice isolation without introducing production dispatch or a guessed threadblock MMA core.
