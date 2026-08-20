# v235 mixed-path audit: original MixLLM versus SHMQ SM75

## Scope

This audit compares the original MixLLM runtime contract with the current SHMQ SM75 implementation before repairing the v234 validation blocker. The objective is to distinguish a genuine SHMQ correctness bug from an intentional ABI difference.

## Evidence table

| Concern | Original MixLLM | Current SHMQ SM75 | Finding |
|---|---|---|---|
| Mixed native output shape | `kernels.cu::gemm` allocates `[N, M]` when both INT8 and INT4 partitions exist; `linear.py` transposes that result after the native call. | `three_level_linear_v2_core` always allocates `[rows, output_width]` and all branches scatter directly into global output columns. | This is an intentional SHMQ ABI redesign. It is correct only if every SHMQ kernel uses the row-major destination and global channel indices consistently. |
| Partition indices | Original constructor keeps `indices_int8` and `indices_int4` as global output-channel indices and vLLM concatenates them into one global `indices` tensor. | SHMQ validates `cat(indices_4, indices_8, indices_16)` as a complete permutation of `[0, out_features)`, and each native branch scatters through the corresponding global index tensor. | The global-index invariant matches the original serving contract. |
| INT4 packed layout | Original `interleave_uint4_for_cutlass` applies a 32-column permutation, then an 8-column lane permutation, then packs adjacent nibbles. | SHMQ prefill fallback receives a cached expanded signed INT8 `[n4, K]`; the pure INT4 fused path receives the packed ABI and performs its own SM75 pair arithmetic. | Expanded fallback avoids relying on the original packed CUTLASS layout, while fused path must preserve the packed ABI. |
| CUTLASS destination stride | Original runner writes the native mixed result in its own `[N, M]`/single-partition orientation and Python handles the mixed transpose. | `sm75_cutlass_testbed.h` writes `ptr_C[global_row * ldc + global_index]`; `Runner::run` passes `matrix_C.stride(0)`, which is `output_width` for SHMQ’s row-major destination. | The SHMQ epilogue mapping is internally consistent with its redesigned output ABI; the historical rows=128 failure is therefore more likely in a tile/fragment mapping or metadata path than in the high-level transpose contract. |
| Large mixed dispatch | Original uses staged iterator-based Tensor Core GEMMs. | SHMQ uses parallel INT4/INT8 CUTLASS streams plus caller-stream FP16; mixed rows >= 32 currently sets `use_fused_int4=false` and uses the v228 CUTLASS fallback. | The observed mixed error predates fused mixed dispatch and must be debugged in the CUTLASS path or its inputs. |
| v234 diagnostic | Not applicable. | The v232 probe showed rows 0–7 written and rows 8–31 left at sentinel. v234 changes the warp mapping from `local_row=item/8, local_channel=warp*8+item%8` to `local_row=warp*8+item/8, local_channel=item%8`. | v234 is the correct candidate fix for the diagnostic failure, but it has not yet been compiled and executed on T4. |

## Most important discrepancy

The original mixed path has a transposed native-output ABI, whereas SHMQ deliberately has a row-major global-scatter ABI. This difference is not by itself a bug because the SHMQ C++ wrapper, CUTLASS epilogue, and direct WMMA path all target `[rows, output_width]`. The invariant that must be protected is: for every partition and every valid tile, `output[row * output_width + global_index]` is written exactly once.

The v232 sentinel failure violated that invariant in the diagnostic fused writeback: only the first eight rows were written. The v234 mapping correction targets exactly this issue. Because the v234 notebook push was not accepted, the next action must be a successful T4 validation of the v234 probe before changing mixed correctness logic.

## Next repair hypothesis after v234 validation

If v234 passes the sentinel probe but the historical mixed rows=128 error remains, compare the CUTLASS runner's fragment-to-row mapping and metadata indexing against the direct WMMA reference. The current CUTLASS epilogue uses `warp_m = warp_id % WarpCount::kM`, `warp_n = warp_id / WarpCount::kM`, and writes `global_row * ldc + index_fragment[...]`. The original upstream path's transpose behavior cannot be copied blindly; instead, the row-major SHMQ ABI must remain explicit and be validated with a partition whose global indices are deliberately non-contiguous.

## Sources

1. Original MixLLM linear module: `external/MixLLM/mixllm/nn/modules/linear.py`, constructor and `interleave_uint4_for_cutlass` at lines 13–157; mixed transpose at lines 202–221.
2. Original MixLLM vLLM adapter: `external/MixLLM/mixllm/nn/modules/linear_for_vllm.py`, combined global indices at lines 59–82.
3. Original MixLLM native GEMM: `external/MixLLM/mixllm/kernels/kernels.cu`, output allocation at lines 515–541.
4. SHMQ three-level module: `external/MixLLM/mixllm/nn/modules/three_level_linear.py`, partition state and SM75 dispatch at lines 139–242.
5. SHMQ SM75 dispatcher and row-major output allocation: `external/MixLLM/mixllm/kernels/three_level_sm75.cu`, lines 1190–1389.
6. SHMQ CUTLASS epilogue and destination stride: `external/MixLLM/mixllm/kernels/sm75_cutlass_testbed.h`, lines 95–197.

## T4 result and newly isolated root cause

The accepted v234 notebook did execute on T4. Its log confirms that the SM75 instruction, packed-load, and fused arithmetic probes passed, but the mixed-stride sentinel assertion still failed. The failure is therefore inside the production `sm75_int4_pair_gemm_kernel` writeback path, not compilation or the legal pair arithmetic.

The current kernel invokes CUTLASS `LowMma` and `HighMma` instruction fragments whose `FragmentC::kElements == 2`, then copies only `x[0]` and `x[1]` into a WMMA accumulator fragment whose logical 8x8 tile is later stored with `wmma::store_matrix_sync`. NVIDIA’s CUDA Programming Guide states that the internal mapping of fragment elements is unspecified and that individual matrix elements must be accessed from memory after calling `store_matrix_sync` [7]. It does not permit assuming that a two-element CUTLASS instruction fragment can be treated as a complete eight-by-eight WMMA accumulator by assigning only two registers. This is the leading explanation for the remaining T4 sentinel mismatch.

The corrected warp mapping in v234 is present and compiled, but it was not sufficient because the accumulator-conversion seam remains invalid. The next local repair should preserve the proven two-register CUTLASS fragment and use a deterministic shared-memory scatter compatible with the instruction fragment’s actual lane/register mapping, or replace the WMMA store conversion with a CUTLASS-supported fragment iterator. No mixed fused dispatch should be enabled until that path passes the T4 sentinel and numerical correctness gates.

## Primary external source

7. NVIDIA, *CUDA C++ Programming Guide*, WMMA API: fragment storage mapping is unspecified; `store_matrix_sync` must be used before elementwise access, and the accumulator fragment’s `x[]` layout is implementation-defined. See the official guide section around lines 8011–8062 in the extracted page: https://docs.nvidia.com/cuda/cuda-c-programming-guide/.

## Fragment-layout evidence

The official CUDA guide says that a WMMA fragment is distributed across the warp and that the mapping of matrix elements into fragment storage is unspecified; it explicitly requires `store_matrix_sync` before accessing individual matrix elements [7]. The extracted CUTLASS SM75 header independently shows that the legal `mma.sync.aligned.m8n8k32` INT4 specializations expose `FragmentC = Array<int, 2>` for each lane. Therefore, the current conversion that assigns only two CUTLASS registers into a larger WMMA accumulator and then relies on `wmma::store_matrix_sync` is not a valid complete-tile conversion. The correct repair must either write the two CUTLASS accumulator registers directly using their instruction-defined lane mapping or use a CUTLASS-supported fragment/epilogue iterator; it must not invent the rest of the WMMA accumulator fragment.

The primary sources consulted are NVIDIA’s CUDA C++ Programming Guide WMMA section and the NVIDIA CUTLASS SM75 `mma_sm75.h` specialization bundled in this repository’s vendor archive. The WMMA guide also notes that elementwise fragment access is only safe when applied uniformly to all fragment elements; it does not authorize partial initialization of a different fragment type.

## Exact PTX accumulator mapping

The NVIDIA PTX ISA documentation provides the missing mapping for `mma.m8n8k32` accumulator fragments [8]. For lane `laneid`, `groupID = laneid >> 2` and `threadID_in_group = laneid % 4`; the two accumulator registers held by that lane correspond to `row = groupID` and `col = (threadID_in_group * 2) + i` for `i = 0, 1`. Thus each lane owns two adjacent columns in one of eight rows. The production fused kernel can scatter the two CUTLASS registers directly as:

```cpp
const int local_row = warp * 8 + (lane >> 2);
const int local_channel = (lane & 3) * 2 + register_index;
```

This is the legal replacement for the invalid partial WMMA-accumulator conversion. It also explains why the v234 `item / 8` mapping was directionally wrong for the CUTLASS two-register fragment: `item` was indexing a fabricated row-major tile rather than the actual per-lane PTX fragment ownership.

8. NVIDIA, *PTX ISA 8.2*, section “Matrix Fragments for mma.m8n8k32,” lines around 32288–32408 in the extracted page: https://docs.nvidia.com/cuda/archive/12.2.0/parallel-thread-execution/index.html.
