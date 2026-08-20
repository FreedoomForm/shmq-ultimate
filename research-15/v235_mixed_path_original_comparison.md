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

## v236 T4 outcome and build-cache audit

The post-repair Kaggle run compiled the current embedded source manifest and ran 79 contract tests successfully. The three existing SM75 instruction probes passed. The mixed-stride probe still failed at the first exact-value assertion, so no correctness or performance gate result is admissible and the version is rejected.

A further differential audit found a validation risk in SHMQ’s loader: it calls `torch.utils.cpp_extension.load(name="mixllm_sm75_backend", sources=[three_level_sm75.cu])` with a fixed extension name and no source-versioned build directory. Kaggle logs repeatedly showed `ninja: no work to do` across notebook versions even when the embedded CUDA source hash changed. Original MixLLM also uses a fixed extension namespace, but its normal build runs in a clean source/build environment; the SHMQ gate’s persistent PyTorch extension cache can therefore reuse stale objects across notebook versions. The next safe repair is to version the JIT extension name from the SHA-256 of the exact embedded CUDA source, preserving the registered `mixllm_sm75` operator namespace while forcing the tested source to compile. This change affects only build provenance, not arithmetic, model quality, or benchmark settings.

## v237 T4 outcome and complete warp-tile audit

The source-digest loader worked: Kaggle v235 logged the exact current CUDA source hash, the versioned extension name `mixllm_sm75_backend_e7101384fbc3e471`, and fresh compilation. The mixed-stride probe still failed, so the problem is not stale compilation.

The remaining discrepancy is in SHMQ’s fused kernel geometry, not the PTX register mapping. Each of the four warps currently loads `A` from `warp * 8` rows and `B` from `warp * 8` channels, then scatters only that same diagonal 8x8 tile. Even after v234/v235 row mapping, every warp writes only its local eight channels; the full 32x32 probe therefore has missing channel columns. The original MixLLM staged GEMM organization covers a complete threadblock tile by iterating warp-level row/column subtiles, rather than assigning each warp only one diagonal subtile.

The safe v238 repair is to keep the legal SM75 m8n8k32 pair instructions and the exact arithmetic, but have each warp own one 8-column channel tile and iterate all four 8-row subtiles. Each warp will maintain four two-register accumulator fragments/partial pairs, load `A` for `row_tile = 0..3`, reuse its `B` channel tile, apply the corresponding row sums and activation scales, and scatter rows `row_tile*8 + lane/4` to columns `warp*8 + 2*(lane%4) + register`. This covers all 32 rows × 32 channels exactly once with no quality or benchmark change; the production grid remains the same.

## v238 T4 outcome and final warp-N mapping discrepancy

Kaggle version 236 compiled the current v238 source (the log contained the current CUDA hash and source-digest extension name), ran 80 contract tests successfully, and passed the three instruction probes. The mixed-stride probe still failed with the same diagonal pattern.

The original CUTLASS warp decomposition assigns both a warp-M and a warp-N coordinate. v238 added the complete four-way warp-M row loop, but its output-channel expression still used `channel_base + (lane & 3)*2 + register_index` without the warp-N offset. Thus warp 0 loaded B channels 0–7, warp 1 loaded B channels 8–15, and so on, but all four warps scattered into output channels 0–7 and overwrote each other. The remaining columns stayed at the sentinel. v239 must add `warp * 8` to the channel base consistently in correction, scale, and final scatter; no other arithmetic or benchmark setting changes are needed.

## v239 T4 outcome and diagnostic requirement

The terminal Kaggle log confirms v239’s exact CUDA source hash and source-digest extension name, so the tested binary is current. It ran 80 embedded tests successfully and passed the three basic SM75 probes, but failed the mixed-stride assertion before native correctness and performance gates. The visible tensor repr is abbreviated and cannot distinguish residual channel-offset, row-tile, or accumulator-value errors. The next iteration will add diagnostic-only printing of `torch.unique` counts and the first mismatch coordinates/values before the existing strict assertions. This changes no kernel, arithmetic, benchmark, or quality path and is required to avoid guessing from a shortened tensor repr.
