# v295 next-seam research: native SM75 INT4 is a coordinated layout/iterator contract

## Primary-source comparison

The original Microsoft mixed-input warp wrapper does not define a native SM75 INT4 instruction. It uses an operand adapter around the legal SM75 `m8n8k16` signed INT8 tensor operation, converting the compact B fragment to INT8 and applying the original A-fragment shuffle. This is why a permutation-only repair cannot establish correctness: the surrounding iterator, shared-memory layout, and fragment ABI must agree with that instruction.

CUTLASS’s authoritative SM75 `DefaultMmaCore` for plain row-major A and column-major B selects `RowMajorTensorOpMultiplicandCrosswise<element_bits, Shape::kK>` for A and `ColumnMajorTensorOpMultiplicandCrosswise<element_bits, Shape::kK>` for B. Its interleaved specialization instead changes the global layout tags to `ColumnMajorInterleaved<InterleavedK>`/`RowMajorInterleaved<InterleavedK>`, uses `TransposePitchLinearThreadMap` for both operands, and still writes through the matching SM75 crosswise shared layouts. The layout’s offset formula includes vector grouping, fundamental-tile partitioning, XOR swizzles, and a stride proportional to `kCrosswise * stage`; these operations are not represented by a register-level fragment shuffle.

CUTLASS’s native Turing INT4 device defaults are threadblock `128x256x128`, warp `64x64x128`, instruction `8x8x32`, two stages, with the legal SM75 `OpMultiplyAddSaturate` operator. NVIDIA’s Turing documentation lists `8x8x32` INT4 and `8x8x16` INT8 as distinct operations and explains that `ldmatrix` distributes shared-memory rows into the exact warp register arrangement consumed by `mma.sync`.

SHMQ’s packed core is structurally different: its global ABI is plain row-major INT8 A / column-major compact uint4 B; its custom `CorePackedInt4M64N64` uses `32x128x64` threadblock and `32x32x64` warp geometry; its adapter advertises a logical K32 load but actually converts uint4 to int8 and issues two K16 s8*s8 instructions; and its synchronous pipeline uses a custom stage/iterator pairing. v294 proved that even after making the operand transform match the original wrapper, large-M output corruption remained (mixed smoke rows32/128 212.17/266.48, Qwen mixed rows128 818.57, pure INT4 rows128 877.45). Therefore the corruption is a deeper cross-layer contract failure, not merely A/B permutation.

## Design consequence

A legitimate packed candidate would need one self-consistent design across: (a) the global packed INT4 layout and its logical matrix tags, (b) the SM75 crosswise shared-memory layout and 128-bit access map, (c) the warp fragment iterator and exact `m8n8k32` or decomposed `m8n8k16` register ABI, (d) the dequantizer’s scale/zero correction, and (e) the row-major output/scatter mapping. Reusing the current synthetic K32 adapter while changing only one layer is rejected by the v294 evidence.

An exact native INT4 decomposition is arithmetically possible in principle for signed INT8 activation `a`: write `a = lo_u4 + 16*hi_s4`, with `lo_u4` in `[0,15]` and `hi_s4` in `[-8,7]`, then use legal `u4*u4` plus `s4*u4` MMAs. However, it would require new activation fragment packing and new scale/zero correction accounting; it is not a safe edit to the current dequantizer or packed weight adapter. No implementation or performance claim is made here.

## Next action

Do not submit a permutation or dispatch patch. Build an isolated host/reference contract test for the decomposition and a CUDA-only probe specification covering all fragment lanes, K halves, interleaved layout offsets, n4 tails, mixed 4/8/16 partitions, and rows 32/128. Only if the exact probe can be implemented and passes should a narrow production branch be considered. In parallel, audit the safe expanded v299 launcher for source-backed overhead reductions that do not alter its arithmetic.

## References

[1]: https://raw.githubusercontent.com/microsoft/MixLLM/main/mixllm/kernels/cutlass_extension/mq_mma_mixed_input_tensor_op.h
[2]: https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/threadblock/default_mma_core_sm75.h
[3]: https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/layout/tensor_op_multiplicand_sm75.h
[4]: https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/device/default_gemm_configuration.h
[5]: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html
