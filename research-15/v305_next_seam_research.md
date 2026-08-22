# v305 next-seam research: narrow native core to a four-warp 32x64x64 tile

The v304 T4 compile reached the native CUTLASS core instantiation and failed before execution. For the proposed 32x128x64 / 32x32x32 / m8n8k32 core, CUTLASS derived an A `PitchLinearWarpRakedThreadMap` with `Shape=<64,32>`, `Threads=256`, `WarpThreadArrangement=<2,16>`, and `ElementsPerAccess=32`; its iteration count is zero. The additional static assertion that the derived U4 fragment has four bytes was also false. Therefore the eight-warp 32x128 geometry is not a valid native subbyte core under this CUTLASS specialization.

The compiler evidence suggests the next bounded compile-only candidate should reduce the threadblock N tile to 64, yielding four warps while retaining the required K64 staging and a warp shape of 32x32x32. This changes no production code or dispatcher. The candidate is only `DefaultMmaCore<32x64x64, 32x32x32, 8x8x32, uint4b_t, RowMajor, uint4b_t, ColumnMajor, int, RowMajor, OpClassTensorOp, 2, OpMultiplyAddSaturate>`, with the expected four-warp resource envelope. The fragment-size assertion is removed because the v304 compiler disproved that assumption; the native CUTLASS Array layout remains the authority.

If v305 compiles, the next probe must inspect the derived iterator access types and actual shared-memory tile loading before production dispatch. If it fails, the native core seam is narrower still and no production packed path should be attempted.

## References

[1]: https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/threadblock/default_mma_core_sm75.h
[2]: https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/array_subbyte.h
