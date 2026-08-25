# v328 CUTLASS accumulator-store contract

## Evidence

CUTLASS's SM75 subbyte WMMA test defines `WarpShape = cutlass::gemm::GemmShape<8, 8, 32>` and `InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>`, then obtains the complete warp operator through `DefaultMmaTensorOpWmma`. The test harness constructs `Mma::IteratorA` and `Mma::IteratorB` using `cutlass::arch::LaneId()`, repeatedly loads `Mma::FragmentA` and `Mma::FragmentB`, invokes the complete `Mma` object, and finally stores `Mma::FragmentC` through `Mma::IteratorC`. This is the authoritative complete path in `cutlass/test/unit/gemm/warp/testbed.h`.

The custom probe had previously instantiated `MmaTensorOpAccumulatorTileIterator` directly while also using a second direct Crosswise probe. v326 failed with explicit missing `kM/kN/kMN` members when `MatrixShape<8,8>` was passed as the instruction shape. v327 changed that argument to `gemm::GemmShape<8,8,32>`, which supplies the required static fields, but T4 still terminated nvcc with exit code 255 and no compiler diagnostic. The current v328 isolation removes the second accumulator-iterator instantiation from the direct b16 Crosswise probe while retaining one accumulator iterator in the separate WMMA/native full-matrix proof.

## Interpretation

The original CUTLASS path couples iterator fragments, MMA policy, and accumulator store mapping as one complete type. It does not use a hand-created accumulator iterator as an independent proof of arbitrary native fragment ownership. Therefore the next T4 result must first establish that the isolated single accumulator iterator compiles. If it does, the direct Crosswise loader/MMA diagnostic and the complete WMMA/native-store matrix proof can be treated as separate seams. If it still fails with code 255, the next correction should replace the hand-instantiated accumulator store with the complete CUTLASS `DefaultMmaTensorOpWmma`/`Mma::IteratorC` path, not another shape or tile guess.

Production remains unchanged: v299 expanded INT4 + staged CUTLASS control, with native v3 dispatch disabled. No correctness, quality, memory, ABI, or benchmark gate is relaxed by this research note.
