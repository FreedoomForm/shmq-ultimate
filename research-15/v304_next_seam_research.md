# v304 next-seam research: instantiate the native SM75 INT4 core before designing a loader

## Source-grounded comparison

CUTLASS’s SM75 `DefaultMmaCore` for row-major A and column-major B derives the thread maps from the element bit width and a 128-bit access size. For native 4-bit operands, the logical thread arrangement and `RowMajorTensorOpMultiplicandCrosswise<4, kK>` / `ColumnMajorTensorOpMultiplicandCrosswise<4, kK>` shared layouts are materially different from SHMQ’s expanded INT8 core. The native warp instruction is m8n8k32 and the two legal operator variants required by the decomposition are U4/U4 and S4/U4. [1] [2]

Microsoft’s original multistage organization supplies the same CUTLASS core through its iterator and pipeline templates, but its SM80 mixed operator cannot be copied to Turing. SHMQ must retain the SM75-specific core and synchronous pipeline; no Ampere `cp.async` behavior is being imported. [3]

The v304 step is therefore limited to compiling a native `DefaultMmaCore<GemmShape<32,128,64>, GemmShape<32,32,32>, GemmShape<8,8,32>, uint4b_t, RowMajor, uint4b_t, ColumnMajor, int, RowMajor, OpClassTensorOp, 2, OpMultiplyAddSaturate>` alias and its derived iterators/policy. It does not consume checkpoint data, alter production dispatch, or claim correctness. This isolates whether the native core geometry itself is accepted by the vendored CUTLASS version before any packed loader or affine correction is attempted.

## Acceptance conditions

The compile-only contract must build for SM75 and expose the expected 4-bit shared layouts, warp count, and m8n8k32 policy. The existing v303 arithmetic probe must continue to pass. Any iterator/core compile failure is evidence that the geometry needs a different tile shape, not a reason to weaken the production route.

## References

[1]: https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/threadblock/default_mma_core_sm75.h
[2]: https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/mma_sm75.h
[3]: https://raw.githubusercontent.com/microsoft/MixLLM/main/mixllm/kernels/cutlass_extension/mq_mma_multistage.h
