# v292 next-seam primary-source research

## Differential finding

Microsoft MixLLM defines two launcher families. Its row-major `gemm_rm()` instantiates `extension_cpp::Testbed<MmaCore, true>`, while the column-major family uses the default accumulator-order flag. The original row-major path therefore explicitly tells the CUTLASS testbed that accumulators must be interpreted in row-major order.

SHMQ’s packed runner uses `LayoutC = RowMajor`, but `CorePackedInt4M64N64` omits the final `AccumulatorsInRowMajor` template argument, leaving it at CUTLASS’s default `false`. The custom `MQMmaPackedInputTensorOpSm75` adapter uses that flag directly when mapping its M/N MMA fragments. v291’s pattern is consistent with this specific defect: rows 1 and 16 use the expanded fallback and are numerically correct, while rows 32/128 reach the packed runner and show large errors. The packed instruction and WMMA probes pass, so the failure is not evidence that the MMA opcode itself is illegal.

## Safe candidate

Instantiate only `CorePackedInt4M64N64` with `AccumulatorsInRowMajor=true`, matching the original row-major testbed contract and the SHMQ output layout. Do not alter the packed permutation, synthetic MMA arithmetic, tile geometry, dispatch threshold, benchmark, model, or fallback. The candidate must first pass large-M correctness at smoke and Qwen shapes; otherwise it is reverted immediately.

## References

[1]: https://raw.githubusercontent.com/microsoft/MixLLM/main/mixllm/kernels/mix_mma_multistage.cuh
[2]: https://raw.githubusercontent.com/microsoft/MixLLM/main/mixllm/kernels/mma_multistage_testbed.h
[3]: https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/layout.html
