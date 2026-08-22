# v293 next-seam primary-source research

## Differential finding

The original Microsoft `MQMmaMixedInputTensorOp::transform()` does two important things: it applies `FragmentShuffler<..., Operand::kA>` to the loaded A fragment, while it deliberately uses `tmp_B = B` and does not shuffle B. Its `operator()` then uses the row-major accumulator flag to select the corresponding fragment mapping [1].

SHMQ’s custom SM75 packed adapter does the opposite: it applies `FragmentShuffler<..., Operand::kB>` to B and copies A without the original A shuffle. The adapter comments therefore describe the wrong operand transformation. The v291 large-M corruption is consistent with this concrete mismatch: the packed path reached the MMA probes successfully but produced large output errors only when the packed runner was used.

## Safe candidate

Change only `MQMmaPackedInputTensorOpSm75::transform()` to mirror the original transformation responsibilities: leave B in the iterator-provided layout and shuffle A with the original `Operand::kA` shuffler before converting its two widened halves. Keep the existing legal SM75 k16 MMA decomposition, packed input conversion, core geometry, dispatch guard, fallback, and benchmark unchanged. The candidate must compile and pass large-M correctness before any speed result is admissible.

## Reference

[1]: https://raw.githubusercontent.com/microsoft/MixLLM/main/mixllm/kernels/cutlass_extension/mq_mma_mixed_input_tensor_op.h
