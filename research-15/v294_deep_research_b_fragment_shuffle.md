# v294 Deep Research: Packed INT4 B-Fragment Shuffle

## v293 evidence

The v293 Colab T4 operator notebook compiled successfully and exposed large-M native packed INT4 corruption: max absolute error was 191–826 for rows >= 32. The Python byte permutation matched the original MixLLM routine byte-for-byte, so the permutation itself was not the immediate suspect.

## Original MixLLM comparison

The authoritative original `MQMmaMixedInputTensorOp::transform` in `kernels/cutlass_extension/mq_mma_mixed_input_tensor_op.h` constructs a `FragmentShuffler` for B but explicitly does not invoke it: the source contains `//tmp_B = shuffler_B(B);` followed by `tmp_B = B`. The original packed INT4 memory preparation already supplies the layout expected by its iterator/warp operand path.

SHMQ’s SM75 adapter in `mq_mma_tensor_op_sm75.h` diverged from that behavior. v280–v291 invoked `FragmentShuffler` on the loaded packed B fragment before conversion and then split the widened fragment into two k16 MMAs. This was an unverified semantic change, despite the earlier comment claiming it matched upstream.

## Safe candidate

For v294, change only the SM75 packed adapter’s B transform to preserve the loaded fragment (`FragmentB tmp_B = B`), matching the original disabled-shuffle behavior. Keep the original two-stage byte permutation, the widened k32 iterator, the A conversion, the two legal SM75 k16 MMAs, stream/event topology, quantization, benchmark, and quality settings unchanged. Add a source contract requiring the no-shuffle assignment. Validate numerically on Colab before considering any speed result.
