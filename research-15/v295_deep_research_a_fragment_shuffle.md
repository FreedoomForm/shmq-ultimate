# v295 Deep Research: Missing Packed INT4 A-Fragment Shuffle

## v294 Colab evidence

Removing SHMQ’s extra B-fragment shuffle did not restore large-M correctness. The official Colab T4 operator notebook compiled successfully and reported 98 tests passing, but native packed INT4 still produced large errors: smoke mixed max error 212.165 at rows 32 and 266.186 at rows 128; Qwen-shaped mixed max error 818.567 at rows 128; pure INT4 max error 877.446 at rows 128 with `timing_integrity=False`. Thus v294 is rejected.

Rows 1 and 16 use the legacy path and remain numerically correct, which localizes the defect to the native packed large-M runner rather than the checkpoint quantization or reference model.

## Original-vs-SHMQ discrepancy

The original `MQMmaMixedInputTensorOp::transform` performs `tmp_A = shuffler_A(A)` before converting A into the instruction fragment. The original B path intentionally preserves `tmp_B = B`. SHMQ’s SM75 adapter previously applied a B shuffle, but it also used `FragmentA tmp_A = A` and never applied the original A shuffle. After v294 removed the incorrect B shuffle, this missing A shuffle is the remaining direct semantic difference in the warp transform.

The widened SM75 adapter must adapt the original A shuffle to its k32 fragment: use `FragmentShuffler<ElementAMma, ElementA, MmaIterations::kRow, FragmentA::kElements, MmaOperandA::kElements, Operand::kA>` on the full widened A fragment, then convert each half as before. This is a correctness-first experiment; no performance or quality claim is made until Colab numerical tests pass.

## Safety boundary

Change only the A-fragment transform. Preserve the original packed memory permutation, no-shuffle B behavior, split two-k16 MMA, core geometry, stream/event ordering, quantization, output mapping, and benchmark. If the full-width shuffler fails compile or correctness, revert to the v271-safe fallback rather than retaining an unsafe native path.
