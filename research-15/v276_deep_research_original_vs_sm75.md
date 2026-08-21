# v276 deep research: complete the embedded include graph

## Finding from v275

The v275 CUDA source did not reach any production gate. Kaggle version 272 compiled the notebook source with `nvcc -gencode=arch=compute_75,code=sm_75` and stopped at the first include from `mq_mma_tensor_op_sm75.h`: `fatal error: mq_mma_mixed_input_tensor_op.h: No such file or directory`.

This is a packaging defect, not evidence against the operator design or its performance. The header exists in the SHMQ tree and is the implementation dependency of the existing `MQMmaMixedInputTensorOp`; it was omitted from the explicit `SOURCE_FILES` manifest used by the notebook builder. The original MixLLM tree keeps this operator header beside the launcher and compiles it through the repository include path, whereas the Kaggle notebook materializes only the manifest entries under `/kaggle/working/mixllm-3level`. Therefore the correct repair is to add the existing header to the manifest, not to alter arithmetic, the SM75 operator specialization, the runner geometry, the cache layout, or any benchmark gate.

## v276 scope

The only implementation change is the addition of `mixllm/kernels/cutlass_extension/mq_mma_mixed_input_tensor_op.h` to the notebook embedding manifest. The v275 source, tests, persistent interleaved INT4 cache, v3 ABI, stream topology, model, quality checks, and benchmark settings remain unchanged. After the manifest repair, the complete local suite must pass and the notebook must be rebuilt from the committed source before one new Kaggle T4 run.

## References inspected

The original mixed-input operator is `/tmp/original-MixLLM/mixllm/kernels/mix_mma_multistage.cuh` and its SM80 custom implementation. The SHMQ staged SM75 path is `shmq-ultimate/external/MixLLM/mixllm/kernels/cutlass_extension/mq_mma_pipelined_sm75.h`; the new v275 adapter includes `mq_mma_mixed_input_tensor_op.h` directly. Kaggle version 272 is the authoritative compiler evidence.
