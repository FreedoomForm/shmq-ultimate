# v263 deep research: avoid per-call FP16 transpose materialization

The source-verified v262 run compiled and passed correctness and timing-integrity. Its Qwen mixed rows=128 E2E improved to `0.388131x`, but the required gates still failed. The v262 diff shows one new avoidable allocation in the hot path: `run_fp16_partition_cublas` calls `weight_fp16.transpose(0, 1).contiguous()` on every forward before `at::mm`.

The official MixLLM source prepares its INT4 operand in a kernel-native layout once in the module constructor and passes it directly to the launcher. SHMQ's FP16 partition is newly using a dense matrix multiply, so the analogous rule is to pass the existing `[n16,K]` module buffer as a transpose view rather than eagerly materializing a `[K,n16]` copy on every call. PyTorch's `at::mm` accepts strided 2D operands; the transpose view has valid strides `(1, n16? no: original row stride K)`, and the cuBLAS-backed dispatcher can either use the view directly or choose its own internal layout conversion. The explicit `.contiguous()` guarantees a new copy and therefore cannot be cheaper than allowing the backend to make that decision.

This is deliberately narrower than adding a new cache or changing the operator ABI. It preserves the exact operands, FP16 output, matrix dimensions, and index scatter. It also keeps the helper on the caller stream after the integer fork. The candidate is only a measurement of whether removing one host-dispatched allocation improves E2E; if the backend performs worse on T4, v262 remains the measured improvement and v263 is rejected.

Primary sources used: original MixLLM `linear.py` lines 99-104 and 121-157, original staged launcher `mix_mma_multistage.cuh` lines 228-258, and SHMQ v262 `three_level_sm75.cu` helper lines 1148-1168.
