# v275 deep research: staged native INT4 operator feasibility on SM75

## Baseline and failed detours

The safe v271 large-M mixed path uses a CUTLASS `Int8Runner` over an eagerly expanded signed-INT8 view of the INT4 weights. Its Qwen rows=128 mixed GEMM is `0.464896 ms` and E2E is `0.733232 ms`, versus dense FP16 `0.245760 ms`; the E2E speedup is only `0.33517x`. The v272 and v274 direct native-pair experiments were rejected: v272 was correct but extremely slow, while v274 corrected v272's coverage bug but still reached only `0.17788x` E2E speedup at rows=128. These results rule out a simple dispatch or direct-WMMA geometry fix.

## Original MixLLM dataflow

The original module interleaves and packs INT4 weights once into a persistent `[n4, K/2]` buffer in `/tmp/original-MixLLM/mixllm/nn/modules/linear.py:99-157`. Its multistage launcher binds that buffer directly to an `ElementB_INT4 = cutlass::uint4b_t` runner in `/tmp/original-MixLLM/mixllm/kernels/mix_mma_multistage.cuh:193-208` and `/tmp/original-MixLLM/mixllm/kernels/mma_multistage_testbed.h:68-80`. The original launch asserts the interleaved `[partial_n_int4, K/2]` shape and passes the packed pointer directly to the staged iterator at `mix_mma_multistage.cuh:223-247`; no signed-INT8 expansion is present in the hot prefill path.

## SHMQ implementation seam

SHMQ's threadblock `DefaultMmaCore` is generic over `ElementB` and `Operator` in `/home/ubuntu/v204-shmq/shmq-ultimate/external/MixLLM/mixllm/kernels/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:241-312`. However, its generic warp dispatcher in `cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:74-111` only constructs the ordinary `arch::Mma` policy, and the file includes only the SM80 specialization at `:121`. The SM80 mixed-input specialization lives in `cutlass_extension/mq_mma_tensor_op_sm80.h:56-201`; it defines the mixed operator tag, selects the wider internal MMA operand, and routes through `MQMmaMixedInputTensorOp`.

The SM75 pipeline itself is largely generic over the iterator and operator. In `cutlass_extension/mq_mma_pipelined_sm75.h:167-185`, dequantizer selection is derived from the operator fragment element types; the copy loop is generic over `IteratorB::AccessType`. This means the missing depth is a narrow internal SM75 warp-operator policy/adapter, not a new public launcher interface. The current `sm75_cutlass_testbed.h:20-24` instead fixes `ElementB=int8_t`, which forces the large-M INT4 branch through the expansion.

## v275 decision

The next candidate should add a minimal SM75 mixed-input warp policy for the legal native `u4*u4` WMMA form, while retaining signed-INT8 activations through the same low/high pair arithmetic used by the proven native kernel. The implementation must not pretend that a single `int8*u4` SM75 WMMA exists: the adapter must issue two legal `u4*u4` operations over packed activation nibbles, reconstruct the signed activation product, and integrate with the staged iterator/metadata seam. If that policy cannot be made to compile and pass the existing probes locally, the candidate must be rejected before Kaggle. The launcher must continue to use the persistent fork/int4/int8 streams and the original output/scatter ABI.

This is the first candidate aimed at the actual upstream-vs-SHMQ divergence: persistent interleaved native INT4 B tiles in a staged, large-M dataflow. It changes no quality thresholds, benchmark shapes, model, weights, or arithmetic contract. The existing v271 path remains the fallback until every local gate passes and a single Kaggle T4 run validates the new branch.

## Primary sources

1. [Original MixLLM linear module](file:///tmp/original-MixLLM/mixllm/nn/modules/linear.py)
2. [Original MixLLM multistage launcher](file:///tmp/original-MixLLM/mixllm/kernels/mix_mma_multistage.cuh)
3. [Original MixLLM multistage testbed](file:///tmp/original-MixLLM/mixllm/kernels/mma_multistage_testbed.h)
4. [SHMQ SM75 threadblock core](file:///home/ubuntu/v204-shmq/shmq-ultimate/external/MixLLM/mixllm/kernels/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h)
5. [SHMQ generic warp dispatcher](file:///home/ubuntu/v204-shmq/shmq-ultimate/external/MixLLM/mixllm/kernels/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h)
6. [SHMQ SM80 mixed warp specialization](file:///home/ubuntu/v204-shmq/shmq-ultimate/external/MixLLM/mixllm/kernels/cutlass_extension/mq_mma_tensor_op_sm80.h)
7. [SHMQ SM75 staged pipeline](file:///home/ubuntu/v204-shmq/shmq-ultimate/external/MixLLM/mixllm/kernels/cutlass_extension/mq_mma_pipelined_sm75.h)
8. [SHMQ SM75 CUTLASS runner](file:///home/ubuntu/v204-shmq/shmq-ultimate/external/MixLLM/mixllm/kernels/sm75_cutlass_testbed.h)
