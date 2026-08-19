# v191 deep research: restore original parallel precision streams

## v190 T4 evidence

v190 preserved correctness and decode gates. Mixed QKV rows=16 measured `0.2932x` E2E speedup and rows=128 `0.2312x`; this is an improvement over v189 rows=128 `0.2197x` but still fails the prefill gate by a wide margin. Pure INT4 rows=128 improved to `0.6390x`, and pure INT8 rows=128 to `0.7062x`, confirming that the unchecked cached CUTLASS wrapper removes some overhead but does not close the mixed-path gap.

The exact Qwen2.5-0.5B quality gate remained `unavailable_environment`: Kaggle accepted the `model_sources` metadata syntax after the full five-component handle was restored, but the expected filesystem path `/kaggle/input/qwen2.5/transformers/0.5b/1` was not mounted. This is now an environment/path discovery issue, not a silent quality pass; the report correctly remains `no_go`.

## Original-versus-SHMQ dataflow comparison

The original MixLLM launcher in `mix_mma_multistage.cuh` creates separate work streams for the precision partitions and joins them with CUDA events. INT4, INT8, and FP16 partition GEMMs write disjoint output-channel ranges, so they can overlap after the common activation/scale tensors are ready. The SHMQ v188-v190 core currently obtains the caller stream and launches the INT4 CUTLASS partition, then INT8 CUTLASS partition, then FP16 direct-WMMA partition sequentially on that same stream. Thus mixed prefill pays the sum of all three partition latencies even when the GPU could overlap independent work.

This is a stronger explanation for the mixed-only gap than activation quantization or metadata allocation: the pure INT4/INT8 paths are materially faster than the mixed path, while the mixed output combines all three precision engines. It also explains why v188's large-M CUTLASS improvement did not approach the theoretical memory-bandwidth bound.

## Safe experiment and correctness constraints

v191 should add an opt-in, shape-limited parallel launch only for rows>=32 and mixed integer/FP16 partitions. It must establish dependencies from the caller stream to worker streams before reading activation/weights, launch each precision partition on a worker stream, record completion events, and make the caller stream wait on all completion events before returning. Output ranges are disjoint by validated partition indices. All input, cached metadata, expanded INT4, and output tensors must be recorded on the worker streams for allocator lifetime. Rows=1 decode and rows<32 direct-WMMA remain unchanged. If the stream API or event dependency cannot be compiled cleanly on SM75, the experiment must be rejected rather than falling back to unsafe asynchronous writes.

The prediction is lower mixed rows>=32 E2E/GEMM time with identical outputs and no change to memory accounting. A negative result will be kept only as evidence and rolled back.

## References

1. Original MixLLM launcher: `external/MixLLM/mixllm/kernels/mix_mma_multistage.cuh`.
2. SHMQ v190 kernel: `external/MixLLM/mixllm/kernels/three_level_sm75.cu`.
3. SHMQ v190 backend: `external/MixLLM/mixllm/sm75_backend.py`.
4. v190 Kaggle T4 gate: `shmq-ultimate/mixllm_3level_kaggle/latest-output-v190-computer/mixllm_3level_gate.json`.
5. NVIDIA CUDA Runtime API stream/event documentation: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html


## Historical rejection guard discovered before implementation

The repository ledger contains the exact experiment that invalidates this otherwise plausible hypothesis: v153 introduced original-style parallel precision launches and failed native correctness because temporary transposed metadata was freed before auxiliary streams finished; v154 added allocator `recordStream` protection, restored correctness, but still measured only `0.20294x` mixed rows=16 and `0.29165x` mixed rows=128 E2E speedup and was rejected by the all-gates rule. Therefore v191 will **not** reintroduce stream parallelism. The new red contract was removed before any implementation. This finding narrows the remaining problem: concurrency alone is insufficient, and the direct-WMMA/CUTLASS arithmetic and tile dataflow must be improved without repeating v153/v154.
