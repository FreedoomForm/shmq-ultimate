

## Additional primary-source findings

The vendored SM75 `mma_sm75.h` explicitly defines the legal `m8n8k32` `uint4b_t * uint4b_t -> int` instruction with two accumulator registers per lane. The generic `mq_mma_tensor_op_dequantizer.h` is specialized for compute capability >=75 and supports a weight dequantizer with half scales and uint8 zero points. Its `apply_zero` path subtracts zero points from transformed operand-B fragments before MMA, and its `apply_scale_accum` path converts the integer accumulator and applies scales. This means the existing staged SM75 pipeline has a plausible direct-uint4 extension seam; the missing piece is a legal `DefaultMmaCore`/iterator/packing contract, not an unavailable hardware instruction.

The official Microsoft source uses `ElementB_INT4 = cutlass::uint4b_t`, `LayoutB = ColumnMajor`, and a custom SM80 mixed-input operator. It passes a packed `[n4,K/2]` tensor directly to the global iterator and asserts `matrix_B_interleaved.size(1) * 2 == K`. SHMQ currently passes an expanded signed-INT8 `[n4,K]` tensor to its staged INT8 runner for mixed prefill. This remains the highest-upside architectural mismatch.

Sources: official Microsoft MixLLM repository `https://github.com/microsoft/MixLLM`; official `mixllm/nn/modules/linear.py`; official `mixllm/kernels/mix_mma_multistage.cuh`; vendored SHMQ `cutlass/arch/mma_sm75.h`; vendored SHMQ `cutlass_extension/mq_mma_tensor_op_dequantizer.h`.


## Hot-path bookkeeping comparison

The official CUDA launcher contains no SHMQ-equivalent per-partition `record_tensor_stream` helper. SHMQ records seven tensors on each auxiliary stream for every integer partition. This is required for allocator lifetime safety when temporary metadata or outputs are used asynchronously, so it cannot be removed globally without a lifetime proof. Immutable module-owned weights, scales, zeros, and indices are initialized before forward and can potentially be recorded once per persistent worker stream; dynamic activation and output tensors still require per-call recording. This is a lower-risk seam than changing arithmetic, but the expected gain is small relative to the rows=128 deficit.

Historical evidence rules out simply re-enabling cached-v3 metadata: v220/v189 cached metadata was numerically correct but reduced mixed rows=128 performance and failed timing integrity; the v188/v200 unchecked path was retained because it measured better. Therefore v258 will not repeat the rejected cached-v3 switch. The direct packed INT4 seam remains the only high-upside discrepancy supported by the primary-source comparison.


## New high-impact discrepancy: output dtype

The official MixLLM test and launcher allocate `matrix_C_computed` as `torch.float16`; its correctness checks compare the kernel result to a FP32 reference only after casting both to half. SHMQ's `three_level_linear_v2_core` allocates a FP32 output and every SM75 kernel stores four-byte floats, after which `ThreeLevelLinear.forward()` casts the result back to the original FP16 activation dtype. This is an avoidable ABI mismatch: it doubles output bandwidth and output storage, adds a separate cast in the module forward path, and can reduce occupancy. Matching upstream by writing FP16 output directly preserves the model-visible result because the module already returns FP16 and the original path uses FP16 output. The change must update all output stores, probes, dtype contracts, and reference tests, but must not alter quantization, weights, partitioning, benchmark settings, or quality thresholds.
