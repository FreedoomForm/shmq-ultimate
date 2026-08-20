# v200 prefill deep research

## Scope
This note compares upstream MixLLM with the current SHMQ-Ultimate SM75 implementation before any v200 code change. The goal is to identify a falsifiable architectural improvement without altering benchmark inputs, arithmetic quality, or gate definitions.

## Primary sources opened

1. Upstream repository: https://github.com/microsoft/MixLLM
2. NVIDIA CUTLASS functionality documentation: https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/functionality.html

## Initial findings

The upstream repository is the authoritative MixLLM implementation and documents a mixed 4-bit/8-bit linear design. The repository page itself does not claim a single monolithic kernel across precisions; the actual launcher and kernel source in the checked-out repository must be compared directly.

The CUTLASS documentation page is a generic device-level GEMM reference. Its public page describes separate device-level GEMM kernel instances and WMMA/TensorOp abstractions, but the page extraction did not provide a claim that INT4, INT8, and FP16 operands can be combined in one GEMM instruction or one CUTLASS kernel. Therefore, a three-precision single-launch design cannot be assumed from CUTLASS support alone; it must be implemented as a custom fused kernel or retained as separate compatible kernels.

## Local-source comparison to complete

The checked-out upstream `mix_mma_multistage.cuh` is the decisive source for the launch topology. The current implementation's `three_level_sm75.cu` and `sm75_backend.py` must be inspected side by side, with special attention to (a) whether the original uses separate streams/events rather than a single launch, (b) the lifetime and shape of expanded INT4 storage, and (c) whether the current large-M path unnecessarily materializes expanded INT4 and transposed metadata before each benchmark call.

## Decisive local-source findings

### Upstream launch topology

The upstream `mix_mma_multistage.cuh` does **not** implement one monolithic three-precision kernel. Its `LinearMixLLM` owns two auxiliary CUDA streams and events. For a mixed W4/W8 operation, it records a fork event on the caller stream, launches the INT4 CUTLASS testbed on `local_stream_int4`, launches the INT8 CUTLASS testbed on `local_stream_int8`, records one completion event per auxiliary stream, and makes the caller stream wait on both events. The original therefore attacks the partition overhead by overlapping the two quantized GEMMs, not by combining INT4 and INT8 instructions into one kernel. The original source has no FP16 partition; its W4/W8 design cannot directly provide a three-precision single instruction.

The upstream launcher also autotunes staged configurations and chooses separate row-major or column-major kernels. Its CUTLASS configuration uses `ThreadblockShape` and `WarpShape` parameters selected per shape, rather than assuming that one direct-WMMA kernel is optimal for every M.

### Current large-M path

The current `three_level_sm75.cu` large-M path expands INT4 into signed INT8 before launch, then calls `run_cutlass_int_partition` once for INT4 and once for INT8, and launches a separate direct WMMA FP16 kernel. This is three launches after activation quantization, with INT4 expansion and optional metadata preparation outside the kernel. The direct fused `three_level_tensorcore_kernel` already handles 4/8/16 in one launch, but its large-M dispatch is bypassed because the code routes rows>=32 with any integer partition to separate CUTLASS calls.

The current host adapter confirms the persistent overhead: `_expanded_int4_for_prefill` materializes an `[n4, K]` int8 cache from packed INT4 and zeros, and `_prefill_metadata_for_cutlass` materializes transposed scale/zero layouts for rows>=32. The v199 mixed Qwen benchmark reports `expanded_int4_bytes=8,601,600` at rows 16 and 128 and `prefill_metadata_bytes=0` for the v2 ABI route, so expansion is a real persistent footprint even when metadata caching is bypassed.

### Storage divergence

The upstream module prepares CUTLASS-facing INT4 data with `interleave_uint4_for_cutlass` during construction. The current three-level module stores simple sequential nibble packing via `_pack_uint4`, then compensates at runtime with the int4 expansion cache. The upstream source's interleaving is tied to its SM80 `OpMultiplyAddMixedAndShuffledInputUpcast` path, so copying it blindly to SM75 is unsafe; however, the architectural lesson is valid: prepare a kernel-native weight layout once, rather than unpacking all INT4 values into an 8.6 MB int8 matrix before every large-M workload.

## Ranked falsifiable v200 hypotheses

1. **Auxiliary-stream overlap is the safest high-leverage experiment.** If the main prefill loss is serialized INT4 and INT8 CUTLASS work, launching them on separate persistent streams with event joins, as upstream does, should lower the quantized portion without changing arithmetic. Prediction: v200 mixed rows>=32 GEMM and E2E times improve while pure INT4/INT8 correctness and timing-integrity remain unchanged. The FP16 launch can remain on the caller stream and wait only where output overlap requires it.

2. **Reusing one fused direct-WMMA kernel for all three partitions will reduce launch count but may lose large-M throughput.** Prediction: forcing the existing fused kernel at rows 32/128 lowers launch overhead but its repeated B loads and per-channel scatter will remain slower than CUTLASS; it is a useful control, not the first implementation choice.

3. **Direct packed-INT4 consumption could remove the 8.6 MB expansion, but the current packed layout and asymmetric zero correction require a new SM75 nibble decode path.** Prediction: a correct implementation must decode nibbles and subtract the group zero while staging B tiles; if it is added without a validated layout/WMMA mapping, correctness risk is high and memory-load savings may be dominated by per-element unpacking.

4. **Precomputing CUTLASS-native INT4 layout and metadata in the module may remove cache work but is unlikely to reach 2.6x alone.** Prediction: it can reduce allocation/transform telemetry but cannot eliminate the three large-M kernel launches or the FP16 scatter.

## Selected v200 experiment

Implement hypothesis 1 first: mechanically revert v199's wide core, preserve v196's timing-integrity and v2 ABI safeguards, and add a stream/event adapter around the two integer CUTLASS partition launches. Keep the FP16 partition on the current stream for correctness and avoid changing the kernel arithmetic, packed ABI, benchmark settings, or model. This is a single-variable architectural experiment aligned with upstream MixLLM's actual launch topology.

## References

[1]: https://github.com/microsoft/MixLLM "Microsoft MixLLM source repository"
[2]: https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/functionality.html "NVIDIA CUTLASS Functionality documentation"
