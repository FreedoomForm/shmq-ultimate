# v287 deep research: persistent stream-recording audit

## Comparison with original MixLLM

The original MixLLM CUDA entrypoint receives persistent INT8/INT4 weights, scales, zero points, and partition indices directly and launches its native GEMM on the current CUDA stream. Its wrapper does not call a per-forward allocator `record_stream` operation for model-owned weight or index tensors. Activation quantization and the GEMM output are the dynamic tensors; persistent checkpoint storage remains owned by the module for the whole forward lifetime.

SHMQ v286 follows the original packed layout and staged stream organization, but its helper functions still call `record_tensor_stream()` for several persistent objects on every mixed prefill. In `run_cutlass_packed_int4_partition`, `weight_int4_interleaved`, `matrix_scale`, and `matrix_zero` are module-owned caches when v3 dispatch is active. In the fallback `run_cutlass_int_partition`, the weight and indices are already intentionally not recorded, but temporary metadata is recorded. In `run_int4_pair_partition`, the comments correctly record only dynamic activation/output tensors and omit persistent weights and metadata.

The v286 Python module guarantees that the packed tensors and cached metadata remain alive for the complete forward call: `prepare_sm75_packed_tensors()` and `prepare_sm75_prefill_metadata()` return module-owned buffers, and the v3 dispatcher obtains those references before launching the auxiliary stream work. Therefore allocator stream recording of those persistent buffers is redundant for the native cached-v3 path. Removing it does not alter tensor values, kernel arithmetic, event ordering, or ownership of dynamic tensors.

## Safe v287 seam

Remove only the redundant `record_tensor_stream()` calls for persistent `weight_int4_interleaved`, cached `matrix_scale`, and cached `matrix_zero` from `run_cutlass_packed_int4_partition`. Keep recording `input_int8`, `scale_act`, and `output`, because these are dynamic tensors whose allocator lifetime can cross auxiliary streams. Keep all fallback temporary metadata recording unchanged. Add source contracts requiring the persistent-buffer distinction so a future refactor cannot accidentally remove dynamic lifetime protection.

This is a small hot-path wrapper optimization aligned with upstream ownership semantics. It must still pass the complete local suite and a fresh Colab T4 compile/correctness run. It is not a performance claim; only the required Kaggle benchmark can establish whether it changes latency or the >=2.6x target.
