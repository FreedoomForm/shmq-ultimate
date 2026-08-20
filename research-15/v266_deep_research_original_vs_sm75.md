# v266 deep research: omit allocator records for cached CUTLASS metadata

The v264 runtime failure confirmed that the project-wide index ABI is int32 and the valid v263 path must keep its custom int32 epilogue. The fresh upstream comparison then isolated a remaining bookkeeping mismatch in the integer path.

The original MixLLM launcher owns persistent INT4/INT8 CUDA streams and persistent module operands. It launches both branches after the fork event and joins only at the caller stream. SHMQ now has the same stream topology, but `run_cutlass_int_partition` still records `matrix_scale` and `matrix_zero` on the auxiliary stream for every call. In the v3 cached-metadata path these tensors are module-owned immutable buffers prepared before forward; recording them again is redundant. In the v2 fallback, the transposed metadata are temporary tensors created on the caller stream, so their records remain required.

v266 will add one boolean seam to `run_cutlass_int_partition`: skip only `matrix_scale` and `matrix_zero` allocator records when `has_cached_metadata` is true, while continuing to record dynamic `input_int8`, activation scales, and output, and continuing to record temporary metadata in v2. This cannot change arithmetic, tensor lifetime, output mapping, precision allocation, or stream ordering. It is intentionally a small measurement; the large performance gap is still expected to be dominated by the expanded signed-INT4 CUTLASS path.

Primary sources: upstream `mix_mma_multistage.cuh` persistent stream/operand organization; SHMQ `three_level_sm75.cu` cached v3 dispatch and `record_tensor_stream`; SHMQ `three_level_linear.py` cached metadata preparation.
