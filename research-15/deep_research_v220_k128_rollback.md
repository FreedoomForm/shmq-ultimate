# Deep research for v220: K=128 rollback decision

## Sources inspected

The comparison used the upstream MixLLM checkout at `/home/ubuntu/mixllm-upstream-research/mixllm/kernels/` and the accepted v200 and current v219 implementations in this repository.

## Original MixLLM dataflow

The upstream `mix_mma_multistage.cuh` defines both INT4 and INT8 staged CUTLASS paths with `ThreadblockShape = GemmShape<BLOCKSIZE_M, BLOCKSIZE_N, 64>` and `WarpShape = GemmShape<TILESIZE_M, TILESIZE_N, 64>`. Its launch geometry derives `logicalGridM` and `logicalGridN` from those shapes, and its metadata contracts are expressed in 128-element activation groups. The upstream launcher records a fork event, starts independent INT4 and INT8 streams, records completion events, and makes the caller stream wait on both. It does not widen the production K tile to 128.

The upstream launcher also performs shape/configuration selection through explicit configuration tables and bounded autotuning. The fallback column-major configuration is `gemm<5, 64, 128, 64, 32>`, again preserving a K=64 threadblock/warp tile. This is the relevant original design: staged producer/consumer movement, fixed K=64 CUTLASS geometry, and metadata advancement synchronized with that geometry.

## SHMQ v200/v219 comparison

The accepted v200 SM75 implementation preserved the same high-level two-stream overlap and always dispatched the SM75 `Int8Runner`, whose testbed alias uses `GemmShape<32, 128, 64>`. v219 added a production branch selecting `KWideInt8Runner` with `GemmShape<32, 128, 128>` for `rows >= 32 && channels >= 128`; the SM75 mainloop was then changed to suppress its second `mac_loop_iter` call for K=128.

The v219 T4 result proves that suppressing the duplicate mainloop call repaired timing-integrity only. Rows=128 still produced `max_abs_error` approximately 340--357 for mixed, pure INT4, and pure INT8 cases, while rows=1/16 remained correct. The failed cases are exactly the K=128 dispatch region. The gate report therefore marks `sm75_native_correctness=failed`, `mixed_prefill_end_to_end_performance=failed`, `operator_production=failed`, and `terminal_decision=no_go`, despite `timing_integrity=passed` and `sm75_native_benchmarks=passed`.

This pattern is inconsistent with a simple double-consumption bug alone. It indicates that the widened K tile still has an incomplete geometry/metadata contract: at least one of the iterator initialization, staged metadata advancement, operand layout, or epilogue assumptions remains written for the original K=64 consumption model. Because pure INT4 and pure INT8 both fail, the defect is in the shared K-wide pipeline rather than one quantization format.

## Safe decision

Remove only the production `KWideInt8Runner` selection from `run_cutlass_int_partition` and restore unconditional v200 `Int8Runner` dispatch. Keep the cached-v2 metadata adapter because it is a host-side `[groups, channels]` metadata-layout optimization independent of K-wide geometry, and preserve the v200 fork/event overlap. Keep the experimental K=128 aliases and guarded mainloop code unselected so the failed experiment remains available for later isolated debugging without affecting production.

The local source contracts must be relaxed only where they require K=128 production selection; contracts for cached-v2 metadata, event ordering, the v200 two-stream topology, and native validation remain mandatory.

## Performance interpretation

v219's corrected rows=128 mixed E2E speedup was 0.381x versus dense FP16, better than v200's approximately 0.307x, but this is not admissible because correctness failed. The user requirement is to retain only changes for which all required gates pass; therefore the speed improvement cannot be accepted as a production result.
