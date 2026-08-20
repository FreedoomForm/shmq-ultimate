# v242 deep research: upstream MixLLM versus SHMQ SM75 after v241

## Observed Kaggle result

The sole v241 submission (Kaggle server version 239) completed with execution passed, embedded contracts passed, SM75 native correctness passed, mixed decode GEMM performance passed, and timing integrity passed. The mixed-stride probe also passed with `SM75_INT4_PAIR_MIXED_STRIDE_STATS [(-999.0, 1024), (128.0, 1024)]`, zero mismatches, and `SM75_INT4_PAIR_MIXED_STRIDE_PROBE_PASS`. The remaining failures are performance gates: mixed decode end-to-end and mixed prefill end-to-end.

For the Qwen QKV mixed partition `{4: 2400, 8: 896, 16: 288}`, the measured end-to-end ratios versus dense FP16 were approximately 1.28x at rows=1, 3.55x at rows=16, and 3.24x at rows=128. Thus v241 is functionally repaired but not a performance candidate. The rows=128 GEMM itself was approximately 3.20x dense, while timing integrity remained true; this is not merely benchmark-event contamination.

## Upstream MixLLM evidence

The original `external/MixLLM/mixllm/kernels/mix_mma_multistage.cuh` creates persistent INT4 and INT8 auxiliary streams and fork/completion events once in `LinearMixLLM` (lines 18–39), records the fork event on the caller stream, launches the two integer GEMMs on their own streams, and joins them on the caller stream (lines 240–259 and 328–346). Its `gemm_launcher` selects between separate column-major and row-major families and has disk-backed shape-keyed autotuning over stages `{5, 11}` and many tile configurations (lines 369–520).

The upstream `mix_mma_config.h` therefore treats tile shape, stage count, output layout, M bin, N, K, and partial-N as runtime configuration dimensions. The generic upstream `mma_multistage_testbed.h` accepts a selected `MmaCore` and keeps the iterator, pipeline, accumulator, and epilogue generic rather than hard-coding one family.

## SHMQ SM75 discrepancy

The current `sm75_cutlass_testbed.h` exposes only two SM75 runner aliases: `Core = GemmShape<32,128,64>` and `CoreN64 = GemmShape<32,64,64>`, both with WarpShape `<32,32,64>`, instruction `<8,8,16>`, and `NumStages=2` (lines 203–224). `Runner::run` launches the fixed core and performs a generic row-major scatter (lines 167–200), but the dispatcher can select only N=128 or N=64. The current `three_level_sm75.cu` tuning cache consequently chooses only between those two fixed families (lines 70–217).

The current mixed-prefill path is not the newly repaired native pair path. In `three_level_linear_v2_core`, rows >= 32 with integer channels call `begin_integer_prefill_overlap` with `use_fused_int4=false` (lines 1322–1341). That launches the CUTLASS-expanded INT4 branch and CUTLASS INT8 branch on auxiliary streams, while FP16 is launched separately on the caller stream. The fused native INT4 pair kernel is available and its standalone mixed-stride coverage now passes, but it is not used for the mixed 4/8/16 production path.

## Safe next seam

The next repair should be limited to enabling the already-existing `sm75_int4_pair_gemm_kernel` for the INT4 branch of mixed prefill, while retaining the upstream-style auxiliary-stream overlap, the existing INT8 CUTLASS branch, the existing FP16 branch, the exact 4/8/16 output-index ABI, and all quality computations. This directly removes the known expanded-INT4 prefill representation and its CUTLASS INT4 path from the mixed hot path without changing model weights, activations, benchmark settings, or partition quality.

The change must be guarded only for rows >= 32 and nonempty INT4 partitions, with the existing fallback retained for unsupported or failing paths. Before any Kaggle run, verify all source contract tests, CPU native INT4 reference proof, Python dispatch contracts, and any available local mixed-path tests. One Kaggle run is then required to establish T4 correctness and all four gates. If full mixed correctness fails, revert the switch and retain v241 as the last known-good functional baseline.
