# v255 deep research: original MixLLM versus SHMQ v254

## Research question

Why does SHMQ v254 remain roughly 11.15x slower than the identical dense FP16 baseline for Qwen/Qwen2.5-0.5B mixed rows=128 despite persistent streams and the new M=128,N=64 stage-2 candidate, and what is the smallest safe correction before attempting a direct INT4 port?

## Primary-source findings

The official Microsoft MixLLM source prepares INT4 weights once in `LinearMixLLM.interleave_uint4_for_cutlass`. It first rearranges K positions in 32-element groups, applies an 8-element sub-interleave, and packs two 4-bit values per byte. The resulting tensor is `uint8` with shape `[partial_n_int4, K/2]`; the original module passes that tensor directly to the CUDA launcher. The original `mix_mma_multistage.cuh` asserts the packed operand shape and instantiates `ElementB_INT4 = cutlass::uint4b_t`, while INT8 remains `int8_t`. Therefore the official large-M path does not materialize a signed-INT8 `[n4, K]` expansion for INT4.

The official launcher creates persistent INT4 and INT8 streams plus fork and completion events once per singleton. It launches the two integer partitions on auxiliary streams after a fork event and waits for both completion events on the caller stream. Its autotuner searches two stage families (`5` and `11`) over broad `gemm_configs` and `gemm_configs_rm` tables. The column-major table includes many legal combinations through `128x128` tiles; the row-major table includes additional `N=256` and `M=256` families. It stores the selected stage and config ID on disk per shape key. This is substantially broader than SHMQ's three SM75 candidates, but the original instruction/core is SM80-only and cannot be copied unchanged to SM75.

The official `mma_multistage_testbed.h` wires `ElementB_INT4` to `cutlass::uint4b_t`, uses a custom iterator and dequantizer, and performs scatter in the epilogue. Its launcher runs INT4 and INT8 as separate overlapping GEMMs. It does not use a single sequential integer GEMM or a second INT4 expansion pass.

SHMQ's `ThreeLevelLinear.prepare_sm75_prefill_cache()` currently unpacks packed checkpoint INT4 into codes, broadcasts zero points across K, subtracts zeros, and stores a persistent signed-INT8 tensor `[n4, K]`. The SM75 mixed path then passes this expanded tensor to the staged `Int8Runner`; direct packed `weight_int4` is only used by the native pair candidate. This is the primary dataflow mismatch with upstream and explains additional memory traffic, although it alone cannot explain every factor of 11 because the expansion is cached before timed execution.

A second, concrete SHMQ inconsistency was found in the current source. The large-M mixed branch comments state that the native pair kernel is reserved for pure INT4 and that mixed work must remain on the measured staged CUTLASS overlap path. However, the actual call passes `use_fused_int4 = n4 > 0`, which routes every mixed INT4 partition through `run_int4_pair_partition` instead of `run_cutlass_int_partition`. The inherited v253 summary said this boolean had been rolled back to `false`, but the committed source still contains `n4 > 0`. This is a source-level regression/inconsistency, not a benchmark intuition.

## Safe next change

v255 should correct only this dispatch seam by passing `false` in the mixed large-M branch. That restores the documented/previously measured staged CUTLASS overlap for mixed prefill while leaving the native pair kernel available for the pure-INT4 candidate. No arithmetic, weight layout, benchmark, model, quality gate, or output ABI changes. Before any Kaggle run, local source contracts, Python contracts, full tests, and the native INT4 reference proof must pass.

A future direct-layout experiment should be isolated behind a new SM75-specific adapter. It must preserve the checkpoint packed ABI and implement a legal SM75 `u4*u4` path with explicit asymmetric-zero correction; it must not silently substitute the SM80 `OpMultiplyAddMixedAndShuffledInputUpcast` or the SM80 stage-5/11 families.

## Sources

1. Official Microsoft MixLLM repository: `https://github.com/microsoft/MixLLM`.
2. Official source file `mixllm/nn/modules/linear.py`, especially `interleave_uint4_for_cutlass` and the packed `weight_int4` construction.
3. Official source file `mixllm/kernels/mix_mma_multistage.cuh`, especially persistent stream/event setup, direct INT4/INT8 launch, and broad autotuning.
4. Official source file `mixllm/kernels/mix_mma_config.h`, especially `gemm_configs` and `gemm_configs_rm`.
5. SHMQ source file `mixllm/nn/modules/three_level_linear.py`, especially `prepare_sm75_prefill_cache`.
6. SHMQ source file `mixllm/kernels/three_level_sm75.cu`, especially `begin_integer_prefill_overlap` and the large-M dispatch in `three_level_linear_v2_core`.
7. Kaggle v254 kernel version 249: functional gates passed, timing integrity passed, mixed rows=128 end-to-end speedup 0.0897x, terminal decision `no_go`.

## Decision

The next repair is the one-line mixed-dispatch correction. Direct INT4 layout remains the highest-upside research direction, but it is not safe to implement without a dedicated SM75 iterator/MMA proof and a complete local contract suite.
