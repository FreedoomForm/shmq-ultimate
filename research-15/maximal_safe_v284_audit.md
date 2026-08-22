# Maximal-safe v284 audit

## Objective

This audit identifies the largest set of changes that can be combined with the v284-derived SHMQ SM75 implementation without changing the model, quantization budgets, benchmark settings, output ABI, or quality criteria. The comparison uses the original Microsoft MixLLM organization, the recorded v271 safe baseline, v284, and every later experiment with available T4 evidence.

## Baseline and current evidence

The last Kaggle-confirmed safe baseline remains v271. v284 is the lazy-expanded-cache candidate, but its exact Kaggle run (kernel version 278, source commit `ccd74338f0bda0c759a523036cc0cc5d17acb103`) failed during CUDA compilation at the packed `M128/N64` CUTLASS alias: `PitchLinearWarpRakedThreadMap` asserted `Number of iterations must be non-zero`. No v284 performance or correctness gates ran.

The later source history already contains the v286 shape-family repair and v287–v291 wrapper optimizations. The current branch also contains v293–v302 packed-native experiments, but the fresh v302 Kaggle run compiled only after refreshing the embedded CUTLASS bundle and then failed native correctness for large M: mixed rows 128 had `max_abs_error=731.8734`, and pure INT4 rows 128 had `max_abs_error=772.9107`. Those native adapter changes cannot be retained as safe performance improvements.

## Original MixLLM comparison

The original Microsoft MixLLM path keeps the launcher/config/cache organization separate from arithmetic. It uses valid CUTLASS shape families, persistent INT4/INT8 streams and events, prepared packed layouts, and direct cached buffers. It does not rely on an invalid packed `M128/N64` SM75 alias, nor does it double-transform fragments that already satisfy the iterator contract. These facts support retaining the launcher/cache improvements while isolating or removing the unproven synthetic SM75 packed-native adapter.

## Compatibility matrix

| Change | Evidence | Compatibility with v284 | Decision |
|---|---|---|---|
| v286: remove packed `M128/N64`; route packed INT4 through `M64/N64`; advance tuning ABI | Colab T4 compile and six native SM75 correctness tests passed | Directly fixes v284’s compile-time failure; leaves ordinary INT8 `M128/N64` intact | **Keep** |
| v287: remove redundant allocator `record_stream` calls for persistent packed buffers | Colab T4 compile/correctness passed | Orthogonal to arithmetic, streams, metadata, and output | **Keep** |
| v288: forward validated packed cache tuple directly | Colab T4 compile/correctness passed | Orthogonal wrapper reduction; fallback remains available | **Keep** |
| v289: avoid duplicate packed-cache signature scan | Colab T4 compile/correctness passed | Orthogonal wrapper reduction; direct callers retain validation | **Keep** |
| v291: hand off completed partition validation | Colab T4 compile/correctness passed | Top-level validation remains mandatory; only duplicate scan is removed | **Keep** |
| v293: allow empty expanded INT4 buffer when packed v3 data is present | Required to reach native packed v3; operator run then exposed large-M corruption | Keep only if native path is not trusted for large M; it must not weaken legacy fallback validation | **Keep conditionally; pair with safe dispatch** |
| v294: remove B fragment shuffle | Colab large-M correctness failed | Conflicts with the already failing packed mapping; no independent proof | **Exclude** |
| v295: restore A fragment shuffle | Colab large-M correctness remained failed | Does not repair the root mapping and adds no demonstrated benefit | **Exclude** |
| v296: widened A iterator shape | Colab large-M correctness remained failed in later native runs | Part of the synthetic adapter under investigation, not independently safe | **Exclude from production native path** |
| v297: vertical visitation | Large-M native correctness remained failed | Changes fragment visitation without restoring the full CUTLASS contract | **Exclude** |
| v298/v300: B register regrouping variants | Large-M native correctness remained failed | Mutually exclusive with the original dequantizer/fragment order and unproven | **Exclude** |
| v299: forced expanded INT4 control | Large-M correctness passed but performance failed | Safe as a correctness fallback, not a target performance solution | **Use only as fallback/control** |
| v301/v302: native dispatch and synthetic-K kgroup bypass | v302 compiled after bundle refresh but large-M correctness failed | Conflicts with production correctness requirement | **Exclude until a new independent proof** |

## Selected maximal-safe set

The maximal safe combination is v284’s lazy cache preparation plus the complete v286–v291 wrapper and shape-family repairs, with v299-style expanded INT4 fallback retained for any shape that cannot be proven safe for native packed execution. The combination keeps the original three-level INT4/INT8/FP16 model representation, persistent streams/events, exact scatter ABI, metadata validation, and all quality/benchmark settings. It does not retain the v294–v302 synthetic fragment transformations or claim native packed performance for large M.

This is the largest set supported by the available evidence. It is not yet a 2.6x performance claim: v291’s Kaggle run remained below dense FP16 on mixed prefill, and v299’s expanded control was correct but slower. A new native design must therefore be tested separately after this safe recovery, not silently combined with the rejected fragment variants.

## Sources

1. Microsoft MixLLM repository: https://github.com/microsoft/MixLLM
2. NVIDIA CUTLASS warp iterator source: https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator.h
3. NVIDIA CUTLASS GEMM API documentation: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html
4. Repository experiment record: `research-15/worklog.md`


## Next native-design research after maximal-safe fallback

The fresh upstream source confirms that original MixLLM’s launcher passes both `matrix_B_int8` and `matrix_B_interleaved` into a CUTLASS mixed GEMM template and separates `gemm()` from `gemm_rm()`, while persistent stream/event lifecycle is owned by `LinearMixLLM`. The original API therefore expects the packed/interleaved weight layout to be consumed directly by a kernel family whose tile and iterator contracts are internally consistent; it does not expand packed INT4 to a signed INT8 matrix in the hot path.

The current SHMQ maximal-safe candidate deliberately uses the expanded INT4 path because the native SM75 synthetic adapter is not correct for large M. The next native optimization must preserve the original packed input and metadata ABI but use a legal SM75 instruction policy end-to-end. The prior v294–v302 variants changed only fragments/visit order while retaining a synthetic K32 iterator contract; their repeated large-M failures show that local permutation edits are insufficient. A promising new seam is a separate SM75 native packed kernel with an explicit k16 policy and independently verified register mapping, kept behind a shape gate and expanded fallback until correctness is proven. No implementation change is made from this note alone.

Fresh source reference: Microsoft MixLLM’s original mixed launcher [https://raw.githubusercontent.com/microsoft/MixLLM/main/mixllm/kernels/mix_mma_multistage.cuh].


## Exact original fragment contract

The original `MQMmaMixedInputTensorOp` defines A and B iterators from the warp `Shape` and the instruction shape, then performs a two-dimensional MMA visitation. For each row fragment it visits N fragments in serpentine order, using the same B fragment index but a different accumulator index depending on accumulator layout. Its transform deliberately leaves B in the iterator-emitted order (`tmp_B = B`) because the persistent INT4 interleave and iterator already establish the required register order; A is passed through the original A fragment shuffler before conversion. The original multistage testbed computes `gemm_k_iterations = ceil(problem_size.k / Mma::Shape::kK)` and invokes the operator once, while the operator itself performs all M/N instruction iterations.

This rules out treating B regrouping or A visitation as isolated fixes. The failed v294–v302 variants changed these pieces independently while retaining an incompatible synthetic K32 policy. A new SM75 packed implementation should instead define a self-consistent native k16 iterator/operator pair, let the mainloop’s K iteration count follow that policy, and copy the original operator’s complete M/N visitation and B no-double-shuffle behavior. Until such an implementation is proven, the v299 expanded fallback remains the safe production control.


## SM80 versus SM75 policy conclusion

The original SM80 implementation has two separate mixed-input specializations. Its packed integer specialization uses a legal `GemmShape<16,8,32>` instruction policy and the generic `MQMmaMixedInputTensorOp`; the iterator shapes and operator M/N loops are derived from that policy. The current SM75 adapter instead wraps a legal `m8n8k16` instruction inside a synthetic `MmaTensorOpPolicyK32`, widens both fragments, and manually invokes two k16 MMAs per logical k32 fragment. The shared-memory pipeline then derives warp-K iteration and iterator behavior from the synthetic policy. This is the structural mismatch behind the large-M corruption, not merely a missing permutation.

The next native path should not modify the existing synthetic adapter in place. It should either use the generic mixed-input operator with a fully consistent SM75 k16 policy and matching iterator shapes, or introduce a separate explicit SM75 packed kernel whose loader and MMA loop are authored together. The existing expanded fallback remains the safety boundary while this new path is independently proven.


## Turing INT4 tile research and measured implication

NVIDIA’s Turing documentation confirms that SM75 natively supports `m8n8k32` INT4 Tensor Core instructions and recommends 128-bit aligned operands; the documented example uses a 128x128 threadblock and 64x64 warp tile with two stages. Microsoft MixLLM’s original launcher also selects large staged tiles (`gemm_rm<5,64,64,32,32>` for row-major and `gemm<5,64,128,64,32>` for column-major) and uses separate persistent INT4/INT8 streams with event joins.

The current SHMQ fused pair kernel is functionally valid but fixes each CTA at 32x64 channels, four warps, and two 8-column subtiles per warp. The completed T4 log shows pure Qwen INT4 rows=128 taking about 2.31 ms versus dense FP16 about 0.156 ms, so the pair kernel’s small tile/repeated shared-memory staging is a direct, measured performance bottleneck. The next candidate should test a larger tile family or reuse the original staged CUTLASS shape for the packed path; any change must preserve the exact low/high nibble decomposition, zero-point correction, indexed scatter, and no-overlap output partition contract.
