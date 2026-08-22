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
