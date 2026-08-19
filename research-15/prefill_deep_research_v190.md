# v190 deep research: v189 metadata-cache regression

## Evidence from identical Kaggle T4 conditions

v189 passed native correctness and decode gates. Compared with v188, mixed QKV rows=16 improved from `0.2884x` to `0.3466x` E2E speedup versus dense FP16, but rows=128 regressed from `0.2762x` to `0.2197x`. The persistent CUTLASS metadata cache is present and reported (`251,776` bytes for mixed rows=128), so the implementation is active rather than a dead-code change. The exact Qwen2.5-0.5B gate reported `unavailable_environment` because the model was not mounted; this was caused by the metadata source handle being inconsistent with Kaggle's documented `{owner}/{model}/{framework}/{instance}` form.

## Original-versus-SHMQ comparison

The original MixLLM's large-M launcher enters its custom multistage GEMM directly after host-side partition preparation and does not perform a full partition validation (`cat/sort/arange`) on every kernel invocation. SHMQ's existing v2 unchecked operator intentionally follows the same hot-path property; Python `_validate_partition` caches the invariant before execution. The new v189 v3 wrapper, however, calls `validate_partition()` inside the CUDA dispatcher before every cached-metadata launch. That validation allocates/operates on concatenated index tensors and sorted expected indices, which is unnecessary after the Python signature cache and can dominate small-to-medium prefill launches. This is a falsifiable explanation for the rows=128 regression: replacing only the v3 validated wrapper with an unchecked v3 wrapper should preserve output and cached dataflow while removing the repeated validation work.

## Safe next experiment

v190 will add `_three_level_linear_v3_unchecked` with the same 16-tensor ABI, route the already-validated Python path to it, retain the public validated `three_level_linear_v3` for external callers, and correct Kaggle metadata to `qwen-lm/qwen2.5/Transformers/0.5b`. No arithmetic, model, allocation budget, benchmark shape, or quality threshold changes. The prediction is that correctness remains unchanged and rows>=32 latency improves; if it does not, the cache will be rejected rather than kept as a performance claim.

## References

1. Kaggle Models documentation: https://www.kaggle.com/docs/models
2. Kaggle `model_sources` metadata format: https://www.kaggle.com/product-feedback/391480
3. Original MixLLM launcher: `external/MixLLM/mixllm/kernels/mix_mma_multistage.cuh`
4. v189 gate artifact: `shmq-ultimate/mixllm_3level_kaggle/latest-output-v189-computer/mixllm_3level_gate.json`
