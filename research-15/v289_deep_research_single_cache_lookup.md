# v289 deep research: eliminate duplicate packed-cache lookup

## Upstream comparison

The original MixLLM prepares its persistent weight layout and invokes the native GEMM wrapper directly. There is no second per-forward cache-signature scan between the top-level linear call and the physical GEMM launch.

SHMQ v288 already calls `module.prepare_sm75_packed_tensors()` in `three_level_linear()` and receives the validated immutable tuple. The same function was then called again inside `three_level_linear_prequantized()` for every forward. Its signature scans tensor identity, version, device, shape, stride, and contiguity for nine persistent buffers. This duplicated Python bookkeeping did not alter GPU arithmetic, but it diverged from the original direct-wrapper organization and could add avoidable end-to-end overhead.

## Safe v289 change

Add an optional `packed_tensors` argument to `three_level_linear_prequantized()`. The top-level `three_level_linear()` forwards the tuple it already validated, so the second signature computation disappears on the production path. Direct callers that omit the argument still perform the existing cache lookup and fallback contiguity conversion. The module-level cache invalidation rules, device checks, activation checks, partition validation, output allocation, native v3 selection, stream/event ordering, precision arithmetic, and benchmark boundaries remain unchanged.

This is a wrapper-only optimization aligned with upstream ownership semantics. It must pass the complete local suite and a fresh Colab T4 compile/correctness run. No performance claim is allowed until a same-condition Kaggle run.
