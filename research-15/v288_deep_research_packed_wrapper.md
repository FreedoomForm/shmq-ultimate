# v288 deep research: persistent packed-tensor wrapper overhead

## Original MixLLM comparison

The original MixLLM hot path prepares its weight and metadata layout before inference, then calls one direct CUDA `gemm` operator with the already prepared tensors. The wrapper allocates only the output for that invocation; it does not repeatedly inspect or conditionally re-contiguous persistent model tensors on every forward.

SHMQ v287 already caches a tuple of immutable packed tensors in `ThreeLevelLinear.prepare_sm75_packed_tensors()`. The cache signature includes tensor identity, version, device, shape, stride, and contiguity, and the method returns `None` if any persistent packed tensor is not contiguous. Nevertheless, `three_level_linear_prequantized()` still rechecks every packed tensor device and calls `tensor.is_contiguous()` for all persistent buffers on every invocation before rebuilding the CUDA argument tuple. The native activation tensors remain dynamic and must retain their device/contiguity checks and conditional conversion.

## Safe seam

When `prepare_sm75_packed_tensors()` returns a non-None cache, its contract already proves that all persistent packed tensors are contiguous and records their devices. Keep the existing device check so cross-device misuse still fails explicitly, but pass the cached tuple directly rather than rechecking and conditionally calling `.contiguous()` on each persistent tensor. Preserve the fallback tuple and all dynamic input/output checks for callers whose packed cache is unavailable.

This changes only Python wrapper bookkeeping; it does not change weight bytes, quantization arithmetic, kernel selection, stream/event ordering, output mapping, benchmark settings, or quality. Add a source contract that the cached path uses direct packed-tensor forwarding and that the fallback still retains conditional contiguity conversion. Validate all local contracts and compile/correctness on Colab T4 before retaining v288. Performance remains unclaimed until the authoritative Kaggle benchmark.
