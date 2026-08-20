# v227 research: wrapper lifecycle and stream bookkeeping

## Comparison

The upstream MixLLM kernel launcher constructs a device guard, output tensor, and kernel launch around its prepared packed tensors. Its performance-critical selection is shape/config based; it does not provide an evidence-based justification for removing allocator stream recording from a PyTorch-integrated adapter.

The SHMQ SM75 wrapper performs explicit device/dtype/contiguity/shape checks, partition validation, contiguous conversions, `record_tensor_stream` calls for every tensor handed to an auxiliary stream, and event waits/joins. These operations are not interchangeable: PyTorch's caching allocator requires stream recording to keep tensors alive until auxiliary-stream work completes, and the v200 event topology establishes the producer/consumer ordering for the three output partitions.

## Safe conclusion

Do not delete `record_tensor_stream`, stream waits, or device guards as an unproven optimization. The legitimate deepening seam is to move immutable invariants and contiguous preparation to module initialization, then keep a compact per-call validation key for tensors whose identity/version/device/shape can change. The v226 module-owned INT4 and metadata caches already move the most expensive persistent preparation out of the first prefill dispatch. A wrapper optimization must preserve all lifetime and capture semantics and should be limited to avoiding repeated `.contiguous()` decisions when the module state is known stable.

## Source references

- Upstream first-party source: `/home/ubuntu/mixllm-upstream-research/mixllm/kernels`
- SHMQ source: `shmq-ultimate/external/MixLLM/mixllm/kernels/three_level_sm75.cu`
- SHMQ backend: `shmq-ultimate/external/MixLLM/mixllm/sm75_backend.py`
- NVIDIA CUDA asynchronous execution guidance: https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html
