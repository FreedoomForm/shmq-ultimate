# Deep research for v224: original epilogue and index scatter

## Sources

The comparison used the upstream MixLLM source at `/home/ubuntu/mixllm-upstream-research/mixllm/kernels/mma_multistage_testbed.h` and the clean SHMQ SM75 source at `shmq-ultimate/external/MixLLM/mixllm/kernels/sm75_cutlass_testbed.h`.

## Original epilogue

Upstream MixLLM also performs indexed output placement rather than a plain contiguous store. Its `kernel_multistage_mma` loads an index fragment for each output-column fragment, constructs a column-major `TensorRef`, and writes `offset_ref.at({row, indicesFrag[...]})`. Therefore SHMQ's per-accumulator indexed scatter is not by itself a discrepancy that can be removed without changing the output contract.

The upstream implementation differs in two details worth preserving as future optimization seams: it hoists the index fragment once per MMA column fragment before the nested accumulator loops, and it converts the float accumulator to the output type through a `NumericConverter` while writing a column-major half output. SHMQ uses an equivalent index lookup in its nested epilogue and writes float output, which is required by the current backend/reference contract. A half-output change would be a quality/ABI change and is out of scope.

## Decision

Do not attempt a speculative scatter removal or output-type change. Any v224 candidate must preserve the index mapping and float output. The only defensible micro-optimization from this comparison would be a scoped index-fragment hoist for repeated `indices[partition_channel]` reads, but it is likely smaller than the dominant large-M GEMM latency and must be measured independently.
