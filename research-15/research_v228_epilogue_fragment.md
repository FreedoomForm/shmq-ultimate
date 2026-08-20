# v228 research: indexed epilogue fragment

The upstream MixLLM epilogue also maps accumulator fragments through `indices` before storing the result. It does not eliminate indexed output placement or change the output type. Its safe optimization is to derive the output-channel mapping once per MMA column fragment and reuse it across accumulator rows.

The v200 SM75 epilogue currently evaluates `indices[partition_channel]` inside the innermost row/column write loop. The exact arithmetic, mask, output stride, and partition semantics can remain unchanged while a fragment stores one index per `(mma_n, mma_m, col)` position. Invalid lanes receive a negative sentinel and retain the existing `partition_channel < problem_size.n()` mask. The optimization must not alter `matrix_indices`, scatter order, float output ABI, or stream behavior.

The previous v224 experiment was rejected only because it was measured before the full v200-derived repair plan and not because its source-level semantics were wrong. This v228 slice will therefore be source- and CPU-contract tested, then retained as an unmeasured local change until the single authorized final T4 run.

Primary reference: `/home/ubuntu/mixllm-upstream-research/mixllm/kernels/mma_multistage_testbed.h`.
