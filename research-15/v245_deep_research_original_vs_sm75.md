# v245 deep research: reject the wider native pair tile

## Kaggle evidence

v244 server version 241 compiled on T4 and passed the mixed-stride probe, embedded contracts, SM75 native correctness, mixed decode GEMM, and timing integrity. It failed mixed decode and mixed prefill end-to-end. On Qwen mixed QKV `{4: 2400, 8: 896, 16: 288}`, end-to-end ratios versus dense FP16 were 1.19x at rows=1, 3.56x at rows=16, and 13.49x at rows=128. This improves over v242's 16.62x rows=128 ratio but remains far worse than v241's approximately 3.24x, so the v244 production change is rejected.

## Differential conclusion

Widening the native pair kernel from a 32-channel CTA to 64 channels reduced the number of channel CTAs but did not change its fundamental dataflow: each block still reloads packed A/B panels for every 8-row subtile and executes the low/high pair instruction serially across the four row tiles. The result confirms that CTA-N width alone is not the dominant large-M bottleneck. Upstream MixLLM obtains its performance from staged iterator dataflow, wider threadblock/warp families, and shape-keyed autotuning, not from merely widening a small handwritten pair kernel.

The safe action is a complete rollback of the v244 production switch and wider pair geometry. Retain only the v241 native probe/final-scatter correctness repairs and the v242/v244 negative evidence in research/worklog. The next candidate must modify the staged CUTLASS dataflow or remove a proven hot-path overhead; another pair-kernel width variant is not justified.
