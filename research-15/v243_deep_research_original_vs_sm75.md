# v243 deep research: why v242 fused mixed INT4 regressed

## Kaggle evidence from v242

v242 compiled on T4 and passed the mixed-stride probe, all embedded contracts, SM75 native correctness, mixed decode GEMM, and timing integrity. It failed both end-to-end performance gates. For Qwen QKV mixed `{4: 2400, 8: 896, 16: 288}`, rows=1 was 1.23x dense end-to-end, rows=16 was 3.52x, and rows=128 was 16.62x. The v241 baseline for the same mixed scenario was approximately 1.28x, 3.55x, and 3.24x respectively. Therefore the v242 switch must be rejected: it preserved correctness but catastrophically regressed large-M mixed prefill.

## Upstream comparison

Upstream MixLLM's large-M integer path uses a generic staged CUTLASS `MmaCore` selected by shape-keyed autotuning. Its `gemm_launcher` searches many tile shapes and stages `{5, 11}` and then uses persistent auxiliary streams for INT4 and INT8. The large-M work is amortized through threadblock tiles and a multistage iterator rather than repeatedly executing a small native pair tile.

The SHMQ fused pair kernel is structurally different. It launches a 32x32 CTA tile with four warps, each warp owning only eight output channels, and iterates four row subtiles serially. For every group and each 32-element K chunk it reloads the CTA A/B panels and executes two 8x8x32 MMAs, then repeats for all four row subtiles. At Qwen width 3584, this means 28 activation groups and 4 K chunks per group, multiplied across 4 row subtiles and 75 channel CTAs for 2400 INT4 channels. This is a correctness-proven native SM75 instruction path, but it is not a competitive large-M GEMM dataflow.

The v241 path instead routes mixed rows>=32 through the measured v188 overlap: INT4 uses expanded signed-INT8 CUTLASS, INT8 uses signed-INT8 CUTLASS, and FP16 remains on the caller stream. That path is slow, but v241's 3.24x rows=128 is materially better than v242's 16.62x, so v242 must be reverted completely.

## Safe conclusion

Do not use the fused pair kernel for production mixed large-M performance. Keep it only for its validated probe and pure-INT4 candidate branch. Restore `use_fused_int4=false` for the mixed overlap path and retain the v242 failure evidence in the worklog. The next performance seam must be upstream-style large-M CUTLASS dataflow or shape/tile selection, not another small native pair kernel dispatch.
