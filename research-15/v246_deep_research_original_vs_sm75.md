# v246 deep research: legal stage-2 N=256 CUTLASS candidate

## Baseline

v245 restores the v241 measured production path after rejecting v244. The remaining mixed Qwen rows=128 result is about 3.24x slower than dense FP16 end-to-end, with correctness and timing integrity passed.

## Upstream discrepancy

The original MixLLM `gemm_configs` catalog includes N=64, N=128, and N=256 threadblock families, and its launcher selects among them by exact shape and persists the best result. The current SM75 runner exposes only `Core` N=128 and `CoreN64` N=64, with a tuner that can choose only those two. The vendored `DefaultMmaCore` specialization is parameterized over Shape and uses stage 2; unlike the rejected stage-3/5 experiments, an N=256 stage-2 core keeps the supported SM75 instruction and pipeline contract.

## Candidate

Add `CoreN256 = GemmShape<32,256,64>` with the existing `WarpShape<32,32,64>`, instruction `<8,8,16>`, row-major accumulator, `NumStages=2`, and the same `Runner` implementation. Add it as a third legal tuner choice keyed by exact shape/device/ABI. This changes only the number of output channels per staged CUTLASS CTA and the compile-time warp count; it preserves signed-INT8 arithmetic, scales/zero ABI, index scatter, streams/events, model, benchmark settings, and quality gates. The existing N=128 candidate remains the deterministic fallback if tuning is unavailable or rejects N=256.

This candidate is a falsifiable test of upstream N-family data reuse, not a claim that wider is automatically faster. If compilation, correctness, timing integrity, or any production gate regresses, revert it.
