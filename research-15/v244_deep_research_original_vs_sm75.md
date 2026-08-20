# v244 deep research: wider native INT4 channel tile

## Baseline and rejected v242

The last measured functional baseline is v241: mixed Qwen QKV rows=128 was approximately 3.24x slower than dense FP16 end-to-end, with all correctness and timing gates passed. v242 routed mixed INT4 through the existing native pair kernel and degraded rows=128 to 16.62x slower, so the small 32-channel pair dispatch is rejected. v243 restores v241 and records the rollback.

## Original MixLLM comparison

The upstream launcher is not limited to one small N tile. Its `gemm_configs` table includes N=64, N=128, and N=256 families, while the fallback large-M column-major family is a 64x128 threadblock with 64x32 warp tiles. The current SM75 CUTLASS path exposes only 32x128 and 32x64 threadblocks, and the native pair path exposes only a 32-channel CTA tile with four warps. This is a concrete data-reuse discrepancy: the current native pair kernel performs a 32-channel CTA per launch, whereas upstream's tuned families are designed to amortize A movement across wider N tiles.

## Candidate

Implement a separate wider native pair kernel with eight warps and a 64-channel CTA tile, retaining the proven SM75 `8x8x32` low/high instruction pair, exact `P_low + 16*P_high - zero*sum(A8)` arithmetic, exact scale and index ABI, and the same 32-row CTA tile. Each warp still owns eight channels and iterates the four eight-row subtiles; only the CTA channel width and launch width change. The row-sum correction remains read from the first four row-tile slots, which are filled by warps 0–3, while warps 4–7 own only additional channel tiles.

This candidate must first be source- and CPU-contract validated. It must be used only as a reversible experiment for the mixed INT4 branch after the kernel is complete; the existing v241 CUTLASS overlap remains the fallback and pure-INT4 path. No quality, quantization, partition, benchmark, model, or timing methodology changes are allowed. If T4 correctness or any gate fails, revert entirely.
