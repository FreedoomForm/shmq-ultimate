# v248 deep research: legal M=64,N=64 stage-2 CUTLASS candidate

## Baseline and exclusions

v245/v247 restore the v241 measured production path. v246's N=256 candidate is rejected at compile time because the vendored SM75 row-major thread map cannot distribute its A tile across eight warps. v199's M=64,N=128 candidate compiled and was rejected on T4 because mixed rows=128 fell from about 0.281x to 0.267x speedup versus dense FP16 and timing integrity also failed at rows=16; that does not prove every M=64 family is invalid.

## Upstream comparison

The original `gemm_configs` and `gemm_configs_rm` both include M=64,N=64,K=64 families. The current SM75 port exposes only M=32,N=128 and M=32,N=64. A stage-2 `GemmShape<64,64,64>` with `WarpShape<32,32,64>` has four warps, the same warp count as the current runners, and avoids the N=256 thread-map failure. It is therefore a legal-shape candidate that changes only CTA M/N coverage and shared-memory tile reuse.

## Candidate and safety boundary

Add `CoreM64N64` and a matching `Int8RunnerM64N64`; expose it as a third tuner candidate with a bumped tuning ABI. The existing N=128 and N=64 runners remain fallbacks. Select the candidate only through the existing exact-shape CUDA-event tuner, so T4 chooses it only if its measured kernel time beats the current options for that exact `(rows, channels, width, device)` key. This preserves all arithmetic, metadata, stream/event ordering, output scatter, benchmark settings, model, quality gates, and deterministic fallback behavior. The candidate is retained only if it compiles and every required T4 gate passes.
