# v306 next-seam research: native SM75 DefaultMmaCore remains incompatible

The v305 T4 compile tested the narrower four-warp `DefaultMmaCore<32x64x64, 32x32x32, 8x8x32>` candidate. CUTLASS still derived an A `PitchLinearWarpRakedThreadMap` with `Shape=<64,32>`, `Threads=128`, `WarpThreadArrangement=<2,16>`, and `ElementsPerAccess=32`, then rejected it with `Number of iterations must be non-zero`. The v304 eight-warp variant failed the same derived iterator contract with `Threads=256`; reducing N and warps did not repair the underlying mismatch.

This is decisive evidence that directly instantiating the generic native subbyte `DefaultMmaCore` with SHMQ’s row-major/column-major aliases is not a viable next seam. The native arithmetic probe remains valid, but its instruction-level success does not imply that the generic CUTLASS shared-memory iterator can load the checkpoint layout under the proposed geometry. The v305 alias and associated static contract are therefore rolled back. No production dispatch was changed, and v299 expanded control remains the current baseline.

Future native work would require an explicitly authored SM75 subbyte iterator/thread map or a complete upstream-compatible native core, not another M/N tile guess. Until that is independently proven, no packed production route should be enabled.
