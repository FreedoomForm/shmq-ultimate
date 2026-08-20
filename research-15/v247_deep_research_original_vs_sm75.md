# v247 deep research: reject N=256 on this SM75 iterator family

## v246 Kaggle result

Kaggle server version 242 failed during CUDA extension compilation before any runtime gate. The exact nvcc diagnostic was `pitch_linear_thread_map.h(298): static assertion failed: Number of iterations must be non-zero`. The instantiation was `PitchLinearShape<64,32>, Threads=256, WarpThreadArrangement=PitchLinearShape<4,8>, ElementsPerAccess=16`, reached through `DefaultMmaCore<GemmShape<32,256,64>, GemmShape<32,32,64>, ...>`.

## Source-grounded cause

The vendored SM75 row-major/column-major `DefaultMmaCore` derives `kThreads` from `WarpCount`, so N=256 with a 32-channel warp tile creates eight warps. Its A-side thread map is built over `PitchLinearShape<Shape::kK, Shape::kM> = <64,32>` and partitions accesses across the eight-warps arrangement. The resulting integer division collapses one iteration dimension to zero, and the static assertion correctly rejects the configuration before runtime. This is a structural legality failure, not a missing compiler flag or a runtime-only issue.

The upstream catalog does contain N=256 entries, but they are in its separate row-major family and use different shape/layout contracts; the generic family most closely mirrored by this SM75 port tops out at N=128. Therefore a drop-in N=256 alias was not faithful to the original family and must not be retained.

## Safe action

Remove the N=256 alias, tuner choice, and ABI bump. Restore the v245/v241 code exactly, preserving only the research/worklog evidence. The next candidate must use a shape/layout family whose SM75 thread-map contract is independently known to compile, or target a non-geometry hot-path discrepancy such as epilogue index-fragment hoisting. No new Kaggle run is justified for the compile-failing candidate.
