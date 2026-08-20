# v253 deep research: four-warp N=64 native pair tile

## v252 rejection

Kaggle server version 247 (v252) compiled and preserved correctness, but the lazy expanded-INT4 bypass did not improve the pair path: mixed Qwen rows=128 remained 10.97x slower than dense FP16 and timing integrity failed. The cache change is therefore rejected for production and must be restored to the v241/v251 ABI behavior.

## Upstream comparison

The original MixLLM configuration table includes a legal `32x64x64` family with the same `32x32` warp tile and K=64 staging family used by this port. The current native pair kernel instead covers only a 32-channel CTA with four warps, so the mixed INT4 partition launches roughly twice as many CTAs as an N=64 tile would require. The rejected v244 experiment widened to 64 channels by using eight warps; its result remained 13.49x slower and the research identified unchanged row-subtile reload dataflow as the main issue. v249 fixed the reload, but the pair still used the narrow geometry.

## Candidate

Keep four warps per CTA and widen the packed native pair tile to N=64. Each warp owns 16 channels, implemented as two 8-column N subtiles, while all four warps continue to iterate the four 8-row subtiles. Pack the 64-channel B panel once per K chunk; reuse it for all row/N subtiles. Accumulate low/high m8n8k32 pairs and apply the exact zero correction and scales to the same indexed output ABI. The candidate reduces channel-CTA count without the eight-warp occupancy cost of v244 and remains within the upstream legal `32x64x64` family.

Restore expanded INT4 preparation for the current direct-WMMA fallback and preserve the v241 metadata/stream contracts. Retain the candidate only if it compiles on SM75 and all required correctness, timing-integrity, and performance gates pass.
