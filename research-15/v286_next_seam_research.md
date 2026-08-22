# v286 next-seam primary-source research

## Differential finding

NVIDIA CUTLASS documents a hierarchical loop in which threadblock tiles are partitioned across warp-N and warp-M, and larger warp tiles are chosen to maximize reuse [1]. Microsoft MixLLM's launcher follows that hierarchy through a `DefaultMmaCore` whose warp shape is independently selected for each configuration [2].

The v284/v285 direct pair kernel instead assigns one warp to a 16-channel strip and repeats the same four 8-row A fragments in every N warp. For `kPairChannels=128`, the 8-warp candidate therefore makes eight independent warp-level A loads for the same 32x32 activation tile, while each warp computes only 16 output channels. v285 removed repeated B loads inside a warp, but the T4 result barely improved, supporting the conclusion that this cross-warp A duplication and the resulting 8-warp resource footprint dominate the direct pair path.

## Safe experiment

Use a 4-warp block for a 128-channel pair tile. Each warp handles 32 channels, represented as four 8-column native `m8n8k32` N subtiles. This retains the legal low/high `u4*u4` and `s4*u4` arithmetic, the exact `a=low+16*high` identity, row-sum zero correction, packed weight source, one-writer-per-output mapping, and all existing stream/fallback behavior. The only change is warp ownership and the resulting reduction in redundant A loads from eight to four per activation tile. The 64-channel family remains 4 warps x 16 channels.

This is not assumed to be faster: four N subtiles increase per-warp B fragment/register pressure. It is admissible only if the T4 gate passes correctness, timing integrity, memory, quality/vLLM availability, and prefill performance. Otherwise restore the safe expanded fallback immediately.

## References

[1]: https://docs.nvidia.com/cutlass/4.3.5/media/docs/cpp/efficient_gemm.html
[2]: https://raw.githubusercontent.com/microsoft/MixLLM/main/mixllm/kernels/mix_mma_multistage.cuh
