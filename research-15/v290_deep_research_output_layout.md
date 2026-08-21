# v290 deep research: output-layout comparison

## Original MixLLM

The preserved original linear path invokes the upstream mixed GEMM with a column-major result whenever both INT8 and INT4 partitions are present. Its forward wrapper then performs a second output-layout operation: a custom CUDA transpose for selected large output widths, or `output.t().contiguous()` for other widths. The original comments explicitly identify this transpose as a possible slow path.

## SHMQ v289

SHMQ’s SM75 CUTLASS and unified scheduler write directly into a row-major `[rows, output_width]` output using the original output-channel indices. The FP16 branch computes `at::mm` into a temporary partial matrix and scatters it directly to row-major output; the integer branches use their row-major indexed epilogue/scatter contracts. The vLLM patch receives that row-major result and only restores leading dimensions/adds bias.

## Decision

The original output transpose is not a missing optimization in SHMQ; it is already eliminated by the SM75 output ABI. Reintroducing a column-major intermediate would add work and violate the direct-output design. No v290 production code change is justified from this seam. The remaining likely cost centers are activation quantization plus separate GEMM launches and the FP16 `at::mm`-plus-scatter path, but changing either requires a larger kernel design and must preserve exact three-level arithmetic and output mapping. This audit finding is recorded without a performance claim; Kaggle remains the only authoritative performance gate.
