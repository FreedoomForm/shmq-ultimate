# v254 deep research: upstream M=128,N=64 stage-2 CUTLASS candidate

## v253 result and production rollback

Kaggle server version 248 (v253) compiled and passed native correctness, the mixed-stride probe, timing integrity, and all non-performance gates, but the four-warp N=64 native pair tile still measured Qwen mixed rows=128 at 12.954x slower than dense FP16. It is rejected as a production path. The next candidate restores v241's `use_fused_int4=false` mixed overlap; the validated native pair remains probe-only/unselected.

## Original-versus-SMQ discrepancy

The upstream `gemm_configs` and `gemm_configs_rm` tables include `M=128,N=64,K=64` families, while the SM75 runner currently exposes only `M=32,N=128` and `M=32,N=64`. The prior v199 experiment tested M=64,N=128 and was rejected at 0.2671x speedup, but it did not test the transposed aspect ratio M=128,N=64. A stage-2 `DefaultMmaCore<GemmShape<128,64,64>, WarpShape<32,32,64>, InstructionShape<8,8,16>>` uses eight warps and remains inside the same legal SM75 instruction/core family.

## Candidate and safety boundary

Add `CoreM128N64`/`Int8RunnerM128N64` to the exact-shape CUDA-event tuner with N=128 and N=64 fallbacks. Bump only the tuner cache ABI. Restore mixed overlap with the existing staged INT4/INT8 runners and caller FP16 stream. No arithmetic, quantization, metadata, stream/event, output ABI, model, benchmark, or quality change is allowed. Retain this candidate only if it compiles on T4 and every required correctness, timing-integrity, and performance gate passes.
