# v256 deep research: larger legal SM75 tile and timing integrity

## Evidence from the v255 Kaggle run

The v255 dispatch correction restored staged CUTLASS for mixed large-M and produced the expected functional result, but the run was rejected. Qwen mixed rows=128 reported integer GEMM p50 `0.8256 ms`, end-to-end p50 `0.4080 ms`, and timing-integrity ratio `2.0235`, so the gate correctly failed. Other row counts and pure-precision cases passed timing integrity. This is not an acceptable performance result because the end-to-end measurement must not be shorter than the GEMM work it contains.

The v255 source identity matched the committed CUDA source. The staged branch still uses the original-style auxiliary INT4/INT8 streams and caller-stream event joins. The official Microsoft launcher uses the same topology: persistent auxiliary streams, a fork event recorded on the caller stream, branch completion events, and caller-stream waits. Therefore the next change must not weaken the timing gate or replace device timing with a favorable host measurement.

## Primary-source tile comparison

The official Microsoft `mix_mma_config.h` contains a broad column-major configuration family including `{128,128,32,64}`, `{128,128,64,32}`, and `{128,128,64,64}` in the form `(blocksize_m, blocksize_n, tilesize_m, tilesize_n)`. The existing SHMQ SM75 port has already confirmed the stage-2 SM75 specialization for `M=128,N=64,K=64` and has rejected `N=256`, stage-3, and a prior `M=64,N=128` attempt. It has not yet tested the legal-looking `M=128,N=128,K=64` family.

The vendored SM75 `DefaultMmaCore` is generic in operand element types and maps the row-major `A`, column-major `B`, TensorOp path to a warp-level default MMA. The SM75 header explicitly provides the native `m8n8k32` `uint4b_t * uint4b_t -> int` specialization, while the current staged runner uses `int8_t * int8_t -> int` at `m8n8k16`. A `M=128,N=128,K=64` stage-2 INT8 candidate preserves the already-validated arithmetic and metadata ABI; it changes only block geometry and tuner selection.

## Safe v256 change

Add `CoreM128N128 = GemmShape<128,128,64>` with the existing `WarpShape<32,32,64>`, instruction shape `<8,8,16>`, SM75 stage-2 operator, and a bounded tuner candidate. Keep N=128, N=64, and M=128,N=64 candidates. Do not alter the mixed dispatch selector, timing gate, arithmetic, benchmark, or model. If the source contracts, full tests, and native proof pass locally, one Kaggle run will determine whether this family improves the valid mixed path and whether timing integrity is stable.

A direct packed INT4 pipeline remains a separate future seam: it must use the legal SM75 `u4*u4` MMA, preserve asymmetric zero correction, and not reuse the SM80 mixed-input instruction. The M=128,N=128 candidate is intentionally lower-risk because it does not change that arithmetic seam.

## Sources

1. Official Microsoft MixLLM: `https://github.com/microsoft/MixLLM`.
2. `mixllm/kernels/mix_mma_config.h`, official broad configuration tables.
3. `mixllm/kernels/mix_mma_multistage.cuh`, official overlap and launcher behavior.
4. SHMQ `three_level_sm75.cu`, current SM75 tuner and mixed overlap.
5. Vendored CUTLASS `default_mma_core_sm75.h` and `mma_sm75.h`, legal SM75 core and MMA specializations.
