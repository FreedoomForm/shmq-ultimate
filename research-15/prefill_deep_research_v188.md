# Deep research: v187 failure and the next SM75 prefill seam

## Question

Why did v187’s shape-stratified direct-WMMA dispatch still fail mixed prefill, and which next experiment addresses a verified difference between the original MixLLM and SHMQ-Ultimate without repeating a rejected geometry-only change?

## Primary-source comparison

| Area | Original MixLLM | SHMQ-Ultimate v187 | Consequence |
|---|---|---|---|
| Launch family | `external/MixLLM/mixllm/kernels/mix_mma_multistage.cuh:353-520` searches and caches a configuration across many M/N/warp families and two stage counts; its fallbacks are `gemm<5,64,128,64,32>` and row-major `gemm_rm<5,64,64,32,32>` | `kernels/three_level_sm75.cu:711-750` selects only two direct-WMMA geometries: 8 warps × 128 channels for rows=16 and 4 warps × 64 channels otherwise | v187 changed only one tile choice. It did not restore the original family-level dataflow or staged operand layout.
| Integer partition concurrency | Original `mix_mma_multistage.cuh:240-260` records a fork event, launches INT4 and INT8 on independent persistent CUDA streams, records completion events, and joins them on the caller stream | v187 sends all precision planes through one direct-WMMA kernel on the caller stream; FP16 and integer work are selected by each CTA’s precision plane | The original overlap is still absent. Earlier v153/v154 proved that concurrency is correctness-sensitive and, with synchronous CUTLASS, insufficient alone; it remains a possible ingredient only when paired with a better mainloop.
| Operand staging | Original CUTLASS path uses tensor-op iterators and a multistage shared-memory mainloop. Its tuned configurations include 16/32/64/128/256 M families and N values up to 256 (`mix_mma_config.h:22-179`) | v187’s active WMMA path loads A and B directly from global memory with `wmma::load_matrix_sync` (`three_level_sm75.cu:104-280`), using a CTA-wide barrier around every 16-K substep | The direct-WMMA path has not adopted the original iterator/shared-memory operand layout. v158’s two-subtile attempt and v159’s per-warp shared-memory attempt were rejected, so the next change must avoid duplicating large fragments or adding private shared arrays.
| INT4 representation | Original path uses a custom CUTLASS dequantizer and shuffled/upcast integer operation in the mixed path (`mix_mma_multistage.cuh:193-205`) | SHMQ expands INT4 into signed INT8 before prefill and uses signed-INT8 WMMA (`three_level_sm75.cu:205-248`) | A native S4/U4 replacement is not safe without proving the exact asymmetric zero/scale identity. Preserve the signed-INT8 ABI for this iteration.
| Output path | Original has separate row-major and column-major families and CUTLASS epilogues (`mix_mma_multistage.cuh:266-349`) | SHMQ always writes indexed results through a manual FP32 scatter (`three_level_sm75.cu:251-279`) | The manual indexed scatter and global operand loads remain unoptimized differences, but changing both at once would obscure causality.

## v187 T4 evidence

Kaggle v187 ran on Tesla T4 / SM75 and passed native correctness, both decode gates, embedded contracts, allocator/import checks, and T4 hardware checks. It failed only mixed prefill end-to-end. For Qwen/Qwen2.5-0.5B’s fixed mixed QKV scenario (3584×3584, partition counts 2400/896/288), the reported speedups versus the identical `torch_fp16_linear` baseline were:

| Rows | v185 4-warp direct WMMA | v187 dispatch | Interpretation |
|---:|---:|---:|---|
| 1 | 1.0438× E2E | 1.2023× E2E | Decode path is unaffected and passes.
| 16 | 0.2661× E2E | 0.3104× E2E | The 8-warp/128-channel direct-WMMA branch helps this shape but remains 3.22× slower than dense.
| 128 | 0.1539× E2E | 0.0995× E2E | Restoring the 8-warp branch only for rows=16 was safe syntactically but the v187 result shows the direct-WMMA family is still badly mismatched to large-M prefill; v185 remains better at rows=128.

The v187 report is archived under `shmq-ultimate/mixllm_3level_kaggle/latest-output-v187-computer/`. Quantization remained about 0.027 ms, so CPU fallback and activation quantization are not the dominant cause. The expanded INT4 cache is a memory/lifetime concern but was already present in v185 and cannot explain the 10× rows=128 compute ratio by itself.

## Historical negative evidence that constrains the next change

The project worklog is the negative-evidence ledger. v145’s active 64×128 synchronous CUTLASS path achieved approximately 0.329× mixed rows=128 E2E; v156’s row-stratified direct-WMMA plus allocator-safe parallel CUTLASS achieved approximately 0.367× rows=128 E2E. Thus the original-style staged CUTLASS mainloop is materially better than direct WMMA for large M, although still far from the gate. v157’s geometry-only reduction did not solve the same regime. v158’s wider direct-WMMA tile was much worse, and v159’s private per-warp shared-memory variant was also worse. v160’s one-thread-per-output DP4A was decisively worse. These results rule out another simple M/N tile widening, another private shared-memory layout, or DP4A as the next primary experiment.

The current compiled CUTLASS helper (`kernels/sm75_cutlass_testbed.h:32-182`) uses a 32×128×64 SM75 tensor-op core with two synchronous stages, iterator-based shared-memory staging, and the same signed-INT8/scales/zero/output-index ABI. It is compiled but not selected by v187 (`three_level_sm75.cu:552-574` defines the helper; `:711-750` selects only direct WMMA). The helper’s active code does not reduce arithmetic, does not alter quality, and can be tested behind the existing host seam.

## NVIDIA constraints

NVIDIA’s first-party [Turing architecture overview](https://developer.nvidia.com/blog/nvidia-turing-architecture-in-depth/) documents that Turing Tensor Cores add INT8 and INT4 inference modes, while its SM has dedicated Tensor Cores and a unified shared-memory/L1 path. The official [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) is the governing source for warp-level synchronization and memory ordering. These sources support using SM75 Tensor Core INT8, but they do not imply that a hand-written direct-global WMMA kernel will match cuBLAS: operand staging, occupancy, epilogue mapping, and launch-family selection remain implementation responsibilities.

## Next isolated hypothesis: dispatch the existing SM75 CUTLASS helper only for rows >= 32

The evidence supports a narrow, reversible hybrid:

1. Restore v185 production dispatch semantics as the control for rows=16 and decode.
2. For rows >= 32, route only the integer partitions through the already compiled `shmq_cutlass_sm75::Int8Runner`, preserving its 32×128×64 two-stage iterator/shared-memory mainloop and allocator-recorded metadata lifetime.
3. Keep FP16 handling and all quantization, scales, zero points, output indices, and benchmark conditions unchanged.
4. Do not combine it with stream concurrency yet. v153/v154 show that auxiliary-stream lifetime and synchronization must be independently verified, and v156’s best large-M result came from a specific allocator-safe path that should not be reconstructed speculatively.

This is not expected to reach 2.6× by itself; it is a causal experiment that tests whether the original’s iterator/staging path is the missing large-M ingredient. The gate remains the decision boundary: retain only if all four gates pass, otherwise revert immediately. A small-M CUTLASS dispatch for rows=16 is not proposed because v152 already failed to compile for a 16×128×64 SM75 core and v145/v156 show that the synchronous path is weak at small M.

## Sources

1. Original MixLLM source: `external/MixLLM/mixllm/kernels/mix_mma_multistage.cuh`, lines 178-349 and 353-520.
2. Original MixLLM configuration catalog: `external/MixLLM/mixllm/kernels/mix_mma_config.h`, lines 22-179.
3. SHMQ active kernel and dispatch: `external/MixLLM/mixllm/kernels/three_level_sm75.cu`, lines 104-280 and 552-753.
4. SHMQ CUTLASS helper: `external/MixLLM/mixllm/kernels/sm75_cutlass_testbed.h`, lines 32-195.
5. Project measurements and rejected experiments: `research-15/worklog.md`, lines 544-707 and 751-773; `research-15/v121_runtime_research.md`, lines 181-230.
6. NVIDIA, “NVIDIA Turing Architecture In-Depth,” 2018: <https://developer.nvidia.com/blog/nvidia-turing-architecture-in-depth/>.
7. NVIDIA, “CUDA Programming Guide,” v13.3, updated 2026: <https://docs.nvidia.com/cuda/cuda-programming-guide/index.html>.

## Decision

v187 is a **no-go** and must not replace v185. The next code change should be the isolated rows>=32 dispatch to the existing SM75 CUTLASS helper, after restoring the v185 WMMA branch and adding a contract test for the new branch. Kaggle T4 is required before any retention decision.
