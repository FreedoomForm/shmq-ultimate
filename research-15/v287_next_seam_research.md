# v287 next-seam primary-source research

## Differential finding

NVIDIA CUTLASS’s Turing guidance describes the threadblock K tile as the mainloop stage and warns that tile geometry must be internally consistent; the SHMQ staged SM75 pipeline is intentionally built around `Shape::kK == 64` because one quantization group is 128 elements and two 64-element halves share metadata [1]. Microsoft MixLLM likewise instantiates a 64-wide threadblock K tile while its native SM80 MMA consumes k32 instruction groups [2].

The direct pair kernel currently stages only a 32-wide packed K tile and executes one legal `m8n8k32` low/high pair per chunk. For each 128-element quantization group this creates four shared-memory barriers and four staging passes, whereas the surrounding CUTLASS dataflow uses K=64 stages. The pair kernel therefore pays an avoidable synchronization/prologue cost even though its arithmetic naturally supports two k32 instructions per 64-wide stage.

## Safe experiment

Change only the pair kernel’s staging tile from packed K=32 to packed K=64. Each staged tile contains 64 logical activation/weight values; the kernel issues two legal native `m8n8k32` low/high MMA pairs at offsets 0 and 32 before the next CTA barrier. Keep group-scale/zero correction outside the pair MMAs exactly as before. This halves stage barriers from four to two per 128-element quantization group, preserves all values and output ownership, and aligns the pair’s stage width with the already validated SM75 K64 pipeline.

The experiment must verify shared-memory alignment, WMMA leading dimensions, native correctness, timing integrity, and performance on T4. It must not be accepted if any gate fails. No claim is made before Kaggle.

## References

[1]: https://docs.nvidia.com/cutlass/4.3.5/media/docs/cpp/efficient_gemm.html
[2]: https://raw.githubusercontent.com/microsoft/MixLLM/main/mixllm/kernels/mix_mma_multistage.cuh
