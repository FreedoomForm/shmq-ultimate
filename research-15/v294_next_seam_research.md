# v294 next-seam primary-source research

## Differential finding

The v293 source repair was not exercised because the safe-control branch still disabled v3. The completed v293 T4 run therefore cannot validate or reject the fragment transformation. The original Microsoft MixLLM launcher confirms that its row-major path directly passes `matrix_B_interleaved` into the INT4 testbed, while its mixed-input warp operator leaves the B fragment unchanged and shuffles A before conversion. SHMQ’s existing v3 ABI and C++ scheduler already have the same conceptual slots and branch, but the branch must be enabled intentionally.

## Safe integrated candidate

v294 combines the two evidence-backed pieces that must be tested together: (1) the v293 adapter transform, which leaves B unchanged and shuffles A with `Operand::kA`; and (2) the v289 Python v3 handoff, which prepares the original-equivalent interleaved INT4 tensor and cached transposed metadata. The v290 C++ validation exemption for a nonempty packed tensor is also required. The expanded v299 route remains the fallback whenever v3 is absent or no integer partition exists. No synthetic fragment permutation, row-major flag experiment, new tile, quantizer, model, benchmark, or quality relaxation is included.

The gate must verify that the packed path is actually selected, then pass smoke and Qwen large-M correctness before performance is admissible.

## References

[1]: https://raw.githubusercontent.com/microsoft/MixLLM/main/mixllm/kernels/mix_mma_multistage.cuh
[2]: https://raw.githubusercontent.com/microsoft/MixLLM/main/mixllm/kernels/cutlass_extension/mq_mma_mixed_input_tensor_op.h
[3]: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html
