# Differential audit before v56 continuation

## Original linear.py
The original constructor sends INT4 weights through interleave_uint4_for_cutlass before packing. The observed 32-column remap uses offsets +12, -4, +8, -8, +4, -12 in 4-wide subranges, followed by the 8-column order i, i+4, i+1, i+5, i+2, i+6, i+3, i+7. This is a CUTLASS-specific B layout, not direct row-major storage.
The original activation_quantization has @torch.compile. It views input as [M,K/128,128], computes abs().amax over each group divided by 127, rounds to int8, transposes the scale matrix, and writes it into padded scale_act. The reference shape is [K/128, M + M % 2].
The original forward invokes mixllm_gemm with quantized activation, padded activation scales, INT4 scales and zeros, precision indices, and packed weights. Its result is column-oriented before a custom mixllm transpose or output.t().contiguous().

## Original quantizer.py
Quantizer.quantize_activation reshapes activation to [-1, hidden_size], uses group size 128 by default, asserts hidden-size divisibility, and applies groupwise parameters. K-grouping is therefore a correctness contract.

## Original mq_mma_multistage.h
The original defines Base::kStages = Stages, double-buffered warp_loaded_frag_A_[2] and warp_loaded_frag_B_[2], smem_write_stage_idx_ and smem_read_stage_idx_. It has a prologue, stage catch-up, predicated cp_async copies, and a mainloop that overlaps global loads with TensorOp work. This is a software-pipelined multistage GEMM, not a single serial loop.

## Original mix_mma_multistage.cu
The original test instantiates separate MmaCore_INT8 and MmaCore_INT4 TensorOp paths. It uses ThreadblockShape K=64, WarpShape K=64, InstructionShape 16x8x32, and NumStages. Separate INT8 and INT4 matrices, scales, zeros, and indices are prepared and synchronized.

## Concrete current mismatch
Our current SM75 kernel is monolithic. Prefill still branches on runtime precision inside the WMMA loop, decode uses DP4A or half2, INT4 uses direct row-major or expanded cache rather than the original interleaved packed B layout, scale_act is checked as [K/128,M] rather than padded M-even, and the prefill path has no cp_async multistage pipeline or separate precision-specialized launches.

## Consequence
The largest audit-backed architectural gap is original CUTLASS multistage pipelining versus our single-pass loops. Any next hypothesis must be a narrowly scoped pipeline or original-layout compatibility change. No intuition-only decode tweak is accepted.
