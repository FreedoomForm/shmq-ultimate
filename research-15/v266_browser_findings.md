# Browser research findings for the next SM75 iteration

The official Microsoft MixLLM repository confirms the upstream project is organized around a dedicated `mixllm` kernel package, vLLM patch, and module-level implementation. The paper's method section states that mixed precision is assigned between output features, making each precision branch a disjoint output-channel subproblem. It also states that MixLLM's system design uses two-step dequantization to reuse fast INT8 Tensor Cores and a software pipeline that overlaps memory access, dequantization, and matrix multiplication. The paper's kernel description therefore supports keeping INT4 and INT8 branches parallel rather than combining them into one serial GEMM.

The upstream CUDA testbed's epilogue constructs an index fragment from the partition indices and writes converted accumulators directly to the global output using those indices inside the fused kernel. SHMQ's SM75 CUTLASS runner follows the same semantic mapping, but its implementation is a custom SM75 port with expanded signed INT4 weights and separate stream launches. SHMQ's FP16 cuBLAS path necessarily remains a two-step partial-plus-scatter path because cuBLAS does not expose the original fused indexed epilogue.

The v264 browser/Kaggle evidence also confirms that ATen `index_copy_` cannot consume the project's int32 index ABI on the target environment; the custom int32 scatter remains the valid baseline. The next research should therefore target the integer path's expanded INT4 and separate CUTLASS launch cost, not repeat the failed ATen epilogue substitution.

References:

1. [Microsoft MixLLM repository](https://github.com/microsoft/MixLLM)
2. [MixLLM paper, arXiv HTML](https://arxiv.org/html/2412.14590v1)
3. Local upstream source: `/tmp/original-MixLLM/mixllm/kernels/mma_multistage_testbed.h`
4. Local SHMQ source: `shmq-ultimate/external/MixLLM/mixllm/kernels/sm75_cutlass_testbed.h`
