# External research: SM75 native INT4

## NVIDIA Turing architecture
Source: [NVIDIA Turing Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-turing-architecture-in-depth/)

NVIDIA states that Turing Tensor Cores add INT8 and INT4 precision modes for inference and that low-precision INT4 matrix operations are supported on Turing. The article’s Turing specification table lists peak INT4 Tensor throughput above INT8 for the architecture family.

## CUTLASS SM75 MMA specializations
Source: [CUTLASS `include/cutlass/arch/mma_sm75.h`](https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/arch/mma_sm75.h)

The SM75 CUTLASS specializations expose native integer Tensor Core operators at `GemmShape<8, 8, 16>` for signed/unsigned INT8 combinations and at `GemmShape<8, 8, 32>` for signed/unsigned INT4 combinations. The file includes `S4*S4`, `U4*S4`, `S4*U4`, and `U4*U4` variants, with S32 accumulation and inline `mma.sync.aligned.m8n8k32...s4/u4.s32` instructions. This confirms that the current SM75 path is not using native INT4 MMA: it expands INT4 codes minus zero points to signed INT8 and uses the slower 8-bit Tensor Core operator.

Source: [CUTLASS `include/cutlass/arch/wmma_sm75.h`](https://raw.githubusercontent.com/NVIDIA/cutlass/master/include/cutlass/arch/wmma_sm75.h)

CUTLASS’s WMMA wrapper maps `int4b_t` to the experimental WMMA sub-byte type and requires the native shape `8x8x32` for the INT4 specialization. The direct current WMMA kernel instead uses a `16x16x16` signed-INT8 WMMA tile, so it also does not exploit the SM75 INT4 instruction.

## PyTorch auxiliary-stream lifetime API
Source: [PyTorch `CUDACachingAllocator.h`](https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.h)

PyTorch exposes `c10::cuda::CUDACachingAllocator::recordStream(const DataPtr&, CUDAStream)` for registering an allocation’s use on an auxiliary stream. Source: [PyTorch CUDAStream C++ API](https://docs.pytorch.org/cppdocs/api/cuda/streams.html). The raw stream can be wrapped with `c10::cuda::getStreamFromExternal(cudaStream_t, DeviceIndex)`. These APIs were used to repair the v153 asynchronous metadata lifetime bug in v154.

## Implication for the next experiment
Native INT4 requires preserving the original 4-bit codes and applying zero-point correction mathematically, rather than feeding `(code-zero)` values into signed INT4 directly because the subtraction can require values outside signed 4-bit range. For each 128-element group, a valid identity is `A @ (code - zero) = A @ code - zero * sum(A)`, with the result multiplied by activation and weight scales. A native INT4 kernel must implement this correction exactly enough for the existing numerical gate; it must not change quantization or benchmark settings.


## PTX instruction research
Source: [NVIDIA PTX ISA 9.3](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)

The official PTX reference was consulted for integer `mma.sync` forms. The vendored CUTLASS SM75 file is the reliable local specialization inventory for this project: it exposes pure INT8 (`m8n8k16`) and pure/mixed signedness INT4 (`m8n8k32`) forms, but no S8-by-S4 or S8-by-U4 operator. The original MixLLM’s `OpMultiplyAddMixedInputUpcast` is an SM80-oriented path in this vendored source, not an immediately portable SM75 operator. Therefore an exact S8-activation/native-4-bit-weight path would require a documented multi-instruction decomposition or a different activation representation; it must not be guessed or substituted with reduced precision.
