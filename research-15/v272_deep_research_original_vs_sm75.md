# v272 deep research: original MixLLM versus SHMQ SM75 large-M dataflow

## Question

Why does the v271 Qwen mixed 4/8/16 path remain slower than dense FP16 on a Tesla T4 even though the exact contracts and native correctness gates pass, and what is the smallest safe next seam to test?

## Evidence from the v271 T4 run

Kaggle version 268 used the committed v271 source and passed the embedded contract suite (`83 tests`, `skipped=2`), native correctness, allocator checks, and timing-integrity checks. The production gates still failed because mixed decode E2E and mixed prefill E2E were false. Qwen mixed 4/8/16 measured GEMM/E2E speedups of `1.26509x/0.85405x` at rows=1, `0.28189x/0.28260x` at rows=16, and `0.52863x/0.33517x` at rows=128. The rows=128 report also showed `expanded_int4_bytes=8601600` and `prefill_metadata_bytes=251776`, while the dense FP16 baseline used the same T4 and benchmark conditions.

## Original MixLLM dataflow

The original module converts INT4 weights to the kernel-facing interleaved CUTLASS layout once during module construction. In `/tmp/original-MixLLM/mixllm/nn/modules/linear.py:99-104`, INT8 weights are made contiguous and INT4 weights pass through `interleave_uint4_for_cutlass()`; the interleave and packing implementation is `/tmp/original-MixLLM/mixllm/nn/modules/linear.py:121-157`. The forward path then passes the prepared weights directly to the fused GEMM at `/tmp/original-MixLLM/mixllm/nn/modules/linear.py:202-205`; it does not expand INT4 to signed INT8 before a large-M call.

The original launcher constructs persistent INT4/INT8 streams and events once in `/tmp/original-MixLLM/mixllm/kernels/mix_mma_multistage.cuh:18-27`. It records one fork event on the caller stream, launches the INT4 and INT8 branches on their auxiliary streams, records branch events, and makes the caller wait at `/tmp/original-MixLLM/mixllm/kernels/mix_mma_multistage.cuh:240-260` and the row-major equivalent at `:328-348`. The launcher owns a broad shape/configuration table and persists a best configuration per shape at `:353-519`; the source table contains dozens of legal configurations in `/tmp/original-MixLLM/mixllm/kernels/mix_mma_config.h:227-317`.

## SHMQ divergence that explains the large-M loss

SHMQ's current prefill backend prepares and passes a signed INT8 expansion of the packed INT4 matrix. The cache owner expands packed codes in `/home/ubuntu/v204-shmq/shmq-ultimate/external/MixLLM/mixllm/nn/modules/three_level_linear.py:93-112`, and the backend requests that expanded matrix for every rows>1 call in `sm75_backend.py:196-207`. The v3 prefill route then passes `expanded_int4` into `/home/ubuntu/v204-shmq/shmq-ultimate/external/MixLLM/mixllm/sm75_backend.py:266-284`.

More importantly, the large-M C++ branch does not instantiate an INT4 CUTLASS runner for that partition. `/home/ubuntu/v204-shmq/shmq-ultimate/external/MixLLM/mixllm/kernels/three_level_sm75.cu:1209-1211` calls `run_cutlass_int_partition()` with `expanded_int4`; that helper at `:1148-1154` selects `run_cutlass_config()`, which dispatches only `Int8Runner`, `Int8RunnerN64`, `Int8RunnerM128N64`, or `Int8RunnerM64N64` in `:139-161`. The runner definition in `/home/ubuntu/v204-shmq/shmq-ultimate/external/MixLLM/mixllm/kernels/sm75_cutlass_testbed.h:20-24` fixes `ElementB=int8_t`, and all four aliases at `:203-244` use `OpMultiplyAddSaturate`. Thus the supposed INT4 large-M branch is an INT8 Tensor Core GEMM over an 8.6 MB expanded weight view, not native 4-bit weight traffic.

SHMQ does possess a native SM75 INT4 pair kernel. Its exact arithmetic is documented in `/home/ubuntu/v204-shmq/shmq-ultimate/external/MixLLM/mixllm/kernels/three_level_sm75.cu:937-1093`: it splits each signed INT8 activation into low/high 4-bit nibbles, performs two native m8n8k32 pair operations, reconstructs `P_low + 16*P_high - zero*sum(A8)`, and applies the activation and weight scales. The launch wrapper is at `:1095-1115`. This is the only currently implemented path that avoids the INT4-to-INT8 expansion for the large-M candidate.

The staged SM75 extension is generic over its CUTLASS policy, but its dequantizer and iterator ABI would require a new paired operator policy; `/home/ubuntu/v204-shmq/shmq-ultimate/external/MixLLM/mixllm/kernels/cutlass_extension/mq_mma_pipelined_sm75.h:124-185` exposes the `Operator::FragmentA/FragmentB` and dequantizer seam. The legal SM75 integer WMMA definition in the vendored CUTLASS header is only the signed `int4b_t * int4b_t` `8x8x32` specialization at `cutlass/include/cutlass/arch/wmma_sm75.h:57-116`; therefore a direct int8*int4 staged operator cannot be assumed from the upstream SM80 mixed operator. The existing SHMQ pair arithmetic is the safe source of truth for a first test.

## Decision for the next candidate

The next candidate should route the INT4 partition of rows>=32 through the existing native pair kernel while retaining the original-style persistent INT8 auxiliary stream, FP16 caller-stream branch, event fork, and caller waits. It should also stop requiring or preparing `expanded_int4` for that rows>=32 branch; rows<32 keep the validated v2 path and its expanded ABI. This changes neither quantization, weights, output mapping, model, benchmark, nor arithmetic. It removes the known INT4-as-INT8 dataflow mismatch and reduces avoidable live memory pressure. It is preferable to immediately inventing a new staged pair iterator because the existing pair kernel already has local correctness coverage and exact zero-point/scaling semantics.

The candidate must be rejected unless all local tests pass and one final T4 run passes every required gate, including timing integrity. If native pair routing improves the INT4 branch but still misses the target, the subsequent seam is a staged paired SM75 operator built from the same two native m8n8k32 operations, not a loosened quality threshold or benchmark change.

## Primary sources

1. [Original MixLLM linear module](file:///tmp/original-MixLLM/mixllm/nn/modules/linear.py)
2. [Original MixLLM multistage launcher](file:///tmp/original-MixLLM/mixllm/kernels/mix_mma_multistage.cuh)
3. [Original MixLLM configuration tables](file:///tmp/original-MixLLM/mixllm/kernels/mix_mma_config.h)
4. [SHMQ SM75 backend](file:///home/ubuntu/v204-shmq/shmq-ultimate/external/MixLLM/mixllm/sm75_backend.py)
5. [SHMQ three-level module](file:///home/ubuntu/v204-shmq/shmq-ultimate/external/MixLLM/mixllm/nn/modules/three_level_linear.py)
6. [SHMQ SM75 kernel](file:///home/ubuntu/v204-shmq/shmq-ultimate/external/MixLLM/mixllm/kernels/three_level_sm75.cu)
7. [SHMQ SM75 CUTLASS runner](file:///home/ubuntu/v204-shmq/shmq-ultimate/external/MixLLM/mixllm/kernels/sm75_cutlass_testbed.h)
8. [Vendored CUTLASS SM75 WMMA definition](file:///home/ubuntu/v204-shmq/shmq-ultimate/external/MixLLM/mixllm/kernels/cutlass/include/cutlass/arch/wmma_sm75.h)
