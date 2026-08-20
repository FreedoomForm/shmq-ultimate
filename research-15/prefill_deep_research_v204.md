# v204 deep research: native homogeneous SM75 INT4 decomposition

## Evidence from v201-v203

The current accepted v200 design expands checkpoint INT4 weights to signed INT8 and then invokes a supported SM75 `s8*s8` Tensor Core runner. v201 and v202 attempted to eliminate expansion by decoding nibbles in a standalone kernel and inside a CUTLASS iterator; both measured approximately `0.065x` at rows=128. v203 vectorized the expansion with aligned 32-bit loads, `__byte_perm`, and `__vsub4`, but the T4 run still measured `0.0658x` at rows=128 and failed timing-integrity because its reported GEMM time exceeded its E2E time. The next candidate must therefore use the GPU's native 4-bit MMA instead of converting each weight to INT8.

## Primary-source findings

NVIDIA CUTLASS documentation states that CUTLASS supports narrow signed and unsigned 4-bit integer types on Turing/SM75 and organizes GEMM through reusable threadblock, warp, iterator, and instruction-level abstractions [1]. The current NVIDIA `default_gemm_configuration.h` contains an explicit SM75 configuration for `int4b_t` by `uint4b_t`: threadblock `128x256x128`, warp `64x64x128`, instruction `8x8x32`, and two stages [2]. This confirms that homogeneous signed-by-unsigned 4-bit Tensor Core math is an officially supported architecture path, not the unsupported SM80 mixed INT8xINT4 instruction rejected in earlier versions.

NVIDIA's SM75 MMA source defines the exact instruction wrapper `mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.u4.s32`, with eight packed 4-bit elements per fragment and two int32 accumulators per warp [3]. The same source also defines the `s4*s4`, `u4*s4`, and `u4*u4` variants. These wrappers prove the arithmetic and fragment contract needed by v204. The vendored project copy contains these instruction wrappers, but its old SM75 `DefaultMmaCore` header has no 4-bit threadblock specialization; v204 therefore must add a narrowly scoped custom core or a direct warp-level kernel rather than pretending that the existing INT8 core can accept 4-bit types.

The quantized checkpoint stores unsigned INT4 code `q` and a per-group zero point `z`, while the activation is signed INT8 `a`. The exact product is `(q-z)*a = q*a - z*a`. A native `s4*u4` MMA can compute the first term by decomposing each signed INT8 activation into two signed 4-bit digits, `a = a_low + 16*a_high`, and issuing two native 4-bit MMAs against the packed unsigned `q`. The correction term is `z * sum(a)` for each 128-element group and can be accumulated alongside the two MMAs using a small integer reduction. This preserves the existing quantization equation exactly; it is not a quality-reducing approximation.

## v204 scope and risk controls

The candidate will be opt-in for only the large-M mixed INT4 partition and will retain v200's expanded INT8 runner as a fallback. The implementation must use the supported `8x8x32` instruction and stage-2 limits, must not introduce `16x8x32` or `Stages=3/5`, and must not alter the model, benchmark shapes, quality tolerances, or timing guard. If the native path cannot be made correct with a minimal custom warp-level kernel, it will be rejected locally before Kaggle rather than weakening any contract.

## References

[1]: https://docs.nvidia.com/cutlass/latest/overview.html "NVIDIA CUTLASS Overview"
[2]: https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/device/default_gemm_configuration.h "NVIDIA CUTLASS default SM75 INT4 configurations"
[3]: https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/mma_sm75.h "NVIDIA CUTLASS SM75 INT4 MMA wrappers"
[4]: https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html "NVIDIA CUDA integer intrinsics"

## Additional source findings

The vendored `default_mma_core_wmma.h` provides the missing SM75 threadblock scaffold for sub-byte WMMA. Its row-major-A/column-major-B specialization uses `OpClassWmmaTensorOp`, `kAccessSizeInBits=128`, `PitchLinearStripminedThreadMap` global/shared iterators, and a stage-2 `MmaPolicy` built around `MmaTensorOpWmma` [5]. The vendored CUTLASS unit tests instantiate a `64x64x128` threadblock with `64x64x128` warp shape and `8x8x32` instruction shape for signed 4-bit WMMA [6]. The same scaffold is applicable to signed-A/unsigned-B because the SM75 MMA wrapper defines `s4*u4` with the same `8x8x32` shape [3].

`array_subbyte.h` confirms that `Array<T,N,false>` packs 4-bit logical elements into byte/word storage: eight 4-bit elements occupy four bytes, and reads/writes use bitfield extraction and insertion [7]. Therefore, the checkpoint's two-nibbles-per-byte weight buffer can match CUTLASS's `uint4b_t` AccessType ABI without another weight expansion. The global iterator performs byte-address arithmetic using `sizeof_bits<Element>`, so its stride must be supplied in logical packed-element units, not as an expanded INT8 stride.

The custom SM75 MQ pipeline's `MmaTensorOpDequantizer` is instantiated with zero-point handling enabled when the warp operator is not `s8*s8`; this is the relevant path for a native `s4*u4` runner. The v204 implementation must verify its exact correction formula against the quantization equation before enabling it, rather than applying a second correction kernel that could double-subtract zero points [8].

[5]: https://github.com/FreedoomForm/shmq-ultimate/blob/audit-v184-computer/shmq-ultimate/external/MixLLM/mixllm/kernels/cutlass/include/cutlass/gemm/threadblock/default_mma_core_wmma.h "Vendored CUTLASS WMMA threadblock core"
[6]: https://github.com/FreedoomForm/shmq-ultimate/blob/audit-v184-computer/shmq-ultimate/external/MixLLM/mixllm/kernels/cutlass/test/unit/gemm/threadblock/mma_pipelined_wmma_sm75.cu "Vendored SM75 sub-byte WMMA tests"
[7]: https://github.com/FreedoomForm/shmq-ultimate/blob/audit-v184-computer/shmq-ultimate/external/MixLLM/mixllm/kernels/cutlass/include/cutlass/array_subbyte.h "Vendored CUTLASS packed sub-byte arrays"
[8]: https://github.com/FreedoomForm/shmq-ultimate/blob/audit-v184-computer/shmq-ultimate/external/MixLLM/mixllm/kernels/cutlass_extension/mq_mma_tensor_op_dequantizer.h "Vendored MixLLM dequantizer"

## v204 audit correction: original-vs-port comparison

The public upstream repository is `microsoft/MixLLM`. Its original `mix_mma_multistage.cuh` launches the INT4 and INT8 partitions on two private CUDA streams after a fork event, then joins both events on the caller stream. It does **not** run both precisions in one CUDA kernel. The upstream launch uses `ThreadblockShape<..., ..., 64>`, `WarpShape<..., ..., 64>`, and instruction shape `16x8x32`; its INT4 path uses the SM80 mixed-input upcast operator (`OpMultiplyAddMixedAndShuffledInputUpcast`) rather than native SM75 s4/u4 WMMA. The upstream testbed accepts packed `uint4b_t` weights and performs the mixed-input conversion inside the warp operator. These facts explain why v200's stream overlap is a faithful structural port while v204's native path is a new implementation, not a direct copy.

A decisive local constraint was found in `mq_mma_pipelined_sm75.h`: `copy_scales_and_advance()` contains `static_assert(Shape::kK == 64)` and advances quantization metadata across two 64-wide halves of each 128-element group. Therefore the proposed `WmmaCore` with threadblock K=128 cannot be passed unchanged to `MQMmaPipelinedSm75`; it would fail the compile-time contract and also mismatch the metadata schedule. Any native WMMA adapter must use a K=64 threadblock shape or a separately audited metadata pipeline. The existing WMMA operator also exposes no `transform()` method, unlike the tensor-op warp operator, so the pipeline needs an explicit identity-fragment adapter before WMMA can compile.

The original MixLLM quantization equation remains `(q-z)*a`; native s4/u4 MMA computes only the `q*a` term. The zero-point term must be preserved exactly, either through a verified dequantizer or a separate correction with an independent reference test. The v204 note's earlier statement that `ApplyWeightZero=false` is automatically safe was therefore not established; it is a hypothesis that must not be shipped without correctness evidence.

References: [1] https://github.com/microsoft/MixLLM/tree/main/mixllm/kernels (upstream kernel directory); [2] https://github.com/microsoft/MixLLM/blob/main/mixllm/kernels/mix_mma_multistage.cuh (upstream stream launch and geometry); [3] https://github.com/microsoft/MixLLM/blob/main/mixllm/kernels/mma_multistage_testbed.h (upstream packed-weight testbed); [4] local `mq_mma_pipelined_sm75.h` lines 426-450 and 662-670.

Decision: do not blindly apply the inherited K=128 runner snippet. First adapt the WMMA path to the existing K=64 metadata contract and make zero-point handling explicit; if that cannot pass compile/correctness locally, retain v200 rather than submitting a known-invalid Kaggle candidate.

## v204 audit correction: WMMA versus lower-level MMA

The vendored `cutlass/arch/wmma_sm75.h` contains only an `int4b_t * int4b_t` WMMA specialization (plus binary WMMA), not an `int4b_t * uint4b_t` specialization. The lower-level `cutlass/arch/mma_sm75.h` does contain the exact `mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.u4.s32` wrapper with `FragmentA=Array<int4b_t,8>`, `FragmentB=Array<uint4b_t,8>`, and `OpMultiplyAddSaturate`. Thus the inherited `WmmaCore` alias using `OpClassWmmaTensorOp` and `uint4b_t` is not a valid instantiation in this vendored tree. A correct native route would need a custom `OpClassTensorOp` core/policy around the lower-level `Mma`, plus exact activation decomposition (`a = low_u4 + 16*high_s4`) and a second MMA accumulation, or a separate direct warp-level kernel. This invalidates the inherited snippet as written and prevents a safe Kaggle submission until the lower-level route is implemented and verified.

## Native lower-level route selected for the next candidate

The only viable native SM75 route in the vendored source is a pair of lower-level TensorOp cores with K=64: `u4*u4` for `low = a mod 16` and `s4*u4` for `high = floor(a/16)`, using `a = low + 16*high` exactly for every signed int8 value. The packed activation buffers use the same two-nibbles-per-byte ABI as the packed checkpoint weights. The existing metadata pipeline can be reused because both cores keep Shape K=64, but the legacy zero mutation must be disabled and the exact `z * a` term must be applied in a separate correction epilogue. This is now a v205-style design despite being developed from the v204 audit; it will not be called v204 or submitted until it passes source/compile/correctness checks.

## Next-loop hypothesis after v204 no-go

The upstream `mix_mma_config.h` contains many large-M geometries, including `64x128` and `128x128`, while the accepted v200 SM75 port keeps a single `32x128x64 / 32x32x64 / 8x8x16 / stage=2` core. v199 showed that a `64x128` large-M core can improve rows=128 relative to its narrow baseline (`0.2671x` versus `0.3072x` was not an improvement in that measured comparison; it was still a no-go), but its rows=16 timing-integrity failed because the version mixed geometry changes with an older dispatch. The next falsifiable hypothesis is narrower: retain v200’s auxiliary-stream overlap and narrow core for rows<64, add only a supported wide `64x128x64 / 32x32x16` stage-2 adapter for rows>=64, and measure whether the combination improves rows=128 without inheriting v199’s small-M timing failure. Prediction: rows=128 GEMM/E2E may improve while rows=16 remains identical to v200; if timing-integrity or correctness fails, reject the combination and retain v200.

## cuBLAS fallback research for the next loop

NVIDIA’s current cuBLAS documentation confirms that `cublasGemmEx()` exposes signed INT8 input types (`CUDA_R_8I`) with 32-bit integer accumulation (`CUDA_R_32I`), so a cuBLAS-backed integer control is technically available on the T4 environment. However, this is only an API capability, not evidence of a performance win over the benchmark’s dense FP16 cuBLAS path. The next safe experiment, if pursued, must measure the exact signed-INT8 GEMM plus the existing per-group scale/zero correction and indexed scatter under the same gate; it must not claim that cuBLAS INT8 is faster solely because the API exists. The historical NVIDIA developer discussions also warn that Turing INT8 GEMM can be slower than FP16 for some shapes, so this is a control measurement rather than an assumed solution.

## 128x128 geometry audit

The vendored row-major-A/column-major-B `DefaultMmaCore` specialization computes warp counts from divisible `Shape::kM / WarpShape::kM` and `Shape::kN / WarpShape::kN`, uses 128-bit sub-byte-aware accesses, and has no explicit 128x128 exclusion. This makes a stage-2 `Shape<128,128,64>` candidate with a legal warp shape a compile-time possibility, unlike v193’s unsupported five-stage/16x8x32 combination. It is not evidence of a performance win: v143/v145/v199 already show that 64x128 geometry alone does not solve the staged pipeline cost. Any 128x128 test must be isolated, source-contract checked, and reverted if it fails compile, correctness, timing-integrity, or mixed-prefill performance.

## SM75 instruction-set constraint confirmed

The vendored `mma_sm75.h` exposes `m8n8k16` signed/unsigned 8-bit combinations and `m8n8k32` 4-bit combinations (`s4*s4`, `u4*s4`, `s4*u4`, `u4*u4`), but no direct signed-8-by-unsigned-4 instruction. Therefore exact int8 activation with 4-bit weights cannot be represented as one native 4-bit MMA. The v204 two-product decomposition was mathematically valid but necessarily doubled the native INT4 GEMM work and added packing/correction overhead; its T4 result confirms that this route is not a performance solution for the present Qwen-shaped prefill.

## Next candidate after v205: tiled mixed DP4A hypothesis

The evidence now separates the bottleneck: v200’s pure INT4/INT8 rows=128 E2E speedups are about `0.615x` and `0.710x`, while the mixed path is `0.307x` because two full integer partitions contend for the same T4 resources even when launched on auxiliary streams. The historical v160 DP4A kernel was rejected because one thread independently reread the activation for every output and fell to `0.0112x` at rows=128. A distinct, falsifiable design is a single mixed integer kernel with an 8-row x 32-channel CTA: load each 8x128 activation group once into shared memory, let each thread compute one output using 32 `__dp4a` operations per group, select INT4-expanded or INT8 weights by channel, and apply the existing per-group activation/weight scales and output indices exactly. This preserves int8 activation and arithmetic, removes the second partition launch and duplicated global activation reads, and does not reduce computations. It is not assumed to be faster than Tensor Cores; it will be kept only if correctness, timing-integrity, and mixed prefill all pass. The rejected v160 one-thread kernel must not be reused unchanged.

## v206 implementation review note

The submitted v206 kernel removes repeated global activation reads, but review of its memory mapping shows each output thread still loads its own weight row directly inside the DP4A loop. Across a warp, those loads are separated by the full `width` stride, so they are not coalesced; this is a concrete follow-up risk distinct from v160’s one-thread output geometry. If v206 fails, the next candidate should stage a 32x128 weight tile cooperatively with contiguous K-major global loads into shared memory, then reuse that tile across the 8 row warps. No source change is made during the active v206 measurement.
