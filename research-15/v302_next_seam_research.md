# v302 next-seam research: native SM75 INT4 decomposition must start as a probe

CUTLASS exposes legal Turing warp instructions `mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32` and `s4.u4.s32`, with eight logical 4-bit values per operand fragment and two INT32 accumulator values per lane. The packed `Array<T,N,false>` stores subbyte elements in 8-, 16-, or 32-bit storage items; for U4/S4 a vector of 8 logical values occupies 32 bits, and raw data is exposed through the packed storage representation. [1] [2]

For an INT8 activation `a` and unsigned 4-bit weight code `w`, an exact decomposition is `a = low4(a) + 16 * high4(a)`, where `low4` is U4 and `high4` is the signed upper nibble. Therefore the code-domain product can be formed as `MMA_U4U4(low4, w) + 16*MMA_S4U4(high4, w)`, followed by the existing affine zero-point correction and scales. This preserves computation only if the lane fragment packing, shared-memory multiplicand layouts, K traversal, signedness, saturation behavior, and accumulator mapping are all self-consistent.

The current SHMQ packed adapter is not that design: it uses a synthetic k32 policy around the legal k16 int8 operator and has already corrupted large-M results under live v294. The next acceptable packed attempt must therefore begin with an isolated T4 probe using the native CUTLASS U4/U4 and S4/U4 operators, explicit lane-fragment initialization, and comparison against a CPU matrix product for both positive and negative activation values. Only after that probe passes for multiple fragments and K tiles may a native shared-memory iterator/core be added behind a narrow guarded dispatcher. No production path is changed by this research step.

## References

[1]: https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/mma_sm75.h
[2]: https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/array_subbyte.h
