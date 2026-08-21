# v271 deep research: boundary-corrected reciprocal quantization

v270 exposed an exact correctness hazard: replacing `values[item] / scale` with `values[item] * (1 / scale)` changed the final rounded int8 code for the random `(rows=7, width=384)` case. The failure happened because a reciprocal product can cross a half-integer rounding boundary by a small floating-point error, even when its numerical error is otherwise negligible.

The original MixLLM quantizer uses a reciprocal multiply followed by inline `cvt.rni.sat.s8.f16`, but SHMQ's established reference contract is based on float division plus `__float2int_rn`. The safe adaptation is therefore not to loosen the contract. v271 will use the reciprocal product as the fast path and detect values close to either adjacent half-integer boundary; only those rare boundary cases will recompute the original float division before rounding. Values away from a boundary retain the faster multiply result. The scale calculation, warp reduction, clamp, output packing, stream, and ABI remain unchanged.

The boundary correction is conservative: it can preserve the exact old integer code for sensitive values while eliminating most divisions. The local and Kaggle embedded backend tests remain mandatory; if any mismatch survives, the optimization is rejected rather than relaxing correctness.
