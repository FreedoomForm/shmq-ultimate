# v280 deep research: restore the original B-fragment shuffle

## T4 evidence

Kaggle version 275 compiled the v279 widened-load and dequantizer changes. The native large-M path then failed correctness while the small-M fallback remained numerically bounded. The errors grew dramatically with native rows: smoke mixed rows 32 and 128 were approximately 212 and 266 maximum absolute error; Qwen mixed rows 128 was approximately 819; pure INT4 rows 128 was approximately 877. This pattern isolates the defect to the native warp fragment mapping rather than compilation, zero-point metadata shape, timing-integrity, or the FP16 fallback.

## Primary-source comparison

The original MixLLM CUTLASS warp operator defines a B-side `FragmentShuffler` for the 4-bit-load to 8-bit-MMA upcast. It performs cross-lane `__shfl_up_sync` and `__shfl_down_sync`, then uses `__byte_perm` with parity-dependent selectors. Its `transform()` constructs the shuffler, applies it to the loaded B fragment, and only then converts `uint4b_t` values to the MMA register type. The local historical mixed-input adapter had the shuffle call commented out, but the original implementation shows that the loaded shared-memory fragment is not itself in the final `mma.sync` register layout.

The v279 packed adapter also converted B directly. Although it correctly widened the iterator to avoid the SM75 uint4 LDSM policy failure and correctly split the transformed fragment for two internal k16 MMAs, it omitted the required warp-level B permutation. The resulting native-only corruption is consistent with this mismatch.

## v280 repair

Restore the original B shuffler at the packed adapter’s transform seam. Because v280 uses two internal k16 instructions, instantiate the original shuffler over `2 * MmaIterations::kColumn` B groups so both widened halves receive the same proven cross-lane/byte permutation. Convert the shuffled fragment afterward. A is not changed: its element width is already the MMA width and the original A shuffler specialization is identity for this path. The dequantizer’s v279 zero broadcast remains unchanged.

No model, quantization boundary, weights, output mapping, launch geometry, benchmark settings, or quality checks are changed. v280 is accepted only if it compiles and all correctness, timing-integrity, end-to-end, performance, and quality gates pass.
