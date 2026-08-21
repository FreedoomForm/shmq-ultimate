# v267 deep research: make the large-M geometry decision explicit

## Evidence from original MixLLM

The upstream launcher uses persistent INT4 and INT8 streams and selects a CUTLASS family from a persisted `(stage, config_id)` table keyed by M/N/K and the precision split. For the row-major mixed path, its safe fallback is `gemm<5, 64, 128, 64, 32>`, while the launcher also exposes multiple tuned families. The upstream kernel's tile is therefore selected by measured shape, not hard-coded globally.

The upstream mixed INT4 branch consumes a direct `uint4b_t` interleaved operand through `OpMultiplyAddMixedAndShuffledInputUpcast`; SHMQ cannot reuse that operator on SM75 because the vendored SM75 headers expose legal `m8n8k32` native forms but not the upstream mixed operator. SHMQ's current large-M integer branch consequently uses the expanded signed-INT8 cache and `OpMultiplyAddSaturate`. This is the principal remaining arithmetic/dataflow divergence.

## Evidence from SHMQ and Kaggle

SHMQ already contains legal `N=128`, `N=64`, `M=128,N=64`, and `M=64,N=64` SM75 families. The tuner tries all four on first use and stores a shape-keyed result. v263 and v266 did not expose the selected family in their logs; v266's allocator-record change did not improve performance and is rejected. The next candidate must therefore avoid another bookkeeping or epilogue micro-change.

The safe research seam is to make the upstream large-M fallback explicit for the Qwen prefill shape: use the legal `M=128,N=64` family for `rows==128` and `channels>=64`, while leaving the existing tuner for all other shapes and preserving the v263 cached-metadata dispatch, streams, arithmetic, indices, and output ABI. This is not a benchmark-setting change; it is a kernel-family dispatch experiment using an already validated family. It may be rejected if the tuner was already selecting it, but it will establish whether the current shape-keyed selection is being polluted or selecting a smaller-M family.

No direct packed-INT4 operator is introduced in v267 because the source audit shows it would require a new SM75 iterator and fragment contract, not a safe one-line port. That larger change remains a later research branch only if explicit M128N64 dispatch fails.
