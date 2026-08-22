# v303 next-seam note: correct the isolated probe’s expected shape

The v302 T4 payload established that the native U4/U4 plus S4/U4 instruction decomposition is numerically correct for both tested activation cases: the first warp produced `[-480, -480]` on all lanes and the second warp produced `[8128, 8128]` on all lanes. The only failure was in the notebook assertion: the kernel returns 64 rows by 2 accumulator values, while `repeat(32, 1)` constructed a 32-by-4 expected tensor.

v303 changes only the expected tensor construction to `repeat_interleave(32, dim=0)`, yielding the exact 64-by-2 contract. It does not alter CUDA source, production dispatch, model computation, partitioning, benchmark settings, or acceptance gates. The corrected run is needed to promote the already observed raw probe values to a clean PASS artifact.

No production packed implementation is justified yet. The native instruction arithmetic is only one layer of the required contract; shared-memory layout, iterator mapping, fragment transforms, K traversal, zero correction, and large-M output mapping remain unproven.
