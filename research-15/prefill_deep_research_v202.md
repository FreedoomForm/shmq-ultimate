# v202 deep research: CUTLASS-pipelined packed INT4 iterator

## Comparison target
v201 proved that a standalone direct packed-INT4 WMMA kernel is functionally correct on T4 but disastrously slow at rows=128 (`0.0646x`). v200's auxiliary-stream overlap remains safe (`0.3072x`). The next hypothesis must preserve CUTLASS's staged global-to-shared pipeline rather than add a second unpipelined WMMA loop.

## Primary source findings

The upstream MixLLM multistage launcher constructs separate `MmaCore_INT4` and `MmaCore_INT8` types, launches them on two auxiliary streams, and passes `matrix_B_interleaved` directly to the INT4 testbed. Its INT4 core uses `ElementB_INT4 = cutlass::uint4b_t` and the custom `OpMultiplyAddMixedAndShuffledInputUpcast` path. The input/weight storage is an upstream-specific interleaved layout prepared by `interleave_uint4_for_cutlass`; the current checkpoint's simple packed layout cannot be relabeled as upstream storage without changing the serialization contract.

The checked-in vendored SM75 ISA definitions do expose native 4-bit MMA instructions, but only homogeneous/unsigned-signed combinations with `mma.sync.aligned.m8n8k32` (`s4*s4`, `u4*s4`, `s4*u4`, `u4*u4`). The current activation is signed INT8, so a direct SM75 `INT8 x INT4` mixed instruction is not available in the vendored SM75 path. The upstream `16x8x32` mixed type is an SM80-oriented abstraction and was already rejected when wired through the SM75 `DefaultMmaCore` in v193.

The current SM75 `Runner<Core,2>` uses a supported `32x128x64 / 32x32x64 / 8x8x16` core and a synchronous two-stage `MQMmaPipelinedSm75`. Its `copy_tiles_and_advance` contract calls `IteratorB.get()` for every global-to-shared access, copies the returned `AccessType` into shared memory, advances the iterator, and then overlaps those copies with warp MMA. The custom `mq_fine_grained_scale_zero_iterator.h` only handles scale/zero metadata; it does not unpack B weights.

## v202 architectural seam

Implement a `PackedInt4Iterator` that has the same thread-map, predicate, tile-offset, iteration-index, `valid`, `clear_mask`, `set_iteration_index`, `operator++`, and `AccessType` contract as the existing `PredicatedTileAccessIterator`, but whose `get()` decodes one vector of packed nibbles into an `Array<int8_t,...>` register fragment and subtracts the corresponding per-group zero. The existing SM75 `Mma` remains INT8 x INT8 and the existing two-stage pipeline remains unchanged. A `PackedRunner<Core,2>` can reuse the current `kernel` and shared-memory iterators, with only the global B iterator and input pointer/Params changed. This is a single-variable test: packed INT4 conversion moves into CUTLASS's already-overlapped global-load stage; no standalone expansion buffer or unpipelined WMMA path is retained.

This is safer than attempting an unsupported mixed SM75 instruction and safer than v201's standalone kernel. It may still lose to expansion because `get()` performs scalar nibble decode per vector, but it can test whether keeping those operations inside the staged pipeline hides enough latency to recover the large-M path. The fallback remains the v200 expanded INT4 CUTLASS path if the packed iterator is unsupported or fails correctness.

## Sources

[1]: https://github.com/microsoft/MixLLM "Microsoft MixLLM upstream source"
[2]: https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/functionality.html "NVIDIA CUTLASS functionality documentation"
[3]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/ "NVIDIA CUDA C++ programming guide"
[4]: https://github.com/NVIDIA/cutlass "NVIDIA CUTLASS source repository"
