# v302 Deep-Research Iteration

## Scope
This iteration compares the original Microsoft MixLLM/CUTLASS organization with SHMQ’s SM75 packed adapter before any code change. v302 is the current candidate at commit `5f0f3222b020f0045e8e0e0948d394a972162ecd`; no new repair has been made in this iteration.

## Authoritative external findings

1. NVIDIA’s CUTLASS GEMM API describes the hierarchy as CTA/tile mainloop, warp-level K groups, MMA-instruction iterations, and iterator advancement. The documented warp-level usage loads A and B from shared-memory iterators, advances both iterators once per K tile, then invokes the warp MMA operator. This makes iterator shape, operator shape, and pipeline K progression one coupled contract, not independent knobs.
2. NVIDIA’s current `default_mma_core_sm75.h` defines SM75 shared-memory layouts and iterators from the declared threadblock/warp/instruction shapes, and passes `WarpCount::kK` into `DefaultMmaTensorOp`. Its SM75 implementation uses the legal instruction shape supplied by the caller rather than inventing an incompatible K policy.
3. NVIDIA’s current `mma_tensor_op.h` exposes the warp shape as `Shape_`, derives iterator tile shapes from the underlying architecture MMA shape, computes `MmaIterations` from the warp shape divided by the instruction shape, and selects vertical visitation on pre-SM80 architectures. Its `transform()` and `operator()` preserve the iterator fragment contract and visit order together.

## Original MixLLM findings

The upstream `mix_mma_multistage.cuh` retains persistent INT4 and INT8 streams, fork/join CUDA events, a best-configuration cache, and separate row-major/column-major launcher families. It instantiates `ThreadblockShape(..., 64)`, `WarpShape(..., 64)`, and `InstructionShape<16,8,32>` for the SM80 mixed-input operator. The upstream path uses CUTLASS’s complete `DefaultMmaCore` and `Testbed`, not a hand-written replacement of iterator progression. The upstream launcher also keeps dispatch/config selection separate from the arithmetic and uses cached autotuning for the shape/config family.

## Current working hypothesis

The strongest remaining correctness seam is still the synthetic SM75 packed adapter: it advertises a widened K=32 policy to emulate an SM80 mixed instruction while internally loading a single-group SM75-compatible fragment. v302 bypasses k-group setters for that adapter, but this has not yet been validated on T4. Before changing code, the next audit will inspect whether the adapter’s synthetic policy also causes the pipeline’s warp-K loop, iterator increment, and fragment conversion to disagree. If so, the safe repair candidate is to preserve original MixLLM’s launcher/stream/metadata/scatter behavior while replacing only the packed adapter’s internal policy with a legal SM75 k16 contract, backed by source tests and a fresh T4 run.

## Sources

- NVIDIA CUTLASS GEMM API: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html
- NVIDIA CUTLASS SM75 default MMA core: https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/threadblock/default_mma_core_sm75.h
- NVIDIA CUTLASS warp tensor-op operator: https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/warp/mma_tensor_op.h
- SHMQ branch copy of original MixLLM launcher: https://github.com/FreedoomForm/shmq-ultimate/blob/unified-three-level-sm75/shmq-ultimate/external/MixLLM/mixllm/kernels/mix_mma_multistage.cuh

## Iterator-level audit evidence

The vendored CUTLASS `mma_tensor_op_tile_iterator.h` makes the coupling explicit. Its fragment length is `Shape::kContiguous * InstructionShape::kStrided / 32`; `Policy::kGroupsPerTile` is `Shape::kStrided / InstructionShape::kStrided`; `operator++()` advances by one instruction-shaped tile and only uses the group counter when `PartitionsK > 1`; and the normal congruous/crosswise SM75 iterator implementations expose `set_kgroup_index()` as a no-op. The packed adapter nevertheless changes the iterator’s instruction-shape argument to a synthetic 32-wide K load while the underlying MMA remains k16, then performs two k16 MMAs per m/n slot.

For the packed runner, the threadblock shape is 64x64x64, the warp shape is 32x32x64, the legal instruction shape is 8x8x16, A is row-major int8, B is column-major uint4, and the custom operator is selected only through the `OpMultiplyAddSm75PackedInputUpcast` specialization. The selected SM75 DefaultMmaCore derives shared-memory layouts and thread maps from the uint4 element width and still passes the declared k16 instruction shape into `DefaultMmaTensorOp`; the custom adapter then overrides only its own iterator shapes and policy.

This reveals a more precise concern than the old k-group-setter hypothesis: the v302 trait may be harmless for the vendored default iterators because their setters are already no-ops, while the synthetic widened iterator still changes `Fragment` size, `LdsmShape`, pointer count, and `operator++()` byte/tile progression. The pipeline also uses the synthetic policy in `kWarpGemmIterations`, per-group copy partitioning, warp offsets, stage wraparound, and the initial-stage advancement. Any next repair must therefore be checked against all of these sites together; changing only the setter guard would not restore the original CUTLASS contract.

## Candidate direction, not yet implemented

The next candidate should not blindly set `kWarpGemmIterations=1`, because the threadblock K tile is 64 and the pipeline’s staged shared-memory choreography currently relies on two subtiles per 64-wide K tile. Instead, the safe seam is to make the packed adapter’s logical warp-level K group a legal k16-based pair while retaining a packed 32-value load only if its iterator can prove that each increment consumes exactly 32 logical K values and the fragment halves map to consecutive k16 MMA operands. A source-level contract test should assert the intended relationship between the packed iterator’s load width, fragment size, two native MMAs, and the pipeline’s two sub-iterations before any T4 run.

## Additional source references

- Vendored CUTLASS warp iterator: `shmq-ultimate/external/MixLLM/mixllm/kernels/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator.h`, especially lines 183-185, 210-212, 335-349, 1353-1355, 1507-1521, and 1640-1650.
- SHMQ packed runner: `shmq-ultimate/external/MixLLM/mixllm/kernels/sm75_cutlass_testbed.h`, especially lines 205-256.
- SHMQ base/pipeline: `shmq-ultimate/external/MixLLM/mixllm/kernels/cutlass_extension/mq_mma_base.h` lines 118-145 and `mq_mma_pipelined_sm75.h` lines 645-809, 785-809, and 871-953.
- Original MixLLM mixed operator: `shmq-ultimate/external/MixLLM/mixllm/kernels/cutlass_extension/mq_mma_mixed_input_tensor_op.h`, lines 165-212 and 245-305.
