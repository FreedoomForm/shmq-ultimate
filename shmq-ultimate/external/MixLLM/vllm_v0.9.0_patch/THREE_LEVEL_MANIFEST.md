# Three-level vLLM integration manifest

- vLLM version: `0.9.0`
- Required vLLM commit: `5fbbfe9a4c13094ad72ed3d6b4ef208a7ddc0fd7`
- MixLLM branch: `mixllm-3level`
- Quantization method: `mixllm_three_level`
- Checkpoint layout owner: `mixllm.nn.modules.three_level_linear.ThreeLevelLinear`
- Config/parser owner: `mixllm.vllm_three_level.VLLMThreeLevelConfig`

## Native capability and serving contract

Patch `0002` is the pinned vLLM integration. Its native execution path is SM75-specific: `get_min_capability()` is 75, `backend=auto` or `backend=sm75` is accepted, and `apply()` checks for CUDA capability `(7, 5)` before loading `mixllm.sm75_backend`. Unsupported devices must fail explicitly rather than silently selecting a native kernel compiled for another architecture.

The vLLM adapter reshapes arbitrary leading dimensions to two dimensions for the native operator and restores the original leading dimensions. Bias is added after the native operator. The standalone `ThreeLevelLinear` module has the same native-on-SM75 and reference-fallback contract.

## Checkpoint and tensor-parallel layout

Weights are loaded in the custom three-level direct output-channel layout owned by `ThreeLevelLinear`: each precision partition stores rows in output-channel order with group-wise scales and zero points. The adapter does not apply the original MixLLM SM80/CUTLASS interleave a second time. Tensor-parallel loading filters global output-channel indices to each shard, restores local indices, and concatenates partition pieces in local output order. Row-parallel input shards must remain divisible by the group size; this is an explicit validation error.

## Kernel and benchmark scope

The current SM75 implementation has separate activation-quantization, optional INT4 expansion, and three-level GEMM launches. It must not be described as a single monolithic activation-quantization-plus-GEMM launch until that fusion is implemented and validated. The current four-way native gate measures the standalone SM75 operator on Tesla T4; GEMM timing uses precomputed activation INT8/scales and warmed INT4 expansion, while end-to-end timing includes activation quantization and reuses the warmed expansion cache. These boundaries are intentional and must remain identical across comparisons.

Full-model Qwen/Qwen2.5-0.5B quality and serving throughput are separate claims from the native microbenchmark. They remain `not_run` unless the final gate explicitly loads that exact model and records the quality and throughput artifacts. No performance result may claim full-model quality or vLLM serving validation without those artifacts.

The old workspace patch `shmq-ultimate/vllm_patch/0005-shmq-3level-t4-support.patch` is not part of this integration. Patch `0002` applies cleanly to the pinned vLLM commit and has a local Python compilation/contract-smoke path; full CUDA execution still requires the final T4 validation.