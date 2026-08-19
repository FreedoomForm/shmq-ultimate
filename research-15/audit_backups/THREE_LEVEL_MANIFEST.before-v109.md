# Three-level vLLM integration manifest

- vLLM version: `0.9.0`
- Required vLLM commit: `5fbbfe9a4c13094ad72ed3d6b4ef208a7ddc0fd7`
- MixLLM branch: `mixllm-3level`
- Quantization method: `mixllm_three_level`
- Checkpoint layout owner: `mixllm.nn.modules.three_level_linear.ThreeLevelLinear`
- Config/parser owner: `mixllm.vllm_three_level.VLLMThreeLevelConfig`

The old workspace patch `shmq-ultimate/vllm_patch/0005-shmq-3level-t4-support.patch`
is not part of this integration. It has no pinned vLLM base and makes unvalidated
single-kernel and SM75 claims.

Patch `0002` provides the vLLM `QuantizationConfig`, packed checkpoint loader,

tensor-parallel remapping, and a correctness-first PyTorch linear adapter. It is

intentionally gated to `backend=reference` until two prerequisites pass:

1. the reference Kaggle notebook passes on T4;
2. a registered native operator passes the same packed ABI tests.

Until then, vLLM 0.9.0 may use only the explicit `reference` backend on T4. The
existing upstream MixLLM backend remains available on SM80+.



End-to-end serving and CUDA graph behavior require a CUDA environment with the

fork's `mixllm` extension installed; they are not validated by CPU contract tests.