# v291 deep research: avoid duplicate partition-validation scan

## Original MixLLM comparison

The original MixLLM linear wrapper passes its prepared partition indices directly into the native GEMM entrypoint. The native entrypoint computes output width and launches; it does not repeat a Python-side complete-partition validation on every forward. Partition ownership is established while the module/checkpoint is prepared.

SHMQ v289 already caches partition validation in `_validate_partition()` using a signature of output size and the three index tensors' identity, version, count, and device. However, `three_level_linear()` calls `_validate_partition()` and then calls `three_level_linear_prequantized()`, which calls `_validate_partition()` again. The second call returns early because the signature is cached, but it still rebuilds the signature tuple and performs the associated Python object/version reads on every forward.

## Safe v291 seam

Add an internal `partition_validated` flag to `three_level_linear_prequantized()`. The top-level `three_level_linear()` forwards `partition_validated=True` after its existing validation. Direct callers that omit the flag retain the existing validation behavior. This removes only a duplicate cached-signature scan on the production path; it does not weaken validation because the top-level call still validates before forwarding and the optional path is private to the backend module.

Keep packed-cache forwarding from v289, all fallback validation, device/dtype checks, activation quantization, output allocation, native v3 selection, stream/event ordering, precision arithmetic, quality, and benchmark settings unchanged. Add a source contract proving the flag is passed only after top-level validation. Validate locally and on Colab T4 before retention. No performance claim is allowed without Kaggle.
