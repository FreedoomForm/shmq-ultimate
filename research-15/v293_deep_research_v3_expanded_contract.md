# v293 Deep Research: Native v3 Expanded-INT4 Contract

## Observed failure

The first operator-only Colab benchmark reached the real SM75 benchmark after 91 tests and failed on the first mixed case with `RuntimeError: expanded_int4 must have shape [n4, K] for prefill`.

## Comparison with original and current SHMQ paths

The current SHMQ Python dispatcher intentionally keeps signed INT4 expansion lazy. For rows >= 32 it prepares the persistent packed/interleaved INT4 tensor and passes an empty `weight_int8[:0]` placeholder in the ABI slot named `expanded_int4`. The CUDA overlap launcher then selects `run_cutlass_packed_int4_partition` whenever the interleaved tensor is defined and non-empty. In that branch, `expanded_int4` is not dereferenced; it is only included in the shared core’s common validation list.

The shared CUDA core nevertheless unconditionally requires `[n4, K]` for `expanded_int4` whenever `rows > 1`, even when the packed interleaved path is selected. This contradicts the intended lazy-expansion organization and is why the contract-only Colab suite missed the bug: no prior test exercised the real mixed rows >= 32 native-v3 benchmark.

## Safe fix

Relax only the shared shape check so that rows > 1 may use an empty expanded buffer when a non-empty packed interleaved INT4 tensor is present; otherwise retain the existing `[n4, K]` requirement for the legacy expanded path. Keep dtype, contiguity, device, packed-weight, metadata, partition, and all arithmetic checks unchanged. Add a source contract test pinning this condition. This restores the memory-preserving native path without allocating or populating an unused signed INT4 expansion.
