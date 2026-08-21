# v277 deep research: stay under Kaggle's kernel-source limit

## Authoritative failure

The v276 submission request was rejected by Kaggle before execution with HTTP 400: `The kernel source must be less than 1 megabytes in size.` The rebuilt notebook's embedded source cell was 1,023,195 bytes by local measurement, which is below 1 MiB but above the service's decimal 1,000,000-byte source limit. This is a packaging-size failure; no CUDA, correctness, timing, or performance result exists for v276.

## Comparison with the original implementation

The original MixLLM uses the same large CUTLASS/vendor payload and source organization, but it is deployed as a filesystem repository rather than serialized into one Kaggle notebook code cell. SHMQ's gate intentionally embeds the complete source and a hash manifest to make the run reproducible. The new v275 native adapter added one custom header dependency, exposing a pre-existing size margin that was already only about 23 KB under one MiB.

The safe repair is to preserve the exact plain-text source files and manifest hashes while serializing the notebook's `sources` dictionary as zlib-compressed, base64-encoded text. The execution cell will decode the dictionary before writing files and will verify the same per-file hashes and aggregate digest. This reduces the embedded cell substantially because the CUDA/Python source contains repeated syntax; even the already-base64 vendor archive shrinks enough to cross the service limit. No kernel source, arithmetic, ABI, benchmark, model, quality, or gate logic changes.

## v277 scope and acceptance

Only `scripts/build_mixllm_3level_kaggle.py` and the worklog/research note change. The v275 source and the completed v276 manifest dependency repair remain intact. Local acceptance requires the complete test suite, Python compilation, notebook builder validation, an embedded decode/hash self-check, and `git diff --check`; only then is one Kaggle submission permitted.
