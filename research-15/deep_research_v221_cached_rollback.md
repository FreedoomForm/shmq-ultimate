# Deep research for v221: cached-v2 rollback

## Differential comparison

The upstream MixLLM implementation uses a fixed K=64 staged CUTLASS geometry for both INT4 and INT8 (`ThreadblockShape ... 64`, `WarpShape ... 64`) and supplies metadata in the layouts asserted by `mix_mma_multistage.cuh`: activation scales are `[groups, padded_rows]`, weight scales and zeros are `[groups, padded_channels]` at the public launcher seam. The original also uses two long-lived auxiliary streams and fork/completion events, with no additional cached-v2 adapter.

The accepted v200 SHMQ source matches the relevant production structure: it dispatches `Int8Runner` (`GemmShape<32,128,64>`) for every integer partition, transposes metadata inside the C++ overlap helper on the caller stream, records the fork only after those transposes, launches INT4 and INT8 on independent auxiliary streams, and joins them on the caller stream. Its T4 reference for rows=128 mixed is E2E speedup `0.307233x`, GEMM speedup `0.310787x`, max error `0.102539`, and timing-integrity ratio `0.988564` (pass).

v220 differs from v200 in the remaining production path only by the v215 cached-v2 adapter: Python constructs or reuses transposed `[groups, channels]` metadata and calls a private cached operator. The v220 T4 result restores native correctness (`max_abs_error=0.102539` mixed rows=128, `0.000193` pure INT4, `0.000198` pure INT8), proving the cached metadata layout is numerically compatible. It does not preserve the performance contract: mixed rows=128 E2E speedup falls to `0.268557x`, GEMM to `0.189709x`, and timing-integrity fails at `1.415619`. The cached path reports `prefill_metadata_bytes=251776`, whereas the v200 path reports zero for this measured core.

The result is a production performance failure, not a correctness failure. The cached adapter can allocate/cache metadata and changes the benchmark's end-to-end execution surface; even if the C++ transpose is avoided on later calls, the observed T4 timing is worse and unstable under the required integrity test. Therefore it cannot be retained under the user's keep-only-if-all-gates-pass rule.

## v221 action

Restore the complete v200 production implementation for the SM75 kernel and backend: remove cached-v2 operator/dispatch, restore the original v200 K=64 pipeline and runner aliases, and restore the v200 source contracts. Keep only unrelated v218 import-safe local-test repairs and the research/worklog evidence. Do not claim v200 reaches the 2.6x target; it is only the clean correctness/timing baseline for another original-first optimization attempt.
