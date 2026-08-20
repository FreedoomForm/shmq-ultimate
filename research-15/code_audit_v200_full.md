# SHMQ-Ultimate v200 Full Code Audit

## Scope and freeze

The audited base is the accepted v200 source tree. v211, v212, v213, and v214 experiments remain historical evidence only; none is active in the production path. Kaggle execution is forbidden until every locally testable contract, correctness path, cache lifecycle, and quality comparison described here passes.

## Confirmed differences from original MixLLM

| Area | Original MixLLM | v200/SHMQ finding | Required repair or proof |
|---|---|---|---|
| Geometry selection | Persistent shape/config tables and first-use CUDA-event autotuning across many M/N/K/stage choices | Fixed local stage-2 geometry with manual threshold | Add a persistent offline/first-use legal-SM75 geometry registry, but never benchmark inside the measured hot call; validate each candidate against reference and fall back to v200 geometry on failure. |
| INT4 representation | Native/interleaved INT4 path in the upstream architecture | Prefill expands packed INT4 into signed INT8 `[n4,K]` | Move expansion to explicit prewarm/initialization, retain a signature/device cache, expose its bytes, and prove no repeated expansion or unsafe graph-capture allocation. |
| Metadata | Kernel-family-specific layout and fused metadata flow | Transpose/cache infrastructure exists, but v200 large mixed adapter bypasses cached-v3 path | Make metadata preparation an explicit prewarm operation and pass cached layouts through the fastest validated ABI, or prove with local tests why v2 fallback is required. |
| Epilogue/output | Upstream supports layout-specific epilogues | Custom SM75 runner scatters through indices and performs scale/zero work | Keep exact scatter semantics; add isolated per-partition epilogue contracts and measure kernel-only versus epilogue cost. |
| Stream lifetime | Persistent streams/events initialized once | Preserved, but seven `recordStream` calls occur per branch | Keep allocator lifetime safety; centralize/guard recording so persistent immutable weights/metadata are not repeatedly recorded, while dynamic input/output tensors remain correctly recorded. |
| Validation | Minimal hot launcher checks after initialization | Python and C++ device/dtype/contiguous/shape checks each call | Split initialization-time invariants from per-call dynamic checks; retain explicit failure for violated contracts and add tests proving invalid state is rejected before launch. |
| Quality | Upstream model path and layer/reference semantics | Operator gates pass, but exact Qwen full-model quality is unavailable in Kaggle | Add local deterministic full-model-equivalent quality tests using the same packed/reference model state; never claim unavailable external model quality as passed. |
| vLLM | Pinned patch and matching runtime contract | Static patch contracts exist; actual vLLM apply can be unavailable | Separate patch-contract, apply-execution, and runtime-execution gates; local tests must reject silent unavailable status. |

## Confirmed non-root causes

Activation quantization is measured around 0.026–0.032 ms and is not the dominant rows=128 failure. Partition validation is signature-cached after the first call and is a correctness guard, not a credible explanation for a 0.49–0.69 ms integer prefill kernel. Removing final CUDA event joins or input dependencies would be a correctness bug, not an optimization.

## Required local proof before Kaggle

All edited Python must compile. Source contracts must pass. CPU/reference model tests must verify exact packed/dequantized semantics, partition coverage, signed zero correction, and deterministic output. Cache tests must cover invalidation on device/state/version changes, graph-capture refusal before prewarm, and no repeated allocation after prewarm. Static CUDA checks must verify no unsupported SM75 instruction or architecture path is introduced. A local quality harness must compare the packed reference and native-facing operator contract on deterministic inputs, with bounded error thresholds unchanged from the existing gate.

## Acceptance rule

No candidate proceeds to Kaggle unless all local proof obligations pass and the source manifest is fresh. On Kaggle, every required gate and timing-integrity must pass. Any performance regression versus v200 or any correctness/quality failure rejects the candidate and restores v200.
