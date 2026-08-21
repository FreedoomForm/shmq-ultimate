# v286 deep research: isolate and remove the unproven M128 packed-INT4 alias

## Evidence from original MixLLM and SHMQ

The original MixLLM uses multiple shape families, but its SM80 implementation does not prove that the same CUTLASS `DefaultMmaCore` template is legal for SM75 `uint4b_t`. SHMQ’s custom SM75 adapter correctly widens the warp-level logical INT4 load to k32 and performs two legal k16 MMAs, but v283/v284 still instantiate the generic CUTLASS threadblock B iterator through `DefaultMmaCore`.

The v283/v284 code added `CorePackedInt4M128N64` and selected it for rows >= 96. Both the M64/N64 and M128/N64 aliases use the same packed `uint4b_t` B operand and k16 instruction shape. The M128 alias was added for upstream shape-family parity, not from a successful SM75 compile measurement.

## Direct Colab T4 isolation

A temporary M64-only clone was created from exact v284 commit `ccd74338f0bda0c759a523036cc0cc5d17acb103`. Only the M128 alias and its rows>=96 dispatch branch were removed; no production source, arithmetic, layout, test inputs, or thresholds were changed. On Tesla T4 / SM75 with PyTorch 2.11.0+cu128 and Ninja installed, the production CUDA extension compiled successfully:

```text
[1/2] nvcc ... -gencode=arch=compute_75,code=sm_75 ...
[2/2] c++ ... -shared ... mixllm_sm75_backend_5d22ce3fe94d59bf.so
Ran 6 tests in 83.846s
OK
COLAB_V285_M64_PROBE_PASS
```

The six native tests covered CUDA graph capture, INT4 tile-width boundaries, mixed and empty partitions, activation quantizer equivalence, non-default stream dependencies, and randomized row/width determinism. Therefore the existing M64/N64 path is compile- and correctness-valid on this T4 environment. The prior v284 failure was specifically introduced by the M128/N64 alias instantiation: `PitchLinearWarpRakedThreadMap<..., PitchLinearShape<64,64>, ..., ElementsPerAccess=32>` reached CUTLASS’s non-zero-iteration assertion through the M128 core.

## Safe decision

Remove the M128/N64 packed-INT4 alias and the rows>=96 dispatch branch from production v286. Route all native packed-INT4 prefill rows through the measured M64/N64 runner. Preserve v284’s lazy expansion semantics, persistent interleaved packed layout, v282 unified scheduler, and all three precision leaves. This is a compile-safety rollback of an unproven shape only; it does not reduce model precision, quality thresholds, benchmark settings, or arithmetic correctness.

Update source contracts to reflect the measured safe family. Keep a future custom M128 core as a separate research direction requiring a bespoke threadblock iterator, not another `DefaultMmaCore<uint4b_t>` alias. v286 may be accepted only after the complete embedded test suite is rerun on Colab T4; it is not accepted based solely on the six-test isolation probe.
