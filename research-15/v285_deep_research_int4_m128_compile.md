# v285 deep research: packed INT4 M128/N64 compile failure

## Scope

This iteration compares the preserved original MixLLM layout/launcher organization with SHMQ v284 before any implementation change. The goal is to diagnose the first real Colab Tesla T4 failure, not to infer performance from a partial run.

## Upstream comparison

The preserved original `LinearMixLLM` stores INT4 weights in the CUTLASS interleaved packed layout during module construction (`interleave_uint4_for_cutlass`) and calls one upstream quantized GEMM operator. Its layout permutation is the same two-stage permutation implemented by SHMQ `prepare_sm75_prefill_int4()`. The original path does not expose a separate SHMQ signed `[n4, K]` expansion for its hot prefill path.

SHMQ v284 correctly preserves that packed layout for the native cached-v3 path and defers the signed expansion to the legacy v2 fallback. The current runtime dispatch selects `PackedInt4RunnerM128N64` for rows >= 96 and `PackedInt4RunnerM64N64` for smaller rows. Both aliases, however, instantiate CUTLASS `DefaultMmaCore` with `ElementB = cutlass::uint4b_t` and `InstructionShape = GemmShape<8, 8, 16>`.

## Direct T4 evidence

The first Colab T4 run reached `Tesla T4`, compute capability `(7, 5)`, and cloned the exact v284 commit `ccd74338f0bda0c759a523036cc0cc5d17acb103`. After installing the missing Ninja dependency, the second complete compile attempt failed in CUTLASS before any SHMQ operator test ran:

```text
pitch_linear_thread_map.h(298): static assertion failed with "Number of iterations must be non-zero"
... PitchLinearWarpRakedThreadMap<... PitchLinearShape<64, 64>, 256, PitchLinearShape<2, 16>, 32>
... RegularTileIterator<MatrixShape<64, 64>, cutlass::uint4b_t, ...>
... DefaultMmaCore<GemmShape<128, 64, 64>, ... uint4b_t ... OpMultiplyAddSm75PackedInputUpcast>
```

The failing instantiation is the new `CorePackedInt4M128N64` chain. The error happens while `DefaultMmaCore` constructs its generic shared-memory B iterator; it occurs before the custom `MQMmaPackedInputTensorOpSm75` adapter can provide its widened logical k32 warp iterator. The custom adapter therefore fixes only the warp-level MMA/fragment side, not the threadblock-level `DefaultMmaCore` iterator construction.

The relevant vendored formula derives the B thread arrangement from the physical `uint4b_t` width and uses `kAccessSizeInBits / sizeof_bits<ElementB>::value = 128 / 4 = 32`. For the `PitchLinearShape<64, 64>` iterator, CUTLASS computes a zero iteration count and deliberately rejects the type. Changing M from 64 to 128 does not make this generic B iterator legal; the M128 alias merely caused the first reported instantiation to surface.

## Decision for v285

Do not modify the arithmetic, benchmark, model, quality thresholds, or packed layout. The safe next seam is a custom SM75 packed-INT4 threadblock core/iterator that makes the widened k32 logical access visible to the threadblock iterator while retaining the original packed data layout and the existing two internal legal `m8n8k16` instructions. Before attempting that larger redesign, remove the unproven M128 family from the compile path only as a diagnostic fallback and determine whether the existing M64 family has the same generic-iterator failure. If M64 fails identically, reject both aliases and keep v284 as the last T4-pending candidate until a bespoke core is locally audited and then compiled on Colab.

## Acceptance policy

The Colab run is **not accepted**: the T4 hardware check passed, but the embedded SM75 test gate failed at CUDA compilation. v284 remains unaccepted and v271 remains the last fully Kaggle-confirmed safe baseline. No speedup claim is made.
