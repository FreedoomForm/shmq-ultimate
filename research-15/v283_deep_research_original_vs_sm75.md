# v283 deep research: native INT4 shape-family selection

The original MixLLM launcher does not force one tile on every M/N shape. It selects among persistent shape families and autotuned configurations. The current SHMQ INT8 path already has `N128`, `N64`, `M128N64`, and `M64N64` legal SM75 families, but the native packed INT4 adapter currently exposes only `M64N64`. That mismatch means the new INT4 leaf cannot reuse the same large-M shape strategy as INT8 and may repeatedly process rows>=128 with a smaller tile.

v283 adds only the missing packed-INT4 `M128N64` alias and a shape-aware native INT4 dispatch helper. The existing INT8 selector, stream topology, arithmetic, packed layout, metadata, output scatter, model, benchmark, and quality gates remain unchanged. Native INT4 remains selected only when the persistent interleaved tensor is present; otherwise v271’s expanded INT8 fallback is retained.

The M128/N64 alias is a compile-time candidate, not an acceptance claim. It must pass SM75 CUDA compilation, native correctness, timing-integrity, memory telemetry, and all unchanged T4 gates before it can replace the M64/N64 leaf. The ABI/tuning identifier is bumped to 283 so no previous tuning selection is reused.
