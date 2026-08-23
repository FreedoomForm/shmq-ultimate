# v307 next-seam research: expanded INT4 GEMM dominates, native work must use manual warp iterators

## Evidence from v299

The recovered Qwen telemetry shows activation quantization is not the main bottleneck: its p50 is approximately 0.026–0.064 ms, while the real-model expanded INT4 GEMM reaches 0.240 ms at the smaller tested shape and 2.325 ms at the larger pure-INT4 shape. The mixed Qwen GEMM reaches 0.485 ms and 0.469 ms for its larger shapes, at 3.56x and 2.72x the dense-FP16 reference respectively. The expanded INT4 cache is 8.60 MB for the mixed Qwen layer and 12.85 MB for the pure-INT4 layer. Therefore, removing Python bookkeeping or activation quantization alone cannot achieve the target; the dominant seam is the INT4 weight data path and its GEMM arithmetic.

The real-model telemetry also shows the small-shape behavior differs materially: mixed QKV at its smallest shape is faster than dense in GEMM terms, while larger shapes become much slower. This is consistent with a kernel/dataflow problem that appears once the expanded weight matrix is large, not a universal stream/event overhead problem.

## Comparison with original MixLLM

Microsoft’s original launcher directly consumes an interleaved packed INT4 matrix and uses a mixed-input CUTLASS core. SHMQ’s v299 control instead expands packed INT4 to signed INT8 once and sends it through the legal SM75 INT8 runner. The original stream/event topology is already reproduced: persistent INT4/INT8 auxiliary streams, a caller-stream fork, and caller-stream joins. The original broad autotune table and stage-5/11 families cannot simply be copied: v288 stage-5 was rejected by correctness/timing gates, v299’s N256 family failed the SM75 thread-map compile contract, and v300 attribute caching regressed E2E gates.

## Native implementation seam

The vendored CUTLASS tree contains explicit warp-level `MmaTensorOpMultiplicandTileIterator` specializations and the project already has `MQMmaPackedInputTensorOpSm75`. That adapter demonstrates the correct manual architecture: load warp fragments with the SM75 iterator, transform them into instruction fragments, and issue legal m8n8k16 operations. It currently expands packed U4 fragments into INT8 registers and has already failed correctness when used as the production packed route, so it cannot be enabled unchanged.

The next defensible native design is a research-only, self-contained threadblock loader built around the explicit warp-level iterator contract, with two legal m8n8k32 instructions (`U4*U4` for the low nibble and `S4*U4` for the signed high nibble), explicit fragment mapping, exact scale/zero correction, tail handling, and the existing indexed scatter epilogue. No generic `DefaultMmaCore` tile substitution or permutation-only patch is justified. Until that complete contract is proven, v299 remains the production baseline.

## Decision

No performance code was changed in v307. The evidence rules out quantization/bookkeeping as the primary explanation for the large-M slowdown and rules out another generic CUTLASS tile guess. The next implementation should begin as an isolated manual warp-iterator/native decomposition probe, then progress to a full block only after lane mapping and tails are independently checked.
