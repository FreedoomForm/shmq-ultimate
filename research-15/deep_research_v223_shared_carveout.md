# Deep research for v223: SM75 shared-memory carveout experiment

## Original-versus-SMQ discrepancy

The upstream MixLLM path is a staged CUTLASS implementation whose central performance assumption is that the threadblock's operand staging remains resident in shared memory while Tensor Core work proceeds. The clean SHMQ SM75 runner also uses a dynamic shared-memory `SharedStorage` object for its two-stage `MQMmaPipelinedSm75` pipeline, but `Runner::run` requests the 100% shared-memory carveout only when `shared_bytes >= 48 KiB`. For the current `32x128x64` / two-stage core, the conditional may leave the T4 at its default shared-memory/L1 partition even though the kernel is explicitly shared-memory staged.

The upstream source does not use an SM75 carveout branch because its original target is SM80 and its launcher delegates to its CUTLASS testbed/configuration. This is a portability seam rather than a new arithmetic design: the SM75 runner owns the dynamic shared-memory kernel attribute and can request the same shared-memory preference for all instances.

## v223 hypothesis and safety boundary

Request `cudaFuncAttributePreferredSharedMemoryCarveout=100` for every SM75 staged runner invocation, while preserving the existing max-dynamic-shared-memory request for kernels at or above 48 KiB. This changes only the hardware cache/shared-memory partition preference; it does not change tensor layouts, instruction selection, numerical arithmetic, synchronization, model inputs, quality thresholds, or benchmark settings. The candidate is admissible only if native correctness and timing-integrity pass and the measured mixed-prefill performance improves over the clean v200 baseline. If it regresses or is neutral, revert immediately.

The stage-5 experiment is not retained: the vendored SM75 `DefaultMmaCore` supports only `NumStages=2`, as proven by the v222 compile failure.
