# v198 deep research: decouple core-layout stage from pipeline stage

v197 attempted to set the vendored SM75 `DefaultMmaCore` stage parameter to 3 and failed during Kaggle compilation. The actual header contains the row-major/column-major int8 Tensor Core specialization only for `arch::OpClassTensorOp, 2, Operator_`; there is no generic stage-3 specialization, so `DefaultMmaCore<...,3>` does not define the expected `SmemLayoutA`/`SmemLayoutB` aliases.

The safer next experiment is to retain the proven v188 `DefaultMmaCore<...,2>` type for all layout, iterator, warp-policy, and shared-memory layout aliases, while parameterizing only `MQMmaPipelinedSm75` and `Runner` with `Stages=3`. This tests whether the custom synchronous pipeline can use three shared-memory stages with the same supported SM75 core policy, without inventing an unsupported CUTLASS core specialization or changing the instruction geometry. It is a controlled compile/runtime experiment, not an assertion that three stages are automatically safe.

The candidate remains valid only if it compiles on T4, preserves native correctness, timing integrity, memory bounds, and all existing gates. If it fails, the experiment is rejected and the supported stage-2 core remains the baseline.
