# v250 deep research: measure the load-hoisted pair in mixed prefill

## v249 interpretation

Kaggle server version 244 (v249) compiled and passed all correctness and timing-integrity checks, but mixed end-to-end gates still failed. Qwen mixed rows=128 was 3.3786x slower than dense FP16, close to the v241 baseline around 3.24x. This result does **not** measure the native pair repair in the mixed scenario: v249 preserved `use_fused_int4=false` for mixed rows>=32, so the Qwen mixed benchmark continued to use the staged CUTLASS overlap. The native load-hoist was exercised only by the pure-INT4 candidate/probes.

## Candidate

Enable `n4 > 0` for the mixed rows>=32 overlap call, using the v249 load-hoisted native pair kernel for INT4 while INT8 remains on the staged CUTLASS auxiliary stream and FP16 remains on the caller stream. Compared with v242, this candidate changes only the native pair dataflow: A/B packing happens once per K chunk rather than once per four row subtiles. Compared with v241, it changes only the INT4 mixed branch; all stream/event ordering, metadata, arithmetic, output ABI, model, benchmark settings, and quality gates remain unchanged.

The expected falsifiable outcome is a large reduction from v242's 16.62x mixed rows=128 ratio and v244's 13.49x ratio. If correctness, timing integrity, or any gate fails, revert and keep v241. If it is still slower than v241, reject it even if it improves the earlier pair versions.
