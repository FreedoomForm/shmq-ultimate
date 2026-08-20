# v251 deep research: isolate the load-hoisted pair after M=64,N=64 regression

## Results that must be disentangled

Kaggle server version 243 (v248) compiled and passed correctness/timing, but the M=64,N=64 tuner candidate regressed Qwen mixed rows=128 to 5.46x slower than dense FP16 versus the v241 baseline around 3.24x. It is rejected.

Server version 244 (v249) still contained the M=64,N=64 candidate, so its 3.38x result cannot be attributed solely to the native load-hoist. Server version 245 (v250) contained both M=64,N=64 and the mixed native pair dispatch and measured 11.78x slower at rows=128. This is also confounded and not evidence against the load-hoist alone.

## Primary-source guardrail

NVIDIA's PTX ISA states that dense integer `.m16n8k32` MMA requires `sm_80` or higher; SM75 supports the smaller `.m8n8k16` INT8 and `.m8n8k32` sub-byte forms used by this project. Therefore no SM75 custom INT8 m16n8k32 replacement is admissible.

## Candidate

Remove the rejected M=64,N=64 alias, tuner option, and ABI bump, restoring the v241 legal N=128/N=64 CUTLASS set. Keep the v249 load-hoisted native pair implementation and its mixed `n4 > 0` dispatch. This produces a controlled measurement of exactly one change versus v241: native mixed INT4 branch load/barrier reuse. If it still fails or regresses, revert the pair dispatch and retain only v241; if it improves, keep only after all four gates and timing integrity pass.
