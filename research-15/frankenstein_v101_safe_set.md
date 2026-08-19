# Frankenstein v101 safe set

The compatibility audit keeps only the cumulative changes already embodied in v51. The retained set is v41 native vLLM dispatch integration, v45/v46 decode mapping and warp configuration as finalized by the v51 lineage, and v51 initialization of the expanded INT4 cache for rows=1 decode. These changes passed native correctness and did not regress the required decode and prefill gates in the validated v51 result.

Excluded changes: v42/v43 correctness or illegal-address failures; v44/v47-v50 regressions below the then-current baseline; v52 aborted; v53-v55 below v51; v56-v66 correctness, compile, or prefill failures; v67/v69/v71/v72 audit-only; v68/v70/v74-v80 prefill regressions; v81-v85 research-only or rejected before implementation; v86-v95 correctness/compile/prefill regressions; v100-v100.9 unvalidated or prefill-failing packed-INT4, diagnostic, or inline-PTX experiments.

Conclusion: no independently validated post-v51 optimization is safe to add. The honest Frankenstein is therefore the cumulative v51-safe set, rebuilt as a separate notebook for a reproducibility test.
