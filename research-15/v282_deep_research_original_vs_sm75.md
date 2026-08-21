# v282 deep research: unified three-level prefill scheduler

## Original-versus-current finding

The original MixLLM launcher separates INT4 and INT8 arithmetic adapters but centralizes their stream/event topology, tile configuration selection, and output tensor. SHMQ v271 already centralizes the integer overlap in `begin_integer_prefill_overlap`/`finish_integer_prefill_overlap`, but the dispatcher still launches the FP16 partition outside that module. That leaves the three-level operation split across two host paths and makes future native INT4/FP16 leaf substitution harder to verify.

The safe first seam is therefore a host-side deep module named `run_unified_prefill`. It owns the logical prefill plan, the integer fork and join, the FP16 launch, and the final completion ordering. It must call the existing INT4, INT8, and FP16 implementations unchanged. This is an architecture refactor, not a performance claim and not a change to arithmetic, memory allocation policy, benchmark conditions, or quality gates.

## Why this seam is safe

The existing v271 integer helper already records activation/output tensors on auxiliary streams, waits for the caller fork event, records completion events, and joins them on the caller stream. The existing FP16 helper writes through the same output tensor and is already launched on the caller stream. Moving those calls into one function preserves ordering exactly while reducing duplicated dispatcher logic. The precision-specific leaves remain independent, so this change cannot silently convert INT4 to INT8 or alter FP16 arithmetic.

## v282 contract

- The validated 4/8/16 partition remains complete and disjoint.
- Rows=1 decode remains unchanged.
- Rows>=32 uses the same native INT4/staged INT8/caller FP16 choices already present in the branch.
- Rows<32 fallback remains unchanged.
- All streams and events join before return.
- The tuning ABI is bumped to 282 because the host seam and native leaf selection are part of the candidate identity.
- Kaggle T4 validation is required; no performance claim is made from local tests.
