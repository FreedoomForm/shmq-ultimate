# v290 repair research

The v289 Kaggle run failed before execution because the Python v3 call correctly supplied the original interleaved packed tensor and cached metadata but the shared C++ core still unconditionally required an expanded `[n4, K]` tensor for every prefill call. This is an ABI-validation bug, not evidence against the packed layout.

The original MixLLM launcher has no expanded INT4 argument on its packed path: it passes `matrix_B_interleaved` plus transposed scale/zero metadata. SHMQ’s v3 signature already mirrors that contract, and `begin_integer_prefill_overlap()` already selects `run_cutlass_packed_int4_partition()` when the interleaved tensor is nonempty. The correct repair is to derive `has_packed_int4` from the v3 interleaved tensor and exempt the expanded-shape check only for that branch. The v2 fallback must continue to require the expanded tensor.

No arithmetic, layout, dispatch threshold, benchmark, or quality criterion changes are needed. After local validation, Kaggle must be rerun to test actual packed correctness.
