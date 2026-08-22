# v291 repair research

The v290 run reached Python dispatch but failed on the pure-FP16 benchmark because v3 metadata preparation was attempted for a layer with no INT4 or INT8 partition. The original MixLLM launcher only enters its integer GEMM path when an integer partition exists; the v3 packed handoff must preserve that contract.

The repair is to guard v3 cache preparation with both `rows >= 32` and `(indices_4.numel() or indices_8.numel())`. Pure FP16 continues directly through the existing v2 path, while mixed/pure-integer cases may use v3. No threshold, benchmark, model, arithmetic, or quality criterion changes.
