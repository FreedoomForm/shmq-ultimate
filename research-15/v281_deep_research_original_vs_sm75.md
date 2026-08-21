# v281 deep research: match the original uint4 shuffler’s 32-bit fragment width

## Why v281 is needed

The v280 source was locally validated and the Kaggle submission was blocked before execution by the weekly GPU quota. A static audit of the restored shuffle found a parameter mismatch that should be corrected before the next permitted T4 compile.

SM75 `m8n8k16` s8*s8 MMA declares `FragmentB = Array<int8_t, 4>`. The widened iterator produces 32 logical uint4 values. CUTLASS’s original upcast `FragmentShuffler` operates on 32-bit words, not on one four-element int8 MMA fragment at a time. Its `MmaFragment` must therefore contain eight uint4 logical values, which occupy four bytes and are safely accessed by the shuffler’s `uint32_t` loads/stores. Four such shuffler groups cover the 32-value widened fragment. After conversion, each eight-value group becomes two consecutive four-int8 SM75 B operand fragments.

## v281 change

The v280 call used eight shuffler groups of four uint4 values. That is inconsistent with the original shuffler’s 32-bit access contract and can either fail compilation or produce an invalid permutation. v281 will instantiate the same shuffler with:

```cpp
detail::FragmentShuffler<
    ElementBMma, ElementB,
    MmaIterations::kColumn,
    FragmentB::kElements,
    2 * MmaOperandB::kElements,
    Operand::kB>
```

For the current 32-value B fragment and four output-channel MMA columns, this is four groups of eight uint4 values. The resulting converted fragment remains 32 int8 values and the existing adapter continues to issue two k16 MMAs per M/N tile. No weight values, zero points, scales, activation quantization, output mapping, benchmark settings, or quality model changes.

## Validation policy

Because Kaggle quota is exhausted, v281 can receive local source/contract validation only. It must not be called accepted or measured. The next available Kaggle T4 run must compile the exact notebook and evaluate correctness before any performance interpretation.
