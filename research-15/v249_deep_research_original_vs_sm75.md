# v249 deep research: hoist native pair tile loads outside row-subtile loop

## Evidence from current v245/v241 kernel

The native pair kernel iterates `row_tile=0..3`, then `chunk=0..3`. Inside that nested loop it reloads the entire 32-row packed activation tile and the entire 32-channel packed weight tile from global memory for every row tile, followed by a CTA barrier. The same `a_low_packed`, `a_high_packed`, and `b_packed` shared arrays are therefore filled four times for each K chunk even though all four row subtiles consume the same K chunk. The B tile is also identical across all row subtiles.

The kernel then loads only the selected 8-row A fragment and the warp's 8-column B fragment and performs the low/high MMA pair. The shared arrays remain valid while row subtiles are processed, so the packing phase can be moved outside the row-tile loop without changing the data or arithmetic. A barrier is still required after the one packing phase and after all row-subtile MMAs before the next chunk overwrites shared memory.

## Original MixLLM comparison

Upstream staged CUTLASS makes a threadblock A/B tile available to multiple warp-level MMA operations before advancing the pipeline. The current native pair path violates that reuse principle by repeatedly reloading the same CTA tile for each row subtile. This is a concrete dataflow discrepancy, unlike the rejected width-only changes.

## Candidate and safety boundary

For each group and 32-element K chunk: pack A and B exactly once, synchronize, run all four row-subtile low/high MMAs using the resident shared tile, then synchronize before the next chunk. Keep the exact low/high decomposition, zero correction, scales, output indices, row sums, grid, thread count, and ABI. This changes only load/barrier placement and is independently source-testable. It should first remain behind the existing native pair dispatch seam; if local contracts and T4 correctness pass, its performance decides whether the pair path can safely replace the mixed INT4 branch. If any correctness or timing gate fails, revert.
