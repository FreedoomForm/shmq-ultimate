// SPDX-License-Identifier: MIT
#pragma once

// v231 compile probe for the native SM75 INT4 pair seam.
// This header intentionally has no runtime dispatch yet. It verifies that the
// vendored SM75 CUTLASS headers expose the two instruction forms required by
// the eventual fused pair: u4*u4 for the low activation nibble and s4*u4 for
// the signed high activation nibble.

#include "cutlass/arch/mma_sm75.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

namespace shmq_cutlass_sm75 {
namespace int4_pair_probe {

using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using ElementC = int;
using MathOperator = cutlass::arch::OpMultiplyAddSaturate;

using LowMma = cutlass::arch::Mma<
    InstructionShape, 32,
    cutlass::uint4b_t, LayoutA,
    cutlass::uint4b_t, LayoutB,
    ElementC, LayoutC,
    MathOperator>;

using HighMma = cutlass::arch::Mma<
    InstructionShape, 32,
    cutlass::int4b_t, LayoutA,
    cutlass::uint4b_t, LayoutB,
    ElementC, LayoutC,
    MathOperator>;

static_assert(LowMma::Shape::kM == 8 && LowMma::Shape::kN == 8 &&
                  LowMma::Shape::kK == 32,
              "SM75 low-nibble MMA must be m8n8k32");
static_assert(HighMma::Shape::kM == 8 && HighMma::Shape::kN == 8 &&
                  HighMma::Shape::kK == 32,
              "SM75 high-nibble MMA must be m8n8k32");
static_assert(LowMma::FragmentA::kElements == 8 &&
                  LowMma::FragmentB::kElements == 8 &&
                  LowMma::FragmentC::kElements == 2,
              "unexpected SM75 low-nibble fragment ABI");
static_assert(HighMma::FragmentA::kElements == 8 &&
                  HighMma::FragmentB::kElements == 8 &&
                  HighMma::FragmentC::kElements == 2,
              "unexpected SM75 high-nibble fragment ABI");

struct InstructionPair {
  LowMma low;
  HighMma high;
};

}  // namespace int4_pair_probe
}  // namespace shmq_cutlass_sm75
