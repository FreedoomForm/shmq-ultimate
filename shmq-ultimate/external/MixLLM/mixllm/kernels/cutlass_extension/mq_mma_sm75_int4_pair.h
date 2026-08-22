// SPDX-License-Identifier: MIT
#pragma once

// v231 compile probe for the native SM75 INT4 pair seam.
// This header intentionally has no runtime dispatch yet. It verifies that the
// vendored SM75 CUTLASS headers expose the two instruction forms required by
// the eventual fused pair: u4*u4 for the low activation nibble and s4*u4 for
// the signed high activation nibble.

#include "cutlass/arch/mma_sm75.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
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

// Compile-only native core contract.  This uses CUTLASS's genuine SM75
// 4-bit multiplicand layouts and m8n8k32 operator; it is intentionally not
// wired to the SHMQ dispatcher until a matching packed global/shared loader
// and affine zero-point path are independently proven.
using NativeCore = cutlass::gemm::threadblock::DefaultMmaCore<
    cutlass::gemm::GemmShape<32, 128, 64>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    InstructionShape,
    cutlass::uint4b_t, LayoutA,
    cutlass::uint4b_t, LayoutB,
    ElementC, LayoutC,
    cutlass::arch::OpClassTensorOp, 2, MathOperator>;

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
static_assert(NativeCore::WarpCount::kCount == 8,
              "native SM75 INT4 core must use eight warps");
static_assert(NativeCore::MmaPolicy::Operator::Shape::kK == 32,
              "native SM75 INT4 core must use m8n8k32 MMA");
static_assert(sizeof(typename NativeCore::MmaPolicy::Operator::FragmentA) == 4,
              "native uint4 fragment must remain four packed bytes");

struct InstructionPair {
  LowMma low;
  HighMma high;
};

}  // namespace int4_pair_probe
}  // namespace shmq_cutlass_sm75
