// SPDX-License-Identifier: MIT
#pragma once

// v231 compile probe for the native SM75 INT4 pair seam.
// This header intentionally has no runtime dispatch yet. It verifies that the
// vendored SM75 CUTLASS headers expose the two instruction forms required by
// the eventual fused pair: u4*u4 for the low activation nibble and s4*u4 for
// the signed high activation nibble.

#include "cutlass/arch/mma_sm75.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator.h"
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

// Research-only loader contract. This deliberately bypasses the generic
// threadblock DefaultMmaCore, whose derived subbyte thread map failed on T4.
// The explicit warp iterator is the same CUTLASS family used by the existing
// manual SM75 adapter and is not connected to production dispatch.
using NativeWarpCongruousTile = cutlass::layout::PitchLinearShape<32, 8>;
using NativeWarpCongruousInstruction = cutlass::layout::PitchLinearShape<32, 8>;
using NativeWarpCongruousLayout = cutlass::layout::TensorOpMultiplicandCongruous<4, 64>;
using NativeWarpCongruousU4AIterator = cutlass::gemm::warp::MmaTensorOpMultiplicandTileIterator<
    NativeWarpCongruousTile, cutlass::gemm::Operand::kA, cutlass::uint4b_t,
    NativeWarpCongruousLayout, NativeWarpCongruousInstruction, 1, 32, 1>;

// The SM75 Crosswise iterator issues LDSM x4 (16 bytes) even for a
// subbyte fragment. Capture that register payload in a 32-element fragment
// (16 bytes) before any diagnostic unpacking; the native MMA operand itself
// remains an eight-element / four-byte register ABI.
using NativeWarpATile = cutlass::MatrixShape<32, 128>;
using NativeWarpBTile = cutlass::MatrixShape<128, 32>;
using NativeWarpAInstruction = cutlass::MatrixShape<8, 32>;
using NativeWarpBInstruction = cutlass::MatrixShape<32, 8>;
using NativeWarpALayout = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<4, 64>;
using NativeWarpBLayout = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<4, 64>;
using NativeWarpU4AIterator = cutlass::gemm::warp::MmaTensorOpMultiplicandTileIterator<
    NativeWarpATile, cutlass::gemm::Operand::kA, cutlass::uint4b_t,
    NativeWarpALayout, NativeWarpAInstruction, 1, 32, 1>;
using NativeWarpS4AIterator = cutlass::gemm::warp::MmaTensorOpMultiplicandTileIterator<
    NativeWarpATile, cutlass::gemm::Operand::kA, cutlass::int4b_t,
    NativeWarpALayout, NativeWarpAInstruction, 1, 32, 1>;
using NativeWarpU4BIterator = cutlass::gemm::warp::MmaTensorOpMultiplicandTileIterator<
    NativeWarpBTile, cutlass::gemm::Operand::kB, cutlass::uint4b_t,
    NativeWarpBLayout, NativeWarpBInstruction, 1, 32, 1>;

// LDSM is a b16 transfer on SM75.  This physical-storage contract models one
// four-bit logical value in the low nibble of each 16-bit shared-memory word,
// then lets the probe explicitly repack eight loaded words into the one 32-bit
// U4/S4 MMA operand register.  It is distinct from the rejected dense-u4
// interpretation above and still uses the production Crosswise swizzles.
using NativeWarpU16ATile = cutlass::MatrixShape<8, 32>;
using NativeWarpU16BTile = cutlass::MatrixShape<32, 8>;
using NativeWarpU16AInstruction = cutlass::MatrixShape<8, 32>;
using NativeWarpU16BInstruction = cutlass::MatrixShape<32, 8>;
using NativeWarpU16ALayout = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16, 64>;
using NativeWarpU16BLayout = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<16, 64>;
using NativeWarpU16AIterator = cutlass::gemm::warp::MmaTensorOpMultiplicandTileIterator<
    NativeWarpU16ATile, cutlass::gemm::Operand::kA, uint16_t,
    NativeWarpU16ALayout, NativeWarpU16AInstruction, 1, 32, 1>;
using NativeWarpS16AIterator = cutlass::gemm::warp::MmaTensorOpMultiplicandTileIterator<
    NativeWarpU16ATile, cutlass::gemm::Operand::kA, uint16_t,
    NativeWarpU16ALayout, NativeWarpU16AInstruction, 1, 32, 1>;
using NativeWarpU16BIterator = cutlass::gemm::warp::MmaTensorOpMultiplicandTileIterator<
    NativeWarpU16BTile, cutlass::gemm::Operand::kB, uint16_t,
    NativeWarpU16BLayout, NativeWarpU16BInstruction, 1, 32, 1>;

static_assert(NativeWarpCongruousU4AIterator::Fragment::kElements > 0 &&
                  NativeWarpU4AIterator::Fragment::kElements > 0 &&
                  NativeWarpS4AIterator::Fragment::kElements > 0 &&
                  NativeWarpU4BIterator::Fragment::kElements > 0,
              "native SM75 subbyte warp iterators must expose fragments");

struct InstructionPair {
  LowMma low;
  HighMma high;
};

}  // namespace int4_pair_probe
}  // namespace shmq_cutlass_sm75
