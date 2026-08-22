
// v280: SM75 packed INT4 uses a widened logical shared-memory load and the
// original MixLLM warp-level B-fragment permutation before upcast.  The
// source tensor is CUTLASS's compact uint4 array representation; the warp
// iterator therefore loads 32 logical uint4 values per iteration (a valid
// 128-bit ldmatrix access) and the warp adapter issues two legal SM75
// m8n8k16 s8*s8 instructions after conversion.

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator.h"
#include "mq_mma_mixed_input_tensor_op.h"
#include "mq_numeric_conversion.h"

namespace cutlass {
namespace arch {

struct OpMultiplyAddSm75PackedInputUpcast {};

}  // namespace arch

namespace gemm {
namespace warp {

/// Policy used by the staged SM75 pipeline.  The policy advertises a k32
/// warp-level load/compute group while its Operator is the legal k16 SM75
/// instruction used twice by the adapter below.
template <typename Operator_, typename OpDelta_>
struct MmaTensorOpPolicyK32 {
  using Operator = Operator_;
  using OpDelta = OpDelta_;
  using MmaShape = GemmShape<8, 8, 32>;
};

/// SM75 packed-INT4 warp adapter.
///
/// The generic CUTLASS iterator derives its LDSM shape from the instruction
/// shape and the element width.  A k16 instruction with uint4b_t would make
/// 16 / (128 / 4) equal zero.  This adapter presents k32 to both iterators,
/// producing 32 logical uint4 values per loaded fragment.  The converted
/// fragment is split into two k16 register groups and accumulated with two
/// native SM75 s8*s8 MMAs.
template <
    typename Shape_,
    typename ElementA_,
    typename LayoutA_,
    typename ElementB_,
    typename LayoutB_,
    typename ElementC_,
    typename LayoutC_,
    typename Policy_,
    int PartitionsK_ = 1,
    bool AccumulatorsInRowMajor_ = false,
    typename Enable = bool>
class MQMmaPackedInputTensorOpSm75 {
 public:
  using Shape = Shape_;
  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using Policy = Policy_;
  using ArchMmaOperator = typename Policy::Operator;
  using ElementAMma = typename ArchMmaOperator::ElementA;
  using ElementBMma = typename ArchMmaOperator::ElementB;
  using MmaElementC = typename ArchMmaOperator::ElementC;
  using MathOperator = typename ArchMmaOperator::Operator;
  using ArchTag = typename ArchMmaOperator::ArchTag;
  using OperatorClass = arch::OpClassTensorOp;
  using InstructionShape = typename ArchMmaOperator::Shape;

  static ComplexTransform const kTransformA = ComplexTransform::kNone;
  static ComplexTransform const kTransformB = ComplexTransform::kNone;
  static int const kThreadCount = 32;
  static int const kPartitionsK = PartitionsK_;

  // MatrixShape is converted to the pitch-linear orientation expected by the
  // row/column-major iterator specializations.  k32 is intentional here:
  // int8 A uses 32/16 = 2 contiguous LDSM groups and uint4 B uses 32/32 = 1.
  using IteratorA = MmaTensorOpMultiplicandTileIterator<
      MatrixShape<Shape::kM, Shape::kK>, Operand::kA, ElementA, LayoutA,
      // Keep the original MixLLM widened A iterator contract (16x32).
      // The legal SM75 MMA remains m8n8k16; only the iterator shape is
      // widened so its row-major crosswise lane/LDSM mapping matches the
      // original k32 adapter before the two k16 calls consume the halves.
      MatrixShape<16, 32>,
      Policy::OpDelta::kRow, kThreadCount, kPartitionsK>;
  using FragmentA = typename IteratorA::Fragment;
  using TransformedFragmentA = Array<ElementAMma, FragmentA::kElements>;
  using MmaOperandA = typename ArchMmaOperator::FragmentA;

  using IteratorB = MmaTensorOpMultiplicandTileIterator<
      MatrixShape<Shape::kK, Shape::kN>, Operand::kB, ElementB, LayoutB,
      MatrixShape<32, ArchMmaOperator::Shape::kN>,
      Policy::OpDelta::kRow, kThreadCount, kPartitionsK>;
  using FragmentB = typename IteratorB::Fragment;
  using TransformedFragmentB = Array<ElementBMma, FragmentB::kElements>;
  using MmaOperandB = typename ArchMmaOperator::FragmentB;

  using IteratorC = MmaTensorOpAccumulatorTileIterator<
      MatrixShape<Shape::kM, Shape::kN>, ElementC, LayoutC,
      typename ArchMmaOperator::Shape, typename Policy::OpDelta>;
  using FragmentC = typename IteratorC::Fragment;
  using MmaOperandC = typename ArchMmaOperator::FragmentC;

  using MmaIterations = MatrixShape<
      (Shape::kM + ArchMmaOperator::Shape::kM - 1) /
          ArchMmaOperator::Shape::kM,
      (Shape::kN + ArchMmaOperator::Shape::kN - 1) /
          ArchMmaOperator::Shape::kN>;

  static_assert(ArchMmaOperator::Shape::kK == 16,
                "SM75 packed adapter requires an internal k16 MMA");
  static_assert(FragmentA::kElements ==
                    2 * MmaIterations::kRow * MmaOperandA::kElements,
                "unexpected widened A fragment ABI");
  static_assert(FragmentB::kElements ==
                    2 * MmaIterations::kColumn * MmaOperandB::kElements,
                "unexpected widened B fragment ABI");

  ArchMmaOperator mma;

  CUTLASS_DEVICE
  MQMmaPackedInputTensorOpSm75() {}

  CUTLASS_DEVICE
  void operator()(FragmentC &D, TransformedFragmentA const &A,
                  TransformedFragmentB const &B, FragmentC const &C) const {
    D = C;

    MmaOperandA const *ptr_A = reinterpret_cast<MmaOperandA const *>(&A);
    MmaOperandB const *ptr_B = reinterpret_cast<MmaOperandB const *>(&B);
    MmaOperandC *ptr_D = reinterpret_cast<MmaOperandC *>(&D);

    constexpr int kASecondK = MmaIterations::kRow;
    // The SM75 B iterator emits two k16 register groups per N tile.  Its
    // flattened fragment is N-major: [n0.k0,n0.k1,n1.k0,n1.k1,...], not
    // [all N tiles.k0, all N tiles.k1].
    constexpr int kBGroupsPerN = 2;

    // CUTLASS uses vertical visitation for all pre-SM80 TensorOp paths.
    // Keep the two legal SM75 k16 MMAs, but follow the SM75 fragment/slot
    // order instead of the SM80 nonvertical m-outer traversal.
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < MmaIterations::kColumn; ++n) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < MmaIterations::kRow; ++m) {
        int m_serpentine =
            ((n % 2) ? (MmaIterations::kRow - 1 - m) : m);

        int d_index;
        if (AccumulatorsInRowMajor_) {
          d_index = n + m_serpentine * MmaIterations::kColumn;
        } else {
          d_index = m_serpentine + n * MmaIterations::kRow;
        }

        // The first and second calls consume the two k16 halves of the
        // widened k32 fragment and accumulate into the same C fragment.
        const int b_group = kBGroupsPerN * n;
        mma(ptr_D[d_index], ptr_A[m_serpentine], ptr_B[b_group],
            ptr_D[d_index]);
        mma(ptr_D[d_index], ptr_A[kASecondK + m_serpentine],
            ptr_B[b_group + 1], ptr_D[d_index]);
      }
    }
  }

  CUTLASS_DEVICE
  void transform(TransformedFragmentA &dst_A, TransformedFragmentB &dst_B,
                 FragmentA const &A, FragmentB const &B) const {
    // The original MixLLM path intentionally preserves the loaded fragment:
    // its persistent memory permutation already matches the iterator contract.
    // Applying FragmentShuffler here would double-transform packed B data.
    FragmentB tmp_B = B;

    // The compact uint4 Array stores two nibbles per byte.  This conversion
    // expands each loaded 32-value logical fragment to 32 signed int8 values.
    detail::FragmentConverter<ElementBMma, ElementB, FragmentB::kElements>
        convert_B;
    dst_B = convert_B(tmp_B);

    // On SM75, the vertical TensorOp iterator already produces the
    // instruction register layout expected by mma.sync; the generic CUTLASS
    // vertical transform likewise preserves A without an extra shuffle.
    FragmentA tmp_A = A;
    Array<ElementA, FragmentA::kElements / 2> const *ptr_tmp_A =
        reinterpret_cast<Array<ElementA, FragmentA::kElements / 2> const *>(&tmp_A);
    Array<ElementAMma, FragmentA::kElements / 2> *ptr_dst_A =
        reinterpret_cast<Array<ElementAMma, FragmentA::kElements / 2> *>(&dst_A);

    detail::FragmentConverter<ElementAMma, ElementA,
                              FragmentA::kElements / 2>
        convert_A;
    ptr_dst_A[0] = convert_A(ptr_tmp_A[0]);
    ptr_dst_A[1] = convert_A(ptr_tmp_A[1]);
  }
};

template <
    typename WarpShape_,
    typename ElementA,
    typename LayoutA,
    typename ElementB,
    typename LayoutB,
    typename ElementC,
    typename LayoutC,
    int PartitionsK,
    bool AccumulatorsInRowMajor>
struct DefaultMmaTensorOp<
    WarpShape_,
    GemmShape<8, 8, 16>,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    arch::OpMultiplyAddSm75PackedInputUpcast,
    PartitionsK,
    AccumulatorsInRowMajor> {
  static_assert(platform::is_same<ElementA, int8_t>::value,
                "SM75 packed INT4 runner requires signed INT8 activations");
  static_assert(platform::is_same<ElementB, uint4b_t>::value,
                "SM75 packed INT4 runner requires uint4 weights");
  static_assert(platform::is_same<ElementC, int>::value,
                "SM75 packed INT4 runner requires integer accumulation");

  using MmaElementA = int8_t;
  using MmaElementB = int8_t;
  using MmaElementC = ElementC;
  using ArchMma = cutlass::arch::Mma<
      GemmShape<8, 8, 16>, 32, MmaElementA, cutlass::layout::RowMajor,
      MmaElementB, cutlass::layout::ColumnMajor, MmaElementC,
      cutlass::layout::RowMajor, arch::OpMultiplyAddSaturate>;
  using Policy = MmaTensorOpPolicyK32<ArchMma, cutlass::MatrixShape<1, 1>>;

  using Type = MQMmaPackedInputTensorOpSm75<
      WarpShape_, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
      Policy, PartitionsK, AccumulatorsInRowMajor>;
};

}  // namespace warp
}  // namespace gemm
}  // namespace cutlass
