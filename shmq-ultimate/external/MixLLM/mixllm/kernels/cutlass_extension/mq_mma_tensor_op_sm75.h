#pragma once

// v275: SM75 has no single mixed s8*u4 mma.sync form.  This specialization
// keeps the original MixLLM packed/interleaved uint4 B storage and uses the
// legal SM75 s8*s8 Tensor Core instruction after the existing mixed-input
// fragment conversion.  It removes the global signed-INT8 expansion without
// changing the quantized arithmetic contract.

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "mq_mma_mixed_input_tensor_op.h"

namespace cutlass {
namespace arch {

struct OpMultiplyAddSm75PackedInputUpcast {};

}  // namespace arch

namespace gemm {
namespace warp {

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

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::Mma<
          GemmShape<8, 8, 16>,
          32,
          MmaElementA,
          cutlass::layout::RowMajor,
          MmaElementB,
          cutlass::layout::ColumnMajor,
          MmaElementC,
          cutlass::layout::RowMajor,
          arch::OpMultiplyAddSaturate>,
      cutlass::MatrixShape<1, 1>>;

  using Type = cutlass::gemm::warp::MQMmaMixedInputTensorOp<
      WarpShape_, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
      Policy, PartitionsK, AccumulatorsInRowMajor>;
};

}  // namespace warp
}  // namespace gemm
}  // namespace cutlass
