#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op.h"
#include "cutlass_extension/mq_mma_pipelined_sm75.h"
#include "cutlass_extension/mq_mma_sm75_int4_pair.h"
#include "cutlass_extension/mq_mma_tensor_op_sm75.h"

namespace shmq_cutlass_sm75 {

using ElementA = int8_t;
using LayoutA = cutlass::layout::RowMajor;
using ElementB = int8_t;
using LayoutB = cutlass::layout::ColumnMajor;
using ElementC = int;
// SHMQ’s native ABI is a row-major [rows, output_width] destination.
// The original MixLLM row-major runner uses the same accumulator layout.
using LayoutC = cutlass::layout::RowMajor;

struct Problem {
  cutlass::gemm::GemmCoord size;
  int partial_n;
  Problem(int m, int n, int k) : size({m, n, k}), partial_n(n) {}
};

template <typename Mma, typename SharedStorage, typename ElementB_>
__global__ void kernel(
    cutlass::gemm::GemmCoord problem_size,
    typename Mma::IteratorA::Params params_A, ElementA* ptr_A,
    typename Mma::IteratorB::Params params_B, ElementB_* ptr_B,
    typename Mma::IteratorScale::Params params_scale,
    typename Mma::ElementScale const* ptr_scale,
    typename Mma::IteratorScaleAct::Params params_scale_act,
    typename Mma::ElementScale const* ptr_scale_act,
    typename Mma::IteratorZero::Params params_zero,
    typename Mma::ElementZero const* ptr_zero,
    __half* ptr_C, typename LayoutC::Stride::Index ldc,
    ElementC const* indices) {
  extern __shared__ int shared_base[];
  SharedStorage* shared = reinterpret_cast<SharedStorage*>(shared_base);

  cutlass::gemm::GemmCoord tb_tile{int(blockIdx.x), int(blockIdx.y), 0};
  cutlass::MatrixCoord offset_A{tb_tile.m() * Mma::Shape::kM, 0};
  cutlass::MatrixCoord offset_B{0, tb_tile.n() * Mma::Shape::kN};
  int warp_id = int(threadIdx.y);
  int tb_thread_id = warp_id * int(blockDim.x) + int(threadIdx.x);

  typename Mma::IteratorA iterator_A(
      params_A, ptr_A, {problem_size.m(), problem_size.k()}, tb_thread_id, offset_A);
  typename Mma::IteratorB iterator_B(
      params_B, ptr_B, {problem_size.k(), problem_size.n()}, tb_thread_id, offset_B);

  int groups = problem_size.k() / 128;
  cutlass::MatrixCoord offset_scale{0, tb_tile.n() * Mma::Shape::kN};
  cutlass::MatrixCoord offset_scale_act{0, tb_tile.m() * Mma::Shape::kM};
  typename Mma::IteratorScale iterator_scale(
      params_scale, const_cast<typename Mma::ElementScale*>(ptr_scale),
      {groups, problem_size.n()}, tb_thread_id, offset_scale, 128);
  typename Mma::IteratorScaleAct iterator_scale_act(
      params_scale_act, const_cast<typename Mma::ElementScale*>(ptr_scale_act),
      {groups, problem_size.m()}, tb_thread_id, offset_scale_act, 128);
  typename Mma::IteratorZero iterator_zero(
      params_zero, const_cast<typename Mma::ElementZero*>(ptr_zero),
      {groups, problem_size.n()}, tb_thread_id, offset_scale, 128);

  Mma mma(shared->main_loop, tb_thread_id, warp_id, int(threadIdx.x));
  typename Mma::FragmentC accum;
  accum.clear();
  int gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;
  mma(gemm_k_iterations, accum, iterator_A, iterator_B, iterator_scale,
      iterator_scale_act, iterator_zero, accum);

  using Operator = typename Mma::Operator;
  using MmaIterations = typename Operator::MmaIterations;
  using InstructionShape = typename Operator::InstructionShape;
  constexpr int kElementsPerAccess = InstructionShape::kN / 4;
  constexpr int kRowsPerTile = 8;
  constexpr int kAccumulatorRows = InstructionShape::kM / kRowsPerTile;
  auto const* converted = reinterpret_cast<typename Mma::AccumFragmentConvert const*>(accum.data());

  int warp_m = warp_id % Mma::WarpCount::kM;
  int warp_n = warp_id / Mma::WarpCount::kM;
  int row_tile = tb_tile.m() * Mma::Shape::kM + warp_m * Operator::Shape::kM;
  int channel_tile = tb_tile.n() * Mma::Shape::kN + warp_n * Operator::Shape::kN;
  int quad = threadIdx.x >> 2;
  int lane_in_quad = threadIdx.x & 3;

  using IndexFragment = cutlass::Array<int,
      MmaIterations::kColumn * kElementsPerAccess>;
  IndexFragment index_fragment;
  for (int mma_n = 0; mma_n < MmaIterations::kColumn; ++mma_n) {
    for (int col = 0; col < kElementsPerAccess; ++col) {
      int local_channel = lane_in_quad * kElementsPerAccess +
                          mma_n * InstructionShape::kN *
                              Operator::IteratorC::OpDelta::kColumn + col;
      int partition_channel = channel_tile + local_channel;
      int fragment_index = mma_n * kElementsPerAccess + col;
      index_fragment[fragment_index] =
          partition_channel < problem_size.n() ? indices[partition_channel] : -1;
    }
  }

  for (int mma_n = 0; mma_n < MmaIterations::kColumn; ++mma_n) {
    for (int mma_m = 0; mma_m < MmaIterations::kRow; ++mma_m) {
      int start = kAccumulatorRows * kElementsPerAccess *
                  (mma_n * MmaIterations::kRow + mma_m);
      for (int row = 0; row < kAccumulatorRows; ++row) {
        for (int col = 0; col < kElementsPerAccess; ++col) {
          int local_row = quad +
                          mma_m * InstructionShape::kM * Operator::IteratorC::OpDelta::kRow +
                          row * kRowsPerTile;
          int global_row = row_tile + local_row;
          int fragment_index = mma_n * kElementsPerAccess + col;
          int index = start + row * kElementsPerAccess + col;
          if (global_row < problem_size.m() &&
              index_fragment[fragment_index] >= 0) {
            ptr_C[global_row * ldc + index_fragment[fragment_index]] =
                __float2half_rn((*converted)[index]);
          }
        }
      }
    }
  }
}

template <typename Core, int Stages, typename ElementB_ = ElementB>
struct Runner {
  using ThreadblockShape = typename Core::Shape;
  using Element = typename Core::ElementA;
  using ElementB = ElementB_;
  using ThreadMapA = typename Core::IteratorThreadMapA;
  using ThreadMapB = typename Core::IteratorThreadMapB;
  using AccessTypeA = cutlass::Array<ElementA, ThreadMapA::kElementsPerAccess>;
  using AccessTypeB = cutlass::Array<ElementB, ThreadMapB::kElementsPerAccess>;
  using IteratorA = cutlass::transform::threadblock::PredicatedTileAccessIterator<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA, LayoutA, 1, ThreadMapA, AccessTypeA>;
  using IteratorB = cutlass::transform::threadblock::PredicatedTileAccessIterator<
      cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
      ElementB, LayoutB, 0, ThreadMapB, AccessTypeB>;
  using SmemIteratorA = cutlass::transform::threadblock::RegularTileAccessIterator<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      typename Core::ElementA, typename Core::SmemLayoutA, 0, ThreadMapA>;
  using SmemIteratorB = cutlass::transform::threadblock::RegularTileAccessIterator<
      cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
      typename Core::ElementB, typename Core::SmemLayoutB, 1, ThreadMapB>;
  using Mma = cutlass::gemm::threadblock::MQMmaPipelinedSm75<
      typename Core::Shape, IteratorA, SmemIteratorA,
      cutlass::arch::CacheOperation::Global, IteratorB, SmemIteratorB,
      cutlass::arch::CacheOperation::Global, ElementC, LayoutC,
      typename Core::MmaPolicy, Stages>;

  union SharedStorage {
    typename Mma::SharedStorage main_loop;
  };

  static int smem_size() { return int(sizeof(SharedStorage)); }

  static void run(
      int rows, int channels, int width,
      at::Tensor& matrix_A, at::Tensor& matrix_B,
      at::Tensor& matrix_scale_act, at::Tensor& matrix_scale,
      at::Tensor& matrix_zero, at::Tensor& matrix_indices,
      at::Tensor& matrix_C, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(rows, channels, width);
    typename IteratorA::Params params_A(matrix_A.stride(0));
    typename IteratorB::Params params_B(problem_size.k());
    typename Mma::IteratorScale::Params params_scale(matrix_scale.stride(0));
    typename Mma::IteratorScaleAct::Params params_scale_act(matrix_scale_act.stride(0));
    typename Mma::IteratorZero::Params params_zero(matrix_zero.stride(0));
    dim3 block(32, Core::WarpCount::kCount, 1);
    dim3 grid((rows + ThreadblockShape::kM - 1) / ThreadblockShape::kM,
              (channels + ThreadblockShape::kN - 1) / ThreadblockShape::kN);
    int shared_bytes = smem_size();
    if (shared_bytes >= (48 << 10)) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          kernel<Mma, SharedStorage, ElementB>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          kernel<Mma, SharedStorage, ElementB>,
          cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    }
    kernel<Mma, SharedStorage, ElementB><<<grid, block, shared_bytes, stream>>>(
        problem_size, params_A, matrix_A.data_ptr<int8_t>(), params_B,
        reinterpret_cast<ElementB*>(matrix_B.data_ptr()), params_scale,
        reinterpret_cast<typename Mma::ElementScale const*>(matrix_scale.data_ptr<at::Half>()),
        params_scale_act,
        reinterpret_cast<typename Mma::ElementScale const*>(matrix_scale_act.data_ptr<at::Half>()),
        params_zero,
        reinterpret_cast<typename Mma::ElementZero const*>(matrix_zero.data_ptr<uint8_t>()),
        reinterpret_cast<__half*>(matrix_C.data_ptr<at::Half>()), matrix_C.stride(0), matrix_indices.data_ptr<int32_t>());
  }
};

using Core = cutlass::gemm::threadblock::DefaultMmaCore<
    cutlass::gemm::GemmShape<32, 128, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
    cutlass::arch::OpClassTensorOp, 2, cutlass::arch::OpMultiplyAddSaturate>;



using Int8Runner = Runner<Core, 2>;

// Candidate family for channel partitions smaller than the v200 N=128 tile.
// It remains K=64 and NumStages=2, the only TensorOp family provided by the
// vendored SM75 DefaultMmaCore specializations.
using CoreN64 = cutlass::gemm::threadblock::DefaultMmaCore<
    cutlass::gemm::GemmShape<32, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
    cutlass::arch::OpClassTensorOp, 2, cutlass::arch::OpMultiplyAddSaturate>;

using Int8RunnerN64 = Runner<CoreN64, 2>;

// Upstream MixLLM also exposes the transposed large-M family. Keep the
// same legal SM75 instruction and stage-2 core while changing only M/N.
using CoreM128N64 = cutlass::gemm::threadblock::DefaultMmaCore<
    cutlass::gemm::GemmShape<128, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
    cutlass::arch::OpClassTensorOp, 2, cutlass::arch::OpMultiplyAddSaturate>;

using Int8RunnerM128N64 = Runner<CoreM128N64, 2>;

using CoreM64N64 = cutlass::gemm::threadblock::DefaultMmaCore<
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
    cutlass::arch::OpClassTensorOp, 2, cutlass::arch::OpMultiplyAddSaturate>;

using Int8RunnerM64N64 = Runner<CoreM64N64, 2>;

using CorePackedInt4M64N64 = cutlass::gemm::threadblock::DefaultMmaCore<
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    ElementA, LayoutA, cutlass::uint4b_t, LayoutB, ElementC, LayoutC,
    cutlass::arch::OpClassTensorOp, 2,
    cutlass::arch::OpMultiplyAddSm75PackedInputUpcast>;

using CorePackedInt4M128N64 = cutlass::gemm::threadblock::DefaultMmaCore<
    cutlass::gemm::GemmShape<128, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    ElementA, LayoutA, cutlass::uint4b_t, LayoutB, ElementC, LayoutC,
    cutlass::arch::OpClassTensorOp, 2,
    cutlass::arch::OpMultiplyAddSm75PackedInputUpcast>;

using PackedInt4RunnerM64N64 = Runner<CorePackedInt4M64N64, 2, cutlass::uint4b_t>;
using PackedInt4RunnerM128N64 = Runner<CorePackedInt4M128N64, 2, cutlass::uint4b_t>;

}  // namespace shmq_cutlass_sm75
