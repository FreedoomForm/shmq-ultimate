#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cstdint>
#include <type_traits>

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

namespace shmq_cutlass_sm75 {

using ElementA = int8_t;
using LayoutA = cutlass::layout::RowMajor;
using ElementB = int8_t;
using LayoutB = cutlass::layout::ColumnMajor;
using ElementC = int;
using LayoutC = cutlass::layout::ColumnMajor;

struct Problem {
  cutlass::gemm::GemmCoord size;
  int partial_n;
  Problem(int m, int n, int k) : size({m, n, k}), partial_n(n) {}
};

// Global B iterator for the v202 experiment. It reuses CUTLASS's validated
// column-major predicates and thread map, but converts one logical int8 vector
// from the checkpoint's [N,K/2] packed-nibble storage. Conversion occurs in the
// existing MQMmaPipelinedSm75 global-to-shared stage; no standalone expansion
// tensor is allocated.
template <typename Shape_, typename ThreadMap_, typename AccessType_>
class PackedInt4Iterator {
 public:
  using Shape = Shape_;
  using Element = int8_t;
  using Layout = LayoutB;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;
  static int const kAccessesPerVector =
      ThreadMap::kElementsPerAccess / AccessType::kElements;
  using TensorCoord = typename Layout::TensorCoord;
  using Underlying = cutlass::transform::threadblock::PredicatedTileAccessIterator<
      Shape, Element, Layout, 0, ThreadMap, AccessType>;

  struct Params {
    typename Underlying::Params base;
    int width = 0;
    int groups = 0;
    const uint8_t* zero = nullptr;

    CUTLASS_HOST_DEVICE
    Params() {}

    CUTLASS_HOST_DEVICE
    Params(int logical_stride, int group_count, const uint8_t* zero_ptr)
        : base(Layout(logical_stride)),
          width(logical_stride),
          groups(group_count),
          zero(zero_ptr) {}
  };

 private:
  Underlying iterator_;
  const int8_t* packed_ = nullptr;
  int width_ = 0;
  int groups_ = 0;
  const uint8_t* zero_ = nullptr;
  mutable AccessType decoded_;

 public:
  CUTLASS_HOST_DEVICE
  PackedInt4Iterator(
      Params const& params, const int8_t* packed, TensorCoord extent,
      int thread_id, TensorCoord const& threadblock_offset)
      : iterator_(params.base, packed, extent, thread_id, threadblock_offset),
        packed_(packed), width_(params.width), groups_(params.groups),
        zero_(params.zero) {
    decoded_.clear();
  }

  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) { iterator_.set_iteration_index(index); }

  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const& tile_offset) {
    iterator_.add_tile_offset(tile_offset);
  }

  CUTLASS_DEVICE
  AccessType* get() const {
    decoded_.clear();
    if (!iterator_.valid()) {
      return &decoded_;
    }
    const int8_t* logical = reinterpret_cast<const int8_t*>(iterator_.get());
    const uintptr_t logical_address = reinterpret_cast<uintptr_t>(logical);
    const uintptr_t packed_address = reinterpret_cast<uintptr_t>(packed_);
    const int logical_offset = static_cast<int>(logical_address - packed_address);
    const int channel = logical_offset / width_;
    const int k_base = logical_offset - channel * width_;
#pragma unroll
    for (int item = 0; item < AccessType::kElements; ++item) {
      const int k = k_base + item;
      const uint8_t byte = reinterpret_cast<const uint8_t*>(packed_)[
          channel * (width_ / 2) + k / 2];
      const int code = (k & 1) ? (byte >> 4) : (byte & 0x0f);
      decoded_[item] = static_cast<int8_t>(
          code - static_cast<int>(zero_[channel * groups_ + k / 128]));
    }
    return &decoded_;
  }

  CUTLASS_HOST_DEVICE
  PackedInt4Iterator& operator++() {
    ++iterator_;
    return *this;
  }

  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) { iterator_.clear_mask(enable); }

  CUTLASS_HOST_DEVICE
  bool valid() const { return iterator_.valid(); }
};

template <typename Mma, typename SharedStorage>
__global__ void kernel(
    cutlass::gemm::GemmCoord problem_size,
    typename Mma::IteratorA::Params params_A, ElementA* ptr_A,
    typename Mma::IteratorB::Params params_B, const uint8_t* ptr_B,
    typename Mma::IteratorScale::Params params_scale,
    typename Mma::ElementScale const* ptr_scale,
    typename Mma::IteratorScaleAct::Params params_scale_act,
    typename Mma::ElementScale const* ptr_scale_act,
    typename Mma::IteratorZero::Params params_zero,
    typename Mma::ElementZero const* ptr_zero,
    float* ptr_C, typename LayoutC::Stride::Index ldc,
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
  auto* typed_ptr_B = reinterpret_cast<typename Mma::IteratorB::Element*>(
      const_cast<uint8_t*>(ptr_B));
  typename Mma::IteratorB iterator_B(
      params_B, typed_ptr_B, {problem_size.k(), problem_size.n()}, tb_thread_id,
      offset_B);

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

  for (int mma_n = 0; mma_n < MmaIterations::kColumn; ++mma_n) {
    for (int mma_m = 0; mma_m < MmaIterations::kRow; ++mma_m) {
      int start = kAccumulatorRows * kElementsPerAccess *
                  (mma_n * MmaIterations::kRow + mma_m);
      for (int row = 0; row < kAccumulatorRows; ++row) {
        for (int col = 0; col < kElementsPerAccess; ++col) {
          int local_row = quad +
                          mma_m * InstructionShape::kM * Operator::IteratorC::OpDelta::kRow +
                          row * kRowsPerTile;
          int local_channel = lane_in_quad * kElementsPerAccess +
                              mma_n * InstructionShape::kN * Operator::IteratorC::OpDelta::kColumn + col;
          int global_row = row_tile + local_row;
          int partition_channel = channel_tile + local_channel;
          int index = start + row * kElementsPerAccess + col;
          if (global_row < problem_size.m() && partition_channel < problem_size.n()) {
            ptr_C[global_row * ldc + indices[partition_channel]] = (*converted)[index];
          }
        }
      }
    }
  }
}

template <typename Core, int Stages, bool PackedInt4 = false>
struct Runner {
  using ThreadblockShape = typename Core::Shape;
  using Element = typename Core::ElementA;
  using ThreadMapA = typename Core::IteratorThreadMapA;
  using ThreadMapB = typename Core::IteratorThreadMapB;
  using AccessTypeA = cutlass::Array<ElementA, ThreadMapA::kElementsPerAccess>;
  using AccessTypeB = cutlass::Array<ElementB, ThreadMapB::kElementsPerAccess>;
  using IteratorA = cutlass::transform::threadblock::PredicatedTileAccessIterator<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA, LayoutA, 1, ThreadMapA, AccessTypeA>;
  using StandardIteratorB = cutlass::transform::threadblock::PredicatedTileAccessIterator<
      cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
      ElementB, LayoutB, 0, ThreadMapB, AccessTypeB>;
  using IteratorB = typename std::conditional<
      PackedInt4,
      PackedInt4Iterator<
          cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
          ThreadMapB, AccessTypeB>,
      StandardIteratorB>::type;
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
      at::Tensor& matrix_C, cudaStream_t stream,
      const at::Tensor* packed_zero = nullptr) {
    cutlass::gemm::GemmCoord problem_size(rows, channels, width);
    typename IteratorA::Params params_A(matrix_A.stride(0));
    typename IteratorB::Params params_B = [&] {
      if constexpr (PackedInt4) {
        TORCH_CHECK(packed_zero && packed_zero->defined(),
                    "packed INT4 runner requires checkpoint zero points");
        return typename IteratorB::Params(
            problem_size.k(), problem_size.k() / 128,
            packed_zero->data_ptr<uint8_t>());
      } else {
        return typename IteratorB::Params(problem_size.k());
      }
    }();
    typename Mma::IteratorScale::Params params_scale(matrix_scale.stride(0));
    typename Mma::IteratorScaleAct::Params params_scale_act(matrix_scale_act.stride(0));
    typename Mma::IteratorZero::Params params_zero(matrix_zero.stride(0));
    dim3 block(32, Core::WarpCount::kCount, 1);
    dim3 grid((rows + ThreadblockShape::kM - 1) / ThreadblockShape::kM,
              (channels + ThreadblockShape::kN - 1) / ThreadblockShape::kN);
    int shared_bytes = smem_size();
    if (shared_bytes >= (48 << 10)) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          kernel<Mma, SharedStorage>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          kernel<Mma, SharedStorage>,
          cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    }
    kernel<Mma, SharedStorage><<<grid, block, shared_bytes, stream>>>(
        problem_size, params_A, matrix_A.data_ptr<int8_t>(), params_B,
        matrix_B.data_ptr<uint8_t>(), params_scale,
        reinterpret_cast<typename Mma::ElementScale const*>(matrix_scale.data_ptr<at::Half>()),
        params_scale_act,
        reinterpret_cast<typename Mma::ElementScale const*>(matrix_scale_act.data_ptr<at::Half>()),
        params_zero,
        reinterpret_cast<typename Mma::ElementZero const*>(matrix_zero.data_ptr<uint8_t>()),
        matrix_C.data_ptr<float>(), matrix_C.stride(0), matrix_indices.data_ptr<int32_t>());
  }
};

using Core = cutlass::gemm::threadblock::DefaultMmaCore<
    cutlass::gemm::GemmShape<32, 128, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
    cutlass::arch::OpClassTensorOp, 2, cutlass::arch::OpMultiplyAddSaturate>;



using Int8Runner = Runner<Core, 2, false>;
using PackedInt4Runner = Runner<Core, 2, true>;

}  // namespace shmq_cutlass_sm75
