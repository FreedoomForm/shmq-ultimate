// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

#include <ATen/cuda/CUDAContext.h>
#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/library.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <mma.h>
#include "sm75_cutlass_testbed.h"

namespace {
namespace wmma = nvcuda::wmma;

constexpr int kWarpSize = 32;
constexpr int kTile = 16;
constexpr int kGroupSize = 128;
// Four 8-lane subwarps share the input stream. This raises decode output
// parallelism without changing the packed checkpoint ABI.
constexpr int kDecodeWarps = 8;
constexpr int kDecodeChannelsPerWarp = 2;
constexpr int kDecodeSubwarp = kWarpSize / kDecodeChannelsPerWarp;
constexpr int kPrefillWarps = 4;
constexpr int kPrefillChannels = kPrefillWarps * kTile;
constexpr int kReuseRows = 2 * kTile;

// Upstream MixLLM overlaps its INT4 and INT8 staged GEMMs on two auxiliary
// streams and joins them on the caller stream.  Keep that topology private to
// the SM75 adapter: the public operator remains one synchronous three-level
// interface, while partition launch and stream lifetime stay in one deep module.
struct IntegerPrefillStreams {
  cudaStream_t int4 = nullptr;
  cudaStream_t int8 = nullptr;
  cudaEvent_t fork = nullptr;
  cudaEvent_t done_int4 = nullptr;
  cudaEvent_t done_int8 = nullptr;
};

std::mutex g_integer_streams_mutex;
std::unordered_map<int, std::unique_ptr<IntegerPrefillStreams>> g_integer_streams;

IntegerPrefillStreams& integer_prefill_streams(int device_index) {
  std::lock_guard<std::mutex> lock(g_integer_streams_mutex);
  auto& slot = g_integer_streams[device_index];
  if (!slot) {
    slot = std::make_unique<IntegerPrefillStreams>();
    C10_CUDA_CHECK(cudaSetDevice(device_index));
    C10_CUDA_CHECK(cudaStreamCreateWithFlags(&slot->int4, cudaStreamNonBlocking));
    C10_CUDA_CHECK(cudaStreamCreateWithFlags(&slot->int8, cudaStreamNonBlocking));
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&slot->fork, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&slot->done_int4, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&slot->done_int8, cudaEventDisableTiming));
  }
  return *slot;
}

enum class CutlassConfig : int {
  kN128 = 0,
  kN64 = 1,
  kM128N64 = 2,
  kM64N64 = 3,
};

constexpr int kCutlassTuningAbi = 286;
constexpr int kCutlassTuningWarmup = 2;
constexpr int kCutlassTuningIterations = 4;
std::mutex g_cutlass_tuning_mutex;
std::unordered_map<std::string, CutlassConfig> g_cutlass_tuning_cache;

std::string cutlass_tuning_key(
    int device_index, int rows, int channels, int width) {
  std::ostringstream key;
  key << "abi=" << kCutlassTuningAbi << ":device=" << device_index
      << ":m=" << rows << ":n=" << channels << ":k=" << width;
  return key.str();
}

std::string cutlass_tuning_cache_path() {
  const char* configured = std::getenv("SHMQ_SM75_TUNE_CACHE");
  if (configured) {
    return std::string(configured);
  }
  return std::string("/tmp/shmq_sm75_tuning.cache");
}

bool load_cutlass_tuning_from_disk(
    const std::string& key, CutlassConfig& config) {
  const std::string path = cutlass_tuning_cache_path();
  if (path.empty()) {
    return false;
  }
  std::ifstream input(path);
  std::string stored_key;
  int stored_config = -1;
  while (input >> stored_key >> stored_config) {
    if (stored_key == key &&
        (stored_config == static_cast<int>(CutlassConfig::kN128) ||
         stored_config == static_cast<int>(CutlassConfig::kN64) ||
         stored_config == static_cast<int>(CutlassConfig::kM128N64) ||
         stored_config == static_cast<int>(CutlassConfig::kM64N64))) {
      config = static_cast<CutlassConfig>(stored_config);
      return true;
    }
  }
  return false;
}

void save_cutlass_tuning_to_disk(
    const std::string& key, CutlassConfig config) {
  const std::string path = cutlass_tuning_cache_path();
  if (path.empty()) {
    return;
  }
  std::ofstream output(path, std::ios::app);
  if (output) {
    output << key << ' ' << static_cast<int>(config) << '\n';
  }
}

bool stream_is_capturing(cudaStream_t stream) {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
  return status != cudaStreamCaptureStatusNone;
}

void run_cutlass_config(
    CutlassConfig config, int rows, int channels, int width,
    at::Tensor& input_int8, at::Tensor& weight,
    at::Tensor& scale_act, at::Tensor& matrix_scale,
    at::Tensor& matrix_zero, at::Tensor& indices,
    at::Tensor& output, cudaStream_t stream) {
  if (config == CutlassConfig::kM64N64) {
    shmq_cutlass_sm75::Int8RunnerM64N64::run(
        rows, channels, width, input_int8, weight, scale_act, matrix_scale,
        matrix_zero, indices, output, stream);
  } else if (config == CutlassConfig::kM128N64) {
    shmq_cutlass_sm75::Int8RunnerM128N64::run(
        rows, channels, width, input_int8, weight, scale_act, matrix_scale,
        matrix_zero, indices, output, stream);
  } else if (config == CutlassConfig::kN64) {
    shmq_cutlass_sm75::Int8RunnerN64::run(
        rows, channels, width, input_int8, weight, scale_act, matrix_scale,
        matrix_zero, indices, output, stream);
  } else {
    shmq_cutlass_sm75::Int8Runner::run(
        rows, channels, width, input_int8, weight, scale_act, matrix_scale,
        matrix_zero, indices, output, stream);
  }
}

CutlassConfig select_cutlass_config(
    int device_index, int rows, int channels, int width,
    at::Tensor& input_int8, at::Tensor& weight,
    at::Tensor& scale_act, at::Tensor& matrix_scale,
    at::Tensor& matrix_zero, at::Tensor& indices,
    at::Tensor& output, cudaStream_t stream, bool allow_tuning) {
  const std::string key = cutlass_tuning_key(device_index, rows, channels, width);
  {
    std::lock_guard<std::mutex> lock(g_cutlass_tuning_mutex);
    auto found = g_cutlass_tuning_cache.find(key);
    if (found != g_cutlass_tuning_cache.end()) {
      return found->second;
    }
    CutlassConfig disk_config = CutlassConfig::kN128;
    if (load_cutlass_tuning_from_disk(key, disk_config)) {
      g_cutlass_tuning_cache.emplace(key, disk_config);
      return disk_config;
    }
  }

  // CUDA graph capture cannot contain the event synchronization or file I/O
  // used by first-run tuning.  The measured v200 family is the safe fallback.
  if (!allow_tuning || stream_is_capturing(stream)) {
    return CutlassConfig::kN128;
  }

  CutlassConfig best = CutlassConfig::kN128;
  float best_ms = std::numeric_limits<float>::infinity();
  for (CutlassConfig candidate : {CutlassConfig::kN128, CutlassConfig::kN64,
                                  CutlassConfig::kM128N64,
                                  CutlassConfig::kM64N64}) {
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    C10_CUDA_CHECK(cudaEventCreate(&begin));
    C10_CUDA_CHECK(cudaEventCreate(&end));
    for (int iteration = 0; iteration < kCutlassTuningWarmup; ++iteration) {
      run_cutlass_config(
          candidate, rows, channels, width, input_int8, weight, scale_act,
          matrix_scale, matrix_zero, indices, output, stream);
    }
    C10_CUDA_CHECK(cudaEventRecord(begin, stream));
    for (int iteration = 0; iteration < kCutlassTuningIterations; ++iteration) {
      run_cutlass_config(
          candidate, rows, channels, width, input_int8, weight, scale_act,
          matrix_scale, matrix_zero, indices, output, stream);
    }
    C10_CUDA_CHECK(cudaEventRecord(end, stream));
    C10_CUDA_CHECK(cudaEventSynchronize(end));
    float elapsed_ms = 0.0f;
    C10_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, begin, end));
    C10_CUDA_CHECK(cudaEventDestroy(begin));
    C10_CUDA_CHECK(cudaEventDestroy(end));
    if (elapsed_ms < best_ms) {
      best_ms = elapsed_ms;
      best = candidate;
    }
  }

  {
    std::lock_guard<std::mutex> lock(g_cutlass_tuning_mutex);
    auto [entry, inserted] = g_cutlass_tuning_cache.emplace(key, best);
    if (!inserted) {
      best = entry->second;
    } else {
      save_cutlass_tuning_to_disk(key, best);
    }
  }
  return best;
}

void record_tensor_stream(const at::Tensor& tensor, cudaStream_t stream) {
  if (!tensor.defined() || !tensor.numel()) {
    return;
  }
  auto external_stream = at::cuda::getStreamFromExternal(
      stream, tensor.device().index());
  c10::cuda::CUDACachingAllocator::recordStream(
      tensor.storage().data_ptr(), external_stream);
}

// Four warps per block; each warp quantizes one contiguous 128-element group.
// This mirrors the upstream MixLLM activation contract without constructing a
// chain of temporary FP32 tensors through the PyTorch dispatcher.
__global__ void quantize_activation_sm75_kernel(
    const __half* input, int8_t* quantized, __half* scales,
    int rows, int width) {
#if __CUDA_ARCH__ >= 750
  const int lane = threadIdx.x % kWarpSize;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / kWarpSize;
  const int groups = width / kGroupSize;
  const int total_groups = rows * groups;
  if (warp >= total_groups) {
    return;
  }
  const int row = warp / groups;
  const int group = warp % groups;
  const int base = row * width + group * kGroupSize;
  const __half2* input2 = reinterpret_cast<const __half2*>(input + base);
  const __half2 pair0 = input2[lane * 2];
  const __half2 pair1 = input2[lane * 2 + 1];
  const float2 converted0 = __half22float2(pair0);
  const float2 converted1 = __half22float2(pair1);
  float values[4] = {converted0.x, converted0.y, converted1.x, converted1.y};
  float maximum = 0.0f;
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    maximum = fmaxf(maximum, fabsf(values[item]));
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
  }
  maximum = __shfl_sync(0xffffffff, maximum, 0);
  const float scale = fmaxf(maximum / 127.0f, 1.0e-8f);
  if (lane == 0) {
    scales[group * rows + row] = __float2half(scale);
  }
  const float inverse_scale = 1.0f / scale;
  uint32_t packed = 0;
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    float scaled = values[item] * inverse_scale;
    const int fast_value = __float2int_rn(scaled);
    const float lower_boundary = static_cast<float>(fast_value) - 0.5f;
    const float upper_boundary = static_cast<float>(fast_value) + 0.5f;
    if (fabsf(scaled - lower_boundary) < 1.0e-3f ||
        fabsf(scaled - upper_boundary) < 1.0e-3f) {
      scaled = values[item] / scale;
    }
    int value = __float2int_rn(scaled);
    value = max(-127, min(127, value));
    packed |= static_cast<uint32_t>(static_cast<uint8_t>(value)) << (item * 8);
  }
  *reinterpret_cast<uint32_t*>(quantized + base + lane * 4) = packed;
#endif
}

__global__ void expand_int4_sm75_kernel(
    const uint8_t* packed, const uint8_t* zeros, int8_t* expanded,
    int channels, int width) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = channels * width;
  if (linear >= elements) {
    return;
  }
  const int channel = linear / width;
  const int k = linear % width;
  const uint8_t byte = packed[channel * (width / 2) + k / 2];
  const int code = k & 1 ? byte >> 4 : byte & 0x0f;
  const int groups = width / kGroupSize;
  expanded[linear] = static_cast<int8_t>(
      code - static_cast<int>(zeros[channel * groups + k / kGroupSize]));
}

__global__ void scatter_fp16_partition_kernel(
    const __half* partial, const int32_t* indices, __half* output,
    int rows, int partition_channels, int output_width) {
#if __CUDA_ARCH__ >= 750
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = rows * partition_channels;
  if (linear >= elements) {
    return;
  }
  const int row = linear / partition_channels;
  const int channel = linear % partition_channels;
  output[row * output_width + indices[channel]] = partial[linear];
#endif
}

// SM75 has no Ampere mixed INT8 x INT4 MMA. Prefill reads a cached signed INT8
// expansion while decode continues to consume the packed checkpoint weights.
template <int PrefillWarps = kPrefillWarps>
__global__ void three_level_tensorcore_kernel(
    const __half* input_fp16, const int8_t* input_int8,
    const __half* scale_act, const int8_t* expanded_int4,
    const __half* scale_int4,
    const int32_t* indices_int4, const int8_t* weight_int8,
    const __half* scale_int8, const int32_t* indices_int8,
    const __half* weight_fp16, const int32_t* indices_fp16, __half* output,
    int rows, int width, int output_width, int n4, int n8, int n16) {
#if __CUDA_ARCH__ >= 750
  constexpr int prefill_channels = PrefillWarps * kTile;
  const int tiles4 = (n4 + prefill_channels - 1) / prefill_channels;
  const int tiles8 = (n8 + prefill_channels - 1) / prefill_channels;
  const int tile_id = blockIdx.x;
  const int row_base = blockIdx.y * kTile;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  int precision;
  int channel_base;
  if (tile_id < tiles4) {
    precision = 4;
    channel_base = tile_id * prefill_channels + warp * kTile;
  } else if (tile_id < tiles4 + tiles8) {
    precision = 8;
    channel_base = (tile_id - tiles4) * prefill_channels + warp * kTile;
  } else {
    precision = 16;
    channel_base = (tile_id - tiles4 - tiles8) * prefill_channels +
                   warp * kTile;
  }

  __shared__ __align__(16) int8_t a_int8[kTile * kTile];
  __shared__ __align__(16) int8_t b_int8[PrefillWarps][kTile * kTile];
  __shared__ __align__(16) int accumulator_int[PrefillWarps][kTile * kTile];
  __shared__ __align__(16) __half a_fp16[kTile * kTile];
  __shared__ __align__(16) __half b_fp16[PrefillWarps][kTile * kTile];
  __shared__ __align__(16) float accumulator_fp32[PrefillWarps][kTile * kTile];

  if (precision == 16) {
    wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    const bool full_rows = row_base + kTile <= rows;
    const bool full_channels = channel_base + kTile <= n16;
    for (int k_base = 0; k_base < width; k_base += kTile) {
      wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, __half,
                     wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, __half,
                     wmma::col_major> b;
      for (int linear = threadIdx.x; linear < kTile * kTile;
           linear += blockDim.x) {
        const int tile_row = linear / kTile;
        const int tile_col = linear % kTile;
        const int row = row_base + tile_row;
        a_fp16[linear] = row < rows
            ? input_fp16[row * width + k_base + tile_col]
            : __float2half(0.0f);
      }
      __syncthreads();
      wmma::load_matrix_sync(a, a_fp16, kTile);
      if (full_channels) {
        wmma::load_matrix_sync(
            b, weight_fp16 + channel_base * width + k_base, width);
      } else {
        for (int linear = lane; linear < kTile * kTile;
             linear += kWarpSize) {
          const int tile_row = linear / kTile;
          const int tile_col = linear % kTile;
          const int channel = channel_base + tile_col;
          b_fp16[warp][tile_col * kTile + tile_row] = channel < n16
              ? weight_fp16[channel * width + k_base + tile_row]
              : __float2half(0.0f);
        }
        __syncwarp();
        wmma::load_matrix_sync(b, b_fp16[warp], kTile);
      }
      wmma::mma_sync(accumulator, a, b, accumulator);
      __syncthreads();
    }
    wmma::store_matrix_sync(accumulator_fp32[warp], accumulator, kTile,
                            wmma::mem_row_major);
    __syncwarp();
    for (int linear = lane; linear < kTile * kTile; linear += kWarpSize) {
      const int row = row_base + linear / kTile;
      const int local_channel = channel_base + linear % kTile;
      if (row < rows && local_channel < n16) {
        output[row * output_width + indices_fp16[local_channel]] =
            __float2half_rn(accumulator_fp32[warp][linear]);
      }
    }
    return;
  }

  float scaled_accumulators[kTile * kTile / kWarpSize] = {};
  const int groups = width / kGroupSize;
  const int partition_size = precision == 4 ? n4 : n8;
  const bool full_rows = row_base + kTile <= rows;
  const bool full_channels = channel_base + kTile <= partition_size;
  for (int group = 0; group < groups; ++group) {
    wmma::fragment<wmma::accumulator, kTile, kTile, kTile, int> accumulator;
    wmma::fill_fragment(accumulator, 0);
    for (int group_k = 0; group_k < kGroupSize; group_k += kTile) {
      const int k_base = group * kGroupSize + group_k;
      wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, signed char,
                     wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, signed char,
                     wmma::col_major> b;
      for (int linear = threadIdx.x; linear < kTile * kTile;
           linear += blockDim.x) {
        const int tile_row = linear / kTile;
        const int tile_col = linear % kTile;
        const int row = row_base + tile_row;
        a_int8[linear] = row < rows
            ? input_int8[row * width + k_base + tile_col]
            : int8_t{0};
      }
      __syncthreads();
      wmma::load_matrix_sync(
          a, reinterpret_cast<signed char*>(a_int8), kTile);
      if (precision == 8 && full_channels) {
        wmma::load_matrix_sync(
            b, reinterpret_cast<const signed char*>(
                   weight_int8 + channel_base * width + k_base), width);
      } else if (precision == 4 && full_channels) {
        wmma::load_matrix_sync(
            b, reinterpret_cast<const signed char*>(
                   expanded_int4 + channel_base * width + k_base), width);
      } else {
        for (int linear = lane; linear < kTile * kTile;
             linear += kWarpSize) {
          const int tile_row = linear / kTile;
          const int tile_col = linear % kTile;
          const int channel = channel_base + tile_col;
          int8_t weight = 0;
          if (channel < partition_size) {
            const int k = k_base + tile_row;
            weight = precision == 4
                ? expanded_int4[channel * width + k]
                : weight_int8[channel * width + k];
          }
          b_int8[warp][tile_col * kTile + tile_row] = weight;
        }
        __syncwarp();
        wmma::load_matrix_sync(
            b, reinterpret_cast<signed char*>(b_int8[warp]), kTile);
      }
      wmma::mma_sync(accumulator, a, b, accumulator);
      __syncthreads();
    }
    wmma::store_matrix_sync(accumulator_int[warp], accumulator, kTile,
                            wmma::mem_row_major);
    __syncwarp();
    for (int linear = lane, item = 0; linear < kTile * kTile;
         linear += kWarpSize, ++item) {
      const int tile_row = linear / kTile;
      const int local_channel = channel_base + linear % kTile;
      if (row_base + tile_row < rows && local_channel < partition_size) {
        const float activation_scale =
            __half2float(scale_act[group * rows + row_base + tile_row]);
        const __half weight_scale = precision == 4
            ? scale_int4[local_channel * groups + group]
            : scale_int8[local_channel * groups + group];
        scaled_accumulators[item] += static_cast<float>(accumulator_int[warp][linear]) *
                                     activation_scale * __half2float(weight_scale);
      }
    }
    __syncwarp();
  }
  for (int linear = lane, item = 0; linear < kTile * kTile;
       linear += kWarpSize, ++item) {
    const int row = row_base + linear / kTile;
    const int local_channel = channel_base + linear % kTile;
    if (row < rows && local_channel < partition_size) {
      const int output_channel = precision == 4
          ? indices_int4[local_channel] : indices_int8[local_channel];
        output[row * output_width + output_channel] = __float2half_rn(scaled_accumulators[item]);
    }
  }
#endif
}

// Balanced prefill variant. Eight warps form a 2x4 grid of 16x16 WMMA tiles.
// Each 32x16 A panel and 16x64 B panel is loaded once per CTA/K step.
__global__ void three_level_tensorcore_reuse_kernel(
    const __half* input_fp16, const int8_t* input_int8,
    const __half* scale_act, const int8_t* expanded_int4,
    const __half* scale_int4,
    const int32_t* indices_int4, const int8_t* weight_int8,
    const __half* scale_int8, const int32_t* indices_int8,
    const __half* weight_fp16, const int32_t* indices_fp16, __half* output,
    int rows, int width, int output_width, int n4, int n8, int n16) {
#if __CUDA_ARCH__ >= 750
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int warp_row = warp / 4;
  const int warp_channel = warp % 4;
  const int row_tile_base = blockIdx.y * kReuseRows;
  const int row_base = row_tile_base + warp_row * kTile;
  const int precision_tile = blockIdx.x;
  const int tiles4 = (n4 + kPrefillChannels - 1) / kPrefillChannels;
  const int tiles8 = (n8 + kPrefillChannels - 1) / kPrefillChannels;
  int precision = 16;
  int cta_channel_base = 0;
  int partition_size = n16;
  if (precision_tile < tiles4) {
    precision = 4;
    cta_channel_base = precision_tile * kPrefillChannels;
    partition_size = n4;
  } else if (precision_tile < tiles4 + tiles8) {
    precision = 8;
    cta_channel_base = (precision_tile - tiles4) * kPrefillChannels;
    partition_size = n8;
  } else {
    cta_channel_base = (precision_tile - tiles4 - tiles8) * kPrefillChannels;
  }
  const int channel_base = cta_channel_base + warp_channel * kTile;
  __shared__ __align__(16) int8_t a_int8[2][kTile * kTile];
  __shared__ __align__(16) int8_t b_int8[4][kTile * kTile];
  __shared__ __align__(16) int accumulator_int[8][kTile * kTile];
  __shared__ __align__(16) __half a_fp16[2][kTile * kTile];
  __shared__ __align__(16) __half b_fp16[4][kTile * kTile];
  __shared__ __align__(16) float accumulator_fp32[8][kTile * kTile];
  const int groups = width / kGroupSize;
  if (precision == 16) {
    wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int k_base = 0; k_base < width; k_base += kTile) {
      for (int linear = threadIdx.x; linear < 2 * kTile * kTile;
           linear += blockDim.x) {
        const int tile = linear / (kTile * kTile);
        const int item = linear % (kTile * kTile);
        const int r = row_tile_base + tile * kTile + item / kTile;
        a_fp16[tile][item] = r < rows
            ? input_fp16[r * width + k_base + item % kTile]
            : __float2half(0.0f);
      }
      for (int linear = threadIdx.x; linear < 4 * kTile * kTile;
           linear += blockDim.x) {
        const int tile = linear / (kTile * kTile);
        const int item = linear % (kTile * kTile);
        const int c = cta_channel_base + tile * kTile + item / kTile;
        b_fp16[tile][item] = c < partition_size
            ? weight_fp16[c * width + k_base + item % kTile]
            : __float2half(0.0f);
      }
      __syncthreads();
      wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, __half,
                     wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, __half,
                     wmma::col_major> b;
      wmma::load_matrix_sync(a, a_fp16[warp_row], kTile);
      wmma::load_matrix_sync(b, b_fp16[warp_channel], kTile);
      wmma::mma_sync(acc, a, b, acc);
      __syncthreads();
    }
    wmma::store_matrix_sync(accumulator_fp32[warp], acc, kTile,
                            wmma::mem_row_major);
    __syncwarp();
    for (int linear = lane; linear < kTile * kTile; linear += kWarpSize) {
      const int r = row_base + linear / kTile;
      const int c = channel_base + linear % kTile;
      if (r < rows && c < partition_size)
        output[r * output_width + indices_fp16[c]] =
            __float2half_rn(accumulator_fp32[warp][linear]);
    }
    return;
  }
  float scaled[kTile * kTile / kWarpSize] = {};
  for (int group = 0; group < groups; ++group) {
    for (int group_k = 0; group_k < kGroupSize; group_k += kTile) {
      const int k_base = group * kGroupSize + group_k;
      for (int linear = threadIdx.x; linear < 2 * kTile * kTile;
           linear += blockDim.x) {
        const int tile = linear / (kTile * kTile);
        const int item = linear % (kTile * kTile);
        const int r = row_tile_base + tile * kTile + item / kTile;
        a_int8[tile][item] = r < rows
            ? input_int8[r * width + k_base + item % kTile] : 0;
      }
      for (int linear = threadIdx.x; linear < 4 * kTile * kTile;
           linear += blockDim.x) {
        const int tile = linear / (kTile * kTile);
        const int item = linear % (kTile * kTile);
        const int c = cta_channel_base + tile * kTile + item / kTile;
        int8_t value = 0;
        if (c < partition_size) {
          const int k = k_base + item % kTile;
          value = precision == 8 ? weight_int8[c * width + k]
                                 : expanded_int4[c * width + k];
        }
        b_int8[tile][item] = value;
      }
      __syncthreads();
      wmma::fragment<wmma::accumulator, kTile, kTile, kTile, int> acc;
      wmma::fill_fragment(acc, 0);
      wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, signed char,
                     wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, signed char,
                     wmma::col_major> b;
      wmma::load_matrix_sync(a, a_int8[warp_row], kTile);
      wmma::load_matrix_sync(b, b_int8[warp_channel], kTile);
      wmma::mma_sync(acc, a, b, acc);
      __syncthreads();
      wmma::store_matrix_sync(accumulator_int[warp], acc, kTile,
                              wmma::mem_row_major);
      __syncwarp();
      for (int linear = lane; linear < kTile * kTile; linear += kWarpSize) {
        const int r = row_base + linear / kTile;
        const int c = channel_base + linear % kTile;
        if (r < rows && c < partition_size) {
          const float as = __half2float(scale_act[group * rows + r]);
          const float ws = __half2float(precision == 4
              ? scale_int4[c * groups + group] : scale_int8[c * groups + group]);
          scaled[linear / kWarpSize] +=
              static_cast<float>(accumulator_int[warp][linear]) * as * ws;
        }
      }
      __syncthreads();
    }
  }
  for (int linear = lane; linear < kTile * kTile; linear += kWarpSize) {
    const int r = row_base + linear / kTile;
    const int c = channel_base + linear % kTile;
    if (r < rows && c < partition_size)
        output[r * output_width + (precision == 4 ? indices_int4[c] : indices_int8[c])] = __float2half_rn(scaled[linear / kWarpSize]);
  }
#endif
}

// Decode is bandwidth-bound and WMMA would compute 15 unused rows for M=1.
// Four 8-lane subwarps share one warp; each subwarp owns one output channel and
// performs four dp4a operations per 128-element activation group. This avoids
// assigning a full warp and five reduction steps to every scalar output.
__global__ void three_level_decode_kernel(
    const __half* input_fp16, const int8_t* input_int8,
    const __half* scale_act, const uint8_t* packed_int4,
    const __half* scale_int4, const uint8_t* zero_int4,
    const int32_t* indices_int4, const int8_t* weight_int8,
    const __half* scale_int8, const int32_t* indices_int8,
    const __half* weight_fp16, const int32_t* indices_fp16, __half* output,
    int width, int output_width, int n4, int n8, int n16) {
#if __CUDA_ARCH__ >= 750
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int local_warp = threadIdx.x / kWarpSize;
  const int subwarp = lane / kDecodeSubwarp;
  const int sublane = lane & (kDecodeSubwarp - 1);
  const int channel = blockIdx.x * (kDecodeWarps * kDecodeChannelsPerWarp) +
                      local_warp * kDecodeChannelsPerWarp + subwarp;
  if (channel >= output_width) {
    return;
  }

  const int groups = width / kGroupSize;
  const unsigned int subwarp_mask = __activemask();
  float result = 0.0f;
  int output_channel;
  if (channel < n4) {
    output_channel = indices_int4[channel];
    const uint8_t* weights = packed_int4 + channel * (width / 2);
    for (int group = 0; group < groups; ++group) {
      const int base = group * kGroupSize;
      const int zero = static_cast<int>(zero_int4[channel * groups + group]);
      int accumulator = 0;
#pragma unroll
      for (int offset = sublane * 4; offset < kGroupSize;
           offset += kDecodeSubwarp * 4) {
        const int k = base + offset;
        const uint8_t packed01 = weights[k / 2];
        const uint8_t packed23 = weights[k / 2 + 1];
        const int w0 = static_cast<int>(packed01 & 0x0f) - zero;
        const int w1 = static_cast<int>(packed01 >> 4) - zero;
        const int w2 = static_cast<int>(packed23 & 0x0f) - zero;
        const int w3 = static_cast<int>(packed23 >> 4) - zero;
        const int packed_weights =
            (w0 & 0xff) | ((w1 & 0xff) << 8) |
            ((w2 & 0xff) << 16) | ((w3 & 0xff) << 24);
        const int activations = *reinterpret_cast<const int*>(input_int8 + k);
        accumulator = __dp4a(packed_weights, activations, accumulator);
      }
      for (int delta = kDecodeSubwarp / 2; delta > 0; delta >>= 1) {
        accumulator += __shfl_down_sync(subwarp_mask, accumulator, delta,
                                         kDecodeSubwarp);
      }
      if (sublane == 0) {
        result += static_cast<float>(accumulator) *
                  __half2float(scale_act[group]) *
                  __half2float(scale_int4[channel * groups + group]);
      }
    }
  } else if (channel < n4 + n8) {
    const int local_channel = channel - n4;
    output_channel = indices_int8[local_channel];
    const int8_t* weights = weight_int8 + local_channel * width;
    for (int group = 0; group < groups; ++group) {
      const int base = group * kGroupSize;
      int accumulator = 0;
#pragma unroll
      for (int offset = sublane * 4; offset < kGroupSize;
           offset += kDecodeSubwarp * 4) {
        const int k = base + offset;
        const int packed_weights = *reinterpret_cast<const int*>(weights + k);
        const int activations = *reinterpret_cast<const int*>(input_int8 + k);
        accumulator = __dp4a(packed_weights, activations, accumulator);
      }
      for (int delta = kDecodeSubwarp / 2; delta > 0; delta >>= 1) {
        accumulator += __shfl_down_sync(subwarp_mask, accumulator, delta,
                                         kDecodeSubwarp);
      }
      if (sublane == 0) {
        result += static_cast<float>(accumulator) *
                  __half2float(scale_act[group]) *
                  __half2float(scale_int8[local_channel * groups + group]);
      }
    }
  } else {
    const int local_channel = channel - n4 - n8;
    output_channel = indices_fp16[local_channel];
    const __half2* input2 = reinterpret_cast<const __half2*>(input_fp16);
    const __half2* weights2 = reinterpret_cast<const __half2*>(
        weight_fp16 + local_channel * width);
    float accumulator = 0.0f;
    for (int k2 = sublane; k2 < width / 2; k2 += kDecodeSubwarp) {
      const float2 product = __half22float2(__hmul2(input2[k2], weights2[k2]));
      accumulator += product.x + product.y;
    }
    for (int delta = kDecodeSubwarp / 2; delta > 0; delta >>= 1) {
      accumulator += __shfl_down_sync(subwarp_mask, accumulator, delta,
                                       kDecodeSubwarp);
    }
    if (sublane == 0) {
      result = accumulator;
    }
  }
  if (sublane == 0) {
    output[output_channel] = __float2half_rn(result);
  }
#endif
}

using PairProbeLowMma = shmq_cutlass_sm75::int4_pair_probe::LowMma;
using PairProbeHighMma = shmq_cutlass_sm75::int4_pair_probe::HighMma;

__global__ void sm75_int4_pair_instruction_probe_kernel(int* output) {
#if __CUDA_ARCH__ >= 750
  PairProbeLowMma::FragmentA low_a;
  PairProbeHighMma::FragmentA high_a;
  PairProbeLowMma::FragmentB weights;
  PairProbeLowMma::FragmentC low_accum;
  PairProbeHighMma::FragmentC high_accum;
  low_a.clear();
  high_a.clear();
  weights.clear();
  low_accum.clear();
  high_accum.clear();
  PairProbeLowMma low_mma;
  PairProbeHighMma high_mma;
  low_mma(low_accum, low_a, weights, low_accum);
  high_mma(high_accum, high_a, weights, high_accum);
  if (threadIdx.x == 0) {
    output[0] = low_accum[0];
    output[1] = high_accum[0];
  }
#endif
}

at::Tensor sm75_int4_pair_instruction_probe_cuda(const at::Tensor& device_tensor) {
  TORCH_CHECK(device_tensor.is_cuda(), "SM75 INT4 probe requires a CUDA tensor argument");
  auto output = at::zeros({2}, device_tensor.options().dtype(at::kInt));
  auto stream = at::cuda::getCurrentCUDAStream();
  sm75_int4_pair_instruction_probe_kernel<<<1, kWarpSize, 0, stream>>>(
      output.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

__global__ void sm75_int4_native_decomposition_probe_kernel(int* output) {
#if __CUDA_ARCH__ >= 750
  const int case_id = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int low_value = case_id == 0 ? 1 : 15;
  const int high_value = case_id == 0 ? -1 : 7;
  const int weight_value = case_id == 0 ? 1 : 2;

  PairProbeLowMma::FragmentA low_a;
  PairProbeHighMma::FragmentA high_a;
  PairProbeLowMma::FragmentB weights;
  PairProbeLowMma::FragmentC low_accum;
  PairProbeHighMma::FragmentC high_accum;
  for (int i = 0; i < low_a.kElements; ++i) {
    low_a[i] = cutlass::uint4b_t(static_cast<uint8_t>(low_value));
    high_a[i] = cutlass::int4b_t(high_value);
    weights[i] = cutlass::uint4b_t(static_cast<uint8_t>(weight_value));
  }
  low_accum.clear();
  high_accum.clear();
  PairProbeLowMma low_mma;
  PairProbeHighMma high_mma;
  low_mma(low_accum, low_a, weights, low_accum);
  high_mma(high_accum, high_a, weights, high_accum);
  output[(case_id * kWarpSize + lane) * 2 + 0] =
      low_accum[0] + 16 * high_accum[0];
  output[(case_id * kWarpSize + lane) * 2 + 1] =
      low_accum[1] + 16 * high_accum[1];
#endif
}

at::Tensor sm75_int4_native_decomposition_probe_cuda(
    const at::Tensor& device_tensor) {
  TORCH_CHECK(device_tensor.is_cuda(), "SM75 INT4 probe requires a CUDA tensor argument");
  auto output = at::zeros({2 * kWarpSize, 2}, device_tensor.options().dtype(at::kInt));
  auto stream = at::cuda::getCurrentCUDAStream();
  sm75_int4_native_decomposition_probe_kernel<<<1, 2 * kWarpSize, 0, stream>>>(
      output.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

__global__ void sm75_int4_pair_wmma_load_probe_kernel(int* output) {
#if __CUDA_ARCH__ >= 750
  namespace precision = wmma::experimental::precision;
  __shared__ __align__(16) uint8_t a_packed[8 * 16];
  __shared__ __align__(16) uint8_t b_packed[8 * 16];
  __shared__ __align__(16) int accumulator[8 * 8];
  const int lane = threadIdx.x;
  for (int item = lane; item < 8 * 16; item += kWarpSize) {
    const int row = item / 16;
    const int byte = item % 16;
    const int value = row + 1;
    a_packed[item] = static_cast<uint8_t>(value | (value << 4));
    const int column = row;
    const int weight = column + 1;
    b_packed[item] = static_cast<uint8_t>(weight | (weight << 4));
  }
  __syncwarp();
  wmma::fragment<wmma::matrix_a, 8, 8, 32, precision::u4,
                 wmma::row_major> a_low;
  wmma::fragment<wmma::matrix_b, 8, 8, 32, precision::u4,
                 wmma::col_major> weights;
  wmma::fragment<wmma::accumulator, 8, 8, 32, int> low_accum;
  wmma::load_matrix_sync(a_low, a_packed, 32);
  wmma::load_matrix_sync(weights, b_packed, 32);
  wmma::fill_fragment(low_accum, 0);
  wmma::mma_sync(low_accum, a_low, weights, low_accum);
  wmma::store_matrix_sync(accumulator, low_accum, 8, wmma::mem_row_major);
  __syncwarp();
  for (int item = lane; item < 8 * 8; item += kWarpSize) {
    output[item] = accumulator[item];
  }
#endif
}

__global__ void sm75_int4_pair_fused_probe_kernel(int* output) {
#if __CUDA_ARCH__ >= 750
  namespace precision = wmma::experimental::precision;
  __shared__ __align__(16) uint8_t a_low_packed[8 * 16];
  __shared__ __align__(16) uint8_t a_high_packed[8 * 16];
  __shared__ __align__(16) uint8_t b_packed[8 * 16];
  const int lane = threadIdx.x;
  for (int item = lane; item < 8 * 16; item += kWarpSize) {
    a_low_packed[item] = 0x11;
    a_high_packed[item] = 0xff;
    b_packed[item] = 0x22;
  }
  __syncwarp();
  wmma::fragment<wmma::matrix_a, 8, 8, 32, precision::u4,
                 wmma::row_major> a_low_u4;
  wmma::fragment<wmma::matrix_a, 8, 8, 32, precision::u4,
                 wmma::row_major> a_high_u4;
  wmma::fragment<wmma::matrix_b, 8, 8, 32, precision::u4,
                 wmma::col_major> b_u4;
  wmma::load_matrix_sync(a_low_u4, a_low_packed, 32);
  wmma::load_matrix_sync(a_high_u4, a_high_packed, 32);
  wmma::load_matrix_sync(b_u4, b_packed, 32);

  PairProbeLowMma::FragmentA low_a;
  PairProbeHighMma::FragmentA high_a;
  PairProbeLowMma::FragmentB weights;
  low_a.clear();
  high_a.clear();
  weights.clear();
  reinterpret_cast<unsigned&>(low_a) = a_low_u4.x[0];
  reinterpret_cast<unsigned&>(high_a) = a_high_u4.x[0];
  reinterpret_cast<unsigned&>(weights) = b_u4.x[0];

  PairProbeLowMma::FragmentC low_accum;
  PairProbeHighMma::FragmentC high_accum;
  low_accum.clear();
  high_accum.clear();
  PairProbeLowMma low_mma;
  PairProbeHighMma high_mma;
  low_mma(low_accum, low_a, weights, low_accum);
  high_mma(high_accum, high_a, weights, high_accum);
  output[lane * 4 + 0] = low_accum[0];
  output[lane * 4 + 1] = low_accum[1];
  output[lane * 4 + 2] = high_accum[0];
  output[lane * 4 + 3] = high_accum[1];
#endif
}

void run_int4_pair_partition(
    const at::Tensor& input_int8, const at::Tensor& scale_act,
    const at::Tensor& weight_int4, const at::Tensor& scale_int4,
    const at::Tensor& zero_int4, const at::Tensor& indices_int4,
    at::Tensor& output, int rows, int width, cudaStream_t stream);

at::Tensor sm75_int4_pair_mixed_stride_probe_cuda(const at::Tensor& device_tensor) {
  TORCH_CHECK(device_tensor.is_cuda(), "SM75 mixed-stride probe requires a CUDA tensor argument");
  const auto options = device_tensor.options();
  constexpr int rows = 32;
  constexpr int width = 128;
  constexpr int channels = 32;
  constexpr int output_width = 64;
  auto input_int8 = at::ones({rows, width}, options.dtype(at::kChar));
  auto scale_act = at::ones({1, rows}, options.dtype(at::kHalf));
  auto weight_int4 = at::full({channels, width / 2}, 0x11, options.dtype(at::kByte));
  auto scale_int4 = at::ones({channels, 1}, options.dtype(at::kHalf));
  auto zero_int4 = at::zeros({channels, 1}, options.dtype(at::kByte));
  auto indices_int4 = at::arange(channels, options.dtype(at::kInt));
  auto output = at::full({rows, output_width}, -999.0, options.dtype(at::kHalf));
  run_int4_pair_partition(
      input_int8, scale_act, weight_int4, scale_int4, zero_int4,
      indices_int4, output, rows, width, at::cuda::getCurrentCUDAStream());
  return output.to(at::kFloat);
}

at::Tensor sm75_int4_pair_fused_probe_cuda(const at::Tensor& device_tensor) {
  TORCH_CHECK(device_tensor.is_cuda(), "SM75 fused INT4 probe requires a CUDA tensor argument");
  auto output = at::empty({kWarpSize, 4}, device_tensor.options().dtype(at::kInt));
  auto stream = at::cuda::getCurrentCUDAStream();
  sm75_int4_pair_fused_probe_kernel<<<1, kWarpSize, 0, stream>>>(
      output.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

__global__ void sm75_int4_pair_warp_iterator_probe_kernel(int* output) {
#if __CUDA_ARCH__ >= 750
  using U4AIterator = shmq_cutlass_sm75::int4_pair_probe::NativeWarpU4AIterator;
  using S4AIterator = shmq_cutlass_sm75::int4_pair_probe::NativeWarpS4AIterator;
  using U4BIterator = shmq_cutlass_sm75::int4_pair_probe::NativeWarpU4BIterator;
  __shared__ __align__(16) uint8_t a_storage[32 * 16];
  __shared__ __align__(16) uint8_t b_storage[32 * 16];
  const int lane = threadIdx.x;
  for (int item = lane; item < 32 * 16; item += kWarpSize) {
    a_storage[item] = 0x11;
    b_storage[item] = 0x22;
  }
  __syncwarp();

  U4AIterator::TensorCoord extent(32, 32);
  U4AIterator::Layout layout = U4AIterator::Layout::packed(extent);
  U4AIterator u4a(U4AIterator::TensorRef(
      reinterpret_cast<cutlass::uint4b_t*>(a_storage), layout), lane);
  S4AIterator s4a(
      S4AIterator::TensorRef(reinterpret_cast<cutlass::int4b_t*>(a_storage),
                              S4AIterator::Layout::packed(extent)),
      lane);
  U4BIterator u4b(U4BIterator::TensorRef(
      reinterpret_cast<cutlass::uint4b_t*>(b_storage), layout), lane);
  U4AIterator::Fragment u4a_fragment;
  S4AIterator::Fragment s4a_fragment;
  U4BIterator::Fragment u4b_fragment;
  u4a.load(u4a_fragment);
  s4a.load(s4a_fragment);
  u4b.load(u4b_fragment);
  output[lane * 3 + 0] = reinterpret_cast<unsigned*>(&u4a_fragment)[0];
  output[lane * 3 + 1] = reinterpret_cast<unsigned*>(&s4a_fragment)[0];
  output[lane * 3 + 2] = reinterpret_cast<unsigned*>(&u4b_fragment)[0];
#endif
}

at::Tensor sm75_int4_pair_warp_iterator_probe_cuda(
    const at::Tensor& device_tensor) {
  TORCH_CHECK(device_tensor.is_cuda(),
              "SM75 warp iterator probe requires a CUDA tensor argument");
  auto output = at::empty({kWarpSize, 3}, device_tensor.options().dtype(at::kInt));
  auto stream = at::cuda::getCurrentCUDAStream();
  sm75_int4_pair_warp_iterator_probe_kernel<<<1, kWarpSize, 0, stream>>>(
      output.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor sm75_int4_pair_wmma_load_probe_cuda(const at::Tensor& device_tensor) {
  TORCH_CHECK(device_tensor.is_cuda(), "SM75 WMMA probe requires a CUDA tensor argument");
  auto output = at::empty({8, 8}, device_tensor.options().dtype(at::kInt));
  auto stream = at::cuda::getCurrentCUDAStream();
  sm75_int4_pair_wmma_load_probe_kernel<<<1, kWarpSize, 0, stream>>>(
      output.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

template <int kPairWarps, int kPairChannels>
__global__ void sm75_int4_pair_gemm_kernel(
    const int8_t* input_int8, const uint8_t* weight_int4,
    const __half* scale_act, const __half* scale_int4,
    const uint8_t* zero_int4, const int32_t* indices_int4,
    __half* output, int rows, int width, int channels, int output_width) {
#if __CUDA_ARCH__ >= 750
  namespace precision = wmma::experimental::precision;
  constexpr int kPairRows = 32;
  constexpr int kPairRowTiles = 4;
  constexpr int kPairNSubtiles = 2;
  constexpr int kPairK = 32;
  constexpr int kPairBytes = kPairK / 2;
  __shared__ __align__(16) uint8_t a_low_packed[kPairRows][kPairBytes];
  __shared__ __align__(16) uint8_t a_high_packed[kPairRows][kPairBytes];
  __shared__ __align__(16) uint8_t b_packed[kPairChannels][kPairBytes];
  __shared__ int row_sums[kPairRowTiles][8];

  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int row_base = blockIdx.y * kPairRows;
  const int channel_base = blockIdx.x * kPairChannels;
  const int groups = width / kGroupSize;
  const int weight_bytes_per_channel = width / 2;
  // PTX m8n8k32 accumulator mapping: each lane owns two adjacent columns
  // in one row, with row = lane >> 2 and col = (lane & 3) * 2 + r.
  // Each warp owns one 8-column channel tile and iterates all four 8-row
  // subtiles so the 4-warp block covers the complete 32x32 output tile.
  float partial[kPairRowTiles][kPairNSubtiles][2] = {};

  for (int group = 0; group < groups; ++group) {
    if (warp < kPairRowTiles && lane < 8) {
      int sum = 0;
      const int row = row_base + warp * 8 + lane;
      if (row < rows) {
#pragma unroll 4
        for (int k = 0; k < kGroupSize; ++k) {
          sum += static_cast<int>(input_int8[row * width + group * kGroupSize + k]);
        }
      }
      row_sums[warp][lane] = sum;
    }
    __syncthreads();

    PairProbeLowMma::FragmentC low_accum[kPairRowTiles][kPairNSubtiles];
    PairProbeHighMma::FragmentC high_accum[kPairRowTiles][kPairNSubtiles];
    for (int row_tile = 0; row_tile < kPairRowTiles; ++row_tile) {
      for (int n_tile = 0; n_tile < kPairNSubtiles; ++n_tile) {
        low_accum[row_tile][n_tile].clear();
        high_accum[row_tile][n_tile].clear();
      }
    }

    for (int chunk = 0; chunk < kGroupSize / kPairK; ++chunk) {
      const int k_base = group * kGroupSize + chunk * kPairK;
      for (int item = threadIdx.x; item < kPairRows * kPairBytes;
           item += blockDim.x) {
        const int row = item / kPairBytes;
        const int pair = item % kPairBytes;
        const int global_row = row_base + row;
        const int k = k_base + pair * 2;
        int8_t a0 = 0;
        int8_t a1 = 0;
        if (global_row < rows) {
          a0 = input_int8[global_row * width + k];
          a1 = input_int8[global_row * width + k + 1];
        }
        const uint8_t low0 = static_cast<uint8_t>(a0) & 0x0f;
        const uint8_t low1 = static_cast<uint8_t>(a1) & 0x0f;
        const uint8_t high0 = static_cast<uint8_t>(static_cast<int>(a0) >> 4) & 0x0f;
        const uint8_t high1 = static_cast<uint8_t>(static_cast<int>(a1) >> 4) & 0x0f;
        a_low_packed[row][pair] = low0 | (low1 << 4);
        a_high_packed[row][pair] = high0 | (high1 << 4);
      }
      for (int item = threadIdx.x; item < kPairChannels * kPairBytes;
           item += blockDim.x) {
        const int local_channel = item / kPairBytes;
        const int pair = item % kPairBytes;
        const int channel = channel_base + local_channel;
        const int source = channel * weight_bytes_per_channel + k_base / 2 + pair;
        b_packed[local_channel][pair] = channel < channels ? weight_int4[source] : 0;
      }
      __syncthreads();

      for (int row_tile = 0; row_tile < kPairRowTiles; ++row_tile) {
        wmma::fragment<wmma::matrix_a, 8, 8, 32, precision::u4,
                       wmma::row_major> a_low_u4;
        wmma::fragment<wmma::matrix_a, 8, 8, 32, precision::u4,
                       wmma::row_major> a_high_u4;
        wmma::load_matrix_sync(a_low_u4, &a_low_packed[row_tile * 8][0], kPairK);
        wmma::load_matrix_sync(a_high_u4, &a_high_packed[row_tile * 8][0], kPairK);
        for (int n_tile = 0; n_tile < kPairNSubtiles; ++n_tile) {
          wmma::fragment<wmma::matrix_b, 8, 8, 32, precision::u4,
                         wmma::col_major> b_u4;
          wmma::load_matrix_sync(
              b_u4, &b_packed[warp * 16 + n_tile * 8][0], kPairK);

          PairProbeLowMma::FragmentA low_a;
          PairProbeHighMma::FragmentA high_a;
          PairProbeLowMma::FragmentB weights;
          low_a.clear();
          high_a.clear();
          weights.clear();
          reinterpret_cast<unsigned&>(low_a) = a_low_u4.x[0];
          reinterpret_cast<unsigned&>(high_a) = a_high_u4.x[0];
          reinterpret_cast<unsigned&>(weights) = b_u4.x[0];
          PairProbeLowMma low_mma;
          PairProbeHighMma high_mma;
          low_mma(low_accum[row_tile][n_tile], low_a, weights,
                  low_accum[row_tile][n_tile]);
          high_mma(high_accum[row_tile][n_tile], high_a, weights,
                   high_accum[row_tile][n_tile]);
        }
      }
      __syncthreads();
    }

    for (int row_tile = 0; row_tile < kPairRowTiles; ++row_tile) {
      const int local_row = row_tile * 8 + (lane >> 2);
      const int local_channel_base = (lane & 3) * 2;
      const int row = row_base + local_row;
      for (int n_tile = 0; n_tile < kPairNSubtiles; ++n_tile) {
        for (int register_index = 0; register_index < 2; ++register_index) {
          const int channel = channel_base + warp * 16 + n_tile * 8 +
                              local_channel_base + register_index;
          if (row < rows && channel < channels) {
            const int correction = static_cast<int>(zero_int4[channel * groups + group]) *
                                   row_sums[row_tile][lane >> 2];
            const int accumulator = low_accum[row_tile][n_tile][register_index] +
                                     16 * high_accum[row_tile][n_tile][register_index] - correction;
            const float value = static_cast<float>(accumulator) *
                __half2float(scale_act[group * rows + row]) *
                __half2float(scale_int4[channel * groups + group]);
            partial[row_tile][n_tile][register_index] += value;
          }
        }
      }
    }
    __syncthreads();
  }

  const int local_channel_base = (lane & 3) * 2;
  for (int row_tile = 0; row_tile < kPairRowTiles; ++row_tile) {
    const int row = row_base + row_tile * 8 + (lane >> 2);
    for (int n_tile = 0; n_tile < kPairNSubtiles; ++n_tile) {
      for (int register_index = 0; register_index < 2; ++register_index) {
        const int channel = channel_base + warp * 16 + n_tile * 8 +
                            local_channel_base + register_index;
        if (row < rows && channel < channels) {
              output[row * output_width + indices_int4[channel]] =
              __float2half_rn(partial[row_tile][n_tile][register_index]);
        }
      }
    }
  }
#endif
}

void run_int4_pair_partition(
    const at::Tensor& input_int8, const at::Tensor& scale_act,
    const at::Tensor& weight_int4, const at::Tensor& scale_int4,
    const at::Tensor& zero_int4, const at::Tensor& indices_int4,
    at::Tensor& output, int rows, int width, cudaStream_t stream) {
  if (indices_int4.numel() == 0) {
    return;
  }
  record_tensor_stream(input_int8, stream);
  record_tensor_stream(scale_act, stream);
  // Module-owned INT4 weights, scales, zeros, and indices remain alive for
  // the complete forward call; only dynamic activation/output tensors need
  // allocator stream recording here.
  record_tensor_stream(output, stream);
  const int channels = static_cast<int>(indices_int4.numel());
  const int output_width = static_cast<int>(output.size(1));
  const dim3 grid(
      (channels + (channels >= 128 ? 127 : 63)) /
          (channels >= 128 ? 128 : 64),
      (rows + 31) / 32);
  if (channels >= 128) {
    sm75_int4_pair_gemm_kernel<8, 128><<<grid, 256, 0, stream>>>(
        input_int8.data_ptr<int8_t>(), weight_int4.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scale_act.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(scale_int4.data_ptr<at::Half>()),
        zero_int4.data_ptr<uint8_t>(), indices_int4.data_ptr<int32_t>(),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()), rows, width,
        channels, output_width);
  } else {
    sm75_int4_pair_gemm_kernel<4, 64><<<grid, 128, 0, stream>>>(
        input_int8.data_ptr<int8_t>(), weight_int4.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scale_act.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(scale_int4.data_ptr<at::Half>()),
        zero_int4.data_ptr<uint8_t>(), indices_int4.data_ptr<int32_t>(),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()), rows, width,
        channels, output_width);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_cuda_contiguous(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(const at::Tensor& input, const at::Tensor& tensor,
                       const char* name) {
  TORCH_CHECK(tensor.device() == input.device(), name,
              " must be on the same CUDA device as input_fp16");
}

void run_cutlass_int_partition(
    at::Tensor input_int8, at::Tensor scale_act,
    at::Tensor weight, at::Tensor matrix_scale,
    at::Tensor matrix_zero, at::Tensor indices,
    at::Tensor& output, int rows, int width, cudaStream_t stream,
    cudaStream_t caller_stream) {
  if (indices.numel() == 0) {
    return;
  }
  record_tensor_stream(input_int8, stream);
  record_tensor_stream(scale_act, stream);
  // `weight` and `indices` are persistent module buffers.  The transposed
  // matrix metadata can be a temporary v2 allocation, so retain its record.
  record_tensor_stream(matrix_scale, stream);
  record_tensor_stream(matrix_zero, stream);
  record_tensor_stream(output, stream);
  CutlassConfig config = select_cutlass_config(
      input_int8.device().index(), rows, static_cast<int>(indices.numel()), width,
      input_int8, weight, scale_act, matrix_scale, matrix_zero, indices, output,
      stream, !stream_is_capturing(caller_stream));
  run_cutlass_config(
      config, rows, static_cast<int>(indices.numel()), width, input_int8, weight, scale_act,
      matrix_scale, matrix_zero, indices, output, stream);
}

void run_cutlass_packed_int4_partition(
    at::Tensor input_int8, at::Tensor scale_act,
    at::Tensor weight_int4_interleaved, at::Tensor matrix_scale,
    at::Tensor matrix_zero, at::Tensor indices,
    at::Tensor& output, int rows, int width, cudaStream_t stream) {
  if (indices.numel() == 0) {
    return;
  }
  record_tensor_stream(input_int8, stream);
  record_tensor_stream(scale_act, stream);
  // Packed weights and cached metadata are persistent module-owned buffers. They
  // remain alive for the complete forward call and do not need allocator
  // stream-recording; only dynamic activation/output tensors are recorded.
  record_tensor_stream(output, stream);
  shmq_cutlass_sm75::PackedInt4RunnerM64N64::run(
      rows, static_cast<int>(indices.numel()), width, input_int8,
      weight_int4_interleaved, scale_act, matrix_scale, matrix_zero, indices,
      output, stream);
}

void run_fp16_partition_cublas(
    const at::Tensor& input_fp16, const at::Tensor& weight_fp16,
    const at::Tensor& indices_fp16, at::Tensor& output,
    int rows, int width, cudaStream_t stream) {
  if (indices_fp16.numel() == 0) {
    return;
  }
  record_tensor_stream(input_fp16, stream);
  record_tensor_stream(output, stream);
  const auto partial = at::mm(
      input_fp16, weight_fp16.transpose(0, 1));
  const int channels = static_cast<int>(indices_fp16.numel());
  constexpr int threads = 256;
  const int blocks = (rows * channels + threads - 1) / threads;
  scatter_fp16_partition_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __half*>(partial.data_ptr<at::Half>()),
      indices_fp16.data_ptr<int32_t>(),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
      rows, channels, static_cast<int>(output.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void begin_integer_prefill_overlap(
    const at::Tensor& input_int8, const at::Tensor& scale_act,
    const at::Tensor& weight_int4, const at::Tensor& weight_int4_interleaved,
    const at::Tensor& expanded_int4, const at::Tensor& scale_int4,
    const at::Tensor& zero_int4, const at::Tensor& indices_int4,
    const at::Tensor& weight_int8, const at::Tensor& scale_int8,
    const at::Tensor& indices_int8, at::Tensor& output,
    int rows, int width, cudaStream_t caller_stream,
    IntegerPrefillStreams& streams,
    const at::Tensor* cached_scale_int4,
    const at::Tensor* cached_zero_int4,
    const at::Tensor* cached_scale_int8,
    bool use_fused_int4) {
  auto matrix_scale4 = cached_scale_int4
      ? *cached_scale_int4 : scale_int4.transpose(0, 1).contiguous();
  auto matrix_scale8 = cached_scale_int8
      ? *cached_scale_int8 : scale_int8.transpose(0, 1).contiguous();
  auto matrix_zero = cached_zero_int4
      ? *cached_zero_int4
      : (zero_int4.numel() ? zero_int4.transpose(0, 1).contiguous() : zero_int4);
  // All fallback transposes above are queued on caller_stream.  Record the fork
  // only after them so both auxiliary streams observe initialized metadata.
  C10_CUDA_CHECK(cudaEventRecord(streams.fork, caller_stream));
  if (indices_int4.numel()) {
    C10_CUDA_CHECK(cudaStreamWaitEvent(streams.int4, streams.fork, 0));
    if (use_fused_int4) {
      run_int4_pair_partition(
          input_int8, scale_act, weight_int4, scale_int4, zero_int4,
          indices_int4, output, rows, width, streams.int4);
    } else if (weight_int4_interleaved.defined() &&
               weight_int4_interleaved.numel()) {
      run_cutlass_packed_int4_partition(
          input_int8, scale_act, weight_int4_interleaved, matrix_scale4,
          matrix_zero, indices_int4, output, rows, width, streams.int4);
    } else {
      run_cutlass_int_partition(
          input_int8, scale_act, expanded_int4, matrix_scale4, matrix_zero,
          indices_int4, output, rows, width, streams.int4, caller_stream);
    }
    C10_CUDA_CHECK(cudaEventRecord(streams.done_int4, streams.int4));
  }
  if (indices_int8.numel()) {
    C10_CUDA_CHECK(cudaStreamWaitEvent(streams.int8, streams.fork, 0));
    run_cutlass_int_partition(
        input_int8, scale_act, weight_int8, matrix_scale8, matrix_zero,
        indices_int8, output, rows, width, streams.int8, caller_stream);
    C10_CUDA_CHECK(cudaEventRecord(streams.done_int8, streams.int8));
  }
}

void finish_integer_prefill_overlap(
    int n4, int n8, cudaStream_t caller_stream,
    IntegerPrefillStreams& streams) {
  if (n4) {
    C10_CUDA_CHECK(cudaStreamWaitEvent(caller_stream, streams.done_int4, 0));
  }
  if (n8) {
    C10_CUDA_CHECK(cudaStreamWaitEvent(caller_stream, streams.done_int8, 0));
  }
}

// v282 deep-module seam: one logical three-level prefill operation owns the
// integer fork/join, FP16 launch, and final completion ordering.  Precision
// arithmetic remains in the existing compile-time-specialized leaves.
struct UnifiedPrefillPlan {
  int rows;
  int width;
  int n4;
  int n8;
  int n16;

  bool has_integer() const { return n4 != 0 || n8 != 0; }
};

void run_unified_prefill(
    const UnifiedPrefillPlan& plan,
    const at::Tensor& input_fp16, const at::Tensor& input_int8,
    const at::Tensor& scale_act, const at::Tensor& weight_int4,
    const at::Tensor& weight_int4_interleaved,
    const at::Tensor& expanded_int4, const at::Tensor& scale_int4,
    const at::Tensor& zero_int4, const at::Tensor& indices_int4,
    const at::Tensor& weight_int8, const at::Tensor& scale_int8,
    const at::Tensor& indices_int8, const at::Tensor& weight_fp16,
    const at::Tensor& indices_fp16, at::Tensor& output,
    cudaStream_t caller_stream,
    const at::Tensor* cached_scale_int4,
    const at::Tensor* cached_zero_int4,
    const at::Tensor* cached_scale_int8) {
  auto& streams = integer_prefill_streams(input_fp16.device().index());
  if (plan.has_integer()) {
    begin_integer_prefill_overlap(
        input_int8, scale_act, weight_int4, weight_int4_interleaved,
        expanded_int4, scale_int4, zero_int4, indices_int4,
        weight_int8, scale_int8, indices_int8, output, plan.rows, plan.width,
        caller_stream, streams, cached_scale_int4, cached_zero_int4,
        cached_scale_int8, false);
  }
  if (plan.n16 > 0) {
    run_fp16_partition_cublas(
        input_fp16, weight_fp16, indices_fp16, output, plan.rows, plan.width,
        caller_stream);
  }
  if (plan.has_integer()) {
    finish_integer_prefill_overlap(
        plan.n4, plan.n8, caller_stream, streams);
  }
}

std::tuple<at::Tensor, at::Tensor> quantize_activation_sm75(
    const at::Tensor& input) {
  check_cuda_contiguous(input, "input");
  TORCH_CHECK(input.dim() == 2 && input.scalar_type() == at::kHalf,
              "input must be a contiguous float16 matrix");
  const int rows = input.size(0);
  const int width = input.size(1);
  TORCH_CHECK(width % kGroupSize == 0,
              "SM75 activation quantization requires K divisible by 128");
  c10::cuda::CUDAGuard device_guard(input.device());
  auto quantized = at::empty(input.sizes(), input.options().dtype(at::kChar));
  auto scales = at::empty({width / kGroupSize, rows},
                          input.options().dtype(at::kHalf));
  constexpr int threads = 128;
  constexpr int warps_per_block = threads / kWarpSize;
  const int total_groups = rows * (width / kGroupSize);
  if (total_groups == 0) {
    return std::make_tuple(quantized, scales);
  }
  const int blocks = (total_groups + warps_per_block - 1) / warps_per_block;
  quantize_activation_sm75_kernel<<<blocks, threads, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
      quantized.data_ptr<int8_t>(),
      reinterpret_cast<__half*>(scales.data_ptr<at::Half>()), rows, width);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(quantized, scales);
}

at::Tensor three_level_linear_v2_core(
    const at::Tensor& input_fp16, const at::Tensor& input_int8,
    const at::Tensor& scale_act, const at::Tensor& weight_int4,
    const at::Tensor& weight_int4_interleaved,
    const at::Tensor& expanded_int4, const at::Tensor& scale_int4,
    const at::Tensor& zero_int4,
    const at::Tensor& indices_int4, const at::Tensor& weight_int8,
    const at::Tensor& scale_int8, const at::Tensor& indices_int8,
    const at::Tensor& weight_fp16, const at::Tensor& indices_fp16,
    const at::Tensor& cached_scale_int4,
    const at::Tensor& cached_zero_int4,
    const at::Tensor& cached_scale_int8) {
  const at::Tensor* tensors[] = {&input_int8, &scale_act, &weight_int4,
      &expanded_int4, &scale_int4, &zero_int4, &indices_int4, &weight_int8,
      &scale_int8, &indices_int8, &weight_fp16, &indices_fp16};
  const char* names[] = {"input_int8", "scale_act", "weight_int4",
      "expanded_int4", "scale_int4", "zero_int4", "indices_int4",
      "weight_int8", "scale_int8", "indices_int8", "weight_fp16",
      "indices_fp16"};
  check_cuda_contiguous(input_fp16, "input_fp16");
  check_cuda_contiguous(input_int8, "input_int8");
  check_cuda_contiguous(scale_act, "scale_act");
  check_cuda_contiguous(weight_int4, "weight_int4");
  check_cuda_contiguous(expanded_int4, "expanded_int4");
  TORCH_CHECK(!weight_int4_interleaved.defined() ||
                  (weight_int4_interleaved.is_cuda() &&
                   weight_int4_interleaved.is_contiguous()),
              "weight_int4_interleaved must be a contiguous CUDA tensor when defined");
  if (weight_int4_interleaved.defined()) {
    check_same_device(input_fp16, weight_int4_interleaved,
                      "weight_int4_interleaved");
  }
  check_cuda_contiguous(scale_int4, "scale_int4");
  check_cuda_contiguous(zero_int4, "zero_int4");
  check_cuda_contiguous(indices_int4, "indices_int4");
  check_cuda_contiguous(weight_int8, "weight_int8");
  check_cuda_contiguous(scale_int8, "scale_int8");
  check_cuda_contiguous(indices_int8, "indices_int8");
  check_cuda_contiguous(weight_fp16, "weight_fp16");
  check_cuda_contiguous(indices_fp16, "indices_fp16");
  for (int i = 0; i < 12; ++i) {
    check_same_device(input_fp16, *tensors[i], names[i]);
  }
  const bool has_cached_metadata = cached_scale_int4.defined() ||
      cached_zero_int4.defined() || cached_scale_int8.defined();
  TORCH_CHECK(
      !has_cached_metadata ||
          (cached_scale_int4.defined() && cached_zero_int4.defined() &&
           cached_scale_int8.defined()),
      "CUTLASS metadata cache must provide all three tensors");
  if (has_cached_metadata) {
    check_cuda_contiguous(cached_scale_int4, "cached_scale_int4");
    check_cuda_contiguous(cached_zero_int4, "cached_zero_int4");
    check_cuda_contiguous(cached_scale_int8, "cached_scale_int8");
    check_same_device(input_fp16, cached_scale_int4, "cached_scale_int4");
    check_same_device(input_fp16, cached_zero_int4, "cached_zero_int4");
    check_same_device(input_fp16, cached_scale_int8, "cached_scale_int8");
  }
  TORCH_CHECK(input_fp16.dim() == 2 && input_fp16.scalar_type() == at::kHalf,
              "input_fp16 must be a float16 matrix");
  TORCH_CHECK(input_int8.sizes() == input_fp16.sizes() &&
                  input_int8.scalar_type() == at::kChar,
              "input_int8 must match input_fp16 and have dtype int8");
  TORCH_CHECK(scale_act.scalar_type() == at::kHalf && scale_act.dim() == 2,
              "scale_act must be a float16 matrix");
  TORCH_CHECK(weight_int4.scalar_type() == at::kByte &&
                  zero_int4.scalar_type() == at::kByte,
              "INT4 codes and zero points must be uint8");
  TORCH_CHECK(weight_int8.scalar_type() == at::kChar,
              "weight_int8 must have dtype int8");
  TORCH_CHECK(expanded_int4.scalar_type() == at::kChar,
              "expanded_int4 must have dtype int8");
  TORCH_CHECK(!weight_int4_interleaved.defined() ||
                  weight_int4_interleaved.scalar_type() == at::kByte,
              "weight_int4_interleaved must have dtype uint8 when defined");
  TORCH_CHECK(scale_int4.scalar_type() == at::kHalf &&
                  scale_int8.scalar_type() == at::kHalf &&
                  weight_fp16.scalar_type() == at::kHalf,
              "weights/scales must use the checkpoint ABI dtypes");
  TORCH_CHECK(indices_int4.scalar_type() == at::kInt &&
                  indices_int8.scalar_type() == at::kInt &&
                  indices_fp16.scalar_type() == at::kInt,
              "all channel indices must be int32");

  const int rows = input_fp16.size(0);
  const int width = input_fp16.size(1);
  const int n4 = indices_int4.numel();
  const int n8 = indices_int8.numel();
  const int n16 = indices_fp16.numel();
  const int output_width = n4 + n8 + n16;
  TORCH_CHECK(output_width > 0, "at least one precision partition is required");
  TORCH_CHECK(width % kGroupSize == 0,
              "SM75 Tensor Core backend requires K divisible by 128");
  TORCH_CHECK(scale_act.size(0) == width / kGroupSize &&
                  scale_act.size(1) == rows,
              "scale_act must have shape [K/128, rows]");
  TORCH_CHECK(weight_int4.size(0) == n4 && weight_int4.size(1) == width / 2,
              "invalid packed INT4 weight shape");
  TORCH_CHECK(rows == 1 ||
                  (expanded_int4.dim() == 2 && expanded_int4.size(0) == n4 &&
                   expanded_int4.size(1) == width),
              "expanded_int4 must have shape [n4, K] for prefill");
  TORCH_CHECK(!weight_int4_interleaved.defined() ||
                  weight_int4_interleaved.numel() == 0 ||
                  (weight_int4_interleaved.dim() == 2 &&
                   weight_int4_interleaved.size(0) == n4 &&
                   weight_int4_interleaved.size(1) == width / 2),
              "invalid interleaved INT4 prefill weight shape");
  TORCH_CHECK(weight_int8.size(0) == n8 && weight_int8.size(1) == width,
              "invalid INT8 weight shape");
  TORCH_CHECK(weight_fp16.size(0) == n16 && weight_fp16.size(1) == width,
              "invalid FP16 weight shape");
  const int groups = width / kGroupSize;
  TORCH_CHECK(scale_int4.size(0) == n4 && scale_int4.size(1) == groups &&
                  zero_int4.size(0) == n4 && zero_int4.size(1) == groups,
              "invalid INT4 metadata shape");
  TORCH_CHECK(scale_int8.size(0) == n8 && scale_int8.size(1) == groups,
              "invalid INT8 scale shape");
  if (has_cached_metadata) {
    TORCH_CHECK(cached_scale_int4.scalar_type() == at::kHalf &&
                    cached_zero_int4.scalar_type() == at::kByte &&
                    cached_scale_int8.scalar_type() == at::kHalf,
                "invalid cached CUTLASS metadata dtype");
    TORCH_CHECK(cached_scale_int4.sizes() == at::IntArrayRef({groups, n4}) &&
                    cached_zero_int4.sizes() == at::IntArrayRef({groups, n4}) &&
                    cached_scale_int8.sizes() == at::IntArrayRef({groups, n8}),
                "invalid cached CUTLASS metadata shape");
  }

  c10::cuda::CUDAGuard device_guard(input_fp16.device());
  auto output = at::empty({rows, output_width}, input_fp16.options().dtype(at::kHalf));
  if (rows == 0) {
    return output;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  if (rows == 1) {
    const int channels_per_block = kDecodeWarps * kDecodeChannelsPerWarp;
    const int blocks = (output_width + channels_per_block - 1) /
                       channels_per_block;
    three_level_decode_kernel<<<blocks, kDecodeWarps * kWarpSize, 0, stream>>>(
        reinterpret_cast<const __half*>(input_fp16.data_ptr<at::Half>()),
        input_int8.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(scale_act.data_ptr<at::Half>()),
        weight_int4.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scale_int4.data_ptr<at::Half>()),
        zero_int4.data_ptr<uint8_t>(), indices_int4.data_ptr<int32_t>(),
        weight_int8.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(scale_int8.data_ptr<at::Half>()),
        indices_int8.data_ptr<int32_t>(),
        reinterpret_cast<const __half*>(weight_fp16.data_ptr<at::Half>()),
        indices_fp16.data_ptr<int32_t>(), reinterpret_cast<__half*>(output.data_ptr<at::Half>()), width,
        output_width, n4, n8, n16);
  } else if (rows >= 32 && (n4 > 0 || n8 > 0)) {
    // The original MixLLM relies on iterator-based, staged Tensor Core GEMMs
    // for larger M. Keep the measured v188 overlap path here: the native
    // packed pair kernel is reserved for the pure-INT4 candidate because its
    // 32x32 small-tile geometry is not competitive for mixed large-M work.
    // The helper is deliberately selected only for rows>=32: the SM75
    // 16x128 CUTLASS geometry is not a valid portable small-M core, and
    // rows==16 remains on the validated direct-WMMA control path.
    const UnifiedPrefillPlan plan{rows, width, n4, n8, n16};
    run_unified_prefill(
        plan, input_fp16, input_int8, scale_act, weight_int4,
        weight_int4_interleaved, expanded_int4, scale_int4, zero_int4,
        indices_int4, weight_int8, scale_int8, indices_int8, weight_fp16,
        indices_fp16, output, stream.stream(),
        has_cached_metadata ? &cached_scale_int4 : nullptr,
        has_cached_metadata ? &cached_zero_int4 : nullptr,
        has_cached_metadata ? &cached_scale_int8 : nullptr);
  } else {
    const int channel_tiles =
        (n4 + kPrefillChannels - 1) / kPrefillChannels +
        (n8 + kPrefillChannels - 1) / kPrefillChannels +
        (n16 + kPrefillChannels - 1) / kPrefillChannels;
    const dim3 grid(channel_tiles, (rows + 15) / 16);
    three_level_tensorcore_kernel<kPrefillWarps>
        <<<grid, kPrefillWarps * kWarpSize, 0, stream>>>(
      reinterpret_cast<const __half*>(input_fp16.data_ptr<at::Half>()),
      input_int8.data_ptr<int8_t>(),
      reinterpret_cast<const __half*>(scale_act.data_ptr<at::Half>()),
      expanded_int4.data_ptr<int8_t>(),
      reinterpret_cast<const __half*>(scale_int4.data_ptr<at::Half>()),
      indices_int4.data_ptr<int32_t>(),
      weight_int8.data_ptr<int8_t>(),
      reinterpret_cast<const __half*>(scale_int8.data_ptr<at::Half>()),
      indices_int8.data_ptr<int32_t>(),
      reinterpret_cast<const __half*>(weight_fp16.data_ptr<at::Half>()),
      indices_fp16.data_ptr<int32_t>(), reinterpret_cast<__half*>(output.data_ptr<at::Half>()), rows, width,
      output_width, n4, n8, n16);
  }
  // timing_integrity: surface asynchronous launch failures before the caller records
  // a benchmark event instead of allowing a deferred CUDA error to corrupt timing.
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
void validate_partition(const at::Tensor& indices_int4,
                        const at::Tensor& indices_int8,
                        const at::Tensor& indices_fp16) {
  const int64_t output_width = indices_int4.numel() + indices_int8.numel() +
                               indices_fp16.numel();
  auto complete = at::cat({indices_int4, indices_int8, indices_fp16});
  auto expected = at::arange(output_width, indices_int4.options());
  TORCH_CHECK(std::get<0>(complete.sort()).equal(expected),
              "4/8/16 indices must form a complete output partition");
}

at::Tensor three_level_linear_v2_unchecked_cuda(
    const at::Tensor& input_fp16, const at::Tensor& input_int8,
    const at::Tensor& scale_act, const at::Tensor& weight_int4,
    const at::Tensor& expanded_int4, const at::Tensor& scale_int4,
    const at::Tensor& zero_int4,
    const at::Tensor& indices_int4, const at::Tensor& weight_int8,
    const at::Tensor& scale_int8, const at::Tensor& indices_int8,
    const at::Tensor& weight_fp16, const at::Tensor& indices_fp16) {
  return three_level_linear_v2_core(
      input_fp16, input_int8, scale_act, weight_int4, at::Tensor(), expanded_int4,
      scale_int4, zero_int4, indices_int4, weight_int8, scale_int8, indices_int8,
      weight_fp16, indices_fp16, at::Tensor(), at::Tensor(), at::Tensor());
}

at::Tensor three_level_linear_v2_cuda(
    const at::Tensor& input_fp16, const at::Tensor& input_int8,
    const at::Tensor& scale_act, const at::Tensor& weight_int4,
    const at::Tensor& expanded_int4, const at::Tensor& scale_int4,
    const at::Tensor& zero_int4,
    const at::Tensor& indices_int4, const at::Tensor& weight_int8,
    const at::Tensor& scale_int8, const at::Tensor& indices_int8,
    const at::Tensor& weight_fp16, const at::Tensor& indices_fp16) {
  validate_partition(indices_int4, indices_int8, indices_fp16);
  return three_level_linear_v2_core(
      input_fp16, input_int8, scale_act, weight_int4, at::Tensor(), expanded_int4,
      scale_int4, zero_int4, indices_int4, weight_int8, scale_int8, indices_int8,
      weight_fp16, indices_fp16, at::Tensor(), at::Tensor(), at::Tensor());
}

at::Tensor three_level_linear_v3_unchecked_cuda(
    const at::Tensor& input_fp16, const at::Tensor& input_int8,
    const at::Tensor& scale_act, const at::Tensor& weight_int4,
    const at::Tensor& weight_int4_interleaved,
    const at::Tensor& expanded_int4, const at::Tensor& scale_int4,
    const at::Tensor& zero_int4,
    const at::Tensor& indices_int4, const at::Tensor& weight_int8,
    const at::Tensor& scale_int8, const at::Tensor& indices_int8,
    const at::Tensor& weight_fp16, const at::Tensor& indices_fp16,
    const at::Tensor& cached_scale_int4,
    const at::Tensor& cached_zero_int4,
    const at::Tensor& cached_scale_int8) {
  return three_level_linear_v2_core(
      input_fp16, input_int8, scale_act, weight_int4, weight_int4_interleaved,
      expanded_int4, scale_int4,
      zero_int4, indices_int4, weight_int8, scale_int8, indices_int8,
      weight_fp16, indices_fp16, cached_scale_int4, cached_zero_int4,
      cached_scale_int8);
}

at::Tensor three_level_linear_v3_cuda(
    const at::Tensor& input_fp16, const at::Tensor& input_int8,
    const at::Tensor& scale_act, const at::Tensor& weight_int4,
    const at::Tensor& weight_int4_interleaved,
    const at::Tensor& expanded_int4, const at::Tensor& scale_int4,
    const at::Tensor& zero_int4,
    const at::Tensor& indices_int4, const at::Tensor& weight_int8,
    const at::Tensor& scale_int8, const at::Tensor& indices_int8,
    const at::Tensor& weight_fp16, const at::Tensor& indices_fp16,
    const at::Tensor& cached_scale_int4,
    const at::Tensor& cached_zero_int4,
    const at::Tensor& cached_scale_int8) {
  validate_partition(indices_int4, indices_int8, indices_fp16);
  return three_level_linear_v2_core(
      input_fp16, input_int8, scale_act, weight_int4, weight_int4_interleaved,
      expanded_int4, scale_int4,
      zero_int4, indices_int4, weight_int8, scale_int8, indices_int8,
      weight_fp16, indices_fp16, cached_scale_int4, cached_zero_int4,
      cached_scale_int8);
}

at::Tensor three_level_linear_legacy_cuda(
    const at::Tensor& input, const at::Tensor& weight_int4,
    const at::Tensor& scale_int4, const at::Tensor& zero_int4,
    const at::Tensor& indices_int4, const at::Tensor& weight_int8,
    const at::Tensor& scale_int8, const at::Tensor& indices_int8,
    const at::Tensor& weight_fp16, const at::Tensor& indices_fp16,
    int64_t group_size) {
  TORCH_CHECK(group_size == kGroupSize,
              "legacy SM75 adapter requires group_size=128");
  check_cuda_contiguous(input, "input");
  TORCH_CHECK(input.dim() == 2 &&
                  (input.scalar_type() == at::kHalf ||
                   input.scalar_type() == at::kFloat),
              "input must be a float16 or float32 matrix");
  auto input_fp16 = input.scalar_type() == at::kHalf
      ? input : input.to(at::kHalf);
  auto activation = quantize_activation_sm75(input_fp16);
  auto expanded_int4 = input_fp16.size(0) == 1
      ? weight_int8
      : at::empty({weight_int4.size(0), input_fp16.size(1)},
                  weight_int8.options());
  const int elements = expanded_int4.numel();
  if (input_fp16.size(0) > 1 && elements > 0) {
    constexpr int threads = 256;
    expand_int4_sm75_kernel<<<(elements + threads - 1) / threads, threads, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        weight_int4.data_ptr<uint8_t>(), zero_int4.data_ptr<uint8_t>(),
        expanded_int4.data_ptr<int8_t>(), weight_int4.size(0), input_fp16.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  validate_partition(indices_int4, indices_int8, indices_fp16);
  return three_level_linear_v2_core(
      input_fp16, std::get<0>(activation), std::get<1>(activation), weight_int4,
      at::Tensor(), expanded_int4, scale_int4, zero_int4, indices_int4, weight_int8,
      scale_int8, indices_int8, weight_fp16, indices_fp16,
      at::Tensor(), at::Tensor(), at::Tensor());
}

}  // namespace

TORCH_LIBRARY(mixllm_sm75, m) {
  m.def("quantize_activation(Tensor input) -> (Tensor, Tensor)");
  m.def("sm75_int4_pair_instruction_probe(Tensor device_tensor) -> Tensor");
  m.def("sm75_int4_native_decomposition_probe(Tensor device_tensor) -> Tensor");
  m.def("sm75_int4_pair_wmma_load_probe(Tensor device_tensor) -> Tensor");
  m.def("sm75_int4_pair_warp_iterator_probe(Tensor device_tensor) -> Tensor");
  m.def("sm75_int4_pair_fused_probe(Tensor device_tensor) -> Tensor");
  m.def("sm75_int4_pair_mixed_stride_probe(Tensor device_tensor) -> Tensor");
  m.def("_three_level_linear_v2_unchecked(Tensor input_fp16, Tensor input_int8, "
        "Tensor scale_act, Tensor weight_int4, Tensor expanded_int4, "
        "Tensor scale_int4, "
        "Tensor zero_int4, Tensor indices_int4, Tensor weight_int8, "
        "Tensor scale_int8, Tensor indices_int8, Tensor weight_fp16, "
        "Tensor indices_fp16) -> Tensor");
  m.def("three_level_linear_v2(Tensor input_fp16, Tensor input_int8, "
        "Tensor scale_act, Tensor weight_int4, Tensor expanded_int4, "
        "Tensor scale_int4, "
        "Tensor zero_int4, Tensor indices_int4, Tensor weight_int8, "
        "Tensor scale_int8, Tensor indices_int8, Tensor weight_fp16, "
        "Tensor indices_fp16) -> Tensor");
  m.def("_three_level_linear_v3_unchecked(Tensor input_fp16, Tensor input_int8, "
        "Tensor scale_act, Tensor weight_int4, Tensor weight_int4_interleaved, "
        "Tensor expanded_int4, "
        "Tensor scale_int4, Tensor zero_int4, Tensor indices_int4, "
        "Tensor weight_int8, Tensor scale_int8, Tensor indices_int8, "
        "Tensor weight_fp16, Tensor indices_fp16, "
        "Tensor cached_scale_int4, Tensor cached_zero_int4, "
        "Tensor cached_scale_int8) -> Tensor");
  m.def("three_level_linear_v3(Tensor input_fp16, Tensor input_int8, "
        "Tensor scale_act, Tensor weight_int4, Tensor weight_int4_interleaved, "
        "Tensor expanded_int4, "
        "Tensor scale_int4, Tensor zero_int4, Tensor indices_int4, "
        "Tensor weight_int8, Tensor scale_int8, Tensor indices_int8, "
        "Tensor weight_fp16, Tensor indices_fp16, "
        "Tensor cached_scale_int4, Tensor cached_zero_int4, "
        "Tensor cached_scale_int8) -> Tensor");
  m.def("three_level_linear(Tensor input, Tensor weight_int4, Tensor scale_int4, "
        "Tensor zero_int4, Tensor indices_int4, Tensor weight_int8, "
        "Tensor scale_int8, Tensor indices_int8, Tensor weight_fp16, "
        "Tensor indices_fp16, int group_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(mixllm_sm75, CUDA, m) {
  m.impl("quantize_activation", &quantize_activation_sm75);
  m.impl("sm75_int4_pair_instruction_probe", &sm75_int4_pair_instruction_probe_cuda);
  m.impl("sm75_int4_native_decomposition_probe", &sm75_int4_native_decomposition_probe_cuda);
  m.impl("sm75_int4_pair_wmma_load_probe", &sm75_int4_pair_wmma_load_probe_cuda);
  m.impl("sm75_int4_pair_warp_iterator_probe", &sm75_int4_pair_warp_iterator_probe_cuda);
  m.impl("sm75_int4_pair_fused_probe", &sm75_int4_pair_fused_probe_cuda);
  m.impl("sm75_int4_pair_mixed_stride_probe", &sm75_int4_pair_mixed_stride_probe_cuda);
  m.impl("_three_level_linear_v2_unchecked", &three_level_linear_v2_unchecked_cuda);
  m.impl("three_level_linear_v2", &three_level_linear_v2_cuda);
  m.impl("_three_level_linear_v3_unchecked", &three_level_linear_v3_unchecked_cuda);
  m.impl("three_level_linear_v3", &three_level_linear_v3_cuda);
  m.impl("three_level_linear", &three_level_linear_legacy_cuda);
}

