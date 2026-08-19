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
  uint32_t packed = 0;
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    int value = __float2int_rn(values[item] / scale);
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

// SM75 has no Ampere mixed INT8 x INT4 MMA. Prefill reads a cached signed INT8
// expansion while decode continues to consume the packed checkpoint weights.
__global__ void three_level_tensorcore_kernel(
    const __half* input_fp16, const int8_t* input_int8,
    const __half* scale_act, const int8_t* expanded_int4,
    const __half* scale_int4,
    const int32_t* indices_int4, const int8_t* weight_int8,
    const __half* scale_int8, const int32_t* indices_int8,
    const __half* weight_fp16, const int32_t* indices_fp16, float* output,
    int rows, int width, int output_width, int n4, int n8, int n16) {
#if __CUDA_ARCH__ >= 750
  const int tiles4 = (n4 + kPrefillChannels - 1) / kPrefillChannels;
  const int tiles8 = (n8 + kPrefillChannels - 1) / kPrefillChannels;
  const int tile_id = blockIdx.x;
  const int row_base = blockIdx.y * kTile;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  int precision;
  int channel_base;
  if (tile_id < tiles4) {
    precision = 4;
    channel_base = tile_id * kPrefillChannels + warp * kTile;
  } else if (tile_id < tiles4 + tiles8) {
    precision = 8;
    channel_base = (tile_id - tiles4) * kPrefillChannels + warp * kTile;
  } else {
    precision = 16;
    channel_base = (tile_id - tiles4 - tiles8) * kPrefillChannels +
                   warp * kTile;
  }

  __shared__ __align__(16) int8_t a_int8[kTile * kTile];
  __shared__ __align__(16) int8_t b_int8[kPrefillWarps][kTile * kTile];
  __shared__ __align__(16) int accumulator_int[kPrefillWarps][kTile * kTile];
  __shared__ __align__(16) __half a_fp16[kTile * kTile];
  __shared__ __align__(16) __half b_fp16[kPrefillWarps][kTile * kTile];
  __shared__ __align__(16) float accumulator_fp32[kPrefillWarps][kTile * kTile];

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
            accumulator_fp32[warp][linear];
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
      output[row * output_width + output_channel] = scaled_accumulators[item];
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
    const __half* weight_fp16, const int32_t* indices_fp16, float* output,
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
            accumulator_fp32[warp][linear];
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
            output[r * output_width + (precision == 4 ? indices_int4[c] : indices_int8[c])] = scaled[linear / kWarpSize];
  }
#endif
}

// Wide direct prefill variant. Four warps form a 1x4 grid, and each warp
// computes two adjacent 16x16 WMMA channel subtiles. This keeps the exact
// signed-INT8 arithmetic of the validated kernel while halving CTA count and
// reusing the activation tile across twice as many output channels.
__global__ void three_level_tensorcore_int_n128_kernel(
    const int8_t* input_int8, const __half* scale_act,
    const int8_t* expanded_int4, const __half* scale_int4,
    const int32_t* indices_int4, const int8_t* weight_int8,
    const __half* scale_int8, const int32_t* indices_int8, float* output,
    int rows, int width, int output_width, int n4, int n8) {
#if __CUDA_ARCH__ >= 750
  constexpr int kWideChannels = 128;
  constexpr int kSubtiles = 2;
  const int tile_id = blockIdx.x;
  const int row_base = blockIdx.y * kTile;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int tiles4 = (n4 + kWideChannels - 1) / kWideChannels;
  const int tiles8 = (n8 + kWideChannels - 1) / kWideChannels;
  const bool is_int4 = tile_id < tiles4;
  const int precision_tile = is_int4 ? tile_id : tile_id - tiles4;
  const int partition_size = is_int4 ? n4 : n8;
  const int channel_base = precision_tile * kWideChannels + warp * 32;
  const int groups = width / kGroupSize;

  __shared__ __align__(16) int8_t a_int8[kTile * kTile];
  __shared__ __align__(16) int8_t b_int8[kPrefillWarps][kSubtiles][kTile * kTile];
  __shared__ __align__(16) int accumulator_int[
      kPrefillWarps][kSubtiles][kTile * kTile];
  float scaled_accumulators[kSubtiles * kTile * kTile / kWarpSize] = {};

  for (int group = 0; group < groups; ++group) {
    wmma::fragment<wmma::accumulator, kTile, kTile, kTile, int> accumulator[kSubtiles];
    for (int subtile = 0; subtile < kSubtiles; ++subtile) {
      wmma::fill_fragment(accumulator[subtile], 0);
    }
    for (int group_k = 0; group_k < kGroupSize; group_k += kTile) {
      const int k_base = group * kGroupSize + group_k;
      for (int linear = threadIdx.x; linear < kTile * kTile;
           linear += blockDim.x) {
        const int tile_row = linear / kTile;
        const int tile_col = linear % kTile;
        const int row = row_base + tile_row;
        a_int8[linear] = row < rows
            ? input_int8[row * width + k_base + tile_col]
            : int8_t{0};
      }
      for (int subtile = 0; subtile < kSubtiles; ++subtile) {
        for (int linear = lane; linear < kTile * kTile;
             linear += kWarpSize) {
          const int tile_row = linear / kTile;
          const int tile_col = linear % kTile;
          const int channel = channel_base + subtile * kTile + tile_col;
          int8_t weight = 0;
          if (channel < partition_size) {
            const int k = k_base + tile_row;
            weight = is_int4
                ? expanded_int4[channel * width + k]
                : weight_int8[channel * width + k];
          }
          b_int8[warp][subtile][tile_col * kTile + tile_row] = weight;
        }
      }
      __syncthreads();
      wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, signed char,
                     wmma::row_major> a;
      wmma::load_matrix_sync(a, reinterpret_cast<signed char*>(a_int8), kTile);
      for (int subtile = 0; subtile < kSubtiles; ++subtile) {
        wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, signed char,
                       wmma::col_major> b;
        wmma::load_matrix_sync(
            b, reinterpret_cast<signed char*>(b_int8[warp][subtile]), kTile);
        wmma::mma_sync(accumulator[subtile], a, b, accumulator[subtile]);
      }
      __syncthreads();
    }
    for (int subtile = 0; subtile < kSubtiles; ++subtile) {
      wmma::store_matrix_sync(
          accumulator_int[warp][subtile], accumulator[subtile], kTile,
          wmma::mem_row_major);
    }
    __syncwarp();
    for (int subtile = 0; subtile < kSubtiles; ++subtile) {
      for (int linear = lane, item = subtile * (kTile * kTile / kWarpSize);
           linear < kTile * kTile; linear += kWarpSize, ++item) {
        const int tile_row = linear / kTile;
        const int local_channel = channel_base + subtile * kTile + linear % kTile;
        if (row_base + tile_row < rows && local_channel < partition_size) {
          const float activation_scale =
              __half2float(scale_act[group * rows + row_base + tile_row]);
          const __half weight_scale = is_int4
              ? scale_int4[local_channel * groups + group]
              : scale_int8[local_channel * groups + group];
          scaled_accumulators[item] +=
              static_cast<float>(accumulator_int[warp][subtile][linear]) *
              activation_scale * __half2float(weight_scale);
        }
      }
    }
    __syncwarp();
  }

  for (int subtile = 0; subtile < kSubtiles; ++subtile) {
    for (int linear = lane, item = subtile * (kTile * kTile / kWarpSize);
         linear < kTile * kTile; linear += kWarpSize, ++item) {
      const int row = row_base + linear / kTile;
      const int local_channel = channel_base + subtile * kTile + linear % kTile;
      if (row < rows && local_channel < partition_size) {
        const int output_channel = is_int4
            ? indices_int4[local_channel]
            : indices_int8[local_channel];
        output[row * output_width + output_channel] = scaled_accumulators[item];
      }
    }
  }
#endif
}

// DP4A integer prefill kernel. Each thread computes one output element using
// the same signed-byte dot products as decode, while retaining every group
// scale and every output index.
__global__ void three_level_dp4a_prefill_kernel(
    const int8_t* input_int8, const __half* scale_act,
    const int8_t* expanded_int4, const __half* scale_int4,
    const int32_t* indices_int4, const int8_t* weight_int8,
    const __half* scale_int8, const int32_t* indices_int8, float* output,
    int rows, int width, int output_width, int n4, int n8) {
#if __CUDA_ARCH__ >= 750
  const int total_channels = n4 + n8;
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * total_channels;
  if (linear >= total) {
    return;
  }
  const int row = linear / total_channels;
  const int channel = linear % total_channels;
  const bool is_int4 = channel < n4;
  const int local_channel = is_int4 ? channel : channel - n4;
  const int groups = width / kGroupSize;
  const int8_t* weights = is_int4
      ? expanded_int4 + local_channel * width
      : weight_int8 + local_channel * width;
  float result = 0.0f;
  for (int group = 0; group < groups; ++group) {
    int accumulator = 0;
    const int base = group * kGroupSize;
#pragma unroll
    for (int offset = 0; offset < kGroupSize; offset += 4) {
      const int activations = *reinterpret_cast<const int*>(
          input_int8 + row * width + base + offset);
      const int packed_weights = *reinterpret_cast<const int*>(weights + base + offset);
      accumulator = __dp4a(activations, packed_weights, accumulator);
    }
    const float activation_scale =
        __half2float(scale_act[group * rows + row]);
    const float weight_scale = __half2float(is_int4
        ? scale_int4[local_channel * groups + group]
        : scale_int8[local_channel * groups + group]);
    result += static_cast<float>(accumulator) * activation_scale * weight_scale;
  }
  const int output_channel = is_int4
      ? indices_int4[local_channel] : indices_int8[local_channel];
  output[row * output_width + output_channel] = result;
#endif
}

// Barrier-free direct WMMA integer kernel. Each warp owns its A and B tile,
// so the integer prefill path needs only warp-local synchronization. This
// avoids CTA-wide barriers without changing the original 16x64 decomposition.
__global__ void three_level_tensorcore_local_kernel(
    const int8_t* input_int8, const __half* scale_act,
    const int8_t* expanded_int4, const __half* scale_int4,
    const int32_t* indices_int4, const int8_t* weight_int8,
    const __half* scale_int8, const int32_t* indices_int8, float* output,
    int rows, int width, int output_width, int n4, int n8) {
#if __CUDA_ARCH__ >= 750
  const int tile_id = blockIdx.x;
  const int row_base = blockIdx.y * kTile;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int tiles4 = (n4 + kPrefillChannels - 1) / kPrefillChannels;
  const int precision_tile = tile_id < tiles4 ? tile_id : tile_id - tiles4;
  const bool is_int4 = tile_id < tiles4;
  const int partition_size = is_int4 ? n4 : n8;
  const int channel_base = precision_tile * kPrefillChannels + warp * kTile;
  const int groups = width / kGroupSize;
  __shared__ __align__(16) int8_t a_int8[kPrefillWarps][kTile * kTile];
  __shared__ __align__(16) int8_t b_int8[kPrefillWarps][kTile * kTile];
  __shared__ __align__(16) int accumulator_int[kPrefillWarps][kTile * kTile];
  float scaled[kTile * kTile / kWarpSize] = {};

  for (int group = 0; group < groups; ++group) {
    wmma::fragment<wmma::accumulator, kTile, kTile, kTile, int> accumulator;
    wmma::fill_fragment(accumulator, 0);
    for (int group_k = 0; group_k < kGroupSize; group_k += kTile) {
      const int k_base = group * kGroupSize + group_k;
      for (int linear = lane; linear < kTile * kTile;
           linear += kWarpSize) {
        const int tile_row = linear / kTile;
        const int tile_col = linear % kTile;
        const int row = row_base + tile_row;
        const int channel = channel_base + tile_col;
        a_int8[warp][linear] = row < rows
            ? input_int8[row * width + k_base + tile_col] : int8_t{0};
        b_int8[warp][tile_col * kTile + tile_row] = channel < partition_size
            ? (is_int4 ? expanded_int4[channel * width + k_base + tile_row]
                       : weight_int8[channel * width + k_base + tile_row])
            : int8_t{0};
      }
      __syncwarp();
      wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, signed char,
                     wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, signed char,
                     wmma::col_major> b;
      wmma::load_matrix_sync(a, reinterpret_cast<signed char*>(a_int8[warp]), kTile);
      wmma::load_matrix_sync(b, reinterpret_cast<signed char*>(b_int8[warp]), kTile);
      wmma::mma_sync(accumulator, a, b, accumulator);
      __syncwarp();
    }
    wmma::store_matrix_sync(accumulator_int[warp], accumulator, kTile,
                            wmma::mem_row_major);
    __syncwarp();
    for (int linear = lane; linear < kTile * kTile; linear += kWarpSize) {
      const int row = row_base + linear / kTile;
      const int channel = channel_base + linear % kTile;
      if (row < rows && channel < partition_size) {
        const float activation_scale =
            __half2float(scale_act[group * rows + row]);
        const float weight_scale = __half2float(is_int4
            ? scale_int4[channel * groups + group]
            : scale_int8[channel * groups + group]);
        scaled[linear / kWarpSize] +=
            static_cast<float>(accumulator_int[warp][linear]) *
            activation_scale * weight_scale;
      }
    }
    __syncwarp();
  }
  for (int linear = lane; linear < kTile * kTile; linear += kWarpSize) {
    const int row = row_base + linear / kTile;
    const int channel = channel_base + linear % kTile;
    if (row < rows && channel < partition_size) {
      output[row * output_width +
             (is_int4 ? indices_int4[channel] : indices_int8[channel])] =
          scaled[linear / kWarpSize];
    }
  }
#endif
}

#if 0
// Balanced prefill variant. Eight warps form a 2x4 grid of 16x16 WMMA tiles.

// Each 32x16 A panel and 16x64 B panel is loaded once per CTA/K step.
__global__ void three_level_tensorcore_reuse_kernel(
    const __half* input_fp16, const int8_t* input_int8,
    const __half* scale_act, const int8_t* expanded_int4,
    const __half* scale_int4,
    const int32_t* indices_int4, const int8_t* weight_int8,
    const __half* scale_int8, const int32_t* indices_int8,
    const __half* weight_fp16, const int32_t* indices_fp16, float* output,
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
            accumulator_fp32[warp][linear];
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
      output[r * output_width +
             (precision == 4 ? indices_int4[c] : indices_int8[c])] =
          scaled[linear / kWarpSize];
  }
#endif
}

#endif

// Decode is bandwidth-bound and WMMA would compute 15 unused rows for M=1.

// Four 8-lane subwarps share one warp; each subwarp owns one output channel and
// performs four dp4a operations per 128-element activation group. This avoids
// assigning a full warp and five reduction steps to every scalar output.
__global__ void three_level_decode_kernel(
    const __half* input_fp16, const int8_t* input_int8,
    const __half* scale_act, const int8_t* expanded_int4,
    const __half* scale_int4, const uint8_t* zero_int4,
    const int32_t* indices_int4, const int8_t* weight_int8,
    const __half* scale_int8, const int32_t* indices_int8,
    const __half* weight_fp16, const int32_t* indices_fp16, float* output,
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
    const int8_t* weights = expanded_int4 + channel * width;
    for (int group = 0; group < groups; ++group) {
      const int base = group * kGroupSize;
      const int zero = static_cast<int>(zero_int4[channel * groups + group]);
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
    output[output_channel] = result;
  }
#endif
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

void run_cutlass_int_partition(
    const at::Tensor& input_int8, const at::Tensor& scale_act,
    const at::Tensor& weight, const at::Tensor& scale,
    const at::Tensor& zero, const at::Tensor& indices,
    at::Tensor& output, int rows, int width, cudaStream_t stream) {
  if (indices.numel() == 0) {
    return;
  }
  auto matrix_A = input_int8;
  auto matrix_B = weight;
  auto matrix_scale_act = scale_act;
  // CUTLASS receives metadata in its own transposed layout when dispatched.
  auto matrix_scale = scale.transpose(0, 1).contiguous();
  auto matrix_zero = zero.numel() ? zero.transpose(0, 1).contiguous() : zero;
  // These tensors are temporary PyTorch allocations. When the launch uses a
  // pooled auxiliary stream, explicitly register their storage with that
  // stream so the caching allocator cannot recycle it before CUTLASS finishes.
  auto allocation_stream = at::cuda::getStreamFromExternal(
      stream, matrix_A.device().index());
  c10::cuda::CUDACachingAllocator::recordStream(
      matrix_scale.storage().data_ptr(), allocation_stream);
  if (matrix_zero.numel()) {
    c10::cuda::CUDACachingAllocator::recordStream(
        matrix_zero.storage().data_ptr(), allocation_stream);
  }
  auto matrix_indices = indices;
  shmq_cutlass_sm75::Int8Runner::run(
      rows, static_cast<int>(indices.numel()), width,
      matrix_A, matrix_B, matrix_scale_act, matrix_scale,
      matrix_zero, matrix_indices, output, stream);
}

at::Tensor three_level_linear_v2_core(

    const at::Tensor& input_fp16, const at::Tensor& input_int8,
    const at::Tensor& scale_act, const at::Tensor& weight_int4,
    const at::Tensor& expanded_int4,
    const at::Tensor& scale_int4,
    const at::Tensor& zero_int4,
    const at::Tensor& indices_int4, const at::Tensor& weight_int8,
    const at::Tensor& scale_int8, const at::Tensor& indices_int8,
    const at::Tensor& weight_fp16, const at::Tensor& indices_fp16) {
  const at::Tensor* tensors[] = {&input_int8, &scale_act, &weight_int4,
      &expanded_int4, &scale_int4, &zero_int4,
      &indices_int4, &weight_int8, &scale_int8, &indices_int8,
      &weight_fp16, &indices_fp16};
  const char* names[] = {"input_int8", "scale_act", "weight_int4",
      "expanded_int4", "scale_int4", "zero_int4",
      "indices_int4", "weight_int8", "scale_int8", "indices_int8",
      "weight_fp16", "indices_fp16"};

  check_cuda_contiguous(input_fp16, "input_fp16");
  check_cuda_contiguous(input_int8, "input_int8");
  check_cuda_contiguous(scale_act, "scale_act");
  check_cuda_contiguous(weight_int4, "weight_int4");
  check_cuda_contiguous(expanded_int4, "expanded_int4");
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
              "expanded INT4 tensors must have shape [n4, K] for prefill");

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

  c10::cuda::CUDAGuard device_guard(input_fp16.device());
  auto output = at::empty({rows, output_width}, input_fp16.options().dtype(at::kFloat));
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
        expanded_int4.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(scale_int4.data_ptr<at::Half>()),
        zero_int4.data_ptr<uint8_t>(), indices_int4.data_ptr<int32_t>(),
        weight_int8.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(scale_int8.data_ptr<at::Half>()),
        indices_int8.data_ptr<int32_t>(),
        reinterpret_cast<const __half*>(weight_fp16.data_ptr<at::Half>()),
        indices_fp16.data_ptr<int32_t>(), output.data_ptr<float>(), width,
        output_width, n4, n8, n16);
    } else if (n4 > 0 || n8 > 0) {
      const int channel_tiles =
          (n4 + kPrefillChannels - 1) / kPrefillChannels +
          (n8 + kPrefillChannels - 1) / kPrefillChannels +
          (n16 + kPrefillChannels - 1) / kPrefillChannels;
      const dim3 grid(channel_tiles, (rows + 15) / 16);
      three_level_tensorcore_kernel<<<grid, kPrefillWarps * kWarpSize, 0, stream>>>(
          reinterpret_cast<const __half*>(input_fp16.data_ptr<at::Half>()),
          input_int8.data_ptr<int8_t>(),
          reinterpret_cast<const __half*>(scale_act.data_ptr<at::Half>()),
          expanded_int4.data_ptr<int8_t>(),
          reinterpret_cast<const __half*>(scale_int4.data_ptr<at::Half>()),
          indices_int4.data_ptr<int32_t>(), weight_int8.data_ptr<int8_t>(),
          reinterpret_cast<const __half*>(scale_int8.data_ptr<at::Half>()),
          indices_int8.data_ptr<int32_t>(),
          reinterpret_cast<const __half*>(weight_fp16.data_ptr<at::Half>()),
          indices_fp16.data_ptr<int32_t>(), output.data_ptr<float>(), rows, width,
          output_width, n4, n8, n16);
    } else if (width % 64 == 0 && (n4 > 0 || n8 > 0)) {
      cudaEvent_t fork_event;
      cudaEvent_t done_int4;
      cudaEvent_t done_int8;
      C10_CUDA_CHECK(cudaEventCreateWithFlags(&fork_event, cudaEventDisableTiming));
      C10_CUDA_CHECK(cudaEventCreateWithFlags(&done_int4, cudaEventDisableTiming));
      C10_CUDA_CHECK(cudaEventCreateWithFlags(&done_int8, cudaEventDisableTiming));
      C10_CUDA_CHECK(cudaEventRecord(fork_event, stream.stream()));
      auto stream_int4 = at::cuda::getStreamFromPool(false, input_fp16.device().index());
      auto stream_int8 = at::cuda::getStreamFromPool(false, input_fp16.device().index());
      if (n4 > 0) {
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream_int4.stream(), fork_event, 0));
        run_cutlass_int_partition(
            input_int8, scale_act, expanded_int4, scale_int4, zero_int4,
            indices_int4, output, rows, width, stream_int4.stream());
        C10_CUDA_CHECK(cudaEventRecord(done_int4, stream_int4.stream()));
      }
      if (n8 > 0) {
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream_int8.stream(), fork_event, 0));
        run_cutlass_int_partition(
            input_int8, scale_act, weight_int8, scale_int8, zero_int4,
            indices_int8, output, rows, width, stream_int8.stream());
        C10_CUDA_CHECK(cudaEventRecord(done_int8, stream_int8.stream()));
      }
      if (n16 > 0) {
        const dim3 grid_fp16(
            (n16 + kPrefillChannels - 1) / kPrefillChannels,
            (rows + 15) / 16);
        three_level_tensorcore_kernel<<<grid_fp16, kPrefillWarps * kWarpSize, 0,
                                       stream>>>(
            reinterpret_cast<const __half*>(input_fp16.data_ptr<at::Half>()),
            input_int8.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(scale_act.data_ptr<at::Half>()),
            expanded_int4.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(scale_int4.data_ptr<at::Half>()),
            indices_int4.data_ptr<int32_t>(), weight_int8.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(scale_int8.data_ptr<at::Half>()),
            indices_int8.data_ptr<int32_t>(),
            reinterpret_cast<const __half*>(weight_fp16.data_ptr<at::Half>()),
            indices_fp16.data_ptr<int32_t>(), output.data_ptr<float>(), rows, width,
            output_width, 0, 0, n16);
      }
      if (n4 > 0) {
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(), done_int4, 0));
      }
      if (n8 > 0) {
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(), done_int8, 0));
      }
      C10_CUDA_CHECK(cudaEventDestroy(fork_event));
      C10_CUDA_CHECK(cudaEventDestroy(done_int4));
      C10_CUDA_CHECK(cudaEventDestroy(done_int8));
    } else {
      const int channel_tiles =
          (n4 + kPrefillChannels - 1) / kPrefillChannels +
          (n8 + kPrefillChannels - 1) / kPrefillChannels +
          (n16 + kPrefillChannels - 1) / kPrefillChannels;
      const dim3 grid(channel_tiles, (rows + kReuseRows - 1) / kReuseRows);
      three_level_tensorcore_reuse_kernel<<<grid, 8 * kWarpSize, 0, stream>>>(
          reinterpret_cast<const __half*>(input_fp16.data_ptr<at::Half>()),
          input_int8.data_ptr<int8_t>(),
          reinterpret_cast<const __half*>(scale_act.data_ptr<at::Half>()),
          expanded_int4.data_ptr<int8_t>(),
          reinterpret_cast<const __half*>(scale_int4.data_ptr<at::Half>()),
          indices_int4.data_ptr<int32_t>(), weight_int8.data_ptr<int8_t>(),
          reinterpret_cast<const __half*>(scale_int8.data_ptr<at::Half>()),
          indices_int8.data_ptr<int32_t>(),
          reinterpret_cast<const __half*>(weight_fp16.data_ptr<at::Half>()),
          indices_fp16.data_ptr<int32_t>(), output.data_ptr<float>(), rows, width,
          output_width, n4, n8, n16);
    }

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

at::Tensor three_level_linear_v2_cuda(
    const at::Tensor& input_fp16, const at::Tensor& input_int8,
    const at::Tensor& scale_act, const at::Tensor& weight_int4,
    const at::Tensor& expanded_int4,
    const at::Tensor& scale_int4,
    const at::Tensor& zero_int4,
    const at::Tensor& indices_int4, const at::Tensor& weight_int8,
    const at::Tensor& scale_int8, const at::Tensor& indices_int8,
    const at::Tensor& weight_fp16, const at::Tensor& indices_fp16) {
  validate_partition(indices_int4, indices_int8, indices_fp16);
  return three_level_linear_v2_core(
      input_fp16, input_int8, scale_act, weight_int4, expanded_int4,
      scale_int4, zero_int4, indices_int4,
      weight_int8, scale_int8, indices_int8, weight_fp16, indices_fp16);

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
      expanded_int4, scale_int4, zero_int4, indices_int4,
      weight_int8, scale_int8, indices_int8, weight_fp16, indices_fp16);


}

}  // namespace

TORCH_LIBRARY(mixllm_sm75, m) {
  m.def("quantize_activation(Tensor input) -> (Tensor, Tensor)");
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
  m.def("three_level_linear(Tensor input, Tensor weight_int4, Tensor scale_int4, "
        "Tensor zero_int4, Tensor indices_int4, Tensor weight_int8, "
        "Tensor scale_int8, Tensor indices_int8, Tensor weight_fp16, "
        "Tensor indices_fp16, int group_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(mixllm_sm75, CUDA, m) {
  m.impl("quantize_activation", &quantize_activation_sm75);
  m.impl("_three_level_linear_v2_unchecked", &three_level_linear_v2_core);
  m.impl("three_level_linear_v2", &three_level_linear_v2_cuda);
  m.impl("three_level_linear", &three_level_linear_legacy_cuda);
}

