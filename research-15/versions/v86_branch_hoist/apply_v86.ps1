$p = 'C:\Users\User\Downloads\Multi-main\Multi-main\shmq-ultimate\external\MixLLM\mixllm\kernels\three_level_sm75.cu'
$s = Get-Content -Raw -LiteralPath $p
$marker = '  float scaled_accumulators[kTile * kTile / kWarpSize] = {};'
$start = $s.IndexOf($marker)
if ($start -lt 0) { throw 'v86 marker not found' }
$endMarker = '  for (int linear = lane; linear < kTile * kTile;'
$end = $s.IndexOf($endMarker, $start)
if ($end -lt 0) { throw 'v86 end marker not found' }
$head = $s.Substring(0, $start)
$tail = $s.Substring($end)
$body = @'
  float scaled_accumulators[kTile * kTile / kWarpSize] = {};
  const int groups = width / kGroupSize;
  const int partition_size = precision == 4 ? n4 : n8;
  const bool full_rows = row_base + kTile <= rows;
  const bool full_channels = channel_base + kTile <= partition_size;
  if (precision == 4 || precision == 8) {
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
          scaled_accumulators[item] +=
              static_cast<float>(accumulator_int[warp][linear]) *
              activation_scale * __half2float(weight_scale);
        }
      }
    }
  }
'@
Set-Content -LiteralPath $p -Value ($head + $body + $tail)
Write-Output 'v86_branch_hoist_applied'
