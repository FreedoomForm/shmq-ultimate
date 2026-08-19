$src = 'C:\Users\User\Downloads\Multi-main\Multi-main\shmq-ultimate\external\MixLLM\mixllm\kernels\three_level_sm75.cu'
$raw = Get-Content -LiteralPath $src -Raw
if ($raw.Contains('wmma::fragment<wmma::matrix_a, 8, 32, kTile, signed char>')) { throw 'v102_already_present' }
if (-not $raw.Contains('const int row_base = blockIdx.y * kTile;')) { throw 'row_base_marker' }
$raw = $raw.Replace('const int row_base = blockIdx.y * kTile;', 'int row_base = blockIdx.y * kTile;')
$oldShared = '__shared__ __align__(16) int8_t b_int8[kPrefillWarps][kTile * kTile];'
if (-not $raw.Contains($oldShared)) { throw 'b_shared_marker' }
$raw = $raw.Replace($oldShared, '__shared__ __align__(16) int8_t b_int8[kPrefillWarps][32 * kTile];')
$start = $raw.IndexOf('  float scaled_accumulators[kTile * kTile / kWarpSize] = {};')
$end = $raw.IndexOf('  return;', $start)
if ($start -lt 0 -or $end -lt 0) { throw 'int_path_bounds' }
$pre = $raw.Substring(0, $start)
$mid = $raw.Substring($start, $end - $start)
$post = $raw.Substring($end)
$mid = $mid.Replace('  float scaled_accumulators[kTile * kTile / kWarpSize] = {};', '  row_base += (warp / 2) * 8;`r`n  channel_base = (tile_id - (precision == 4 ? 0 : tiles4)) * kPrefillChannels + (warp % 2) * 32;`r`n  float scaled_accumulators[kTile * kTile / kWarpSize] = {};')
$mid = $mid.Replace('wmma::fragment<wmma::accumulator, kTile, kTile, kTile, int>', 'wmma::fragment<wmma::accumulator, 8, 32, kTile, int>')
$mid = $mid.Replace('wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, signed char>', 'wmma::fragment<wmma::matrix_a, 8, 32, kTile, signed char>')
$mid = $mid.Replace('wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, signed char>', 'wmma::fragment<wmma::matrix_b, 8, 32, kTile, signed char>')
$mid = $mid.Replace('const bool full_rows = row_base + kTile <= rows;', 'const bool full_rows = row_base + 8 <= rows;')
$mid = $mid.Replace('const bool full_channels = channel_base + kTile <= partition_size;', 'const bool full_channels = channel_base + 32 <= partition_size;')
$mid = $mid.Replace('for (int linear = threadIdx.x; linear < kTile * kTile;', 'for (int linear = threadIdx.x; linear < 8 * kTile;')
$newPartial = @'
        for (int linear = lane; linear < 32 * kTile;
             linear += kWarpSize) {
          const int channel_offset = linear / kTile;
          const int k_offset = linear % kTile;
          const int channel = channel_base + channel_offset;
          int8_t weight = 0;
          if (channel < partition_size) {
            const int k = k_base + k_offset;
            weight = precision == 4
                ? expanded_int4[channel * width + k]
                : weight_int8[channel * width + k];
          }
          b_int8[warp][channel_offset * kTile + k_offset] = weight;
        }
'@
$pattern = '(?ms)        for \(int linear = lane; linear < kTile \* kTile;.*?        }\r?\n        __syncwarp\(\);'
$mid2 = [regex]::Replace($mid, $pattern, ($newPartial.TrimEnd() + "`r`n        __syncwarp();"), 1)
if ($mid2 -eq $mid) { throw 'partial_loop_regex_marker' }
$mid = $mid2
$mid = $mid.Replace('wmma::store_matrix_sync(accumulator_int[warp], accumulator, kTile,', 'wmma::store_matrix_sync(accumulator_int[warp], accumulator, 32,')
$outPattern = '(?ms)    for \(int linear = lane, item = 0; linear < kTile \* kTile;\r?\n         linear \+= kWarpSize, \+\+item\) \{\r?\n      const int tile_row = linear / kTile;\r?\n      const int local_channel = channel_base \+ linear % kTile;'
$outReplacement = "    for (int linear = lane, item = 0; linear < 8 * 32;`r`n         linear += kWarpSize, ++item) {`r`n      const int tile_row = linear / 32;`r`n      const int local_channel = channel_base + linear % 32;"
$mid2 = [regex]::Replace($mid, $outPattern, $outReplacement, 1)
if ($mid2 -eq $mid) { throw 'output_loop_regex_marker' }
$mid = $mid2
[IO.File]::WriteAllText($src, $pre + $mid + $post, (New-Object Text.UTF8Encoding($false)))
Write-Output 'v102_m8n32_applied'
