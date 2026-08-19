$p = 'C:\Users\User\Downloads\Multi-main\Multi-main\shmq-ultimate\external\MixLLM\mixllm\kernels\three_level_sm75.cu'
$raw = Get-Content -LiteralPath $p -Raw
$startMarker = '      for (int linear = (warp % 2) * kWarpSize + lane;'
$endMarker = '      if (precision == 8 && full_channels) {'
$start = $raw.IndexOf($startMarker)
$end = $raw.IndexOf($endMarker)
if ($start -lt 0 -or $end -le $start) { throw 'v1027_a_stage_markers' }
$replacement = @'
      if (full_rows) {
        wmma::load_matrix_sync(
            a, reinterpret_cast<const signed char*>(
                   input_int8 + row_base * width + k_base), width);
      } else {
        for (int linear = (warp % 2) * kWarpSize + lane;
             linear < 8 * kTile; linear += 2 * kWarpSize) {
          const int tile_row = linear / kTile;
          const int tile_col = linear % kTile;
          const int row = row_base + tile_row;
          a_int8[warp / 2][linear] = row < rows
              ? input_int8[row * width + k_base + tile_col]
              : int8_t{0};
        }
        __syncthreads();
        wmma::load_matrix_sync(
            a, reinterpret_cast<signed char*>(a_int8[warp / 2]), kTile);
      }
'@
$raw = $raw.Substring(0, $start) + $replacement + $raw.Substring($end)
[IO.File]::WriteAllText($p, $raw, (New-Object Text.UTF8Encoding($false)))
Write-Output 'v1027_direct_a_fullrows_applied'
