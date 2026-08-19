$root = 'C:\Users\User\Downloads\Multi-main\Multi-main'
$cu = Join-Path $root 'shmq-ultimate\external\MixLLM\mixllm\kernels\three_level_sm75.cu'
$backup = Join-Path $root 'research-15\audit_backups\three_level_sm75.before-v104-uniform-barrier.cu'
New-Item -ItemType Directory -Force -Path (Split-Path $backup -Parent) | Out-Null
Copy-Item -LiteralPath $cu -Destination $backup -Force
$text = Get-Content -LiteralPath $cu -Raw
$old = @'
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
$new = @'
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
      }
      __syncthreads();
      if (!full_rows) {
        wmma::load_matrix_sync(
            a, reinterpret_cast<signed char*>(a_int8[warp / 2]), kTile);
      }
'@
if (-not $text.Contains($old)) { throw 'Expected v102.7 A staging block was not found exactly' }
$text = $text.Replace($old, $new)
[IO.File]::WriteAllText($cu, $text, (New-Object Text.UTF8Encoding($false)))
Write-Output ('CUDA='+$cu)
Write-Output ('BACKUP='+$backup)
Write-Output ('UNCONDITIONAL_BARRIER='+$text.Contains("      __syncthreads();`r`n      if (!full_rows) {"))
Write-Output ('PARTIAL_BARRIER_REMOVED='+(-not $text.Contains("        __syncthreads();`r`n        wmma::load_matrix_sync(`r`n            a, reinterpret_cast<signed char*>(a_int8[warp / 2]), kTile);")))
Write-Output ('DIRECT_A_RETAINED='+$text.Contains('input_int8 + row_base * width + k_base'))
