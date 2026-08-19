$root = 'C:\Users\User\Downloads\Multi-main\Multi-main'
$src = Join-Path $root 'shmq-ultimate\external\MixLLM\mixllm\kernels\three_level_sm75.cu'
$v51 = Join-Path $root 'research-15\versions\v86_branch_hoist\three_level_sm75.before.cu'
Copy-Item -LiteralPath $v51 -Destination $src -Force
$a = [Collections.ArrayList](Get-Content -LiteralPath $src)
$raw = Get-Content -LiteralPath $src -Raw
if ($raw.Contains('wmma::fragment<wmma::accumulator, 8, 32')) { throw 'v102_already_present' }
for ($i = 0; $i -lt $a.Count; $i++) {
  if ($a[$i] -eq '  const int row_base = blockIdx.y * kTile;') { $a[$i] = '  int row_base = blockIdx.y * kTile;' }
  if ($a[$i].Contains('int8_t b_int8[kPrefillWarps][kTile * kTile];')) { $a[$i] = $a[$i].Replace('[kTile * kTile]', '[32 * kTile]') }
}
$start = -1
for ($i = 0; $i -lt $a.Count; $i++) { if ($a[$i].Contains('float scaled_accumulators[kTile * kTile / kWarpSize] = {};')) { $start = $i; break } }
if ($start -lt 0) { throw 'int_path_start' }
$a.Insert($start, '  row_base += (warp / 2) * 8;')
$a.Insert($start + 1, '  channel_base = (tile_id - (precision == 4 ? 0 : tiles4)) * kPrefillChannels + (warp % 2) * 32;')
$end = -1
for ($i = $start + 2; $i -lt $a.Count; $i++) { if ($a[$i] -eq '#endif') { $end = $i; break } }
if ($end -lt 0) { throw 'int_path_end' }
$storeSeen = $false
for ($i = $start; $i -lt $end; $i++) {
  if ($a[$i] -like '*wmma::fragment<wmma::accumulator,*kTile*int>*') { $a[$i] = $a[$i].Replace('kTile, kTile, kTile, int', '8, 32, kTile, int') }
  if ($a[$i] -like '*wmma::fragment<wmma::matrix_a,*kTile*signed char,*') { $a[$i] = $a[$i].Replace('kTile, kTile, kTile, signed char,', '8, 32, kTile, signed char,') }
  if ($a[$i] -like '*wmma::fragment<wmma::matrix_b,*kTile*signed char,*') { $a[$i] = $a[$i].Replace('kTile, kTile, kTile, signed char,', '8, 32, kTile, signed char,') }
  if ($a[$i] -eq '  const bool full_rows = row_base + kTile <= rows;') { $a[$i] = '  const bool full_rows = row_base + 8 <= rows;' }
  if ($a[$i] -eq '  const bool full_channels = channel_base + kTile <= partition_size;') { $a[$i] = '  const bool full_channels = channel_base + 32 <= partition_size;' }
  if ($a[$i] -like '*for (int linear = threadIdx.x; linear < kTile * kTile;*') { $a[$i] = $a[$i].Replace('kTile * kTile', '8 * kTile') }
  if ($a[$i].Contains('wmma::store_matrix_sync(accumulator_int[warp], accumulator, kTile,')) { $a[$i] = $a[$i].Replace('accumulator, kTile,', 'accumulator, 32,'); $storeSeen = $true }
  if ($storeSeen -and $a[$i] -like '*for (int linear = lane, item = 0; linear < kTile * kTile;*') { $a[$i] = $a[$i].Replace('kTile * kTile', '8 * 32') }
  if ($storeSeen -and $a[$i] -eq '      const int tile_row = linear / kTile;') { $a[$i] = '      const int tile_row = linear / 32;' }
  if ($storeSeen -and $a[$i] -eq '      const int local_channel = channel_base + linear % kTile;') { $a[$i] = '      const int local_channel = channel_base + linear % 32;' }
}
$partial = -1
for ($i = $start; $i -lt $end; $i++) { if ($a[$i] -like '*for (int linear = lane; linear < kTile * kTile;*') { $partial = $i; break } }
if ($partial -lt 0) { throw 'partial_start' }
$partialEnd = -1
for ($i = $partial; $i -lt $end; $i++) { if ($a[$i] -eq '        __syncwarp();') { $partialEnd = $i; break } }
if ($partialEnd -lt 0) { throw 'partial_end' }
$new = [string[]]@(
'        for (int linear = lane; linear < 32 * kTile;',
'             linear += kWarpSize) {',
'          const int channel_offset = linear / kTile;',
'          const int k_offset = linear % kTile;',
'          const int channel = channel_base + channel_offset;',
'          int8_t weight = 0;',
'          if (channel < partition_size) {',
'            const int k = k_base + k_offset;',
'            weight = precision == 4',
'                ? expanded_int4[channel * width + k]',
'                : weight_int8[channel * width + k];',
'          }',
'          b_int8[warp][channel_offset * kTile + k_offset] = weight;',
'        }',
'        __syncwarp();'
)
for ($i = $partialEnd; $i -ge $partial; $i--) { $a.RemoveAt($i) }
for ($j = $new.Count - 1; $j -ge 0; $j--) { $a.Insert($partial, $new[$j]) }
[IO.File]::WriteAllLines($src, [string[]]$a, (New-Object Text.UTF8Encoding($false)))
Write-Output 'v102_tile_line_patch_applied'
