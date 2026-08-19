$p = 'C:\Users\User\Downloads\Multi-main\Multi-main\shmq-ultimate\external\MixLLM\mixllm\kernels\three_level_sm75.cu'
$lines = [Collections.ArrayList](Get-Content -LiteralPath $p)
$found = $false
for ($i = 0; $i -lt $lines.Count; $i++) {
  $line = [string]$lines[$i]
  if ($line.Contains('for (int linear = lane, item = 0; linear < kTile * kTile;')) {
    $lines[$i] = $line.Replace('linear < kTile * kTile', 'linear < 8 * 32')
    $found = $true
    if ($i + 2 -ge $lines.Count) { throw 'v1022_bounds' }
    $rowLine = [string]$lines[$i + 2]
    $chanLine = [string]$lines[$i + 3]
    if (-not $rowLine.Contains('const int row = row_base + linear / kTile;')) { throw 'v1022_row_marker' }
    if (-not $chanLine.Contains('const int local_channel = channel_base + linear % kTile;')) { throw 'v1022_channel_marker' }
    $lines[$i + 2] = $rowLine.Replace('linear / kTile', 'linear / 32')
    $lines[$i + 3] = $chanLine.Replace('linear % kTile', 'linear % 32')
    break
  }
}
if (-not $found) { throw 'v1022_output_loop_marker' }
[IO.File]::WriteAllLines($p, [string[]]$lines, (New-Object Text.UTF8Encoding($false)))
Write-Output 'v1022_output_indexing_line_applied'
