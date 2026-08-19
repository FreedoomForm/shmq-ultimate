$p = 'C:\Users\User\Downloads\Multi-main\Multi-main\shmq-ultimate\external\MixLLM\mixllm\kernels\three_level_sm75.cu'
$raw = Get-Content -LiteralPath $p -Raw
$old = @'
  for (int linear = lane, item = 0; linear < kTile * kTile;
       linear += kWarpSize, ++item) {
    const int row = row_base + linear / kTile;
    const int local_channel = channel_base + linear % kTile;
'@
$new = @'
  for (int linear = lane, item = 0; linear < 8 * 32;
       linear += kWarpSize, ++item) {
    const int row = row_base + linear / 32;
    const int local_channel = channel_base + linear % 32;
'@
if (-not $raw.Contains($old)) { throw 'v1022_output_loop_marker' }
$raw = $raw.Replace($old, $new)
[IO.File]::WriteAllText($p, $raw, (New-Object Text.UTF8Encoding($false)))
Write-Output 'v1022_output_indexing_applied'
