$p = 'C:\Users\User\Downloads\Multi-main\Multi-main\shmq-ultimate\external\MixLLM\mixllm\kernels\three_level_sm75.cu'
$raw = Get-Content -LiteralPath $p -Raw
$pattern = 'for \(int linear = threadIdx\.x; linear < 8 \* kTile;\s+linear \+= blockDim\.x\) \{'
$replacement = 'for (int linear = (warp % 2) * kWarpSize + lane; linear < 8 * kTile;`n           linear += 2 * kWarpSize) {'
if (([regex]::Matches($raw, $pattern)).Count -ne 1) { throw 'v1025_a_loop_marker' }
$raw = [regex]::Replace($raw, $pattern, $replacement, 1)
[IO.File]::WriteAllText($p, $raw, (New-Object Text.UTF8Encoding($false)))
Write-Output 'v1025_pair_local_a_applied'
