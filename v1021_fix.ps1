$p = 'C:\Users\User\Downloads\Multi-main\Multi-main\shmq-ultimate\external\MixLLM\mixllm\kernels\three_level_sm75.cu'
$raw = Get-Content -LiteralPath $p -Raw
$old = 'b, reinterpret_cast<signed char*>(b_int8[warp]), 32);'
$new = 'b, reinterpret_cast<signed char*>(b_int8[warp]), kTile);'
$count = ([regex]::Matches($raw, [regex]::Escape($old))).Count
if ($count -ne 1) { throw "shared_b_ldm_marker_count=$count" }
$raw = $raw.Replace($old, $new)
[IO.File]::WriteAllText($p, $raw, (New-Object Text.UTF8Encoding($false)))
Write-Output 'v1021_shared_b_ldm16_applied'
