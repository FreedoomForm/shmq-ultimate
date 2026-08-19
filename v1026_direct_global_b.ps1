$p = 'C:\Users\User\Downloads\Multi-main\Multi-main\shmq-ultimate\external\MixLLM\mixllm\kernels\three_level_sm75.cu'
$raw = Get-Content -LiteralPath $p -Raw
$old1 = 'if (false && precision == 8 && full_channels) {'
$old2 = '} else if (false && precision == 4 && full_channels) {'
$new1 = 'if (precision == 8 && full_channels) {'
$new2 = '} else if (precision == 4 && full_channels) {'
if (([regex]::Matches($raw, [regex]::Escape($old1))).Count -ne 1) { throw 'v1026_i8_guard_marker' }
if (([regex]::Matches($raw, [regex]::Escape($old2))).Count -ne 1) { throw 'v1026_i4_guard_marker' }
$raw = $raw.Replace($old1, $new1).Replace($old2, $new2)
[IO.File]::WriteAllText($p, $raw, (New-Object Text.UTF8Encoding($false)))
Write-Output 'v1026_direct_global_b_applied'
