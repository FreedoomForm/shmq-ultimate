$p = 'C:\Users\User\Downloads\Multi-main\Multi-main\shmq-ultimate\external\MixLLM\mixllm\kernels\three_level_sm75.cu'
$raw = Get-Content -LiteralPath $p -Raw
$literal = '`n'
if (([regex]::Matches($raw, [regex]::Escape($literal))).Count -ne 1) { throw 'v1025_literal_newline_marker' }
$raw = $raw.Replace($literal, [Environment]::NewLine)
[IO.File]::WriteAllText($p, $raw, (New-Object Text.UTF8Encoding($false)))
Write-Output 'v1025_literal_newline_fixed'
