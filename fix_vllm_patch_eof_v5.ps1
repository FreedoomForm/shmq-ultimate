$root = 'C:\Users\User\Downloads\Multi-main\Multi-main'
$patch = Join-Path $root 'shmq-ultimate\external\MixLLM\vllm_v0.9.0_patch\0002-add-mixllm-three-level-support.patch'
$text = Get-Content -LiteralPath $patch -Raw
if (-not $text.EndsWith("`n")) { $text += "`n" }
[IO.File]::WriteAllText($patch, $text, (New-Object Text.UTF8Encoding($false)))
Write-Output ('FINAL_NEWLINE='+$text.EndsWith("`n"))
Write-Output ('LAST_CHARS='+$text.Substring([Math]::Max(0,$text.Length-40)))
