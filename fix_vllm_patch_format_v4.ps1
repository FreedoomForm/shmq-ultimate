$root = 'C:\Users\User\Downloads\Multi-main\Multi-main'
$patch = Join-Path $root 'shmq-ultimate\external\MixLLM\vllm_v0.9.0_patch\0002-add-mixllm-three-level-support.patch'
$backup = Join-Path $root 'research-15\audit_backups\0002-add-mixllm-three-level-support.before-v107-format.patch'
New-Item -ItemType Directory -Force -Path (Split-Path $backup -Parent) | Out-Null
Copy-Item -LiteralPath $patch -Destination $backup -Force
$text = Get-Content -LiteralPath $patch -Raw
$old1 = ' 2 files changed, 243 insertions(+)'
$new1 = ' 2 files changed, 242 insertions(+)'
if (-not $text.Contains($old1)) { throw 'Patch summary count 243 was not found' }
$text = $text.Replace($old1, $new1)
$old2 = '@@ -0,0 +1,240 @@'
$new2 = '@@ -0,0 +1,239 @@'
if (-not $text.Contains($old2)) { throw 'New-file hunk count 240 was not found' }
$text = $text.Replace($old2, $new2)
[IO.File]::WriteAllText($patch, $text, (New-Object Text.UTF8Encoding($false)))
Write-Output ('PATCH='+$patch)
Write-Output ('BACKUP='+$backup)
Write-Output ('HUNK_FIXED='+$text.Contains('@@ -0,0 +1,239 @@'))
Write-Output ('SUMMARY_FIXED='+$text.Contains('2 files changed, 242 insertions(+)'))
