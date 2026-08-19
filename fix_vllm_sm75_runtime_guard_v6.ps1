$root = 'C:\Users\User\Downloads\Multi-main\Multi-main'
$patch = Join-Path $root 'shmq-ultimate\external\MixLLM\vllm_v0.9.0_patch\0002-add-mixllm-three-level-support.patch'
$test = Join-Path $root 'shmq-ultimate\external\MixLLM\mixllm\test\test_vllm_patch_contract.py'
$patchBackup = Join-Path $root 'research-15\audit_backups\0002-add-mixllm-three-level-support.before-v108-sm75-guard.patch'
$testBackup = Join-Path $root 'research-15\audit_backups\test_vllm_patch_contract.before-v108-sm75-guard.py'
New-Item -ItemType Directory -Force -Path (Split-Path $patchBackup -Parent) | Out-Null
Copy-Item -LiteralPath $patch -Destination $patchBackup -Force
Copy-Item -LiteralPath $test -Destination $testBackup -Force
$text = Get-Content -LiteralPath $patch -Raw
$nl = if ($text.Contains("`r`n")) { "`r`n" } else { "`n" }
$needle = '+            layer._sm75_fp16_placeholders = None'
$idx = $text.IndexOf($needle)
if ($idx -lt 0) { throw 'FP16 placeholder marker was not found' }
$guard = $nl + '+        capability = tuple(torch.cuda.get_device_capability(x.device))' + $nl + '+        if capability != (7, 5):' + $nl + '+            raise RuntimeError(' + $nl + '+                f"MixLLM native vLLM path requires SM75, got {capability}"' + $nl + '+            )'
$text = $text.Insert($idx + $needle.Length, $guard)
$text = $text.Replace(' 2 files changed, 242 insertions(+)', ' 2 files changed, 247 insertions(+)')
$text = $text.Replace('@@ -0,0 +1,239 @@', '@@ -0,0 +1,244 @@')
if (-not $text.EndsWith("`n")) { $text += "`n" }
[IO.File]::WriteAllText($patch, $text, (New-Object Text.UTF8Encoding($false)))
$testText = Get-Content -LiteralPath $test -Raw
$testAnchor = '        self.assertIn("+        if not hasattr(layer, "_sm75_fp16_placeholders"):", text)'
$testIdx = $testText.IndexOf($testAnchor)
if ($testIdx -lt 0) { throw 'Contract-test insertion anchor was not found' }
$testNl = if ($testText.Contains("`r`n")) { "`r`n" } else { "`n" }
$testText = $testText.Insert($testIdx + $testAnchor.Length, $testNl + '        self.assertIn("get_device_capability", text)')
[IO.File]::WriteAllText($test, $testText, (New-Object Text.UTF8Encoding($false)))
Write-Output ('PATCH_GUARD='+$text.Contains('get_device_capability(x.device)'))
Write-Output ('HUNK_244='+$text.Contains('@@ -0,0 +1,244 @@'))
Write-Output ('SUMMARY_247='+$text.Contains('2 files changed, 247 insertions(+)'))
Write-Output ('TEST_GUARD='+$testText.Contains('get_device_capability'))
