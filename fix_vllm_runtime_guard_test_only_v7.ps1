$root = 'C:\Users\User\Downloads\Multi-main\Multi-main'
$test = Join-Path $root 'shmq-ultimate\external\MixLLM\mixllm\test\test_vllm_patch_contract.py'
$backup = Join-Path $root 'research-15\audit_backups\test_vllm_patch_contract.before-v108-test-only.py'
New-Item -ItemType Directory -Force -Path (Split-Path $backup -Parent) | Out-Null
Copy-Item -LiteralPath $test -Destination $backup -Force
$text = Get-Content -LiteralPath $test -Raw
if ($text.Contains('self.assertIn("get_device_capability", text)')) { Write-Output 'TEST_GUARD_ALREADY_PRESENT=True'; exit 0 }
$marker = '        self.assertNotIn("gated to backend=reference", text)'
$idx = $text.IndexOf($marker)
if ($idx -lt 0) { throw 'Existing native contract test marker was not found' }
$nl = if ($text.Contains("`r`n")) { "`r`n" } else { "`n" }
$text = $text.Insert($idx, '        self.assertIn("get_device_capability", text)' + $nl)
[IO.File]::WriteAllText($test, $text, (New-Object Text.UTF8Encoding($false)))
Write-Output ('TEST_GUARD='+$text.Contains('self.assertIn("get_device_capability", text)'))
