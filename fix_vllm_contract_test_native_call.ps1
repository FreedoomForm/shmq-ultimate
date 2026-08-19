$root = 'C:\Users\User\Downloads\Multi-main\Multi-main'
$test = Join-Path $root 'shmq-ultimate\external\MixLLM\mixllm\test\test_vllm_patch_contract.py'
$text = Get-Content -LiteralPath $test -Raw
$old = '        self.assertTrue("F.linear" in text or "torch.nn.functional.linear" in text)'
$new = '        self.assertIn("three_level_linear", text)'
if (-not $text.Contains($old)) { throw 'Stale linear assertion was not found' }
$text = $text.Replace($old, $new)
[IO.File]::WriteAllText($test, $text, (New-Object Text.UTF8Encoding($false)))
Write-Output ('TEST='+$test)
Write-Output ('NATIVE_CALL_ASSERTION='+$text.Contains('self.assertIn("three_level_linear", text)'))
