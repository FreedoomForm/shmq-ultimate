$root = 'C:\Users\User\Downloads\Multi-main\Multi-main'
$test = Join-Path $root 'shmq-ultimate\external\MixLLM\mixllm\test\test_vllm_patch_contract.py'
$text = Get-Content -LiteralPath $test -Raw
$old = '        self.assertRegex(text, r"(?:F\\.linear|torch\\.nn\\.functional\\.linear)")'
$new = '        self.assertTrue("F.linear" in text or "torch.nn.functional.linear" in text)'
if (-not $text.Contains($old)) { throw 'Escaped linear assertion was not found exactly' }
$text = $text.Replace($old, $new)
[IO.File]::WriteAllText($test, $text, (New-Object Text.UTF8Encoding($false)))
Write-Output ('TEST='+$test)
Write-Output ('SIMPLE_ASSERTION='+$text.Contains('self.assertTrue("F.linear" in text or "torch.nn.functional.linear" in text)'))
