$root = 'C:\Users\User\Downloads\Multi-main\Multi-main'
$test = Join-Path $root 'shmq-ultimate\external\MixLLM\mixllm\test\test_vllm_patch_contract.py'
$backup = Join-Path $root 'research-15\audit_backups\test_vllm_patch_contract.before-v106.py'
New-Item -ItemType Directory -Force -Path (Split-Path $backup -Parent) | Out-Null
Copy-Item -LiteralPath $test -Destination $backup -Force
$text = Get-Content -LiteralPath $test -Raw
$old1 = '        self.assertIn("F.linear", text)'
$new1 = '        self.assertRegex(text, r"(?:F\\.linear|torch\\.nn\\.functional\\.linear)")'
if (-not $text.Contains($old1)) { throw 'Stale F.linear assertion was not found' }
$text = $text.Replace($old1, $new1)
$old2 = '        self.assertIn("gated to backend=reference", text)'
$new2 = '        self.assertIn("requires backend=auto or sm75", text)'
if (-not $text.Contains($old2)) { throw 'Stale reference-gate assertion was not found' }
$text = $text.Replace($old2, $new2)
$anchor = '    def test_patch_does_not_modify_cuda_sources(self):'
$extra = @'
    def test_patch_advertises_native_sm75_contract(self):
        text = PATCH.read_text(encoding="utf-8")
        self.assertIn("+        return 75", text)
        self.assertIn('+        if contract.backend not in {"auto", "sm75"}:', text)
        self.assertIn('+        if not hasattr(layer, "_sm75_int4_expanded"):', text)
        self.assertIn('+        if not hasattr(layer, "_sm75_fp16_placeholders"):', text)
        self.assertNotIn("gated to backend=reference", text)

'@
if (-not $text.Contains($anchor)) { throw 'Test insertion anchor was not found' }
$text = $text.Replace($anchor, $extra + $anchor)
[IO.File]::WriteAllText($test, $text, (New-Object Text.UTF8Encoding($false)))
Write-Output ('TEST='+$test)
Write-Output ('BACKUP='+$backup)
Write-Output ('NEW_CAPABILITY_ASSERTION='+$text.Contains('test_patch_advertises_native_sm75_contract'))
Write-Output ('NEW_LINEAR_ASSERTION='+$text.Contains('assertRegex(text'))
