$builder = 'C:\Users\User\Downloads\Multi-main\Multi-main\scripts\build_mixllm_3level_kaggle.py'
$backup = 'C:\Users\User\Downloads\Multi-main\Multi-main\research-15\audit_backups\build_mixllm_3level_kaggle.before-v114.py'
Copy-Item -LiteralPath $builder -Destination $backup -Force
$text = [IO.File]::ReadAllText($builder)
$old = @'\nas
sert gates['vllm_apply_path'] == 'passed', 'patched vLLM apply path did not pass
'@ 
if (-not $text.Contains($old)) {
    $old = "\\nas`nassert gates['vllm_apply_path'] == 'passed', 'patched vLLM apply path did not pass`n'"
}
if (-not $text.Contains($old)) { throw 'literal C7 assertion not found' }
$text = $text.Replace($old, '')
[IO.File]::WriteAllText($builder, $text, (New-Object Text.UTF8Encoding($false)))
Write-Output 'LITERAL_C7_ASSERT_REMOVED'
