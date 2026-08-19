$builder = 'C:\Users\User\Downloads\Multi-main\Multi-main\scripts\build_mixllm_3level_kaggle.py'
$text = [IO.File]::ReadAllText($builder)
$bad = '    "mixllm/test/test_model_gate.py", "mixllm/test/test_vllm_three_level.py",\n    "vllm_v0.9.0_patch/0002-add-mixllm-three-level-support.patch",\n    "vllm_v0.9.0_patch/THREE_LEVEL_MANIFEST.md")'
$good = @'
    "mixllm/test/test_model_gate.py", "mixllm/test/test_vllm_three_level.py",
    "vllm_v0.9.0_patch/0002-add-mixllm-three-level-support.patch",
    "vllm_v0.9.0_patch/THREE_LEVEL_MANIFEST.md")
'@.TrimEnd()
if (-not $text.Contains($bad)) { throw 'Corrupt SOURCE_FILES text not found' }
$text = $text.Replace($bad, $good)
[IO.File]::WriteAllText($builder, $text, (New-Object Text.UTF8Encoding($false)))
Write-Output 'SOURCE_FILES_REPAIRED'
