$builder = 'C:\Users\User\Downloads\Multi-main\Multi-main\scripts\build_mixllm_3level_kaggle.py'
$text = [IO.File]::ReadAllText($builder)
$old = @'
        smoke_code = """import sys, torch
'@.TrimEnd()
$new = @'
        smoke_code = '''import sys, torch
'@.TrimEnd()
if (-not $text.Contains($old)) { throw 'smoke_code opening quote not found' }
$text = $text.Replace($old, $new)
$oldEnd = @'
print('VLLM_APPLY_SMOKE_PASS', tuple(y.shape))
"""
        smoke_file
'@.TrimEnd()
$newEnd = @'
print('VLLM_APPLY_SMOKE_PASS', tuple(y.shape))
'''
        smoke_file
'@.TrimEnd()
if (-not $text.Contains($oldEnd)) { throw 'smoke_code closing quote not found' }
$text = $text.Replace($oldEnd, $newEnd)
[IO.File]::WriteAllText($builder, $text, (New-Object Text.UTF8Encoding($false)))
Write-Output 'C7_QUOTES_REPAIRED'
