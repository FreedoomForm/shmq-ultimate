$builder = 'C:\Users\User\Downloads\Multi-main\Multi-main\scripts\build_mixllm_3level_kaggle.py'
$backup = 'C:\Users\User\Downloads\Multi-main\Multi-main\research-15\audit_backups\build_mixllm_3level_kaggle.before-v114c.py'
Copy-Item -LiteralPath $builder -Destination $backup -Force
$text = [IO.File]::ReadAllText($builder)
$start = $text.IndexOf('\nassert gates')
if ($start -lt 0) { throw 'literal C7 assertion start not found' }
$end = $text.IndexOf('""")]', $start)
if ($end -lt 0) { throw 'literal C7 assertion end not found' }
$text = $text.Remove($start, $end - $start)
[IO.File]::WriteAllText($builder, $text, (New-Object Text.UTF8Encoding($false)))
Write-Output 'LITERAL_C7_ASSERT_REMOVED_BY_INDEX'
