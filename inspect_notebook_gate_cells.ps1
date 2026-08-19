$p = 'C:\Users\User\Downloads\Multi-main\Multi-main\shmq-ultimate\mixllm_3level_kaggle\mixllm_3level_gate.ipynb'
$nb = Get-Content -LiteralPath $p -Raw | ConvertFrom-Json
foreach ($idx in 4,5,7) {
    $c = $nb.cells[$idx]
    Write-Output ('---CELL ' + ($idx + 1) + '---')
    [Console]::WriteLine(($c.source -join ''))
}
