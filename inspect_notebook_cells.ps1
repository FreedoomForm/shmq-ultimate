$p = 'C:\Users\User\Downloads\Multi-main\Multi-main\shmq-ultimate\mixllm_3level_kaggle\mixllm_3level_gate.ipynb'
$nb = Get-Content -LiteralPath $p -Raw | ConvertFrom-Json
Write-Output ('CELLS=' + $nb.cells.Count)
$i = 0
foreach ($c in $nb.cells) {
    $i++
    $src = ($c.source -join '')
    $first = ($src -split "`n" | Select-Object -First 1)
    $short = $first.Substring(0, [Math]::Min(160, $first.Length))
    Write-Output ($i.ToString() + ':' + $c.cell_type + ':' + $short)
}
