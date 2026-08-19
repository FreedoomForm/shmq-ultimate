$nb = 'C:\Users\User\Downloads\Multi-main\Multi-main\shmq-ultimate\mixllm_3level_kaggle\mixllm_3level_gate.ipynb'
$doc = Get-Content -LiteralPath $nb -Raw | ConvertFrom-Json
foreach ($cell in $doc.cells) {
  if ($cell.cell_type -eq 'code') {
    $text = ($cell.source -join '')
    if ($text -match "mixllm/kernels/three_level_sm75\.cu': '([0-9a-f]{64})'") {
      Write-Output ('embedded_cuda_sha=' + $Matches[1])
    }
    if ($text -match "source_sha256': '([0-9a-f]{64})'") {
      Write-Output ('embedded_source_sha=' + $Matches[1])
    }
  }
}
