$nb = 'C:\Users\User\Downloads\Multi-main\Multi-main\research-15\versions\v51_decode_expanded_cache_contract\mixllm_3level_gate.before.ipynb'
$doc = Get-Content -LiteralPath $nb -Raw | ConvertFrom-Json
foreach ($cell in $doc.cells) {
  if ($cell.cell_type -eq 'code') {
    $text = ($cell.source -join '')
    if ($text -match "mixllm/kernels/three_level_sm75\.cu': '([0-9a-f]{64})'") { Write-Output ('embedded_cuda_sha=' + $Matches[1]) }
    if ($text -match "source_sha256': '([0-9a-f]{64})'") { Write-Output ('embedded_source_sha=' + $Matches[1]) }
    if ($text -match 'kDecodeChannelsPerWarp = ([0-9]+)') { Write-Output ('decode_channels=' + $Matches[1]) }
    if ($text -match 'mixed_prefill_end_to_end_performance') { Write-Output 'has_prefill_gate=true' }
  }
}
