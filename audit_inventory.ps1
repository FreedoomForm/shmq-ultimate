$root = 'C:\Users\User\Downloads\Multi-main\Multi-main'
$out = Join-Path $root 'research-15\audit_inventory_v1.txt'
$sb = New-Object System.Text.StringBuilder
[void]$sb.AppendLine('AUDIT INVENTORY v1')
[void]$sb.AppendLine('ROOT='+$root)
[void]$sb.AppendLine('GENERATED='+[DateTime]::UtcNow.ToString('o'))
[void]$sb.AppendLine('')
[void]$sb.AppendLine('== VERSION SNAPSHOTS ==')
$versions = Get-ChildItem -LiteralPath (Join-Path $root 'research-15\versions') -Directory | Sort-Object Name
foreach ($v in $versions) {
  $files = Get-ChildItem -LiteralPath $v.FullName -File -Recurse -ErrorAction SilentlyContinue | Sort-Object FullName
  [void]$sb.AppendLine('VERSION '+$v.Name+' files='+$files.Count)
  foreach ($f in $files) {
    $sha = (Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash
    $rel = $f.FullName.Substring($v.FullName.Length+1)
    [void]$sb.AppendLine('  '+$rel+' | '+$f.Length+' | '+$sha)
  }
}
[void]$sb.AppendLine('')
[void]$sb.AppendLine('== CURRENT KEY FILES ==')
$keyFiles = @(
  (Join-Path $root 'shmq-ultimate\external\MixLLM\mixllm\kernels\three_level_sm75.cu'),
  (Join-Path $root 'shmq-ultimate\external\MixLLM\mixllm\sm75_backend.py'),
  (Join-Path $root 'shmq-ultimate\external\MixLLM\mixllm\quantization\three_level.py'),
  (Join-Path $root 'shmq-ultimate\external\MixLLM\mixllm\nn\modules\three_level_linear.py'),
  (Join-Path $root 'shmq-ultimate\external\MixLLM\mixllm\vllm_three_level.py'),
  (Join-Path $root 'shmq-ultimate\mixllm_3level_kaggle\mixllm_3level_gate.ipynb'),
  (Join-Path $root 'research-15\worklog.md')
)
foreach ($f in $keyFiles) {
  if (Test-Path -LiteralPath $f) {
    $i = Get-Item -LiteralPath $f
    [void]$sb.AppendLine($f+' | '+$i.Length+' | '+(Get-FileHash -LiteralPath $f -Algorithm SHA256).Hash)
  } else { [void]$sb.AppendLine('MISSING '+$f) }
}
[void]$sb.AppendLine('')
[void]$sb.AppendLine('== WORKLOG HEADINGS ==')
$wl = Join-Path $root 'research-15\worklog.md'
if (Test-Path -LiteralPath $wl) {
  Select-String -LiteralPath $wl -Pattern '^## |^### ' | ForEach-Object { [void]$sb.AppendLine($_.LineNumber.ToString()+':'+$_.Line) }
}
[void]$sb.AppendLine('')
[void]$sb.AppendLine('== KAGGLE GATE ARTIFACTS ==')
$kr = Join-Path $root 'shmq-ultimate\mixllm_3level_kaggle'
Get-ChildItem -LiteralPath $kr -Directory -Filter 'output-*' | Sort-Object Name | ForEach-Object {
  $g = Get-ChildItem -LiteralPath $_.FullName -Recurse -File -Filter 'mixllm_3level_gate.json' -ErrorAction SilentlyContinue | Select-Object -First 1
  $b = Get-ChildItem -LiteralPath $_.FullName -Recurse -File -Filter 'mixllm_3level_benchmarks.json' -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($g) {
    try {
      $j = Get-Content -LiteralPath $g.FullName -Raw | ConvertFrom-Json
      $gate = ($j.gates.PSObject.Properties | ForEach-Object { $_.Name+'='+[string]$_.Value }) -join ';'
    } catch { $gate='GATE_PARSE_ERROR' }
    $bench = ''
    if ($b) {
      try {
        $x = Get-Content -LiteralPath $b.FullName -Raw | ConvertFrom-Json
        $shapes = $x.scenarios.qwen_qkv_mixed_4_8_16.shapes | ForEach-Object { 'r'+$_.rows+':e2e='+$_.end_to_end_p50_speedup_vs_dense+',gemm='+$_.gemm_p50_speedup_vs_dense+',err='+$_.max_abs_error }
        $bench = ($shapes -join '|')
      } catch { $bench='BENCH_PARSE_ERROR' }
    }
    [void]$sb.AppendLine($_.Name+' | '+$gate+' | '+$bench)
  }
}
[IO.File]::WriteAllText($out, $sb.ToString(), (New-Object Text.UTF8Encoding($false)))
Write-Output $out
