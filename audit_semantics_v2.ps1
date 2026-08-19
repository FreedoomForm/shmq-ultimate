$root = 'C:\Users\User\Downloads\Multi-main\Multi-main'
$out = Join-Path $root 'research-15\audit_semantics_v2.txt'
$sb = New-Object System.Text.StringBuilder
function AddLine($s) { [void]$sb.AppendLine($s) }
AddLine 'SEMANTIC AUDIT v2'
AddLine ('GENERATED='+[DateTime]::UtcNow.ToString('o'))
AddLine ''
AddLine '== SOURCE-BEARING VERSION SNAPSHOTS =='
$seen = @{}
$versionRoot = Join-Path $root 'research-15\versions'
Get-ChildItem -LiteralPath $versionRoot -Directory | Sort-Object Name | ForEach-Object {
  $v = $_
  $cu = Get-ChildItem -LiteralPath $v.FullName -File -Recurse -ErrorAction SilentlyContinue | Where-Object -Property Extension -EQ '.cu' | Select-Object -First 1
  if (-not $cu) { return }
  $hash = (Get-FileHash -LiteralPath $cu.FullName -Algorithm SHA256).Hash
  if ($seen.ContainsKey($hash)) { AddLine ($v.Name+'|same_as='+$seen[$hash]+'|sha='+$hash.Substring(0,16)); return }
  $seen[$hash] = $v.Name
  AddLine ($v.Name+'|sha='+$hash.Substring(0,16)+'|file='+$cu.Name)
  $lines = Get-Content -LiteralPath $cu.FullName
  $patterns = @('constexpr int kTile','constexpr int kGroupSize','constexpr int kPrefillWarps','constexpr int kPrefillChannels','constexpr int kReuseRows','__global__ void','wmma::fragment<wmma::accumulator','wmma::fragment<wmma::matrix_a','wmma::fragment<wmma::matrix_b','wmma::load_matrix_sync','wmma::store_matrix_sync','wmma::mma_sync','__syncthreads','__syncwarp','full_rows','full_channels','expanded_int4','three_level_tensorcore_kernel<<<','const dim3 grid')
  $picked = @()
  foreach ($line in $lines) {
    foreach ($pat in $patterns) {
      if ($line.Contains($pat)) { $picked += $line.Trim(); break }
    }
  }
  foreach ($line in ($picked | Select-Object -First 80 -Unique)) { AddLine ('  '+$line) }
}
AddLine ''
AddLine '== CURRENT AND V51 PYTHON LAYER SIGNATURES =='
$paths = @(
  (Join-Path $root 'research-15\versions\v86_branch_hoist\sm75_backend.before.py'),
  (Join-Path $root 'shmq-ultimate\external\MixLLM\mixllm\sm75_backend.py'),
  (Join-Path $root 'shmq-ultimate\external\MixLLM\mixllm\quantization\three_level.py'),
  (Join-Path $root 'shmq-ultimate\external\MixLLM\mixllm\nn\modules\three_level_linear.py'),
  (Join-Path $root 'shmq-ultimate\external\MixLLM\mixllm\nn\modules\ops.py'),
  (Join-Path $root 'shmq-ultimate\external\MixLLM\mixllm\nn\modules\mixllm_config.py'),
  (Join-Path $root 'shmq-ultimate\external\MixLLM\mixllm\vllm_three_level.py')
)
foreach ($p in $paths) {
  if (-not (Test-Path -LiteralPath $p)) { AddLine ('MISSING '+$p); continue }
  $i = Get-Item -LiteralPath $p
  AddLine ($p+'|sha='+((Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash)+'|lines='+((Get-Content -LiteralPath $p).Count))
  $lines = Get-Content -LiteralPath $p
  foreach ($line in $lines) {
    if ($line -match '^(def |class |    def |    async def |@|\s*torch\.ops|\s*return |\s*if .*rows|\s*if .*precision|\s*expanded_int4|\s*scale_|\s*indices_|\s*load_sm75|\s*quantize_|\s*register|\s*apply|\s*patch|\s*SUPPORTED|\s*LEVELS|\s*ThreeLevel)') { AddLine ('  '+$line.Trim()) }
  }
}
AddLine ''
AddLine '== CURRENT NOTEBOOK MARKERS =='
$nb = Join-Path $root 'shmq-ultimate\mixllm_3level_kaggle\mixllm_3level_gate.ipynb'
if (Test-Path -LiteralPath $nb) {
  $nbtext = Get-Content -LiteralPath $nb -Raw
  foreach ($marker in @('embedded_file_count','source_sha256','workspace_commit','three_level_sm75.cu','sm75_backend.py','Qwen/Qwen2.5-0.5B','compute_75','rows=1','rows=16','rows=128')) {
    AddLine ($marker+'|count='+([regex]::Matches($nbtext,[regex]::Escape($marker))).Count)
  }
}
[IO.File]::WriteAllText($out,$sb.ToString(),(New-Object Text.UTF8Encoding($false)))
Write-Output $out
