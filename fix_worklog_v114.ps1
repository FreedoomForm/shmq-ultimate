$path = 'C:\Users\User\Downloads\Multi-main\Multi-main\research-15\worklog.md'
$existing = [IO.File]::ReadAllText($path)
if ($existing.Contains('## v114 —')) { throw 'v114 already logged' }
$entry = @'

## v114 — remove stale generated vLLM assertion from repaired v129 gate
Date: 2026-08-18. Change: the v113 notebook still contained a literal generated `\\nassert gates['vllm_apply_path'] == 'passed'` in its final cell, so `unavailable_environment` was converted back into an AssertionError. Removed only that stale generated assertion; retained patch-contract verification and honest unavailable-environment reporting. Validation: 15 local contract tests, Python compilation, patch applicability, notebook rebuild, provenance check, and stale-assertion scan passed. Kaggle: version 131 pushed; it remained queued during the polling window, so no T4 benchmark result is attributed to v114 yet.
'@
Add-Content -LiteralPath $path -Value $entry -Encoding utf8
Write-Output 'WORKLOG_V114_UPDATED'
