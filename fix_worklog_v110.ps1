$root = 'C:\Users\User\Downloads\Multi-main\Multi-main'
$path = Join-Path $root 'research-15\worklog.md'
$backup = Join-Path $root 'research-15\audit_backups\worklog.before-v105-v110.md'
New-Item -ItemType Directory -Force -Path (Split-Path $backup -Parent) | Out-Null
Copy-Item -LiteralPath $path -Destination $backup -Force
$entries = @'

## v105 — native ThreeLevelLinear SM75 forward path
Date: 2026-08-18. Change: `ThreeLevelLinear.forward()` now selects the native `mixllm.sm75_backend.three_level_linear` path only for CUDA capability `(7, 5)`, restores leading dimensions, and applies bias after the operator; other devices retain the reference fallback. Validation: static source inspection and Python compilation passed. Kaggle: not run because the audit was still open.

## v106 — vLLM contract tests aligned with native SM75 execution
Date: 2026-08-18. Change: replaced stale reference-only expectations in the vLLM contract tests with assertions for `three_level_linear`, native SM75 backend selection, and the accepted `backend=auto|sm75` contract. Validation: all 11 local vLLM contract tests passed. Kaggle: not run.

## v107 — vLLM patch format corrected
Date: 2026-08-18. Change: corrected patch hunk counts and restored the final newline. Validation: `git apply --check` passed against the pinned vLLM commit `5fbbfe9a4c13094ad72ed3d6b4ef208a7ddc0fd7`. Kaggle: not run.

## v108 — runtime SM75 guard and contract assertion
Date: 2026-08-18. Change: vLLM `apply()` now verifies `torch.cuda.get_device_capability(x.device) == (7, 5)` before loading the native backend; the test asserts the guard is present. Validation: Python compilation, 11 local vLLM tests, and `git apply --check` passed. Kaggle: not run.

## v109 — manifest contract reconciliation
Date: 2026-08-18. Change: rewrote `THREE_LEVEL_MANIFEST.md` to document native SM75-only execution, direct output-channel checkpoint layout, tensor-parallel remapping, separate activation/INT4-expansion launches, and honest benchmark boundaries. Validation: stale reference-only manifest claim removed; a disposable patch-application check still passed. Kaggle: not run.

## v110 — CUDA source-contract test repair and invariant coverage
Date: 2026-08-18. Change: replaced the malformed source-contract test with a clean test module covering the native kernel count, m8n32 fragment shapes, per-panel A staging, warp ownership, output indexing, block-uniform `__syncthreads()`, and native `ThreeLevelLinear.forward()`. Validation: Python compilation and all 4 source-contract tests passed. Kaggle: not run.
'@
Add-Content -LiteralPath $path -Value $entries -Encoding utf8
Write-Output ('WORKLOG='+$path)
Write-Output ('ADDED_VERSIONS=v105,v106,v107,v108,v109,v110')
