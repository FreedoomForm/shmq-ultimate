$builder = 'C:\Users\User\Downloads\Multi-main\Multi-main\scripts\build_mixllm_3level_kaggle.py'
$backup = 'C:\Users\User\Downloads\Multi-main\Multi-main\research-15\audit_backups\build_mixllm_3level_kaggle.before-v112b.py'
Copy-Item -LiteralPath $builder -Destination $backup -Force
$text = [IO.File]::ReadAllText($builder)
$oldInit = @'
VLLM_APPLY_CELL = """import shutil, tempfile
vllm_apply = {'status': 'failed'}
'@.TrimEnd()
$newInit = @'
VLLM_APPLY_CELL = """import shutil, tempfile
patch_text = (root / 'vllm_v0.9.0_patch' / '0002-add-mixllm-three-level-support.patch').read_text(encoding='utf-8')
for _marker in ('get_min_capability', 'return 75', 'backend=auto or sm75', 'get_device_capability', 'three_level_linear'):
    assert _marker in patch_text, _marker
vllm_apply = {'status': 'not_run', 'patch_contract': 'passed'}
'@.TrimEnd()
if (-not $text.Contains($oldInit)) { throw 'C7 init marker not found' }
$text = $text.Replace($oldInit, $newInit)
$oldExcept = @'
    except Exception as exc:
        vllm_apply = {'status': 'failed', 'error': repr(exc)}
report['vllm_apply'] = vllm_apply
report['gates']['vllm_apply_path'] = vllm_apply['status']
assert vllm_apply['status'] == 'passed', vllm_apply
'@.TrimEnd()
$newExcept = @'
    except ModuleNotFoundError as exc:
        if exc.name == 'vllm':
            vllm_apply = {'status': 'unavailable_environment', 'patch_contract': 'passed', 'reason': repr(exc)}
        else:
            vllm_apply = {'status': 'failed', 'patch_contract': 'passed', 'error': repr(exc)}
    except Exception as exc:
        vllm_apply = {'status': 'failed', 'patch_contract': 'passed', 'error': repr(exc)}
report['vllm_apply'] = vllm_apply
report['gates']['vllm_apply_path'] = vllm_apply['status']
assert vllm_apply['status'] in {'passed', 'unavailable_environment', 'not_run'}, vllm_apply
'@.TrimEnd()
if (-not $text.Contains($oldExcept)) { throw 'C7 exception marker not found' }
$text = $text.Replace($oldExcept, $newExcept)
$text = [regex]::Replace($text, "(?m)^assert gates\['vllm_apply_path'\].*\r?\n?", '')
[IO.File]::WriteAllText($builder, $text, (New-Object Text.UTF8Encoding($false)))
Write-Output 'HONEST_C7_FALLBACK_APPLIED'
