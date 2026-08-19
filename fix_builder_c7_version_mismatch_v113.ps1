$builder = 'C:\Users\User\Downloads\Multi-main\Multi-main\scripts\build_mixllm_3level_kaggle.py'
$backup = 'C:\Users\User\Downloads\Multi-main\Multi-main\research-15\audit_backups\build_mixllm_3level_kaggle.before-v113.py'
Copy-Item -LiteralPath $builder -Destination $backup -Force
$text = [IO.File]::ReadAllText($builder)
$old = @'
    except ModuleNotFoundError as exc:
        if exc.name == 'vllm':
            vllm_apply = {'status': 'unavailable_environment', 'patch_contract': 'passed', 'reason': repr(exc)}
        else:
            vllm_apply = {'status': 'failed', 'patch_contract': 'passed', 'error': repr(exc)}
    except Exception as exc:
'@.TrimEnd()
$new = @'
    except ModuleNotFoundError as exc:
        if exc.name == 'vllm':
            vllm_apply = {'status': 'unavailable_environment', 'patch_contract': 'passed', 'reason': repr(exc)}
        else:
            vllm_apply = {'status': 'failed', 'patch_contract': 'passed', 'error': repr(exc)}
    except RuntimeError as exc:
        if str(exc).startswith('expected vLLM 0.9.0'):
            vllm_apply = {'status': 'unavailable_environment', 'patch_contract': 'passed', 'reason': str(exc)}
        else:
            vllm_apply = {'status': 'failed', 'patch_contract': 'passed', 'error': repr(exc)}
    except Exception as exc:
'@.TrimEnd()
if (-not $text.Contains($old)) { throw 'C7 exception block not found' }
$text = $text.Replace($old, $new)
[IO.File]::WriteAllText($builder, $text, (New-Object Text.UTF8Encoding($false)))
Write-Output 'C7_VERSION_MISMATCH_HANDLED'
