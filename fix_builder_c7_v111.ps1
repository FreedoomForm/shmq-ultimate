$root = 'C:\Users\User\Downloads\Multi-main\Multi-main'
$builder = Join-Path $root 'scripts\build_mixllm_3level_kaggle.py'
$backup = Join-Path $root 'research-15\audit_backups\build_mixllm_3level_kaggle.before-v111.py'
New-Item -ItemType Directory -Force -Path (Split-Path $backup -Parent) | Out-Null
Copy-Item -LiteralPath $builder -Destination $backup -Force
$text = [IO.File]::ReadAllText($builder)
$oldSource = '"mixllm/test/test_model_gate.py", "mixllm/test/test_vllm_three_level.py")'
$newSource = '"mixllm/test/test_model_gate.py", "mixllm/test/test_vllm_three_level.py",\n    "vllm_v0.9.0_patch/0002-add-mixllm-three-level-support.patch",\n    "vllm_v0.9.0_patch/THREE_LEVEL_MANIFEST.md")'
if (-not $text.Contains($oldSource)) { throw 'SOURCE_FILES marker not found' }
$text = $text.Replace($oldSource, $newSource)
$marker = 'def build_notebook(sources, manifest):'
$cell = @'
VLLM_APPLY_CELL = """import shutil, tempfile
vllm_apply = {'status': 'failed'}
if not cuda_available or capability != (7, 5):
    vllm_apply['reason'] = 'requires Tesla T4 / SM75'
else:
    try:
        import vllm
        vllm_version = str(getattr(vllm, '__version__', ''))
        if not vllm_version.startswith('0.9.0'):
            raise RuntimeError(f'expected vLLM 0.9.0, got {vllm_version!r}')
        package_root = Path(vllm.__file__).resolve().parents[1]
        smoke_root = Path('/kaggle/working/vllm_sm75_apply_smoke')
        if smoke_root.exists(): shutil.rmtree(smoke_root)
        smoke_root.mkdir(parents=True)
        shutil.copytree(package_root / 'vllm', smoke_root / 'vllm')
        patch_path = root / 'vllm_v0.9.0_patch' / '0002-add-mixllm-three-level-support.patch'
        init = subprocess.run(['git', 'init'], cwd=smoke_root, text=True, capture_output=True, check=True)
        subprocess.run(['git', 'add', 'vllm/model_executor/layers/quantization/__init__.py'], cwd=smoke_root, check=True)
        subprocess.run(['git', 'config', 'user.email', 'gate@example.invalid'], cwd=smoke_root, check=True)
        subprocess.run(['git', 'config', 'user.name', 'MixLLM gate'], cwd=smoke_root, check=True)
        subprocess.run(['git', 'commit', '-m', 'baseline'], cwd=smoke_root, text=True, capture_output=True, check=True)
        check = subprocess.run(['git', 'apply', '--check', str(patch_path)], cwd=smoke_root, text=True, capture_output=True)
        if check.returncode != 0:
            raise RuntimeError('git apply --check failed: ' + check.stderr[-2000:])
        subprocess.run(['git', 'apply', str(patch_path)], cwd=smoke_root, text=True, capture_output=True, check=True)
        smoke_code = """import sys, torch
sys.path.insert(0, SMOKE_ROOT)
sys.path.insert(0, MIX_ROOT)
from mixllm.nn.modules.three_level_linear import ThreeLevelLinear
from vllm.model_executor.layers.quantization.mixllm_three_level import MixLLMThreeLevelConfig, MixLLMThreeLevelLinearMethod
config = {'quant_method': 'mixllm_three_level', 'precision_percentages': {'4': 0, '8': 0, '16': 100}, 'group_size': 128, 'backend': 'sm75'}
quant_config = MixLLMThreeLevelConfig.from_config(config)
assert quant_config.get_min_capability() == 75
method = MixLLMThreeLevelLinearMethod(quant_config)
layer = ThreeLevelLinear(128, 1, 128).cuda()
layer.mixllm_output_partition_sizes = [0, 0, 1]
layer.weight_fp16 = torch.ones((1, 128), device='cuda', dtype=torch.float16)
layer.indices_16 = torch.tensor([0], device='cuda', dtype=torch.int32)
layer.weight_int8 = torch.empty((0, 128), device='cuda', dtype=torch.int8)
layer.scale_int8 = torch.empty((0, 1), device='cuda', dtype=torch.float16)
layer.indices_8 = torch.empty((0,), device='cuda', dtype=torch.int32)
layer.weight_int4 = torch.empty((0, 64), device='cuda', dtype=torch.uint8)
layer.scale_int4 = torch.empty((0, 1), device='cuda', dtype=torch.float16)
layer.zero_int4 = torch.empty((0, 1), device='cuda', dtype=torch.uint8)
layer.indices_4 = torch.empty((0,), device='cuda', dtype=torch.int32)
x = torch.ones((2, 128), device='cuda', dtype=torch.float16)
y = method.apply(layer, x)
assert tuple(y.shape) == (2, 1), y.shape
assert torch.isfinite(y).all().item()
print('VLLM_APPLY_SMOKE_PASS', tuple(y.shape))
"""
        smoke_file = smoke_root / 'vllm_apply_smoke.py'
        smoke_file.write_text(smoke_code.replace('SMOKE_ROOT', repr(str(smoke_root))).replace('MIX_ROOT', repr(str(root))), encoding='utf-8')
        env = os.environ.copy()
        env['PYTHONPATH'] = str(smoke_root) + os.pathsep + str(root) + os.pathsep + env.get('PYTHONPATH', '')
        run = subprocess.run([sys.executable, str(smoke_file)], cwd=smoke_root, env=env, text=True, capture_output=True, timeout=600)
        print(run.stdout); print(run.stderr)
        if run.returncode != 0:
            raise RuntimeError('patched vLLM apply smoke failed')
        vllm_apply = {'status': 'passed', 'version': vllm_version, 'pinned_commit': '5fbbfe9a4c13094ad72ed3d6b4ef208a7ddc0fd7', 'patch_check': 'passed', 'apply_execution': 'passed'}
    except Exception as exc:
        vllm_apply = {'status': 'failed', 'error': repr(exc)}
report['vllm_apply'] = vllm_apply
report['gates']['vllm_apply_path'] = vllm_apply['status']
assert vllm_apply['status'] == 'passed', vllm_apply
"""
'@
if (-not $text.Contains($marker)) { throw 'build_notebook marker not found' }
$text = $text.Replace($marker, $cell + "`r`n" + $marker)
$oldTransition = '"""), cell("code", """from mixllm.model_gate import run_model_gate'
$newTransition = '"""), cell("code", VLLM_APPLY_CELL), cell("code", """from mixllm.model_gate import run_model_gate'
if (-not $text.Contains($oldTransition)) { throw 'cell transition marker not found' }
$text = $text.Replace($oldTransition, $newTransition)
$oldAssert = "assert execution, 'T4 gate did not execute completely; inspect artifact'"
$newAssert = "assert execution, 'T4 gate did not execute completely; inspect artifact'\nassert gates['vllm_apply_path'] == 'passed', 'patched vLLM apply path did not pass'"
if (-not $text.Contains($oldAssert)) { throw 'final assertion marker not found' }
$text = $text.Replace($oldAssert, $newAssert)
[IO.File]::WriteAllText($builder, $text, (New-Object Text.UTF8Encoding($false)))
Write-Output ('BUILDER='+$builder)
Write-Output ('C7_MARKER='+$text.Contains("vllm_apply_path"))
Write-Output ('PATCH_SOURCE='+$text.Contains('vllm_v0.9.0_patch/0002-add-mixllm-three-level-support.patch'))
