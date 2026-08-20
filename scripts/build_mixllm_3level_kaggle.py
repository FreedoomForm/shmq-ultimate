"""Build and validate the self-contained MixLLM T4 gate notebook."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / "shmq-ultimate" / "external" / "MixLLM"
OUT = ROOT / "shmq-ultimate" / "mixllm_3level_kaggle"
NOTEBOOK = OUT / "mixllm_3level_gate.ipynb"
METADATA = OUT / "kernel-metadata.json"
SOURCE_FILES = (
    "mixllm/__init__.py", "mixllm/quantization/__init__.py", "mixllm/nn/__init__.py",
    "mixllm/nn/modules/__init__.py", "mixllm/quantization/three_level.py",
    "mixllm/nn/modules/mixllm_config.py", "mixllm/nn/modules/three_level_linear.py",
    "mixllm/nn/modules/ops.py", "mixllm/runtime_capability.py", "mixllm/sm75_backend.py",
    "mixllm/model_gate.py", "mixllm/vllm_three_level.py", "mixllm/kernels/three_level_sm75.cu",
    "mixllm/kernels/cutlass_sm75_vendor.b64",
    "mixllm/kernels/sm75_cutlass_testbed.h",
    "mixllm/kernels/cutlass_extension/mq_mma_pipelined_sm75.h",
    "mixllm/kernels/cutlass_extension/mq_mma_sm75_int4_pair.h",
    "mixllm/kernels/cutlass_extension/mq_mma_base.h",
    "mixllm/kernels/cutlass_extension/mq_mma_tensor_op_dequantizer.h",
    "mixllm/kernels/cutlass_extension/mq_fine_grained_scale_zero_iterator.h",
    "mixllm/kernels/cutlass_extension/mq_numeric_conversion.h",
    "mixllm/test/test_three_level.py", "mixllm/test/test_runtime_capability.py",
    "mixllm/test/test_sm75_backend.py", "mixllm/test/test_sm75_source.py",
    "mixllm/test/test_model_gate.py", "mixllm/test/test_vllm_three_level.py",
    "mixllm/test/test_v51_audit_contract.py",
    "vllm_v0.9.0_patch/0002-add-mixllm-three-level-support.patch",
    "vllm_v0.9.0_patch/THREE_LEVEL_MANIFEST.md")
# CUTLASS and custom headers are unpacked from the embedded vendor archive
# before torch.utils.cpp_extension.load() runs on Kaggle.
def git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()
def git_dirty(cwd, ignored=()):
    repo_root = Path(git(["rev-parse", "--show-toplevel"], cwd)).resolve()
    ignored = {path.resolve() for path in ignored}
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=cwd, check=True,
        capture_output=True, text=True,
    ).stdout.splitlines()
    for line in status:
        relative = line[3:].split(" -> ")[-1]
        if (repo_root / relative).resolve() not in ignored:
            return True
    return False
def provenance(sources):
    hashes = {n: hashlib.sha256(t.encode()).hexdigest() for n, t in sources.items()}
    digest = hashlib.sha256()
    for n in sorted(sources):
        digest.update(n.encode() + b"\0" + sources[n].encode() + b"\0")
    return {"algorithm": "sha256", "source_sha256": digest.hexdigest(), "files": hashes,
            "workspace_commit": git(["rev-parse", "HEAD"], ROOT),
            "mixllm_commit": git(["rev-parse", "HEAD"], FORK),
            "workspace_dirty": git_dirty(ROOT, (NOTEBOOK,)),
            "mixllm_dirty": git_dirty(FORK, (NOTEBOOK,))}
def cell(kind, source):
    result = {"cell_type": kind, "id": hashlib.sha256((kind+"\0"+source).encode()).hexdigest()[:12],
              "metadata": {}, "source": source.splitlines(True)}
    if kind == "code": result.update(execution_count=None, outputs=[])
    return result
VLLM_APPLY_CELL = """import shutil, tempfile
patch_text = (root / 'vllm_v0.9.0_patch' / '0002-add-mixllm-three-level-support.patch').read_text(encoding='utf-8')
for _marker in ('get_min_capability', 'return 75', 'backend=auto or sm75', 'get_device_capability', 'three_level_linear'):
    assert _marker in patch_text, _marker
vllm_apply = {'status': 'not_run', 'patch_contract': 'passed'}
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
        smoke_code = '''import sys, torch
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
'''
        smoke_file = smoke_root / 'vllm_apply_smoke.py'
        smoke_file.write_text(smoke_code.replace('SMOKE_ROOT', repr(str(smoke_root))).replace('MIX_ROOT', repr(str(root))), encoding='utf-8')
        env = os.environ.copy()
        env['PYTHONPATH'] = str(smoke_root) + os.pathsep + str(root) + os.pathsep + env.get('PYTHONPATH', '')
        run = subprocess.run([sys.executable, str(smoke_file)], cwd=smoke_root, env=env, text=True, capture_output=True, timeout=600)
        print(run.stdout); print(run.stderr)
        if run.returncode != 0:
            raise RuntimeError('patched vLLM apply smoke failed')
        vllm_apply = {'status': 'passed', 'version': vllm_version, 'pinned_commit': '5fbbfe9a4c13094ad72ed3d6b4ef208a7ddc0fd7', 'patch_check': 'passed', 'apply_execution': 'passed'}
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
        vllm_apply = {'status': 'failed', 'patch_contract': 'passed', 'error': repr(exc)}
report['vllm_apply'] = vllm_apply
report['gates']['vllm_apply_path'] = vllm_apply['status']
assert vllm_apply['status'] in {'passed', 'unavailable_environment', 'not_run'}, vllm_apply
"""
def build_notebook(sources, manifest):
    cells = [cell("markdown", """# MixLLM 4/8/16 real T4 gate

This notebook embeds the current Python and CUDA sources and validates them on NVIDIA T4 / SM75. It runs model_gate import/allocator checks and native operator benchmarks for mixed and pure precision partitions. Operator timings are not model throughput. Full-model Qwen quality remains not_run unless separately measured.
"""), cell("code", """import hashlib, json, os, platform, subprocess, sys
from pathlib import Path
import torch
ARTIFACT_DIR = Path('/kaggle/working')
print('Python', sys.version)
print('STARTUP_HEARTBEAT', flush=True); print('PyTorch', torch.__version__, flush=True); print('DEVICE_COUNT', torch.cuda.device_count(), flush=True); print('ACTIVE_DEVICE', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, flush=True)
"""), cell("code", f"""sources = {sources!r}
source_manifest = {manifest!r}
root = ARTIFACT_DIR / 'mixllm-3level'
for relative, text in sources.items():
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source_manifest['files'][relative]
digest = hashlib.sha256()
for relative in sorted(sources): digest.update(relative.encode() + b'\\0' + sources[relative].encode() + b'\\0')
assert digest.hexdigest() == source_manifest['source_sha256']
source_manifest['embedded_file_count'] = len(sources)
(ARTIFACT_DIR / 'mixllm_3level_source_manifest.json').write_text(json.dumps(source_manifest, indent=2, sort_keys=True))
sys.path.insert(0, str(root))
print('Embedded source SHA-256:', source_manifest['source_sha256'])
"""), cell("code", """cuda_available = torch.cuda.is_available()
capability = tuple(torch.cuda.get_device_capability(0)) if cuda_available else None
gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
is_t4 = capability == (7, 5) and gpu_name and 'T4' in gpu_name.upper()
report = {'schema_version': 4, 'target': 'NVIDIA T4 / SM75', 'provenance': source_manifest,
          'environment': {'cuda_available': cuda_available, 'gpu_name': gpu_name, 'capability': capability,
                          'torch_version': torch.__version__, 'python_version': platform.python_version()},
          'gates': {'t4_hardware': 'passed' if is_t4 else 'failed',
                    'full_model_qwen_quality': 'not_run', 'full_model_qwen_throughput': 'not_run'},
          'claims': {'benchmark_scope': 'native_operator_microbenchmark', 'full_model_qwen_quality_claimed': False}}
"""), cell("code", """test_env = os.environ.copy()
test_env['PYTHONPATH'] = str(root) + os.pathsep + test_env.get('PYTHONPATH', '')
if capability == (7, 5): test_env['MIXLLM_TEST_SM75'] = '1'
test_modules = [
    'mixllm.test.test_three_level',
    'mixllm.test.test_runtime_capability',
    'mixllm.test.test_sm75_backend',
    'mixllm.test.test_sm75_source',
    'mixllm.test.test_model_gate',
    'mixllm.test.test_vllm_three_level',
    'mixllm.test.test_v51_audit_contract',
]
tests = subprocess.run([sys.executable, '-m', 'unittest', '-v', *test_modules], cwd=root, env=test_env, text=True, capture_output=True, timeout=600)
print(tests.stdout); print(tests.stderr)
report['tests'] = {'returncode': tests.returncode, 'model_gate_test_embedded': 'mixllm/test/test_model_gate.py' in sources}
report['gates']['embedded_contract_tests'] = 'passed' if tests.returncode == 0 else 'failed'
"""), cell("code", VLLM_APPLY_CELL), cell("code", """from mixllm.model_gate import run_model_gate
from mixllm.quantization.three_level import ThreeLevelBudget, allocate_channels, allocate_model_channels, allocate_model_channels_auto, estimate_channel_losses
assert callable(run_model_gate)
torch.manual_seed(1234); x = torch.randn(4, 3, 128); w = torch.randn(8, 128)
losses, awq_stat = estimate_channel_losses(x, w)
allocation = allocate_channels(losses, ThreeLevelBudget(50, 25, 25)); allocation.verify(w.shape[0])
fixed = allocate_model_channels({'layer': losses}, ThreeLevelBudget(50, 25, 25))
automatic, allocation_summary = allocate_model_channels_auto({'layer': losses}, 8.0)
fixed['layer'].verify(w.shape[0]); automatic['layer'].verify(w.shape[0])
assert allocation_summary['achieved_average_bits'] <= 8.0 and torch.isfinite(awq_stat).all()
report['gates'].update(model_gate_import='passed', fixed_allocator='passed', auto_allocator='passed')
report['allocator_check'] = allocation_summary
"""), cell("code", """quality = {
    'status': 'unavailable_environment',
    'model_id': 'Qwen/Qwen2.5-0.5B',
    'backend': 'not_run',
    'reason': 'requires the exact Kaggle Qwen2.5-0.5B model input',
}
if capability == (7, 5):
    expected_model = {
        'model_type': 'qwen2', 'hidden_size': 896, 'num_hidden_layers': 24,
        'vocab_size': 151936, 'intermediate_size': 4864,
        'num_attention_heads': 14,
    }
    model_roots = [
        Path('/kaggle/input/qwen2.5/transformers/0.5b/1'),
        Path('/kaggle/input/qwen2-5/transformers/0.5b/1'),
    ]
    # Never recursively scan the whole Kaggle input tree: model mounts are
    # deterministic for this notebook and an unbounded scan can stall startup.
    discovered = []
    for candidate in model_roots:
        config_path = candidate / 'config.json'
        if not config_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        if all(config.get(key) == value for key, value in expected_model.items()):
            discovered.append(candidate)
    model_root = next(iter(dict.fromkeys(discovered)), None)
    quality['model_candidates'] = [str(path) for path in discovered]
    if model_root is not None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from mixllm.model_gate import run_model_gate
            tokenizer = AutoTokenizer.from_pretrained(str(model_root), local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(model_root), torch_dtype=torch.float16, local_files_only=True,
            ).cuda()
            calibration_ids = tokenizer(
                'Mixed precision protects important channels.\\n'
                'A reproducible benchmark separates quality from speed.',
                return_tensors='pt', truncation=True, max_length=64,
            ).input_ids
            evaluation_ids = tokenizer(
                'The model must preserve quality while using less memory.',
                return_tensors='pt', truncation=True, max_length=64,
            ).input_ids
            result = run_model_gate(
                'Qwen/Qwen2.5-0.5B', tokenizer, model,
                calibration_ids, evaluation_ids, target_average_bits=8.0,
                group_size=128, calibration_rows=64,
                timing_warmup=2, timing_iterations=5,
            )
            result['quality_thresholds'] = {
                'max_loss_delta': 0.05,
                'max_last_token_logit_error': 5.0,
            }
            result['status'] = 'passed' if (
                result['finite'] and result['deterministic'] and
                result['loss_delta'] <= 0.05 and
                result['max_last_token_logit_error'] <= 5.0 and
                result['quantized_forward_ms'] is not None
            ) else 'failed'
            result['backend'] = 'native_capability_selected'
            quality = result
            del model
            torch.cuda.empty_cache()
        except ModuleNotFoundError as exc:
            quality['reason'] = f'missing runtime dependency: {exc.name}'
        except (OSError, RuntimeError) as exc:
            quality['reason'] = repr(exc)
            quality['status'] = 'failed' if isinstance(exc, RuntimeError) else 'unavailable_environment'
        except Exception as exc:
            quality.update(status='failed', reason=repr(exc))
    else:
        quality['reason'] = 'exact Qwen2.5-0.5B config fingerprint not found under /kaggle/input'
else:
    quality['reason'] = 'requires Tesla T4 / SM75'
report['full_model_quality'] = quality
report['gates']['full_model_qwen_quality'] = quality['status']
report['gates']['full_model_qwen_throughput'] = (
    'passed' if quality['status'] == 'passed' else quality['status']
)
report['claims']['full_model_qwen_quality_claimed'] = quality['status'] == 'passed'
"""), cell("code", """benchmarks = {'status': 'not_run', 'baseline': 'torch_fp16_linear', 'scenarios': {}}
if capability == (7, 5):
    from mixllm.nn.modules.three_level_linear import ThreeLevelLinear
    from mixllm.quantization.three_level import ThreeLevelAllocation, ThreeLevelBudget
    from mixllm.sm75_backend import benchmark_sm75_backend, load_sm75_backend
    load_sm75_backend(torch)
    def make_case(n, width, counts, rows):
        n4, n8, n16 = counts; assert n4 + n8 + n16 == n
        alloc = ThreeLevelAllocation(indices={4: tuple(range(n4)), 8: tuple(range(n4, n4+n8)), 16: tuple(range(n4+n8, n))}, scores={b: (0.0,) * n for b in (4, 8, 16)}, budget=ThreeLevelBudget(*(100*c/n for c in counts)))
        packed = ThreeLevelLinear.from_weight(torch.randn(n, width, device='cuda', dtype=torch.float16), alloc).cuda()
        return benchmark_sm75_backend(packed, rows=rows, torch_module=torch, warmup=10, iterations=50)
    cases = {'smoke_mixed_4_8_16': (96, 512, (64, 24, 8), (1, 8, 32, 128)), 'qwen_qkv_mixed_4_8_16': (3584, 3584, (2400, 896, 288), (1, 16, 128)), 'qwen_qkv_pure_int4': (3584, 3584, (3584, 0, 0), (1, 16, 128)), 'qwen_qkv_pure_int8': (3584, 3584, (0, 3584, 0), (1, 16, 128)), 'qwen_qkv_pure_fp16': (3584, 3584, (0, 0, 3584), (1, 16, 128))}
    benchmarks['scenarios'] = {name: make_case(*args) for name, args in cases.items()}
    benchmarks['status'] = 'measured'
report['benchmarks'] = benchmarks
(ARTIFACT_DIR / 'mixllm_3level_benchmarks.json').write_text(json.dumps(benchmarks, indent=2, sort_keys=True))
"""), cell("code", """gates = report['gates']
if benchmarks['status'] == 'measured':
    shapes = [shape for case in benchmarks['scenarios'].values() for shape in case['shapes']]
    mixed = benchmarks['scenarios']['qwen_qkv_mixed_4_8_16']['shapes']
    correctness = all(s['max_abs_error'] <= 0.15 for s in shapes)
    decode_gemm = all(s['gemm_p50_ratio_vs_dense'] <= 1.05 for s in mixed if s['rows'] == 1)
    decode_e2e = all(s['end_to_end_p50_ratio_vs_dense'] <= 1.05 for s in mixed if s['rows'] == 1)
    prefill_e2e = all(s['end_to_end_p50_ratio_vs_dense'] <= 1.05 for s in mixed if s['rows'] > 1)
    timing_integrity = all(s.get('timing_integrity', False) for s in mixed)
    gates.update(sm75_native_benchmarks='passed', sm75_native_correctness='passed' if correctness else 'failed', mixed_decode_gemm_performance='passed' if decode_gemm else 'failed', mixed_decode_end_to_end_performance='passed' if decode_e2e else 'failed', mixed_prefill_end_to_end_performance='passed' if prefill_e2e else 'failed', timing_integrity='passed' if timing_integrity else 'failed')
    operator_production = correctness and decode_e2e and prefill_e2e and timing_integrity
    model_vllm_production = (
        operator_production and
        gates.get('full_model_qwen_quality') == 'passed' and
        gates.get('full_model_qwen_throughput') == 'passed' and
        gates.get('vllm_apply_path') == 'passed'
    )
    production_ready = model_vllm_production
else:
    gates.update(sm75_native_benchmarks='not_run', sm75_native_correctness='not_run'); operator_production = False; model_vllm_production = False; production_ready = False
report['gates']['operator_production'] = 'passed' if operator_production else 'failed'
report['gates']['model_vllm_production'] = 'passed' if model_vllm_production else 'failed'
print('TESTS_RETURNCODE', tests.returncode, flush=True); print('TESTS_STDOUT_TAIL', tests.stdout[-2000:], flush=True); print('TESTS_STDERR_TAIL', tests.stderr[-2000:], flush=True); print('IS_T4', is_t4, flush=True); print('GATES_PRE_EXEC', gates, flush=True); print('BENCHMARKS_PRE_EXEC', benchmarks, flush=True); execution = bool(is_t4 and tests.returncode == 0 and gates.get('model_gate_import') == 'passed' and benchmarks['status'] == 'measured')
report['gate_status'] = {'execution': 'passed' if execution else 'failed', 'operator_production': 'passed' if operator_production else 'failed', 'model_vllm_production': 'passed' if model_vllm_production else 'failed', 't4_production': 'passed' if production_ready else 'failed', 'terminal_decision': 'go' if production_ready else 'no_go', 'reason': 'gate evaluation complete'}
(ARTIFACT_DIR / 'mixllm_3level_gate.json').write_text(json.dumps(report, indent=2, sort_keys=True))
print(json.dumps(report['gate_status'], indent=2)); print('Full-model Qwen quality:', gates['full_model_qwen_quality'])
assert execution, 'T4 gate did not execute completely; inspect artifact'""")]
    return {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.12'}}, 'nbformat': 4, 'nbformat_minor': 5}
def metadata():
    return {
        'id': 'freedomform/mixllm-3-level-real-t4-gate',
        'title': 'MixLLM 3 Level Real T4 Gate',
        'code_file': NOTEBOOK.name,
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': True,
        'enable_gpu': True,
        'enable_internet': False,
        'machine_shape': 'NvidiaTeslaT4',
        # Exact base model requested by the project; never substitute an
        # instruction-tuned or differently sized Qwen checkpoint.
        'model_sources': ['qwen-lm/qwen2.5/Transformers/0.5b/1'],
    }
def build():
    sources = {n: (FORK / n).read_text(encoding='utf-8') for n in SOURCE_FILES}
    OUT.mkdir(parents=True, exist_ok=True); NOTEBOOK.write_text(json.dumps(build_notebook(sources, provenance(sources)), indent=1), encoding='utf-8'); METADATA.write_text(json.dumps(metadata(), indent=2), encoding='utf-8'); print(NOTEBOOK)
def validate():
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8')); assert json.loads(METADATA.read_text()) == metadata(); sources = {n: (FORK / n).read_text(encoding='utf-8') for n in SOURCE_FILES}; assert notebook == build_notebook(sources, provenance(sources)), 'notebook is stale; rebuild it'
    text = ''.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
    for marker in ('model_gate.py', 'three_level_sm75.cu', 'test_model_gate.py', 'test_v51_audit_contract.py', 'source_sha256', 'workspace_commit', 'qwen_qkv_mixed_4_8_16', 'qwen_qkv_pure_int4', 'qwen_qkv_pure_int8', 'qwen_qkv_pure_fp16', 'full_model_qwen_quality', 'terminal_decision', 'mixllm_3level_source_manifest.json', 'mixllm_3level_benchmarks.json'): assert marker in text, marker
    for c in notebook['cells']:
        if c['cell_type'] == 'code': compile(''.join(c['source']), c['id'], 'exec')
    print(f'Validated {NOTEBOOK} ({len(SOURCE_FILES)} embedded files)')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true'); args = parser.parse_args(); validate() if args.check else build()
