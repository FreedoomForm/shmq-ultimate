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
    "mixllm/test/test_three_level.py", "mixllm/test/test_runtime_capability.py",
    "mixllm/test/test_sm75_backend.py", "mixllm/test/test_sm75_source.py",
    "mixllm/test/test_model_gate.py", "mixllm/test/test_vllm_three_level.py")
def git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()
def provenance(sources):
    hashes = {n: hashlib.sha256(t.encode()).hexdigest() for n, t in sources.items()}
    digest = hashlib.sha256()
    for n in sorted(sources):
        digest.update(n.encode() + b"\0" + sources[n].encode() + b"\0")
    return {"algorithm": "sha256", "source_sha256": digest.hexdigest(), "files": hashes,
            "workspace_commit": git(["rev-parse", "HEAD"], ROOT),
            "mixllm_commit": git(["rev-parse", "HEAD"], FORK),
            "workspace_dirty": bool(git(["status", "--porcelain"], ROOT)),
            "mixllm_dirty": bool(git(["status", "--porcelain"], FORK))}
def cell(kind, source):
    result = {"cell_type": kind, "id": hashlib.sha256((kind+"\0"+source).encode()).hexdigest()[:12],
              "metadata": {}, "source": source.splitlines(True)}
    if kind == "code": result.update(execution_count=None, outputs=[])
    return result
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
tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(root / 'mixllm/test'), '-p', 'test_three_level.py'], cwd=root, env=test_env, text=True, capture_output=True, timeout=600)
print(tests.stdout); print(tests.stderr)
report['tests'] = {'returncode': tests.returncode, 'model_gate_test_embedded': 'mixllm/test/test_model_gate.py' in sources}
report['gates']['embedded_contract_tests'] = 'passed' if tests.returncode == 0 else 'failed'
"""), cell("code", """from mixllm.model_gate import run_model_gate
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
    gates.update(sm75_native_benchmarks='passed', sm75_native_correctness='passed' if correctness else 'failed', mixed_decode_gemm_performance='passed' if decode_gemm else 'failed', mixed_decode_end_to_end_performance='passed' if decode_e2e else 'failed', mixed_prefill_end_to_end_performance='passed' if prefill_e2e else 'failed')
    production_ready = correctness and decode_e2e and prefill_e2e
else:
    gates.update(sm75_native_benchmarks='not_run', sm75_native_correctness='not_run'); production_ready = False
print('TESTS_RETURNCODE', tests.returncode, flush=True); print('TESTS_STDOUT_TAIL', tests.stdout[-2000:], flush=True); print('TESTS_STDERR_TAIL', tests.stderr[-2000:], flush=True); execution = bool(is_t4 and tests.returncode == 0 and gates.get('model_gate_import') == 'passed' and benchmarks['status'] == 'measured')
report['gate_status'] = {'execution': 'passed' if execution else 'failed', 't4_production': 'passed' if production_ready else 'failed', 'terminal_decision': 'go' if production_ready else 'no_go', 'reason': 'all native correctness and mixed end-to-end gates passed' if production_ready else 'production requires native correctness plus mixed decode and prefill end-to-end performance'}
(ARTIFACT_DIR / 'mixllm_3level_gate.json').write_text(json.dumps(report, indent=2, sort_keys=True))
print(json.dumps(report['gate_status'], indent=2)); print('Full-model Qwen quality:', gates['full_model_qwen_quality'])
assert execution, 'T4 gate did not execute completely; inspect artifact'
""")]
    return {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.12'}}, 'nbformat': 4, 'nbformat_minor': 5}
def metadata():
    return {'id': 'freedomform/mixllm-3-level-real-t4-gate', 'title': 'MixLLM 3 Level Real T4 Gate', 'code_file': NOTEBOOK.name, 'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_internet': False, 'machine_shape': 'NvidiaTeslaT4'}
def build():
    sources = {n: (FORK / n).read_text(encoding='utf-8') for n in SOURCE_FILES}
    OUT.mkdir(parents=True, exist_ok=True); NOTEBOOK.write_text(json.dumps(build_notebook(sources, provenance(sources)), indent=1), encoding='utf-8'); METADATA.write_text(json.dumps(metadata(), indent=2), encoding='utf-8'); print(NOTEBOOK)
def validate():
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8')); assert json.loads(METADATA.read_text()) == metadata(); sources = {n: (FORK / n).read_text(encoding='utf-8') for n in SOURCE_FILES}; assert notebook == build_notebook(sources, provenance(sources)), 'notebook is stale; rebuild it'
    text = ''.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
    for marker in ('model_gate.py', 'three_level_sm75.cu', 'test_model_gate.py', 'source_sha256', 'workspace_commit', 'qwen_qkv_mixed_4_8_16', 'qwen_qkv_pure_int4', 'qwen_qkv_pure_int8', 'qwen_qkv_pure_fp16', 'full_model_qwen_quality', 'terminal_decision', 'mixllm_3level_source_manifest.json', 'mixllm_3level_benchmarks.json'): assert marker in text, marker
    for c in notebook['cells']:
        if c['cell_type'] == 'code': compile(''.join(c['source']), c['id'], 'exec')
    print(f'Validated {NOTEBOOK} ({len(SOURCE_FILES)} embedded files)')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true'); args = parser.parse_args(); validate() if args.check else build()
