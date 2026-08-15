"""Build the first correctness notebook for the clean MixLLM three-level fork."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / "shmq-ultimate" / "external" / "MixLLM"
OUT = ROOT / "shmq-ultimate" / "mixllm_3level_kaggle"


def cell(kind: str, source: str) -> dict:
    cell_id = hashlib.sha256(f"{kind}\0{source}".encode("utf-8")).hexdigest()[:12]
    result = {
        "cell_type": kind,
        "id": cell_id,
        "metadata": {},
        "source": source.splitlines(True),
    }
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def build() -> None:
    sources = {}
    for relative in (
        "mixllm/__init__.py",
        "mixllm/quantization/__init__.py",
        "mixllm/nn/__init__.py",
        "mixllm/nn/modules/__init__.py",
        "mixllm/quantization/three_level.py",
        "mixllm/nn/modules/mixllm_config.py",
        "mixllm/nn/modules/three_level_linear.py",
        "mixllm/nn/modules/ops.py",
        "mixllm/runtime_capability.py",
        "mixllm/sm75_backend.py",
        "mixllm/vllm_three_level.py",
        "mixllm/kernels/three_level_sm75.cu",
        "mixllm/test/test_three_level.py",
        "mixllm/test/test_runtime_capability.py",
        "mixllm/test/test_sm75_backend.py",
        "mixllm/test/test_vllm_three_level.py",
    ):
        sources[relative] = (FORK / relative).read_text(encoding="utf-8")
    embedded = repr(sources)
    cells = [
        cell("markdown", "# MixLLM 4/8/16 staged gate\n\n"
             "This notebook runs deterministic reference, allocation, packing and capability gates. "
             "Model quality and throughput are reported only when the requested model and runtime are available; "
             "unsupported native backends remain explicit rather than being counted as passes. "
             "SM75 measurements are operator microbenchmarks, not model throughput claims.\n"),
        cell("code", "import json, os, platform, subprocess, sys, tempfile\n"
             "from pathlib import Path\nimport torch\n"
             "print('Python', sys.version)\nprint('PyTorch', torch.__version__)\n"
             "print('CUDA available', torch.cuda.is_available())\n"
             "if torch.cuda.is_available():\n"
             "    print('GPU', torch.cuda.get_device_name(0))\n"
             "    print('Capability', torch.cuda.get_device_capability(0))\n"),
        cell("code", f"sources = {embedded}\n"
             "root = Path('/kaggle/working/mixllm-3level')\n"
             "for relative, text in sources.items():\n"
             "    path = root / relative\n    path.parent.mkdir(parents=True, exist_ok=True)\n"
             "    path.write_text(text, encoding='utf-8')\n"
             "sys.path.insert(0, str(root))\nprint('Embedded source files:', len(sources))\n"),
        cell("code", "test_env = os.environ.copy()\n"
             "test_env['PYTHONPATH'] = str(root) + os.pathsep + test_env.get('PYTHONPATH', '')\n"
             "cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None\n"
             "if cap == (7, 5): test_env['MIXLLM_TEST_SM75'] = '1'\n"
             "result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', "
             "'-s', str(root / 'mixllm/test'), '-p', 'test_*.py', '-v'], cwd=root, env=test_env, "
             "text=True, capture_output=True)\nprint(result.stdout)\nprint(result.stderr)\n"
             "assert result.returncode == 0\n"),
        cell("code", "from mixllm.runtime_capability import RuntimeCapability\n"
             "cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None\n"
             "gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None\n"
             "backend = RuntimeCapability(*cap).select_backend() if cap else 'reference'\n"
             "is_t4 = cap == (7, 5) and gpu_name is not None and 'T4' in gpu_name\n"
             "sm75_test_requested = test_env.get('MIXLLM_TEST_SM75') == '1'\n"
              "report = {'schema_version': 3, 'capability': cap, 'gpu_name': gpu_name,\n"
             "          'selected_production_backend': backend,\n"
             "          'cuda_available': torch.cuda.is_available(),\n"
              "          'reference_contract_gate': 'passed', 't4_hardware_gate': is_t4,\n"
              "          'sm75_correctness_kernel_gate': ('passed' if sm75_test_requested else 'not_run'),\n"
              "          'sm75_production_backend_ready': False,\n"
              "          'native_three_level_gate': ('correctness_only' if sm75_test_requested else 'not_run'),\n"
              "          'vllm_plugin_gate': 'contract_only',\n"
              "          'model_quality_gate': 'not_run',\n"
              "          'throughput_gate': 'not_run'}\n"
             "Path('/kaggle/working/mixllm_3level_gate.json').write_text(json.dumps(report, indent=2))\n"
             "print(report)\n"
             "if cap == (7, 5):\n"
             "    assert backend == 'reference'\n"
             "    assert report['sm75_correctness_kernel_gate'] == 'passed'\n"
             "print('MIXLLM THREE-LEVEL REFERENCE GATE: PASS')\n"),
        cell("code", "# Deterministic fake-quant and allocation gate, independent of CUDA kernels.\n"
              "from mixllm.quantization.three_level import (ThreeLevelBudget, allocate_channels,\n"
              "    estimate_channel_losses, allocate_model_channels)\n"
              "torch.manual_seed(1234)\n"
              "x = torch.randn(4, 3, 128)\n"
              "w = torch.randn(8, 128)\n"
              "losses, awq_stat = estimate_channel_losses(x, w)\n"
              "allocation = allocate_channels(losses, ThreeLevelBudget(50, 25, 25))\n"
              "allocation.verify(w.shape[0])\n"
              "assert torch.isfinite(awq_stat).all() and torch.isfinite(losses[4]).all()\n"
              "model_alloc = allocate_model_channels({'layer': losses}, ThreeLevelBudget(50, 25, 25))\n"
              "assert sum(len(model_alloc['layer'].indices[b]) for b in (4, 8, 16)) == w.shape[0]\n"
              "report['fake_quant_gate'] = 'passed'\n"
              "print('FAKE QUANT / GLOBAL ALLOCATION GATE: PASS')\n"),
        cell("code", "# CUDA-event operator benchmark on the real SM75 implementation.\n"
              "sm75_benchmark = {'status': 'not_run'}\n"
              "if cap == (7, 5):\n"
              "    from mixllm.nn.modules.three_level_linear import ThreeLevelLinear\n"
              "    from mixllm.quantization.three_level import ThreeLevelAllocation, ThreeLevelBudget\n"
              "    from mixllm.sm75_backend import benchmark_sm75_backend, load_sm75_backend\n"
              "    load_sm75_backend(torch)\n"
              "    n = 96; width = 512\n"
              "    alloc = ThreeLevelAllocation(indices={4: tuple(range(0, 64)), 8: tuple(range(64, 88)), 16: tuple(range(88, 96))}, scores={b: (0.0,) * n for b in (4, 8, 16)}, budget=ThreeLevelBudget(67, 25, 8))\n"
              "    packed = ThreeLevelLinear.from_weight(torch.randn(n, width, device='cuda', dtype=torch.float16), alloc).cuda()\n"
              "    sm75_benchmark = benchmark_sm75_backend(packed, rows=(1, 8, 32, 128), torch_module=torch, warmup=10, iterations=50)\n"
              "    errors_ok = all(shape['max_abs_error'] <= 0.05 for shape in sm75_benchmark['shapes'])\n"
              "    performance_ok = all(shape['p50_ratio_vs_dense'] <= 1.05 for shape in sm75_benchmark['shapes'])\n"
              "    report['sm75_graph_capture_gate'] = 'passed'\n"
              "    report['sm75_production_backend_ready'] = bool(errors_ok and performance_ok)\n"
              "    report['throughput_gate'] = 'operator_microbenchmark_measured'\n"
              "report['sm75_operator_benchmark'] = sm75_benchmark\n"
              "Path('/kaggle/working/mixllm_3level_gate.json').write_text(json.dumps(report, indent=2))\n"
              "print(json.dumps(report, indent=2))\n"),
    ]
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
    metadata = {"id": "freedomform/mixllm-3-level-reference-gate", "title": "MixLLM 3 Level Reference Gate", "code_file": "mixllm_3level_gate.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mixllm_3level_gate.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    (OUT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    build()