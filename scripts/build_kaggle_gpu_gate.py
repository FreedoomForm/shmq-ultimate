"""Build a self-contained Kaggle notebook for the strict CUDA gate test."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "shmq-ultimate" / "src" / "shmq" / "inference" / "shmq_3level_kernel.py"
OUT_DIR = ROOT / "shmq-ultimate" / "kaggle_gpu_gate"
NOTEBOOK = OUT_DIR / "shmq_gpu_gate.ipynb"
METADATA = OUT_DIR / "kernel-metadata.json"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True),
    }


def build() -> None:
    kernel_source = SOURCE.read_text(encoding="utf-8")
    embedded_source = repr(kernel_source)
    cells = [
        markdown(
            "# SHMQ-Ultimate strict GPU gate\n\n"
            "This notebook is a correctness gate for the three-level CUDA kernel.\n"
            "It deliberately fails when CUDA/CuPy/NVRTC is unavailable or when the\n"
            "implementation silently falls back to PyTorch. It does not claim tensor\n"
            "core performance or Qwen quality; those require separate benchmarks.\n"
        ),
        code(
            "import os, platform, sys, time\n"
            "import torch\n\n"
            "print('Python:', sys.version.split()[0])\n"
            "print('PyTorch:', torch.__version__)\n"
            "assert torch.cuda.is_available(), 'CUDA is unavailable'\n"
            "name = torch.cuda.get_device_name(0)\n"
            "cap = torch.cuda.get_device_capability(0)\n"
            "props = torch.cuda.get_device_properties(0)\n"
            "print('GPU:', name)\n"
            "print('Compute capability:', cap)\n"
            "print('VRAM GiB:', props.total_memory / 2**30)\n"
            "assert cap == (7, 5), f'Expected T4 sm_75, got sm_{cap[0]}{cap[1]}'\n"
            "assert props.total_memory >= 15 * 2**30, 'Expected a 16 GiB-class T4'\n"
            "import cupy as cp\n"
            "print('CuPy:', cp.__version__)\n"
            "print('CUDA runtime:', cp.cuda.runtime.runtimeGetVersion())\n"
            "assert cp.cuda.runtime.getDevice() == torch.cuda.current_device()\n"
        ),
        code(
            "# Load exactly the source committed in this workspace; no repository clone or fallback copy.\n"
            f"_source = {embedded_source}\n"
            "module = type(sys)('shmq_kernel_under_test')\n"
            "module.__file__ = 'embedded:shmq_3level_kernel.py'\n"
            "sys.modules[module.__name__] = module\n"
            "exec(compile(_source, module.__file__, 'exec'), module.__dict__)\n"
            "print('Loaded kernel source:', len(module.SHMQ_3LEVEL_KERNEL_CUDA), 'CUDA chars')\n"
            "assert 'mma.sync' not in module.SHMQ_3LEVEL_KERNEL_CUDA\n"
            "print('Confirmed: this gate measures the implemented CUDA-cores path, not an unverified MMA claim.')\n"
        ),
        code(
            "# Strict correctness tests. Every call requires the actual RawKernel path.\n"
            "torch.manual_seed(1234)\n"
            "cases = [(1, 128, 1, 1, 1), (7, 256, 17, 19, 23), (33, 384, 32, 32, 32), (64, 512, 31, 37, 29)]\n"
            "results = []\n"
            "for M, K, N16, N8, N4 in cases:\n"
            "    device = 'cuda'\n"
            "    X = torch.randn(M, K, dtype=torch.float16, device=device) * 0.1\n"
            "    W16 = torch.randn(N16, K, dtype=torch.float16, device=device) * 0.1\n"
            "    W8 = torch.randint(-127, 128, (N8, K), dtype=torch.int8, device=device)\n"
            "    W4_codes = torch.randint(-8, 8, (N4, K), dtype=torch.int8, device=device)\n"
            "    W4 = module._pack_int4_on_gpu(W4_codes)\n"
            "    groups = K // 128\n"
            "    S8 = torch.rand(N8, groups, dtype=torch.float16, device=device) * 0.02\n"
            "    S4 = torch.rand(N4, groups, dtype=torch.float16, device=device) * 0.1\n"
            "    t0 = time.perf_counter()\n"
            "    actual = module.shmq_3level_gemm(X, W16, W8, W4, S8, S4, require_cuda_kernel=True)\n"
            "    torch.cuda.synchronize()\n"
            "    elapsed = time.perf_counter() - t0\n"
            "    reference = module._pytorch_fallback(X, W16, W8, W4, S8, S4, True, N16, N8, N4, K, N16 + N8 + N4)\n"
            "    diff = (actual.float() - reference.float()).abs().max().item()\n"
            "    results.append((M, K, N16, N8, N4, diff, elapsed))\n"
            "    print(f'M={M:3d} K={K:3d} regions=({N16},{N8},{N4}) max_abs_diff={diff:.6g} elapsed={elapsed:.3f}s')\n"
            "    assert diff <= 2e-2, f'kernel mismatch: {diff}'\n"
            "print('STRICT CUDA CORRECTNESS: PASS')\n"
        ),
        code(
            "# Test the public verification helper as an additional no-fallback check.\n"
            "_, _, helper_diff = module.verify_against_pytorch(M=64, K=256, N=96, N16=32, N8=32, N4=32)\n"
            "print('verify_against_pytorch max_abs_diff:', helper_diff)\n"
            "assert helper_diff <= 2e-2\n"
            "print('GPU GATE: PASS')\n"
        ),
    ]
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}, "nbformat": 4, "nbformat_minor": 5}
    metadata = {"id": "freedomform/shmq-ultimate-gpu-gate", "title": "SHMQ Ultimate GPU Gate", "code_file": "shmq_gpu_gate.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(NOTEBOOK)
    print(METADATA)


if __name__ == "__main__":
    build()