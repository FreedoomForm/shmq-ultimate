#!/usr/bin/env python3
"""Run the complete SHMQ gate on a fresh Colab T4 VM.

This follows googlecolab/google-colab-cli's documented ``colab run`` model:
the local script is sent to a fresh named T4 runtime and the VM is released by
the CLI after completion. The gate notebook remains the single source of
benchmark inputs, thresholds, and production decision logic.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/FreedoomForm/shmq-ultimate.git"
BRANCH = "unified-three-level-sm75"
EXPECTED_COMMIT = "7be8661"
ROOT = Path("/content/shmq-ultimate-v292")
REPO_ROOT = ROOT / "shmq-ultimate"
NOTEBOOK = REPO_ROOT / "mixllm_3level_kaggle" / "mixllm_3level_gate.ipynb"
ARTIFACT_DIR = Path("/kaggle/working")
MODEL_ROOT = Path("/content/qwen2.5/transformers/0.5b/1")


def run(*args: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(args), flush=True)
    return subprocess.run(args, cwd=cwd, env=env, text=True, check=True)


def install_runtime_dependencies() -> None:
    """Install only dependencies absent from the Colab runtime."""
    missing = []
    for module, package in (
        ("transformers", "transformers>=4.45"),
        ("huggingface_hub", "huggingface_hub>=0.25"),
        ("nbconvert", "nbconvert"),
    ):
        try:
            __import__(module)
        except ModuleNotFoundError:
            missing.append(package)
    if missing:
        run(sys.executable, "-m", "pip", "install", "--quiet", *missing)


def prepare_qwen_model() -> None:
    """Download the public exact Qwen2.5-0.5B model into a writable path."""
    from huggingface_hub import snapshot_download

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    config_path = MODEL_ROOT / "config.json"
    if config_path.is_file():
        print("QWEN_MODEL_CACHE", "present", flush=True)
        return
    print("QWEN_MODEL_DOWNLOAD", "Qwen/Qwen2.5-0.5B", flush=True)
    snapshot_download(
        repo_id="Qwen/Qwen2.5-0.5B",
        local_dir=str(MODEL_ROOT),
        allow_patterns=[
            "config.json", "generation_config.json", "*.safetensors",
            "*.safetensors.index.json", "tokenizer*", "special_tokens_map.json",
            "vocab.json", "merges.txt",
        ],
    )
    if not config_path.is_file():
        raise RuntimeError(f"Qwen download did not create {config_path}")


def prepare_paths() -> None:
    """Create the writable artifact path expected by the notebook."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    run("git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(ROOT))
    run("git", "-C", str(ROOT), "fetch", "--unshallow")
    run("git", "-C", str(ROOT), "checkout", "--detach", EXPECTED_COMMIT)
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
    ).strip()
    print("SHMQ_COMMIT", commit, flush=True)
    if not commit.startswith(EXPECTED_COMMIT):
        raise RuntimeError(f"expected v291 commit prefix {EXPECTED_COMMIT}, got {commit}")

    prepare_paths()
    install_runtime_dependencies()
    import torch

    print("TORCH_VERSION", torch.__version__, flush=True)
    print("CUDA_AVAILABLE", torch.cuda.is_available(), flush=True)
    print("DEVICE_COUNT", torch.cuda.device_count(), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Colab did not allocate a CUDA device")
    capability = tuple(torch.cuda.get_device_capability(0))
    name = torch.cuda.get_device_name(0)
    print("GPU_NAME", name, flush=True)
    print("GPU_CAPABILITY", capability, flush=True)
    if capability != (7, 5) or "T4" not in name.upper():
        raise RuntimeError(f"expected Tesla T4 / SM75, got {name} {capability}")

    prepare_qwen_model()
    colab_notebook = Path("/content/mixllm_3level_gate_colab_v292.ipynb")
    notebook_text = NOTEBOOK.read_text(encoding="utf-8")
    notebook_text = notebook_text.replace(
        "/kaggle/input/qwen2.5/transformers/0.5b/1", str(MODEL_ROOT),
    )
    notebook_text = notebook_text.replace(
        "/kaggle/input/qwen2-5/transformers/0.5b/1", str(MODEL_ROOT),
    )
    colab_notebook.write_text(notebook_text, encoding="utf-8")
    output_name = "mixllm_3level_gate_colab_v292_output.ipynb"
    command = [
        sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
        "--execute", str(colab_notebook), "--output", output_name,
        "--output-dir", "/content", "--ExecutePreprocessor.timeout=1800",
        "--ExecutePreprocessor.kernel_name=python3",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "external" / "MixLLM") + os.pathsep + env.get("PYTHONPATH", "")
    env["MIXLLM_TEST_SM75"] = "1"
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    print("NOTEBOOK_STDOUT_BEGIN", flush=True)
    print(result.stdout[-12000:], flush=True)
    print("NOTEBOOK_STDOUT_END", flush=True)
    print("NOTEBOOK_STDERR_BEGIN", flush=True)
    print(result.stderr[-12000:], flush=True)
    print("NOTEBOOK_STDERR_END", flush=True)
    print("NOTEBOOK_RETURN_CODE", result.returncode, flush=True)

    report_path = ARTIFACT_DIR / "mixllm_3level_gate.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(
            "COLAB_GATE_STATUS",
            json.dumps(report.get("gate_status", {}), sort_keys=True),
            flush=True,
        )
        print(
            "COLAB_GATE_BENCHMARKS",
            json.dumps(report.get("benchmarks", {}), sort_keys=True),
            flush=True,
        )
        print(
            "COLAB_GATE_QUALITY",
            json.dumps(report.get("full_model_quality", {}), sort_keys=True),
            flush=True,
        )
    else:
        print("COLAB_GATE_ARTIFACT", "missing", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
