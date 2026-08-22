#!/usr/bin/env python3
"""Run the complete SHMQ gate on a fresh Colab T4 VM.

This follows googlecolab/google-colab-cli's documented ``colab run`` and
``colab exec`` model. The gate notebook remains the single source of
benchmark inputs, thresholds, and production decision logic. Its code cells
are executed directly in the current Colab kernel, avoiding a nested Jupyter
server inside the remote kernel.
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
EXPECTED_COMMIT = "d4d841a"
ROOT = Path("/content/shmq-ultimate-v300")
REPO_ROOT = ROOT / "shmq-ultimate"
NOTEBOOK = REPO_ROOT / "mixllm_3level_kaggle" / "mixllm_3level_gate.ipynb"
ARTIFACT_DIR = Path("/kaggle/working")
MODEL_ROOT = Path("/content/qwen2.5/transformers/0.5b/1")


def run(*args: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(args), flush=True)
    return subprocess.run(args, cwd=cwd, env=env, text=True, check=True)


def install_runtime_dependencies() -> None:
    """Install dependencies absent from the Colab runtime."""
    missing = []
    for module, package in (
        ("transformers", "transformers>=4.45"),
        ("huggingface_hub", "huggingface_hub>=0.25"),
    ):
        try:
            __import__(module)
        except ModuleNotFoundError:
            missing.append(package)
    if shutil.which("ninja") is None:
        missing.append("ninja")
    if missing:
        run(sys.executable, "-m", "pip", "install", "--quiet", *missing)


def prepare_qwen_model() -> None:
    """Download the public exact Qwen2.5-0.5B model into writable storage."""
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
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


def execute_gate_cells(colab_notebook: Path) -> int:
    """Execute the gate notebook's code cells in this Colab kernel."""
    notebook = json.loads(colab_notebook.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__", "__file__": str(colab_notebook)}
    operator_only = os.environ.get("SHMQ_OPERATOR_ONLY") == "1"
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        if operator_only and index == 7:
            print("COLAB_SKIP_GATE_CELL 7 operator_only", flush=True)
            continue
        print(f"COLAB_EXEC_GATE_CELL {index}", flush=True)
        source = "".join(cell.get("source", []))
        exec(compile(source, f"{colab_notebook}:cell-{index}", "exec"), namespace, namespace)
    return 0


def main() -> int:
    notebook_override = os.environ.get("SHMQ_NOTEBOOK_PATH")
    if notebook_override:
        notebook_source = Path(notebook_override)
        if not notebook_source.is_file():
            raise RuntimeError(f"uploaded notebook not found: {notebook_source}")
        print("SHMQ_NOTEBOOK_SOURCE", notebook_source, flush=True)
    else:
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
            raise RuntimeError(f"expected v300 commit prefix {EXPECTED_COMMIT}, got {commit}")
        notebook_source = NOTEBOOK

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

    if os.environ.get("SHMQ_OPERATOR_ONLY") == "1":
        print("QWEN_MODEL_STAGE", "skipped_operator_only", flush=True)
    else:
        prepare_qwen_model()
    colab_notebook = Path("/content/mixllm_3level_gate_colab_v300.ipynb")
    notebook_text = notebook_source.read_text(encoding="utf-8")
    notebook_text = notebook_text.replace(
        "/kaggle/input/qwen2.5/transformers/0.5b/1", str(MODEL_ROOT),
    )
    notebook_text = notebook_text.replace(
        "/kaggle/input/qwen2-5/transformers/0.5b/1", str(MODEL_ROOT),
    )
    colab_notebook.write_text(notebook_text, encoding="utf-8")
    return execute_gate_cells(colab_notebook)


if __name__ == "__main__":
    raise SystemExit(main())
