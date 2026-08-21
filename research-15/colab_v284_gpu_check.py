#!/usr/bin/env python3
"""Run the committed v284 SM75 contracts on a fresh Colab GPU VM.

This script deliberately runs the existing repository tests and does not change
benchmark inputs, precision, model, or acceptance thresholds. It is a compile /
operator-correctness check; Kaggle remains the authoritative performance gate.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/FreedoomForm/shmq-ultimate.git"
BRANCH = "unified-three-level-sm75"
EXPECTED_COMMIT = "7be8661"
ROOT = pathlib.Path("/content/shmq-ultimate-v291")
FORK = ROOT / "shmq-ultimate" / "external" / "MixLLM"


def run(*args: str, cwd: pathlib.Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(args), flush=True)
    return subprocess.run(args, cwd=cwd, env=env, text=True, check=True)


def main() -> int:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    run("git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(ROOT))
    run("git", "-C", str(ROOT), "fetch", "--unshallow")
    run("git", "-C", str(ROOT), "checkout", "--detach", EXPECTED_COMMIT)
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    print("SHMQ_COMMIT", commit, flush=True)
    if not commit.startswith(EXPECTED_COMMIT):
        raise RuntimeError(f"expected v286 commit prefix {EXPECTED_COMMIT}, got {commit}")

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
    if capability != (7, 5):
        raise RuntimeError(f"expected SM75/T4 for this check, got {name} {capability}")

    if shutil.which("ninja") is None:
        run(sys.executable, "-m", "pip", "install", "--quiet", "ninja")
    if shutil.which("ninja") is None:
        raise RuntimeError("Ninja installation did not provide an executable")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(FORK) + os.pathsep + env.get("PYTHONPATH", "")
    env["MIXLLM_TEST_SM75"] = "1"
    modules = [
        "mixllm.test.test_three_level",
        "mixllm.test.test_runtime_capability",
        "mixllm.test.test_sm75_backend",
        "mixllm.test.test_sm75_source",
        "mixllm.test.test_model_gate",
        "mixllm.test.test_vllm_three_level",
        "mixllm.test.test_v51_audit_contract",
    ]
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "-v", *modules],
        cwd=FORK,
        env=env,
        text=True,
        capture_output=True,
    )
    print("UNITTEST_STDOUT_BEGIN", flush=True)
    print(result.stdout, flush=True)
    print("UNITTEST_STDOUT_END", flush=True)
    print("UNITTEST_STDERR_BEGIN", flush=True)
    print(result.stderr, flush=True)
    print("UNITTEST_STDERR_END", flush=True)
    print("TEST_RETURN_CODE", result.returncode, flush=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    print("COLAB_V291_SM75_CHECK_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

