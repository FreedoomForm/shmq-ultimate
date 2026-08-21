#!/usr/bin/env python3
"""Diagnostic only: test the pre-existing packed INT4 M64/N64 path on SM75.

The production checkout is cloned unchanged, then the temporary Colab clone
removes the v283 M128/N64 alias and routes all packed-INT4 rows through M64/N64.
This isolates whether the generic CUTLASS uint4 iterator failure is M128-only or
fundamental to both aliases. It is not an acceptance run and does not change the
repository source or benchmark gates.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/FreedoomForm/shmq-ultimate.git"
BRANCH = "unified-three-level-sm75"
EXPECTED_COMMIT = "ccd7433"
ROOT = pathlib.Path("/content/shmq-ultimate-v285-m64-probe")
FORK = ROOT / "shmq-ultimate" / "external" / "MixLLM"


def run(*args: str, cwd: pathlib.Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(args), flush=True)
    return subprocess.run(args, cwd=cwd, env=env, text=True, check=True)


def patch_m64_only() -> None:
    testbed_path = FORK / "mixllm" / "kernels" / "sm75_cutlass_testbed.h"
    testbed = testbed_path.read_text(encoding="utf-8")
    alias_block = """using CorePackedInt4M128N64 = cutlass::gemm::threadblock::DefaultMmaCore<
    cutlass::gemm::GemmShape<128, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    ElementA, LayoutA, cutlass::uint4b_t, LayoutB, ElementC, LayoutC,
    cutlass::arch::OpClassTensorOp, 2,
    cutlass::arch::OpMultiplyAddSm75PackedInputUpcast>;

"""
    runner_alias = "using PackedInt4RunnerM128N64 = Runner<CorePackedInt4M128N64, 2, cutlass::uint4b_t>;\n"
    if testbed.count(alias_block) != 1 or testbed.count(runner_alias) != 1:
        raise RuntimeError("v283 M128 aliases were not found exactly once")
    testbed_path.write_text(testbed.replace(alias_block, "").replace(runner_alias, ""), encoding="utf-8")

    cuda_path = FORK / "mixllm" / "kernels" / "three_level_sm75.cu"
    cuda = cuda_path.read_text(encoding="utf-8")
    old_dispatch = """  if (rows >= 96) {
    shmq_cutlass_sm75::PackedInt4RunnerM128N64::run(
        rows, static_cast<int>(indices.numel()), width, input_int8,
        weight_int4_interleaved, scale_act, matrix_scale, matrix_zero, indices,
        output, stream);
  } else {
    shmq_cutlass_sm75::PackedInt4RunnerM64N64::run(
        rows, static_cast<int>(indices.numel()), width, input_int8,
        weight_int4_interleaved, scale_act, matrix_scale, matrix_zero, indices,
        output, stream);
  }
"""
    new_dispatch = """  shmq_cutlass_sm75::PackedInt4RunnerM64N64::run(
      rows, static_cast<int>(indices.numel()), width, input_int8,
      weight_int4_interleaved, scale_act, matrix_scale, matrix_zero, indices,
      output, stream);
"""
    if cuda.count(old_dispatch) != 1:
        raise RuntimeError("v283 M128 dispatch was not found exactly once")
    cuda_path.write_text(cuda.replace(old_dispatch, new_dispatch), encoding="utf-8")
    print("TEMPORARY_M64_ONLY_PATCH_APPLIED", flush=True)


def main() -> int:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    run("git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(ROOT))
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    print("SHMQ_COMMIT", commit, flush=True)
    if not commit.startswith(EXPECTED_COMMIT):
        raise RuntimeError(f"expected v284 commit prefix {EXPECTED_COMMIT}, got {commit}")
    patch_m64_only()

    import torch

    print("TORCH_VERSION", torch.__version__, flush=True)
    print("GPU_NAME", torch.cuda.get_device_name(0), flush=True)
    capability = tuple(torch.cuda.get_device_capability(0))
    print("GPU_CAPABILITY", capability, flush=True)
    if capability != (7, 5):
        raise RuntimeError(f"expected SM75/T4, got {capability}")
    if shutil.which("ninja") is None:
        run(sys.executable, "-m", "pip", "install", "--quiet", "ninja")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(FORK) + os.pathsep + env.get("PYTHONPATH", "")
    env["MIXLLM_TEST_SM75"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "-v", "mixllm.test.test_sm75_backend.SM75BackendTest"],
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
    print("COLAB_V285_M64_PROBE_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
