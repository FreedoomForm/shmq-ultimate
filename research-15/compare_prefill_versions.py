from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent
paths = {
    "v200": ROOT / "kaggle-v200-output" / "mixllm_3level_gate.json",
    "v204": ROOT / "kaggle-v204-output" / "mixllm_3level_gate.json",
    "v205": ROOT / "kaggle-v205-output" / "mixllm_3level_gate.json",
}
scenario = "qwen_qkv_mixed_4_8_16"
for version, path in paths.items():
    data = json.loads(path.read_text())
    report = data["benchmarks"]["scenarios"][scenario]
    print(version, data["gate_status"]["terminal_decision"])
    for shape in report["shapes"]:
        print(
            " rows={rows} dense={dense:.6f} gemm={gemm:.6f} e2e={e2e:.6f} "
            "gemm_ratio={gr:.4f} e2e_ratio={er:.4f} timing={ti}({tir:.4f}) error={err:.6f}".format(
                rows=shape["rows"],
                dense=shape["dense_fp16"]["p50_ms"],
                gemm=shape["sm75_gemm"]["p50_ms"],
                e2e=shape["sm75_end_to_end"]["p50_ms"],
                gr=shape["gemm_p50_ratio_vs_dense"],
                er=shape["end_to_end_p50_ratio_vs_dense"],
                ti=shape["timing_integrity"],
                tir=shape["timing_integrity_ratio"],
                err=shape["max_abs_error"],
            )
        )
    print("gates", data["gates"])
    print()
