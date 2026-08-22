from __future__ import annotations

import ast
import base64
import hashlib
import json
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / "shmq-ultimate" / "external" / "MixLLM"
NOTEBOOKS = [
    ROOT / "shmq-ultimate" / "mixllm_3level_kaggle" / "mixllm_3level_gate.ipynb",
    ROOT / "research-15" / "mixllm_3level_gate_colab_v300_operator.ipynb",
]

for notebook_path in NOTEBOOKS:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = "".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    marker = "embedded_sources = "
    start = code.index(marker) + len(marker)
    end = code.index("\nsources =", start)
    embedded = ast.literal_eval(code[start:end])
    sources = {
        name: zlib.decompress(base64.b64decode(payload)).decode("utf-8")
        for name, payload in embedded.items()
    }
    digest = hashlib.sha256()
    for name in sorted(sources):
        digest.update(name.encode() + b"\0" + sources[name].encode() + b"\0")
    current = {
        name: (FORK / name).read_text(encoding="utf-8")
        for name in sources
    }
    current_digest = hashlib.sha256()
    for name in sorted(current):
        current_digest.update(name.encode() + b"\0" + current[name].encode() + b"\0")
    assert digest.hexdigest() == current_digest.hexdigest(), notebook_path
    assert sources == current, notebook_path
    print(f"V300_NOTEBOOK_SOURCE_MATCH {notebook_path.name} files={len(sources)} digest={digest.hexdigest()}")
    print("V300_NOTEBOOK_HAS_WIDENED_A", "MatrixShape<16, 32>" in sources["mixllm/kernels/cutlass_extension/mq_mma_tensor_op_sm75.h"])

operator = json.loads(NOTEBOOKS[1].read_text(encoding="utf-8"))
quality_cell = "".join(operator["cells"][7]["source"])
assert "'status': 'not_run'" in quality_cell
print("V300_OPERATOR_QUALITY_CELL_NOT_RUN true")
