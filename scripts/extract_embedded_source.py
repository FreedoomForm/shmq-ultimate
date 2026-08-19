import ast
import hashlib
import json
import re
from pathlib import Path

notebook = Path(r"C:\Users\User\Downloads\Multi-main\Multi-main\research-15\versions\v51_decode_expanded_cache_contract\mixllm_3level_gate.before.ipynb")
out = notebook.parent / "embedded_three_level_sm75.cu"
doc = json.loads(notebook.read_text(encoding="utf-8"))
for cell in doc["cells"]:
    if cell.get("cell_type") != "code":
        continue
    text = "".join(cell.get("source", []))
    match = re.search(r"sources = (.*)\nsource_manifest =", text, re.DOTALL)
    if match:
        sources = ast.literal_eval(match.group(1))
        source = sources["mixllm/kernels/three_level_sm75.cu"]
        out.write_text(source, encoding="utf-8", newline="\n")
        print("embedded_path=" + str(out))
        print("normalized_sha256=" + hashlib.sha256(source.encode("utf-8")).hexdigest())
        for line in source.splitlines():
            if "kDecodeChannelsPerWarp" in line or "kPrefillWarps" in line or "packed_int4" in line or "expanded_int4" in line:
                print(line)
        break
else:
    raise SystemExit("sources cell not found")
