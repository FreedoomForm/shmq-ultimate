from __future__ import annotations
import base64
import io
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / 'shmq-ultimate' / 'external' / 'MixLLM'
manifest = ROOT / 'scripts' / 'cutlass_sm75_sources.txt'
out = FORK / 'mixllm' / 'kernels' / 'cutlass_sm75_vendor.b64'
paths = [line.strip() for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for rel in paths:
        archive.writestr(rel, (FORK / rel).read_bytes())
encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
out.write_text(encoded, encoding='ascii')
print(f'files={len(paths)}')
print(f'zip_bytes={len(buffer.getvalue())}')
print(f'b64_bytes={len(encoded)}')
print(out)
