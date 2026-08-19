from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / 'shmq-ultimate' / 'external' / 'MixLLM'
KERNEL_ROOT = FORK / 'mixllm' / 'kernels'
CUTLASS_INCLUDE = KERNEL_ROOT / 'cutlass' / 'include'
OUT = ROOT / 'scripts' / 'cutlass_sm75_sources.txt'
SEEDS = [
    KERNEL_ROOT / 'sm75_cutlass_testbed.h',
    KERNEL_ROOT / 'cutlass_extension' / 'mq_mma_pipelined_sm75.h',
    KERNEL_ROOT / 'cutlass_extension' / 'mq_mma_base.h',
    KERNEL_ROOT / 'cutlass_extension' / 'mq_mma_tensor_op_dequantizer.h',
    KERNEL_ROOT / 'cutlass_extension' / 'mq_fine_grained_scale_zero_iterator.h',
    KERNEL_ROOT / 'cutlass_extension' / 'mq_numeric_conversion.h',
]
include_re = re.compile(r'^\s*#\s*include\s*["<]([^">]+)[">]')
roots = [KERNEL_ROOT, CUTLASS_INCLUDE]
seen: set[Path] = set()
queue = list(SEEDS)
while queue:
    path = queue.pop()
    path = path.resolve()
    if path in seen or not path.is_file():
        continue
    seen.add(path)
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        match = include_re.match(line)
        if not match:
            continue
        name = match.group(1)
        candidates = [path.parent / name] + [root / name for root in roots]
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate.is_file():
                if candidate not in seen:
                    queue.append(candidate)
                break

relative = []
for path in sorted(seen):
    rel = path.relative_to(FORK).as_posix()
    if rel.startswith('mixllm/kernels/'):
        relative.append(rel)
OUT.write_text('\n'.join(relative) + '\n', encoding='utf-8')
print(f'headers={len(relative)}')
print(f'bytes={sum((FORK / r).stat().st_size for r in relative)}')
print(OUT)
