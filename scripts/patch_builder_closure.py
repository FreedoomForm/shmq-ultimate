from pathlib import Path
root = Path(__file__).resolve().parents[1]
path = root / 'scripts' / 'build_mixllm_3level_kaggle.py'
text = path.read_text(encoding='utf-8')
old = '''CUTLASS_INCLUDE = FORK / "mixllm" / "kernels" / "cutlass" / "include"
CUTLASS_SOURCE_FILES = tuple(
    "mixllm/kernels/cutlass/include/" + p.relative_to(CUTLASS_INCLUDE).as_posix()
    for p in sorted(CUTLASS_INCLUDE.rglob("*")) if p.is_file()
)
SOURCE_FILES = SOURCE_FILES + CUTLASS_SOURCE_FILES
'''
new = '''CUTLASS_SOURCE_MANIFEST = ROOT / "scripts" / "cutlass_sm75_sources.txt"
CUTLASS_SOURCE_FILES = tuple(
    line.strip() for line in CUTLASS_SOURCE_MANIFEST.read_text(encoding="utf-8").splitlines()
    if line.strip()
)
SOURCE_FILES = SOURCE_FILES + CUTLASS_SOURCE_FILES
'''
if old not in text:
    raise SystemExit('builder block not found')
path.write_text(text.replace(old, new, 1), encoding='utf-8')
print('builder patched')
