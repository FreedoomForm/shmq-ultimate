from pathlib import Path
import py_compile
import tempfile

patch = Path(r"C:\Users\User\Downloads\Multi-main\Multi-main\shmq-ultimate\external\MixLLM\vllm_v0.9.0_patch\0002-add-mixllm-three-level-support.patch")
lines = patch.read_text(encoding='utf-8', errors='replace').splitlines()
start = next(i for i, line in enumerate(lines) if line.startswith('diff --git a/vllm/model_executor/layers/quantization/mixllm_three_level.py'))
end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith('diff --git ')), len(lines))
body = []
for line in lines[start:end]:
    if line.startswith('+++') or line.startswith('---') or line.startswith('@@') or line.startswith('diff --git'):
        continue
    if line.startswith('+'):
        body.append(line[1:])
    elif line.startswith(' '):
        body.append(line[1:])
path = Path(tempfile.gettempdir()) / 'mixllm_three_level_extracted.py'
path.write_text('\n'.join(body) + '\n', encoding='utf-8')
try:
    py_compile.compile(str(path), doraise=True)
    print('SYNTAX=PASS')
except py_compile.PyCompileError as exc:
    print('SYNTAX=FAIL')
    print(str(exc))
print('EXTRACTED='+str(path))
for i, line in enumerate(body, 1):
    if 'hasattr(layer' in line or 'backend' in line or 'get_min_capability' in line:
        print(f'{i}:{line}')
