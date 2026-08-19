from __future__ import annotations
import ast, difflib, hashlib, json, re
from pathlib import Path

ROOT = Path(r"C:\Users\User\Downloads\Multi-main\Multi-main")
VERSION_ROOT = ROOT / "research-15" / "versions"
OUT = ROOT / "research-15" / "audit_static_v1.json"

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace('\\','/')

def cuda_signature(path: Path) -> dict:
    text = path.read_text(encoding='utf-8', errors='replace')
    lines = text.splitlines()
    pats = {
        'constants': r'constexpr int (kTile|kGroupSize|kPrefillWarps|kPrefillChannels|kReuseRows)\s*=.*',
        'kernels': r'__global__ void\s+([A-Za-z0-9_]+)',
        'wmma_fragments': r'wmma::fragment<wmma::(accumulator|matrix_a|matrix_b),\s*([^>]+)>',
        'wmma_loads': r'wmma::load_matrix_sync\((.*)',
        'wmma_stores': r'wmma::store_matrix_sync\((.*)',
        'precision_branches': r'(if|else if)\s*\(.*precision.*',
        'row_channel_branches': r'.*(full_rows|full_channels|row_base|channel_base).*',
        'shared_arrays': r'__shared__.*',
        'launches': r'.*(<<<.*|const dim3 grid.*)',
    }
    result = {'path': rel(path), 'sha256': sha(path), 'lines': len(lines), 'matches': {}}
    for name, pat in pats.items():
        rx = re.compile(pat)
        hits = []
        for idx, line in enumerate(lines, 1):
            if rx.search(line):
                hits.append({'line': idx, 'text': line.strip()})
        result['matches'][name] = hits[:160]
    result['counts'] = {
        'syncthreads': text.count('__syncthreads'),
        'syncwarp': text.count('__syncwarp'),
        'mma_sync': text.count('wmma::mma_sync'),
        'direct_global_load': len(re.findall(r'wmma::load_matrix_sync\([^;]*\n?[^;]*(?:input_int8|weight_int8|expanded_int4|weight_fp16)', text)),
        'shared_a': len(re.findall(r'__shared__[^;]*a_int8', text)),
        'shared_b': len(re.findall(r'__shared__[^;]*b_int8', text)),
        'false_precision_guards': text.count('false && precision'),
    }
    return result

def py_signature(path: Path) -> dict:
    text = path.read_text(encoding='utf-8', errors='replace')
    result = {'path': rel(path), 'sha256': sha(path), 'lines': len(text.splitlines()), 'syntax_error': None, 'classes': [], 'functions': [], 'imports': [], 'key_lines': []}
    try:
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef): result['classes'].append({'name': node.name, 'line': node.lineno})
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result['functions'].append({'name': node.name, 'line': node.lineno, 'args': [a.arg for a in node.args.args]})
            elif isinstance(node, ast.Import): result['imports'] += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom): result['imports'].append((node.module or '') + ':' + ','.join(a.name for a in node.names))
    except SyntaxError as e:
        result['syntax_error'] = str(e)
    keys = ('load_sm75_backend','quantize_activation_native','three_level_linear_prequantized','mixllm_sm75','ThreeLevelLinear','indices_4','indices_8','indices_16','weight_int4','weight_int8','weight_fp16','scale_int4','scale_int8','zero_int4','group_size','precision_percentages','tensor_parallel')
    for idx, line in enumerate(text.splitlines(), 1):
        if any(k in line for k in keys): result['key_lines'].append({'line': idx, 'text': line.strip()})
    return result

def main():
    result = {'root': str(ROOT), 'versions': [], 'cuda_hash_groups': {}, 'python_files': [], 'current_vs_v51_diff': [], 'worklog_headings': []}
    for v in sorted([p for p in VERSION_ROOT.iterdir() if p.is_dir()]):
        cus = sorted(v.rglob('*.cu'))
        pyfiles = sorted(v.rglob('*.py'))
        entry = {'version': v.name, 'files': []}
        for f in sorted(v.rglob('*')):
            if f.is_file(): entry['files'].append({'path': str(f.relative_to(v)).replace('\\','/'), 'sha256': sha(f), 'bytes': f.stat().st_size})
        if cus:
            sig = cuda_signature(cus[0]); entry['cuda'] = sig
            result['cuda_hash_groups'].setdefault(sig['sha256'], []).append(v.name)
        if pyfiles: entry['python'] = [py_signature(p) for p in pyfiles]
        result['versions'].append(entry)
    current = ROOT / 'shmq-ultimate' / 'external' / 'MixLLM' / 'mixllm'
    for p in sorted(current.rglob('*.py')):
        if '__pycache__' not in p.parts: result['python_files'].append(py_signature(p))
    v51 = VERSION_ROOT / 'v86_branch_hoist' / 'three_level_sm75.before.cu'
    curcu = current / 'kernels' / 'three_level_sm75.cu'
    if v51.exists() and curcu.exists():
        a = v51.read_text(encoding='utf-8', errors='replace').splitlines()
        b = curcu.read_text(encoding='utf-8', errors='replace').splitlines()
        diff = list(difflib.unified_diff(a, b, fromfile='v51', tofile='current', lineterm=''))
        result['current_vs_v51_diff'] = diff
    wl = ROOT / 'research-15' / 'worklog.md'
    if wl.exists():
        for idx, line in enumerate(wl.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
            if line.startswith('## ') or line.startswith('### '): result['worklog_headings'].append({'line': idx, 'text': line})
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(OUT)

if __name__ == '__main__': main()
