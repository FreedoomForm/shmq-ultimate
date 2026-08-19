"""Build, validate, submit, and download the private Kaggle T4 gate."""
from __future__ import annotations
import os, subprocess, sys, time
import json
import argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / 'scripts' / 'build_mixllm_3level_kaggle.py'
KERNEL_DIR = ROOT / 'shmq-ultimate' / 'mixllm_3level_kaggle'
def kernel_id():
    metadata = json.loads((KERNEL_DIR / 'kernel-metadata.json').read_text(
        encoding='utf-8'))
    return metadata['id']
def load_dotenv():
    path = ROOT / '.env'
    if not path.exists():
        return
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        name, value = line.split('=', 1)
        name = name.strip()
        if name in ('KAGGLE_API_TOKEN', 'KAGGLE_USERNAME', 'KAGGLE_KEY'):
            os.environ.setdefault(name, value.strip().strip('"\''))
def redact(text):
    for name in ('KAGGLE_KEY', 'KAGGLE_API_TOKEN'):
        if os.environ.get(name): text = text.replace(os.environ[name], '<redacted>')
    return text
def run(args):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    if result.stdout: print(redact(result.stdout), end='')
    if result.stderr: print(redact(result.stderr), end='', file=sys.stderr)
    result.check_returncode(); return result
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    load_dotenv()
    if not (os.environ.get('KAGGLE_API_TOKEN') or (os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'))): raise SystemExit('Set KAGGLE_API_TOKEN or KAGGLE_USERNAME and KAGGLE_KEY in the environment.')
    kaggle = [sys.executable, '-m', 'kaggle']; run([sys.executable, str(BUILDER)]); run([sys.executable, str(BUILDER), '--check']); current_kernel_id = kernel_id()
    if not args.resume:
        run([*kaggle, 'kernels', 'push', '-p', str(KERNEL_DIR)])
    start_time = time.time()
    max_runtime = int(os.environ.get('KAGGLE_MAX_RUNTIME_SECONDS', '900'))
    while True:
        status = run([*kaggle, 'kernels', 'status', current_kernel_id]).stdout.lower()
        if 'complete' in status: break
        if any(word in status for word in ('error', 'failed', 'cancel')): raise SystemExit('Kaggle gate failed.')
        if time.time() - start_time > max_runtime:
            raise SystemExit(f'Kaggle gate exceeded watchdog limit of {max_runtime}s')
        time.sleep(30)
    output = KERNEL_DIR / f'output-{int(time.time())}'; output.mkdir(); run([*kaggle, 'kernels', 'output', current_kernel_id, '-p', str(output)]); print(output)
if __name__ == '__main__': main()
