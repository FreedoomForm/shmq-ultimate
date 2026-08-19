import importlib.util
from pathlib import Path
import subprocess
import sys
import time
spec = importlib.util.spec_from_file_location('runner', Path(__file__).with_name('run_mixllm_3level_kaggle.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.load_dotenv()
out = runner.ROOT / 'shmq-ultimate' / 'mixllm_3level_kaggle' / ('latest-output-' + str(int(time.time())))
out.mkdir(parents=True, exist_ok=True)
cmd = [sys.executable, '-m', 'kaggle', 'kernels', 'output', runner.kernel_id(), '-p', str(out)]
result = subprocess.run(cmd, cwd=runner.ROOT, text=True, capture_output=True, timeout=300)
print(result.stdout)
print(result.stderr, file=sys.stderr)
print('RETURN_CODE', result.returncode)
print('OUTPUT_DIR', out)
