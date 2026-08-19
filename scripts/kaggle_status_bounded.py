import importlib.util
from pathlib import Path
import subprocess
import sys
spec = importlib.util.spec_from_file_location('runner', Path(__file__).with_name('run_mixllm_3level_kaggle.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.load_dotenv()
cmd = [sys.executable, '-m', 'kaggle', 'kernels', 'status', runner.kernel_id()]
try:
    result = subprocess.run(cmd, cwd=runner.ROOT, text=True, capture_output=True, timeout=60)
    print(result.stdout)
    print(result.stderr, file=sys.stderr)
    print('RETURN_CODE', result.returncode)
except subprocess.TimeoutExpired as exc:
    print('STATUS_TIMEOUT', exc.timeout)
