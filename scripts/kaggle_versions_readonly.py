import importlib.util
from pathlib import Path
import subprocess
import sys
spec = importlib.util.spec_from_file_location('runner', Path(__file__).with_name('run_mixllm_3level_kaggle.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.load_dotenv()
cmd = [sys.executable, '-m', 'kaggle', 'kernels', 'list', '-p', '50']
result = subprocess.run(cmd, cwd=runner.ROOT, text=True, capture_output=True, timeout=180)
print(result.stdout)
print(result.stderr, file=sys.stderr)
print('RETURN_CODE', result.returncode)
