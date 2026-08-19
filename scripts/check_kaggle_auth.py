import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('runner', Path(__file__).with_name('run_mixllm_3level_kaggle.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.load_dotenv()
print({
    "has_api_token": bool(runner.os.environ.get("KAGGLE_API_TOKEN")),
    "has_username_key": bool(runner.os.environ.get("KAGGLE_USERNAME") and runner.os.environ.get("KAGGLE_KEY")),
    "root": str(runner.ROOT),
    "kernel_id": runner.kernel_id(),
})
