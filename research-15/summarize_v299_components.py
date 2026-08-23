import json
from pathlib import Path

for filename in [
    Path('research-15/colab_v299_output/mixllm_3level_benchmarks.json'),
    Path('research-15/colab_v299_output/mixllm_3level_gate.json'),
]:
    print(f'=== {filename} ===')
    data = json.loads(filename.read_text())
    def walk(obj, path='root'):
        if isinstance(obj, dict):
            keys = set(obj)
            if {'memory'} <= keys or any(k in keys for k in ('sm75_gemm','activation_quantization','gemm_p50_ratio_vs_dense')):
                print(path, {k: obj[k] for k in sorted(obj) if k in {'memory','sm75_gemm','activation_quantization','gemm_p50_ratio_vs_dense','gemm_p50_speedup_vs_dense','e2e_p50_ratio_vs_dense','e2e_p50_speedup_vs_dense'}})
            for key, value in obj.items():
                walk(value, f'{path}.{key}')
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                walk(value, f'{path}[{i}]')
    walk(data)
