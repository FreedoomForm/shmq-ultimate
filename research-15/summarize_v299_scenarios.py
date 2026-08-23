import json
from pathlib import Path

data = json.loads(Path('research-15/colab_v299_output/mixllm_3level_gate.json').read_text())
scenarios = data.get('benchmarks', {}).get('scenarios', {})
for name, value in scenarios.items():
    print(f'[{name}]')
    for i, item in enumerate(value.get('shapes', [])):
        memory = item.get('memory', {})
        gemm = item.get('sm75_gemm', {})
        quant = item.get('activation_quantization', {})
        print(i, 'gemm_ms=', gemm.get('p50_ms'), 'ratio=', item.get('gemm_p50_ratio_vs_dense'), 'quant_ms=', quant.get('p50_ms'), 'expanded=', memory.get('expanded_int4_bytes'), 'gemm_peak=', memory.get('peak_cuda_memory', {}).get('gemm_peak_allocated_bytes'), 'e2e_peak=', memory.get('peak_cuda_memory', {}).get('end_to_end_peak_allocated_bytes'))
