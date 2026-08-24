import json
from pathlib import Path

path = Path('research-15/kaggle-v309-output/mixllm_3level_gate.json')
data = json.loads(path.read_text(encoding='utf-8'))
print('gate_status=', data.get('gate_status'))
print('gates=', data.get('gates'))
scenarios = data.get('benchmarks', {}).get('scenarios', {})
for name, value in scenarios.items():
    print(f'[{name}]')
    for i, item in enumerate(value.get('shapes', [])):
        memory = item.get('memory', {})
        peak = memory.get('peak_cuda_memory', {})
        print(i, 'rows=', item.get('rows'), 'gemm_ms=', item.get('sm75_gemm', {}).get('p50_ms'), 'e2e_ms=', item.get('sm75_end_to_end', {}).get('p50_ms'), 'gemm_ratio=', item.get('gemm_p50_ratio_vs_dense'), 'e2e_ratio=', item.get('end_to_end_p50_ratio_vs_dense'), 'max_abs=', item.get('max_abs_error'), 'expanded=', memory.get('expanded_int4_bytes'), 'gemm_peak=', peak.get('gemm_peak_allocated_bytes'), 'e2e_peak=', peak.get('end_to_end_peak_allocated_bytes'))
