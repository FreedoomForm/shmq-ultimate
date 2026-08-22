#!/usr/bin/env python3
import json
from pathlib import Path

source = Path('/home/ubuntu/v204-shmq-unified/shmq-ultimate/mixllm_3level_kaggle/mixllm_3level_gate.ipynb')
target = Path('/home/ubuntu/v204-shmq-unified/research-15/mixllm_3level_gate_colab_v300_operator.ipynb')
data = json.loads(source.read_text(encoding='utf-8'))
replacement = """quality = {
    'status': 'not_run',
    'model_id': 'Qwen/Qwen2.5-0.5B',
    'backend': 'not_run',
    'reason': 'operator-only Colab run; full-model quality is a separate gate',
}
report['full_model_quality'] = quality
report['gates']['full_model_qwen_quality'] = quality['status']
report['gates']['full_model_qwen_throughput'] = quality['status']
report['claims']['full_model_qwen_quality_claimed'] = False
"""
data['cells'][7]['source'] = replacement.splitlines(keepends=True)
target.write_text(json.dumps(data, indent=1), encoding='utf-8')
print(target)
