from pathlib import Path
import ast
import hashlib
import json
import re

LOG = Path('/tmp/kaggle-v257-final.log')
EXPECTED_CU = Path('/home/ubuntu/v204-shmq/shmq-ultimate/external/MixLLM/mixllm/kernels/three_level_sm75.cu')
records = json.loads(LOG.read_text(errors='replace'))
stdout = ''.join(r.get('data', '') for r in records if r.get('stream_name') == 'stdout')
Path('/tmp/kaggle-v257-stdout.txt').write_text(stdout)
expected_digest = hashlib.sha256(EXPECTED_CU.read_bytes()).hexdigest()[:16]
observed_digests = sorted(set(re.findall(r'mixllm_sm75_backend_([0-9a-f]+)', stdout)))
print('stdout_bytes', len(stdout))
print('expected_source_digest', expected_digest)
print('observed_source_digests', observed_digests)
print('source_identity_match', expected_digest in observed_digests)
start = stdout.find('BENCHMARKS_PRE_EXEC ')
if start < 0:
    raise SystemExit('BENCHMARKS_PRE_EXEC marker missing')
start += len('BENCHMARKS_PRE_EXEC ')
end = stdout.find('\n{\n  "execution"', start)
report = ast.literal_eval(stdout[start:end if end >= 0 else len(stdout)].strip())
for name, scenario in report.get('scenarios', {}).items():
    if 'qwen_qkv_mixed' in name or 'qwen_qkv_pure_fp16' in name:
        print('SCENARIO', name, scenario.get('partition_counts'))
        for shape in scenario.get('shapes', []):
            print('SHAPE', shape['rows'],
                  'gemm_ms', shape['sm75_gemm']['p50_ms'],
                  'e2e_ms', shape['sm75_end_to_end']['p50_ms'],
                  'dense_ms', shape['dense_fp16']['p50_ms'],
                  'e2e_ratio', shape['end_to_end_p50_ratio_vs_dense'],
                  'e2e_speedup', shape['end_to_end_p50_speedup_vs_dense'],
                  'integrity_ratio', shape['timing_integrity_ratio'],
                  'integrity', shape['timing_integrity'])
for marker in ('SM75_INT4_PAIR_MIXED_STRIDE_PROBE_PASS', 'TESTS_RETURNCODE', 'GATES_PRE_EXEC'):
    pos = stdout.find(marker)
    print('MARKER', marker, pos)
    if pos >= 0:
        print(stdout[pos:pos + 700])
for line in stdout.splitlines():
    if line.startswith('GATES_') or 'terminal_decision' in line or line.startswith('Full-model'):
        print(line)
