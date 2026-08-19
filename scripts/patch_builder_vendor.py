from pathlib import Path
root = Path(__file__).resolve().parents[1]
path = root / 'scripts' / 'build_mixllm_3level_kaggle.py'
text = path.read_text(encoding='utf-8')
anchor = '    "mixllm/kernels/cutlass_sm75_vendor.b64",\n'
extra = '''    "mixllm/kernels/sm75_cutlass_testbed.h",
    "mixllm/kernels/cutlass_extension/mq_mma_pipelined_sm75.h",
    "mixllm/kernels/cutlass_extension/mq_mma_base.h",
    "mixllm/kernels/cutlass_extension/mq_mma_tensor_op_dequantizer.h",
    "mixllm/kernels/cutlass_extension/mq_fine_grained_scale_zero_iterator.h",
    "mixllm/kernels/cutlass_extension/mq_numeric_conversion.h",
'''
if extra.strip() not in text:
    if anchor not in text:
        raise SystemExit('vendor anchor not found')
    text = text.replace(anchor, anchor + extra, 1)
    path.write_text(text, encoding='utf-8')
    print('custom headers inserted')
else:
    print('custom headers already present')
