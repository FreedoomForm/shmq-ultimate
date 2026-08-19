"""Build and invoke the single-launch SM75 three-level Tensor Core backend."""

from __future__ import annotations

from pathlib import Path
from statistics import median
from typing import Dict, Iterable, Optional


_LOADED = False


def load_sm75_backend(torch_module, build_directory: Optional[str | Path] = None) -> None:
    """JIT-build the SM75-only torch operator and load it into this process."""
    global _LOADED
    if _LOADED:
        return
    if not torch_module.cuda.is_available():
        raise RuntimeError("SM75 backend requires CUDA")
    capability = tuple(torch_module.cuda.get_device_capability())
    if capability != (7, 5):
        raise RuntimeError(f"SM75 backend requires capability 7.5, got {capability}")

    from torch.utils.cpp_extension import load

    kernel_root = Path(__file__).resolve().parent / "kernels"
    source = kernel_root / "three_level_sm75.cu"
    vendor_blob = kernel_root / "cutlass_sm75_vendor.b64"
    vendor_root = kernel_root / "cutlass_sm75_vendor"
    vendor_marker = vendor_root / ".ready"
    if not vendor_marker.exists() or vendor_marker.stat().st_mtime < vendor_blob.stat().st_mtime:
        import base64
        import io
        import zipfile
        vendor_root.mkdir(parents=True, exist_ok=True)
        payload = base64.b64decode(vendor_blob.read_text(encoding="ascii"))
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extractall(vendor_root)
        vendor_marker.write_text("ready", encoding="ascii")
    vendor_kernel_root = vendor_root / "mixllm" / "kernels"
    cutlass_include = vendor_kernel_root / "cutlass" / "include"
    cutlass_extension = vendor_kernel_root / "cutlass_extension"
    kwargs = {}

    if build_directory is not None:
        directory = Path(build_directory)
        directory.mkdir(parents=True, exist_ok=True)
        kwargs["build_directory"] = str(directory)
    try:
        load(
          name="mixllm_sm75_backend",
          sources=[str(source)],
          extra_cuda_cflags=[
              "-O3", "-lineinfo", "-gencode=arch=compute_75,code=sm_75",
              f"-I{vendor_kernel_root}", f"-I{cutlass_include}",
              f"-I{cutlass_extension}",
          ],

          extra_cflags=["-O3"],
          is_python_module=False,
          verbose=True,
          **kwargs,
      )
        _LOADED = True
    except Exception:
        import traceback
        import subprocess
        roots = [Path('/root/.cache/torch_extensions'), Path('/kaggle/working'), Path('/tmp')]
        candidates = [p for root in roots if root.exists() for p in root.rglob('build.ninja')]
        if candidates:
            ninja_file = candidates[0]
            diag = subprocess.run(['ninja', '-v'], cwd=str(ninja_file.parent), capture_output=True, text=True)
            Path('/kaggle/working/sm75_ninja_diagnostic.txt').write_text(diag.stdout + '\n' + diag.stderr, encoding='utf-8')
        Path('/kaggle/working/sm75_build_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise

def quantize_activation(x, torch_module, group_size: int = 128):
    """Reference symmetric per-row, per-group INT8 activation quantization."""
    if x.dim() != 2 or x.shape[1] % group_size:
        raise ValueError("activation must be 2D with width divisible by group_size")
    rows, width = x.shape
    if rows == 0:
        return (
            torch_module.empty_like(x, dtype=torch_module.int8),
            torch_module.empty(
                width // group_size, 0, device=x.device,
                dtype=torch_module.float16,
            ),
        )
    grouped = x.float().reshape(rows, width // group_size, group_size)
    scales = (grouped.abs().amax(dim=-1) / 127.0).clamp_min(1e-8)
    quantized = (grouped / scales.unsqueeze(-1)).round().clamp(-127, 127)
    return (quantized.to(torch_module.int8).reshape(rows, width),
            scales.t().to(torch_module.float16).contiguous())


def quantize_activation_native(x, torch_module, group_size: int = 128):
    """Run the native SM75 activation quantizer used by the inference path."""
    if not _LOADED:
        raise RuntimeError("call load_sm75_backend before using the SM75 operator")
    if x.dim() != 2 or x.shape[1] % group_size:
        raise ValueError("activation must be 2D with width divisible by group_size")
    if group_size != 128 or x.dtype != torch_module.float16:
        raise ValueError("native SM75 quantization requires FP16 and group_size=128")
    if x.shape[0] == 0:
        return (
            torch_module.empty_like(x, dtype=torch_module.int8),
            torch_module.empty(
                x.shape[1] // group_size, 0, device=x.device,
                dtype=torch_module.float16,
            ),
        )
    return torch_module.ops.mixllm_sm75.quantize_activation(x.contiguous())


def quantized_reference_prequantized(
    module, x, quantized, scales, torch_module,
):
    """Reference the GEMM using a caller-provided activation quantization."""
    rows, width = x.shape
    reconstructed = (
        quantized.float().reshape(rows, -1, module.group_size)
        * scales.t().float().unsqueeze(-1)
    ).reshape(rows, width)
    output = torch_module.empty(
        rows, module.out_features, device=x.device, dtype=torch_module.float32,
    )
    dense = module.dequantize_weight()
    for bit, activation in ((4, reconstructed), (8, reconstructed), (16, x)):
        indices = getattr(module, f"indices_{bit}")
        if not indices.numel():
            continue
        weight = dense.index_select(0, indices.long())
        values = torch_module.nn.functional.linear(
            activation if bit != 16 else activation.to(torch_module.float16),
            weight if bit != 16 else weight.to(torch_module.float16),
        ).float()
        output.index_copy_(1, indices.long(), values)
    return output


def quantized_reference(module, x, torch_module):
    """Reference the complete mixed activation contract used by the SM75 op."""
    quantized, scales = quantize_activation(x, torch_module, module.group_size)
    return quantized_reference_prequantized(
        module, x, quantized, scales, torch_module,
    )


def _validate_partition(module, x, torch_module):
    signature = (
        int(module.out_features),
        *((id(indices), int(indices._version), int(indices.numel()), indices.device)
          for indices in (module.indices_4, module.indices_8, module.indices_16)),
    )
    if getattr(module, "_sm75_partition_validation", None) == signature:
        return
    if x.is_cuda and torch_module.cuda.is_current_stream_capturing():
        raise RuntimeError("SM75 partition must be validated before CUDA graph capture")
    indices = (module.indices_4, module.indices_8, module.indices_16)
    complete = torch_module.cat(indices).long()
    expected = torch_module.arange(module.out_features, device=x.device)
    if complete.numel() != module.out_features or not torch_module.equal(
            complete.sort().values, expected):
        raise ValueError(
            "4/8/16 indices must form a complete output partition without "
            "duplicates, missing channels, or out-of-range indices"
        )
    module._sm75_partition_validation = signature


def _expanded_int4_for_prefill(module, x, torch_module):
    """Expand packed INT4 once per packed state/device for the SM75 prefill path."""
    if not module.indices_4.numel():
        # Decode does not read this argument. Empty INT4 prefill still needs the
        # ABI-compatible [0, K] shape without allocating a separate tensor.
        return module.weight_int8[:0]
    signature = (
        module.weight_int4.device,
        id(module.weight_int4), int(module.weight_int4._version),
        id(module.zero_int4), int(module.zero_int4._version),
        tuple(module.weight_int4.shape), tuple(module.zero_int4.shape),
    )
    cached = module._sm75_int4_expanded
    if cached is not None and cached[0] == signature:
        return cached[1]
    if x.is_cuda and torch_module.cuda.is_current_stream_capturing():
        raise RuntimeError("SM75 INT4 prefill cache must be initialized before CUDA graph capture")
    packed = module.weight_int4
    expanded = torch_module.empty(
        packed.shape[0], packed.shape[1] * 2,
        dtype=torch_module.uint8, device=packed.device,
    )
    expanded[:, 0::2] = packed & 0x0f
    expanded[:, 1::2] = packed >> 4
    zeros = module.zero_int4.repeat_interleave(
        module.group_size, dim=1,
    ).to(torch_module.int16)
    expanded = (expanded.to(torch_module.int16) - zeros).to(
        torch_module.int8,
    ).contiguous()
    module._sm75_int4_expanded = (signature, expanded)
    return expanded


def _prefill_metadata(module, torch_module):
    """Cache the [groups, channels] metadata layout required by CUTLASS prefill."""
    signature = (
        module.scale_int4.device,
        id(module.scale_int4), int(module.scale_int4._version),
        id(module.zero_int4), int(module.zero_int4._version),
        id(module.scale_int8), int(module.scale_int8._version),
        tuple(module.scale_int4.shape), tuple(module.zero_int4.shape),
        tuple(module.scale_int8.shape),
    )
    cached = module._sm75_prefill_metadata
    if cached is not None and cached[0] == signature:
        return cached[1], cached[2], cached[3]
    scale_int4 = module.scale_int4.transpose(0, 1).contiguous()
    zero_int4 = module.zero_int4.transpose(0, 1).contiguous()
    scale_int8 = module.scale_int8.transpose(0, 1).contiguous()
    cached = (signature, scale_int4, zero_int4, scale_int8)
    module._sm75_prefill_metadata = cached
    return scale_int4, zero_int4, scale_int8


def three_level_linear_prequantized(module, x, input_int8, scale_act, torch_module):

    """Run the physical GEMM kernel with precomputed activation quantization."""
    if not _LOADED:
        raise RuntimeError("call load_sm75_backend before using the SM75 operator")
    if x.dim() != 2:
        raise ValueError("SM75 correctness backend currently requires a 2D input")
    packed_tensors = (
        module.weight_int4, module.scale_int4, module.zero_int4, module.indices_4,
        module.weight_int8, module.scale_int8, module.indices_8,
        module.weight_fp16, module.indices_16,
    )
    tensors = (input_int8, scale_act, *packed_tensors)
    if any(tensor.device != x.device for tensor in tensors):
        raise ValueError("input and all operator tensors must be on the same device")
    if x.dtype != torch_module.float16:
        raise ValueError("SM75 Tensor Core backend requires float16 activation input")
    if module.group_size != 128:
        raise ValueError("SM75 Tensor Core backend currently requires group_size=128")
    _validate_partition(module, x, torch_module)
    if x.shape[0] == 0:
        return torch_module.empty(
            0, module.out_features, device=x.device, dtype=torch_module.float32,
        )
    expanded_int4 = _expanded_int4_for_prefill(module, x, torch_module)
    return torch_module.ops.mixllm_sm75._three_level_linear_v2_unchecked(
        x if x.is_contiguous() else x.contiguous(),
        input_int8 if input_int8.is_contiguous() else input_int8.contiguous(),
        scale_act if scale_act.is_contiguous() else scale_act.contiguous(),
        module.weight_int4,
        expanded_int4,
        module.scale_int4,
        module.zero_int4,
        module.indices_4,
        module.weight_int8,
        module.scale_int8,
        module.indices_8,
        module.weight_fp16,
        module.indices_16,
    )


def _fp16_abi_placeholders(module, x, torch_module):
    """Return reusable tensors required but unread by the pure-FP16 kernel path."""
    key = (x.device, tuple(x.shape))
    cached = module._sm75_fp16_placeholders
    if cached is None or cached[0] != key:
        rows, width = x.shape
        cached = (
            key,
            torch_module.empty_like(x, dtype=torch_module.int8),
            torch_module.empty(
                width // module.group_size, rows, device=x.device,
                dtype=torch_module.float16,
            ),
        )
        module._sm75_fp16_placeholders = cached
    return cached[1], cached[2]


def three_level_linear(module, x, torch_module):
    """Dispatch by shape/partition and run the single three-level GEMM ABI."""
    if not _LOADED:
        raise RuntimeError("call load_sm75_backend before using the SM75 operator")
    if x.dim() != 2:
        raise ValueError("SM75 correctness backend currently requires a 2D input")
    packed_tensors = (
        module.weight_int4, module.scale_int4, module.zero_int4, module.indices_4,
        module.weight_int8, module.scale_int8, module.indices_8,
        module.weight_fp16, module.indices_16,
    )
    if any(tensor.device != x.device for tensor in packed_tensors):
        raise ValueError("input and all packed tensors must be on the same device")
    _validate_partition(module, x, torch_module)
    if x.shape[0] == 0:
        return torch_module.empty(
            0, module.out_features, device=x.device, dtype=torch_module.float32,
        )
    if not module.indices_4.numel() and not module.indices_8.numel():
        input_int8, scale_act = _fp16_abi_placeholders(module, x, torch_module)
        return three_level_linear_prequantized(
            module, x, input_int8, scale_act, torch_module,
        )
    input_int8, scale_act = quantize_activation_native(
        x, torch_module, module.group_size,
    )
    return three_level_linear_prequantized(
        module, x, input_int8, scale_act, torch_module,
    )


def benchmark_sm75_backend(
    module,
    rows: Iterable[int],
    torch_module,
    warmup: int = 10,
    iterations: int = 50,
) -> Dict[str, object]:
    """Measure the SM75 operator against its dequantized dense reference.

    Timings use per-iteration CUDA events and synchronize only after all events
    have been recorded. This is a kernel-level benchmark, not tokens/second.
    """
    if warmup < 1 or iterations < 2:
        raise ValueError("benchmark requires warmup >= 1 and iterations >= 2")
    if not _LOADED:
        raise RuntimeError("call load_sm75_backend before benchmarking")
    dense_weight = module.dequantize_weight().to(torch_module.float16)
    results = []
    partition_counts = {
        bit: int(getattr(module, f"indices_{bit}").numel())
        for bit in (4, 8, 16)
    }
    total_channels = sum(partition_counts.values())
    average_weight_bits = (
        sum(bit * partition_counts[bit] for bit in (4, 8, 16))
        / total_channels
    )

    def measure(callable_):
        for _ in range(warmup):
            callable_()
        torch_module.cuda.synchronize()
        pairs = []
        for _ in range(iterations):
            start = torch_module.cuda.Event(enable_timing=True)
            end = torch_module.cuda.Event(enable_timing=True)
            start.record()
            callable_()
            end.record()
            pairs.append((start, end))
        torch_module.cuda.synchronize()
        values = sorted(float(start.elapsed_time(end)) for start, end in pairs)
        p95_index = min(len(values) - 1, int(0.95 * len(values)))
        return {"p50_ms": median(values), "p95_ms": values[p95_index]}

    for row_count in rows:
        if int(row_count) <= 0:
            raise ValueError("benchmark row counts must be positive")
        x = torch_module.randn(
            int(row_count), module.in_features, device=module.weight_fp16.device,
            dtype=torch_module.float16,
        )
        input_int8, scale_act = quantize_activation_native(
            x, torch_module, module.group_size,
        )
        actual = three_level_linear_prequantized(
            module, x, input_int8, scale_act, torch_module,
        )
        # Attribute asynchronous extension faults before the independent
        # reference GEMM; this is outside all timed measurement loops.
        torch_module.cuda.synchronize()
        operator_reference = quantized_reference_prequantized(
            module, x, input_int8, scale_act, torch_module,
        )

        dense_reference = torch_module.nn.functional.linear(x, dense_weight).float()
        max_error = float((actual - operator_reference).abs().max().item())
        dense_semantic_error = float((actual - dense_reference).abs().max().item())
        quantization = measure(
            lambda: quantize_activation_native(x, torch_module, module.group_size),
        )
        gemm = measure(lambda: three_level_linear_prequantized(
            module, x, input_int8, scale_act, torch_module,
        ))
        end_to_end = measure(lambda: three_level_linear(module, x, torch_module))
        dense = measure(lambda: torch_module.nn.functional.linear(x, dense_weight))
        results.append({
            "rows": int(row_count),
            "activation_quantization": quantization,
            "sm75_gemm": gemm,
            "sm75_end_to_end": end_to_end,
            "dense_fp16": dense,
            "gemm_p50_ratio_vs_dense": (
                gemm["p50_ms"] / max(dense["p50_ms"], 1e-9)
            ),
            "gemm_p50_speedup_vs_dense": (
                dense["p50_ms"] / max(gemm["p50_ms"], 1e-9)
            ),
            "end_to_end_p50_ratio_vs_dense": (
                end_to_end["p50_ms"] / max(dense["p50_ms"], 1e-9)
            ),
            "end_to_end_p50_speedup_vs_dense": (
                dense["p50_ms"] / max(end_to_end["p50_ms"], 1e-9)
            ),
            "max_abs_error": max_error,
            "max_abs_error_vs_dense_fp16": dense_semantic_error,
        })
    return {
        "status": "measured",
        "in_features": int(module.in_features),
        "out_features": int(module.out_features),
        "partition_counts": partition_counts,
        "average_weight_bits": average_weight_bits,
        "weight_bandwidth_upper_bound_vs_fp16": 16.0 / average_weight_bits,
        "shapes": results,
    }
