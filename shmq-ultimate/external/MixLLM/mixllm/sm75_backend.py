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

    source = Path(__file__).resolve().parent / "kernels" / "three_level_sm75.cu"
    vendor_archive = source.parent / "cutlass_sm75_vendor.b64"
    vendor_root = source.parent / "cutlass"
    vendor_include = vendor_root / "include"
    vendor_header = vendor_include / "cutlass" / "array.h"
    if not vendor_header.exists():
        if not vendor_archive.exists():
            raise RuntimeError(
                "SM75 CUTLASS headers are missing; expected embedded vendor archive "
                f"at {vendor_archive}"
            )
        import base64
        import io
        import shutil
        import zipfile

        # The archive is a source bundle, not a safe project-root overlay: it also
        # contains historical copies of sm75_cutlass_testbed.h and the custom
        # cutlass_extension headers.  Extract into a private staging directory and
        # copy only its vendor CUTLASS subtree, so current project sources remain
        # authoritative.
        staging_root = source.parent / ".cutlass_vendor_staging"
        staged_vendor_root = staging_root / "mixllm" / "kernels" / "cutlass"
        if staging_root.exists():
            shutil.rmtree(staging_root)
        staging_root.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(
                io.BytesIO(base64.b64decode(vendor_archive.read_bytes()))
            ) as archive:
                for member in archive.infolist():
                    target = (staging_root / member.filename).resolve()
                    if target != staging_root and staging_root not in target.parents:
                        raise RuntimeError(
                            f"refusing unsafe CUTLASS archive member: {member.filename}"
                        )
                archive.extractall(staging_root)
            if not staged_vendor_root.is_dir():
                raise RuntimeError(
                    "CUTLASS vendor archive did not provide its expected vendor subtree"
                )
            if vendor_root.exists():
                shutil.rmtree(vendor_root)
            shutil.copytree(staged_vendor_root, vendor_root)
        finally:
            if staging_root.exists():
                shutil.rmtree(staging_root)
        if not vendor_header.exists():
            raise RuntimeError(
                f"CUTLASS vendor archive did not provide {vendor_header}"
            )
    kwargs = {
        "extra_include_paths": [str(vendor_include)],
    }
    if build_directory is not None:
        directory = Path(build_directory)
        directory.mkdir(parents=True, exist_ok=True)
        kwargs["build_directory"] = str(directory)
    load(
        name="mixllm_sm75_backend",
        sources=[str(source)],
        extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_75,code=sm_75"],
        extra_cflags=["-O3"],
        is_python_module=False,
        verbose=True,
        **kwargs,
    )
    _LOADED = True


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
    if x.shape[0] == 1 or not module.indices_4.numel():
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


def _prefill_metadata_for_cutlass(module, x, torch_module):
    """Cache CUTLASS's [groups, channels] prefill metadata layout.

    Decode keeps the checkpoint [channels, groups] layout.  The cache is
    invalidated by tensor identity/version/device/shape changes and must be
    initialized before CUDA graph capture, just like the INT4 expansion cache.
    """
    if x.shape[0] < 32:
        raise ValueError("CUTLASS prefill metadata is only used for rows >= 32")
    signature = (
        x.device,
        id(module.scale_int4), int(module.scale_int4._version),
        id(module.zero_int4), int(module.zero_int4._version),
        id(module.scale_int8), int(module.scale_int8._version),
        tuple(module.scale_int4.shape), tuple(module.zero_int4.shape),
        tuple(module.scale_int8.shape),
    )
    cached = module._sm75_prefill_metadata
    if cached is not None and cached[0] == signature:
        return cached
    if x.is_cuda and torch_module.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "SM75 CUTLASS metadata cache must be initialized before CUDA graph capture"
        )
    cached = (
        signature,
        module.scale_int4.transpose(0, 1).contiguous(),
        module.zero_int4.transpose(0, 1).contiguous(),
        module.scale_int8.transpose(0, 1).contiguous(),
    )
    module._sm75_prefill_metadata = cached
    return cached


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
    arguments = (
        x if x.is_contiguous() else x.contiguous(),
        input_int8 if input_int8.is_contiguous() else input_int8.contiguous(),
        scale_act if scale_act.is_contiguous() else scale_act.contiguous(),
        module.weight_int4,
        expanded_int4,
        *(tensor if tensor.is_contiguous() else tensor.contiguous()
          for tensor in packed_tensors[1:]),
    )
    if x.shape[0] >= 32 and (module.indices_4.numel() or module.indices_8.numel()):
        metadata = _prefill_metadata_for_cutlass(module, x, torch_module)
        native_v3 = getattr(torch_module.ops.mixllm_sm75,
                            "three_level_linear_v3", None)
        if native_v3 is not None:
            return native_v3(*arguments, *metadata[1:])
    return torch_module.ops.mixllm_sm75._three_level_linear_v2_unchecked(*arguments)


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
    def tensor_bytes(tensor):
        return int(tensor.numel() * tensor.element_size())

    def peak_allocated(callable_, device):
        torch_module.cuda.synchronize(device)
        torch_module.cuda.reset_peak_memory_stats(device)
        callable_()
        torch_module.cuda.synchronize(device)
        return int(torch_module.cuda.max_memory_allocated(device))

    dense_weight = module.dequantize_weight().to(torch_module.float16)
    packed_names = (
        "weight_int4", "scale_int4", "zero_int4", "indices_4",
        "weight_int8", "scale_int8", "indices_8",
        "weight_fp16", "indices_16",
    )
    packed_storage_bytes = {
        name: tensor_bytes(getattr(module, name)) for name in packed_names
    }
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
        expanded_cache = getattr(module, "_sm75_int4_expanded", None)
        expanded_tensor = expanded_cache[1] if expanded_cache is not None else None
        metadata_cache = getattr(module, "_sm75_prefill_metadata", None)
        metadata_tensors = metadata_cache[1:] if metadata_cache is not None else ()
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
        device = x.device
        peak_memory = {
            "quantization_peak_allocated_bytes": peak_allocated(
                lambda: quantize_activation_native(x, torch_module, module.group_size),
                device,
            ),
            "gemm_peak_allocated_bytes": peak_allocated(
                lambda: three_level_linear_prequantized(
                    module, x, input_int8, scale_act, torch_module,
                ),
                device,
            ),
            "end_to_end_peak_allocated_bytes": peak_allocated(
                lambda: three_level_linear(module, x, torch_module),
                device,
            ),
            "dense_peak_allocated_bytes": peak_allocated(
                lambda: torch_module.nn.functional.linear(x, dense_weight),
                device,
            ),
        }
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
            "memory": {
                "packed_storage_bytes": dict(packed_storage_bytes),
                "packed_storage_total_bytes": sum(packed_storage_bytes.values()),
                "dense_weight_bytes": tensor_bytes(dense_weight),
                "activation_fp16_bytes": tensor_bytes(x),
                "activation_quantized_bytes": tensor_bytes(input_int8),
                "activation_scale_bytes": tensor_bytes(scale_act),
                "expanded_int4_bytes": (
                    tensor_bytes(expanded_tensor) if expanded_tensor is not None else 0
                ),
                "prefill_metadata_bytes": sum(
                    tensor_bytes(tensor) for tensor in metadata_tensors
                ),
                "output_bytes": tensor_bytes(actual),
                "peak_cuda_memory": peak_memory,
            },
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
