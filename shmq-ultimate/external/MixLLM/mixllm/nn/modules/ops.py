# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

import torch

__all__ = ["quantize", "transpose", "mixllm_gemm", "mixllm_three_level_gemm"]


def transpose(a):
    return torch.ops.kernels_mixllm.transpose(a)


def quantize(a):
    return torch.ops.kernels_mixllm.quantize(a)


def mixllm_gemm(a, scale_act, zero, scale_int8, scale_int4, indices_int8,
                indices_int4, b_int8, b_int4):
    return torch.ops.kernels_mixllm.gemm(a, scale_act, zero, scale_int8,
                                         scale_int4, indices_int8, indices_int4,
                                         b_int8, b_int4)


def mixllm_three_level_gemm(a, scale_act, zero_int4, scale_int8,
                            scale_int4, indices_int8, indices_int4,
                            indices_fp16, b_int8, b_int4, b_fp16):
    """Correctness-first 4/8/16 ABI using the existing MixLLM quantized op.

    The quantized partitions run through the upstream CUDA extension. The FP16
    partition is computed with PyTorch GEMM, and all outputs are scattered to
    their original output-feature positions. A future fused op can replace this
    implementation without changing the checkpoint contract.
    """
    partitions = (indices_int4, indices_int8, indices_fp16)
    total_n = sum(indices.numel() for indices in partitions)
    if total_n == 0:
        raise ValueError("at least one output partition is required")
    complete = torch.cat(partitions).long()
    if complete.unique().numel() != total_n or complete.min().item() != 0 or complete.max().item() != total_n - 1:
        raise ValueError("4/8/16 indices must form a complete output-channel partition")

    m, k = a.shape
    if k % 128:
        raise ValueError("current MixLLM CUDA ABI requires K divisible by 128")
    output = torch.empty((m, total_n), dtype=torch.float16, device=a.device)
    n8, n4 = indices_int8.numel(), indices_int4.numel()
    if n8 + n4:
        local_int8 = torch.arange(n8, dtype=torch.int32, device=a.device)
        local_int4 = torch.arange(n8, n8 + n4, dtype=torch.int32, device=a.device)
        quantized = mixllm_gemm(
            a, scale_act, zero_int4, scale_int8, scale_int4,
            local_int8, local_int4, b_int8, b_int4,
        )
        if n8 and n4:
            quantized = quantized.t().contiguous()
        output.index_copy_(1, torch.cat((indices_int8, indices_int4)).long(), quantized)

    if indices_fp16.numel():
        groups = k // 128
        activation = (
            a.float().reshape(m, groups, 128)
            * scale_act[:, :m].t().reshape(m, groups, 1).float()
        ).reshape(m, k)
        fp16_result = activation @ b_fp16.float().t()
        output.index_copy_(1, indices_fp16.long(), fp16_result.to(torch.float16))
    return output


def _register_fake_if_defined(qualified_name):
    """Register a fake implementation only when the optional op exists."""
    namespace, name = qualified_name.split("::", 1)
    try:
        getattr(getattr(torch.ops, namespace), name)
    except AttributeError:
        return lambda function: function
    return torch.library.register_fake(qualified_name)


@_register_fake_if_defined("kernels_mixllm::quantize")
def quantize_abstract(a):
    torch._check(a.dim() == 2, "Input must be a 2D tensor")
    m = a.shape[0]
    n = a.shape[1]
    group_size = 128
    torch._check(a.is_cuda, "Input must be on CUDA device")
    torch._check(a.dtype == torch.float16, "Input must be float16")
    torch._check(
        n % group_size == 0,
        "Input must have a second dimension that is a multiple of group_size")

    m_round_even = m + (m % 2)
    return (torch.empty((m, n), dtype=torch.int8, device="cuda:0"),
            torch.empty((n // group_size, m_round_even),
                        dtype=torch.float16,
                        device="cuda:0"))


@_register_fake_if_defined("kernels_mixllm::transpose")
def transpose_abstract(a):
    torch._check(a.dim() == 2, "Input must be a 2D tensor")
    m = a.shape[0]
    n = a.shape[1]
    torch._check(a.is_cuda, "Input must be on CUDA device")
    torch._check(a.dtype == torch.float16, "Input must be float16")
    return torch.empty((n, m), dtype=torch.float16, device="cuda:0")


@_register_fake_if_defined("kernels_mixllm::gemm")
def mixllm_gemm_abstract(a, scale_act, zero, scale_int8, scale_int4,
                         indices_int8, indices_int4, b_int8, b_int4):
    torch._check(a.is_cuda, "Input tensor A must be on CUDA device")
    torch._check(a.dtype == torch.int8, "Input tensor A must be int8")
    torch._check(scale_act.dtype == torch.float16,
                 "Scale activation tensor must be float16")
    torch._check(zero.dtype == torch.uint8, "Zero tensor must be uint8")
    torch._check(scale_int8.dtype == torch.float16,
                 "Scale int8 tensor must be float16")
    torch._check(scale_int4.dtype == torch.float16,
                 "Scale int4 tensor must be float16")
    torch._check(indices_int8.dtype == torch.int32,
                 "Indices int8 tensor must be int32")
    torch._check(indices_int4.dtype == torch.int32,
                 "Indices int4 tensor must be int32")
    torch._check(b_int8.dtype == torch.int8, "B int8 tensor must be int8")
    torch._check(b_int4.dtype == torch.uint8, "B int4 tensor must be uint8")
    torch._check(b_int8.is_cuda, "B int8 tensor must be on CUDA device")
    torch._check(b_int4.is_cuda, "B int4 tensor must be on CUDA device")
    torch._check(a.dim() == 2, "Input tensor A must be 2D")
    torch._check(scale_act.dim() == 2, "Scale activation tensor must be 2D")
    torch._check(zero.dim() == 2, "Zero tensor must be 2D")
    torch._check(scale_int8.dim() == 2, "Scale int8 tensor must be 2D")
    torch._check(scale_int4.dim() == 2, "Scale int4 tensor must be 2D")
    torch._check(indices_int8.dim() == 1, "Indices int8 tensor must be 1D")
    torch._check(indices_int4.dim() == 1, "Indices int4 tensor must be 1D")
    torch._check(b_int8.dim() == 2, "B int8 tensor must be 2D")
    torch._check(b_int4.dim() == 2, "B int4 tensor must be 2D")

    m = a.shape[0]
    n = (indices_int4.numel() + indices_int8.numel())
    is_row_major = (indices_int4.numel() == 0 or indices_int8.numel() == 0)
    if is_row_major:
        c = torch.empty((m, n), dtype=torch.float16, device="cuda:0")
    else:
        c = torch.empty((n, m), dtype=torch.float16, device="cuda:0")

    return c
