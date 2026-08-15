"""CPU contract tests for the three-level kernel/adapter boundary."""
import os
import sys

import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shmq.inference.shmq_3level_kernel import (  # noqa: E402
    _pack_int4_on_gpu,
    _pytorch_fallback,
)
from shmq.mixllm.adapter import (  # noqa: E402
    _mixllm_int4_to_signed_packed,
    convert_linear_to_mixllm,
    pack_int4_weights,
)


def _unpack_signed_nibbles(packed: torch.Tensor) -> torch.Tensor:
    low = (packed & 0x0F).to(torch.int16)
    high = ((packed >> 4) & 0x0F).to(torch.int16)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    return torch.stack((low, high), dim=-1).flatten(1).to(torch.int8)


def test_mixllm_offset_binary_is_converted_to_signed_nibbles():
    codes = torch.arange(-8, 8, dtype=torch.int8).reshape(1, 16)
    offset = (codes.to(torch.int16) + 8).to(torch.uint8)
    mixllm_packed = (offset[:, 1::2] << 4) | offset[:, 0::2]

    signed_packed = _mixllm_int4_to_signed_packed(mixllm_packed)

    assert torch.equal(_unpack_signed_nibbles(signed_packed), codes)
    assert torch.equal(signed_packed, _pack_int4_on_gpu(codes))


def test_adapter_pack_matches_three_level_reference_math():
    torch.manual_seed(7)
    weight = torch.randn(5, 128, dtype=torch.float32) * 0.2
    mixllm_packed, scales_mixllm, _ = pack_int4_weights(weight)
    signed_packed = _mixllm_int4_to_signed_packed(mixllm_packed)
    signed_codes = _unpack_signed_nibbles(signed_packed).float()
    scales = scales_mixllm.t().contiguous()

    x = torch.randn(3, 128, dtype=torch.float16)
    expected_weight = (
        signed_codes.reshape(5, 1, 128) * scales.float().unsqueeze(-1)
    ).half().float().reshape(5, 128)
    expected = (x.float() @ expected_weight.t()).half()
    actual = _pytorch_fallback(
        x, None, None, signed_packed, None, scales,
        True, 0, 0, 5, 128, 5,
    )

    assert torch.equal(actual, expected)


def test_three_level_reference_combines_all_precisions():
    torch.manual_seed(11)
    k = 128
    x = torch.randn(2, k, dtype=torch.float16)
    w16 = torch.randn(2, k, dtype=torch.float16)
    w8 = torch.randint(-127, 128, (3, k), dtype=torch.int8)
    w4_codes = torch.randint(-8, 8, (4, k), dtype=torch.int8)
    w4 = _pack_int4_on_gpu(w4_codes)
    s8 = torch.rand(3, 1, dtype=torch.float16) * 0.02
    s4 = torch.rand(4, 1, dtype=torch.float16) * 0.1

    actual = _pytorch_fallback(
        x, w16, w8, w4, s8, s4, True, 2, 3, 4, k, 9,
    )
    expected_weights = torch.cat(
        (
            w16.float(),
            (w8.float() * s8.float()).half().float(),
            (w4_codes.float() * s4.float()).half().float(),
        ),
        dim=0,
    )
    expected = (x.float() @ expected_weights.t()).half()

    assert torch.equal(actual, expected)


def test_adapter_rejects_input_channel_clusters():
    layer = nn.Linear(128, 256, bias=False, dtype=torch.float16)
    with pytest.raises(ValueError, match="input channels.*output rows"):
        convert_linear_to_mixllm(
            layer, 4, cluster_sizes={16: 0, 8: 64, 4: 64}, cluster_axis="input"
        )


def test_adapter_rejects_output_cluster_size_mismatch():
    layer = nn.Linear(128, 256, bias=False, dtype=torch.float16)
    with pytest.raises(ValueError, match="expected out_features=256"):
        convert_linear_to_mixllm(
            layer, 4, cluster_sizes={16: 0, 8: 64, 4: 64}, cluster_axis="output"
        )