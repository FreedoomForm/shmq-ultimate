import torch
import torch.nn as nn

from shmq.inference.k_partition_converter import convert_model_to_k_partition
from shmq.inference.k_partition_linear import SHMQKPartitionLinear
from shmq.permutation.decoupled import decoupled_permutation_3level
from shmq.permutation.rmsnorm_fusion import fuse_permutation_into_rmsnorm


def test_exact_cluster_sizes_drive_permutation():
    sensitivity = torch.arange(12, dtype=torch.float32)
    metric = torch.arange(12, dtype=torch.float32).flip(0)
    perm, sizes = decoupled_permutation_3level(
        sensitivity, metric, 0.0, 0.0, exact_cluster_sizes={16: 2, 8: 4, 4: 6}
    )
    assert sizes == {16: 2, 8: 4, 4: 6}
    assert set(perm[:2].tolist()) == {10, 11}
    assert set(perm[2:6].tolist()) == {6, 7, 8, 9}


def test_k_partition_forward_matches_dequantized_weight():
    torch.manual_seed(7)
    weight = torch.randn(5, 16)
    bias = torch.randn(5)
    layer = SHMQKPartitionLinear.from_weight(
        weight, {16: 4, 8: 4, 4: 8}, group_size=4, bias=bias
    )
    x = torch.randn(2, 3, 16)
    actual = layer(x)
    expected = torch.nn.functional.linear(x, layer.dequantize_weight(), bias)
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-3, rtol=2e-3)


def test_state_dict_roundtrip_preserves_output():
    torch.manual_seed(11)
    layer = SHMQKPartitionLinear.from_weight(
        torch.randn(3, 16), {16: 4, 8: 4, 4: 8}, group_size=4
    )
    clone = SHMQKPartitionLinear(16, 3, 4, 4, group_size=4)
    clone.load_state_dict(layer.state_dict())
    x = torch.randn(7, 16)
    torch.testing.assert_close(layer(x), clone(x), atol=0, rtol=0)


class _RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(width))
        self.eps = 1e-6

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class _Attention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)


class _Layer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.input_layernorm = _RMSNorm(width)
        self.self_attn = _Attention(width)


class _Inner(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(width)])


class _ToyModel(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.model = _Inner(width)


def test_rmsnorm_layout_transform_preserves_linear_output():
    torch.manual_seed(19)
    model = _ToyModel()
    norm = model.model.layers[0].input_layernorm
    linear = model.model.layers[0].self_attn.q_proj
    x = torch.randn(2, 3, 8)
    expected = linear(norm(x))
    perm = torch.tensor([3, 0, 7, 4, 1, 6, 2, 5])
    linear.weight.data = linear.weight.data[:, perm]
    fuse_permutation_into_rmsnorm(
        model, {"model.layers.0.self_attn.q_proj": perm}
    )
    actual = linear(model.model.layers[0].input_layernorm(x))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_explicit_input_gather_preserves_non_fusable_linear_output():
    torch.manual_seed(23)
    model = _ToyModel()
    linear = model.model.layers[0].self_attn.o_proj
    x = torch.randn(4, 8)
    expected = linear(x)
    perm = torch.tensor([5, 2, 0, 7, 1, 6, 4, 3])
    linear.weight.data = linear.weight.data[:, perm]
    fuse_permutation_into_rmsnorm(
        model, {"model.layers.0.self_attn.o_proj": perm}
    )
    torch.testing.assert_close(linear(x), expected, atol=1e-6, rtol=1e-6)


def test_converter_preserves_input_gather_and_step8_codes():
    torch.manual_seed(29)
    model = _ToyModel(width=16)
    name = "model.layers.0.self_attn.o_proj"
    linear = model.model.layers[0].self_attn.o_proj
    original_weight = linear.weight.detach().clone()
    x = torch.randn(2, 16)
    perm = torch.arange(15, -1, -1)
    linear.weight.data = linear.weight.data[:, perm]
    fuse_permutation_into_rmsnorm(model, {name: perm})
    expected = linear(x)
    linear._shmq_segment_codes = {
        8: torch.zeros(16, 4, dtype=torch.int8),
        4: torch.ones(16, 8, dtype=torch.int8),
    }
    linear._shmq_segment_scales = {
        8: torch.ones(16, 1, dtype=torch.float16),
        4: torch.full((16, 2), 0.25, dtype=torch.float16),
    }
    convert_model_to_k_partition(
        model, [name], {name: {16: 4, 8: 4, 4: 8}}, group_size=4, verbose=False
    )
    packed = model.model.layers[0].self_attn.o_proj
    assert torch.equal(packed.input_permutation, perm)
    assert torch.count_nonzero(packed.weight8) == 0
    assert torch.all(packed.scale4 == 0.25)
    manual = torch.nn.functional.linear(x[:, perm], packed.dequantize_weight(), packed.bias)
    torch.testing.assert_close(packed(x).float(), manual.float(), atol=2e-3, rtol=2e-3)


def test_rejects_misaligned_quantized_segments():
    try:
        SHMQKPartitionLinear(16, 3, 4, 3, group_size=4)
    except ValueError as exc:
        assert "align" in str(exc)
    else:
        raise AssertionError("misaligned K segment was accepted")