"""Packed FP16/INT8/INT4 output-feature linear for MixLLM."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from mixllm.quantization.three_level import LEVELS, ThreeLevelAllocation


def _pack_uint4(codes: torch.Tensor) -> torch.Tensor:
    if codes.shape[-1] % 2:
        raise ValueError("INT4 input width must be even")
    values = codes.to(torch.uint8)
    return (values[..., 0::2] | (values[..., 1::2] << 4)).contiguous()


def _unpack_uint4(packed: torch.Tensor) -> torch.Tensor:
    output = torch.empty((*packed.shape[:-1], packed.shape[-1] * 2),
                         dtype=torch.uint8, device=packed.device)
    output[..., 0::2] = packed & 0x0F
    output[..., 1::2] = packed >> 4
    return output


class ThreeLevelLinear(nn.Module):
    """Reference backend and serialization contract for a three-level op."""

    quant_method = "mixllm_three_level"

    def __init__(self, in_features: int, out_features: int, group_size: int = 128,
                 bias: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0 or group_size <= 0:
            raise ValueError("linear dimensions and group_size must be positive")
        if in_features % group_size:
            raise ValueError("in_features must be divisible by group_size")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.register_buffer("weight_fp16", torch.empty(0, in_features, dtype=torch.float16))
        self.register_buffer("weight_int8", torch.empty(0, in_features, dtype=torch.int8))
        self.register_buffer("scale_int8", torch.empty(0, in_features // group_size, dtype=torch.float16))
        self.register_buffer("weight_int4", torch.empty(0, in_features // 2, dtype=torch.uint8))
        self.register_buffer("scale_int4", torch.empty(0, in_features // group_size, dtype=torch.float16))
        self.register_buffer("zero_int4", torch.empty(0, in_features // group_size, dtype=torch.uint8))
        for bit in LEVELS:
            self.register_buffer(f"indices_{bit}", torch.empty(0, dtype=torch.int32))
        self.register_buffer("bias", None if bias is None else bias.detach().to(torch.float16))
        # SM75 runtime caches are not buffers because they contain no model state.
        self._sm75_fp16_placeholders = None
        self._sm75_int4_expanded = None
        self._sm75_prefill_metadata = None

    def _apply(self, fn, recurse=True):
        self._sm75_fp16_placeholders = None
        self._sm75_int4_expanded = None
        self._sm75_prefill_metadata = None
        return super()._apply(fn, recurse)

    @classmethod
    @torch.no_grad()
    def from_weight(cls, weight: torch.Tensor, allocation: ThreeLevelAllocation,
                    group_size: int = 128,
                    bias: Optional[torch.Tensor] = None) -> "ThreeLevelLinear":
        if weight.dim() != 2:
            raise ValueError("weight must have shape [out_features, in_features]")
        out_features, in_features = weight.shape
        allocation.verify(out_features)
        layer = cls(in_features, out_features, group_size, bias).to(weight.device)
        groups = in_features // group_size

        for bit in LEVELS:
            indices = torch.tensor(allocation.indices[bit], dtype=torch.int32,
                                   device=weight.device)
            setattr(layer, f"indices_{bit}", indices)
            if not indices.numel():
                continue
            chunk = weight.index_select(0, indices.long()).float()
            if bit == 16:
                layer.weight_fp16 = chunk.to(torch.float16).contiguous()
            elif bit == 8:
                viewed = chunk.reshape(-1, groups, group_size)
                scale = (viewed.abs().amax(-1) / 127).clamp_min(1e-5)
                codes = (viewed / scale.unsqueeze(-1)).round().clamp(-128, 127)
                layer.weight_int8 = codes.to(torch.int8).reshape(-1, in_features).contiguous()
                layer.scale_int8 = scale.to(torch.float16).contiguous()
            else:
                viewed = chunk.reshape(-1, groups, group_size)
                minimum, maximum = viewed.amin(-1), viewed.amax(-1)
                scale = ((maximum - minimum) / 15).clamp_min(1e-5)
                zero = (-minimum / scale).round().clamp(0, 15)
                codes = ((viewed / scale.unsqueeze(-1)).round() + zero.unsqueeze(-1)).clamp(0, 15)
                layer.weight_int4 = _pack_uint4(codes.to(torch.uint8).reshape(-1, in_features))
                layer.scale_int4 = scale.to(torch.float16).contiguous()
                layer.zero_int4 = zero.to(torch.uint8).contiguous()
        return layer

    @torch.no_grad()
    def dequantize_weight(self) -> torch.Tensor:
        device = self.weight_fp16.device
        output = torch.empty(self.out_features, self.in_features,
                             dtype=torch.float32, device=device)
        if self.indices_16.numel():
            output.index_copy_(0, self.indices_16.long(), self.weight_fp16.float())
        if self.indices_8.numel():
            scales = self.scale_int8.float().repeat_interleave(self.group_size, dim=1)
            output.index_copy_(0, self.indices_8.long(), self.weight_int8.float() * scales)
        if self.indices_4.numel():
            codes = _unpack_uint4(self.weight_int4).float()
            scales = self.scale_int4.float().repeat_interleave(self.group_size, dim=1)
            zeros = self.zero_int4.float().repeat_interleave(self.group_size, dim=1)
            output.index_copy_(0, self.indices_4.long(), (codes - zeros) * scales)
        return output

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self._sm75_fp16_placeholders = None
        self._sm75_int4_expanded = None
        self._sm75_prefill_metadata = None
        # Two-level checkpoints predate the FP16 partition. Treat its omitted
        # tensors as an empty partition while preserving strict loading for all
        # other packed state.
        legacy_defaults = {
            "weight_fp16": self.weight_fp16,
            "indices_16": self.indices_16,
        }
        for name, value in legacy_defaults.items():
            state_dict.setdefault(prefix + name, value)
        variable_buffers = (
            "weight_fp16", "weight_int8", "scale_int8", "weight_int4",
            "scale_int4", "zero_int4", "indices_4", "indices_8", "indices_16",
        )
        for name in variable_buffers:
            key = prefix + name
            if key in state_dict:
                setattr(self, name, torch.empty_like(state_dict[key], device=self.weight_fp16.device))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {x.shape[-1]}")
        if x.is_cuda and tuple(torch.cuda.get_device_capability(x.device)) == (7, 5):
            from mixllm.sm75_backend import load_sm75_backend, three_level_linear

            load_sm75_backend(torch)
            flat = x.reshape(-1, self.in_features)
            result = three_level_linear(self, flat, torch)
            if self.bias is not None:
                result = result + self.bias.float()
            return result.reshape(*x.shape[:-1], self.out_features).to(x.dtype)
        result = torch.nn.functional.linear(x.float(), self.dequantize_weight(),
                                            None if self.bias is None else self.bias.float())
        return result.to(x.dtype)