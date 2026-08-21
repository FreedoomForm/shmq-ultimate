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


def _cutlass_uint4_interleave_indices(width: int, device: torch.device) -> torch.Tensor:
    first = []
    for index in range(width):
        sub = index % 32
        if 4 <= sub < 8:
            first.append(index + 12)
        elif 8 <= sub < 12:
            first.append(index - 4)
        elif 12 <= sub < 16:
            first.append(index + 8)
        elif 16 <= sub < 20:
            first.append(index - 8)
        elif 20 <= sub < 24:
            first.append(index + 4)
        elif 24 <= sub < 28:
            first.append(index - 12)
        else:
            first.append(index)
    second = []
    for base in range(0, width, 8):
        second.extend((base, base + 4, base + 1, base + 5,
                       base + 2, base + 6, base + 3, base + 7))
    return torch.tensor(first, dtype=torch.int64, device=device)[second]


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
        self._sm75_int4_interleaved = None
        self._sm75_prefill_metadata = None
        self._sm75_packed_tensors = None

    def _apply(self, fn, recurse=True):
        self._sm75_fp16_placeholders = None
        self._sm75_int4_expanded = None
        self._sm75_int4_interleaved = None
        self._sm75_prefill_metadata = None
        self._sm75_packed_tensors = None
        result = super()._apply(fn, recurse)
        if self.weight_int4.is_cuda and self.indices_4.numel():
            self.prepare_sm75_prefill_cache()
            self.prepare_sm75_prefill_int4()
        if self.weight_int4.is_cuda and (self.indices_4.numel() or self.indices_8.numel()):
            self.prepare_sm75_prefill_metadata()
        return result

    @torch.no_grad()
    def prepare_sm75_packed_tensors(self):
        """Cache the immutable packed tensors used by the SM75 operator ABI."""
        tensors = (
            self.weight_int4, self.scale_int4, self.zero_int4, self.indices_4,
            self.weight_int8, self.scale_int8, self.indices_8,
            self.weight_fp16, self.indices_16,
        )
        signature = tuple(
            (id(tensor), int(tensor._version), tensor.device,
             tuple(tensor.shape), tuple(tensor.stride()), tensor.is_contiguous())
            for tensor in tensors
        )
        cached = self._sm75_packed_tensors
        if cached is not None and cached[0] == signature:
            return cached[1]
        if not all(tensor.is_contiguous() for tensor in tensors):
            self._sm75_packed_tensors = None
            return None
        self._sm75_packed_tensors = (signature, tensors)
        return tensors

    @torch.no_grad()
    def prepare_sm75_prefill_cache(self):
        """Prepare the signed INT8 INT4 representation outside the hot forward path."""
        if not self.indices_4.numel() or not self.weight_int4.numel():
            return None
        signature = (
            self.weight_int4.device,
            id(self.weight_int4), int(self.weight_int4._version),
            id(self.zero_int4), int(self.zero_int4._version),
            tuple(self.weight_int4.shape), tuple(self.zero_int4.shape),
        )
        cached = self._sm75_int4_expanded
        if cached is not None and cached[0] == signature:
            return cached[1]
        codes = _unpack_uint4(self.weight_int4)
        zeros = self.zero_int4.repeat_interleave(
            self.group_size, dim=1,
        ).to(torch.int16)
        expanded = (codes.to(torch.int16) - zeros).to(torch.int8).contiguous()
        self._sm75_int4_expanded = (signature, expanded)
        return expanded

    @torch.no_grad()
    def prepare_sm75_prefill_int4(self):
        """Prepare original MixLLM's persistent interleaved packed INT4 layout."""
        if not self.indices_4.numel() or not self.weight_int4.numel():
            return None
        signature = (
            self.weight_int4.device,
            id(self.weight_int4), int(self.weight_int4._version),
            tuple(self.weight_int4.shape), tuple(self.weight_int4.stride()),
        )
        cached = self._sm75_int4_interleaved
        if cached is not None and cached[0] == signature:
            return cached[1]
        codes = _unpack_uint4(self.weight_int4)
        permutation = _cutlass_uint4_interleave_indices(
            self.in_features, self.weight_int4.device,
        )
        interleaved = _pack_uint4(codes.index_select(1, permutation))
        self._sm75_int4_interleaved = (signature, interleaved)
        return interleaved

    @torch.no_grad()
    def prepare_sm75_prefill_metadata(self):
        """Prepare the [groups, channels] CUTLASS metadata layout before forward."""
        if not self.indices_4.numel() and not self.indices_8.numel():
            return None
        signature = (
            self.scale_int4.device,
            id(self.scale_int4), int(self.scale_int4._version),
            id(self.zero_int4), int(self.zero_int4._version),
            id(self.scale_int8), int(self.scale_int8._version),
            tuple(self.scale_int4.shape), tuple(self.zero_int4.shape),
            tuple(self.scale_int8.shape),
        )
        cached = self._sm75_prefill_metadata
        if cached is not None and cached[0] == signature:
            return cached
        cached = (
            signature,
            self.scale_int4.transpose(0, 1).contiguous(),
            self.zero_int4.transpose(0, 1).contiguous(),
            self.scale_int8.transpose(0, 1).contiguous(),
        )
        self._sm75_prefill_metadata = cached
        return cached

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
        layer.prepare_sm75_packed_tensors()
        if layer.weight_int4.is_cuda:
            layer.prepare_sm75_prefill_cache()
            layer.prepare_sm75_prefill_int4()
            layer.prepare_sm75_prefill_metadata()
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
        self._sm75_int4_interleaved = None
        self._sm75_prefill_metadata = None
        self._sm75_packed_tensors = None
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
        self.prepare_sm75_packed_tensors()
        if self.weight_int4.is_cuda:
            self.prepare_sm75_prefill_cache()
            self.prepare_sm75_prefill_int4()
            self.prepare_sm75_prefill_metadata()

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