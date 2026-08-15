"""Correct SHMQ three-level linear for an input-channel (K-axis) partition.

SHMQ partitions the reduction dimension of a linear operator.  This module is
the executable reference contract for that layout.  It deliberately uses
PyTorch GEMMs until a CUDA kernel with this exact ABI is validated; silently
using an output-row MixLLM layout would produce numerically wrong models.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .weight_packing import _symmetric_quantize_int, pack_int4, unpack_int4


class SHMQKPartitionLinear(nn.Module):
    """Linear whose input K dimension is split into FP16, INT8 and INT4."""

    quant_method = "shmq_k_partition_reference"

    def __init__(self, in_features: int, out_features: int, k16: int,
                 k8: int, group_size: int = 128,
                 bias: Optional[torch.Tensor] = None,
                 device: Optional[torch.device] = None,
                 input_permutation: Optional[torch.Tensor] = None):
        super().__init__()
        if min(k16, k8) < 0 or k16 + k8 > in_features:
            raise ValueError("invalid K-partition sizes")
        k4 = in_features - k16 - k8
        if any(k % group_size for k in (k8, k4)):
            raise ValueError("INT8 and INT4 K segments must align to group_size")
        if k4 % 2:
            raise ValueError("INT4 K segment must have an even width")
        dev = device or torch.device("cpu")
        self.in_features, self.out_features = in_features, out_features
        self.k16, self.k8, self.k4, self.group_size = k16, k8, k4, group_size
        self.register_buffer("weight16", torch.empty(out_features, k16, dtype=torch.float16, device=dev))
        self.register_buffer("weight8", torch.empty(out_features, k8, dtype=torch.int8, device=dev))
        self.register_buffer("scale8", torch.empty(out_features, k8 // group_size, dtype=torch.float16, device=dev))
        self.register_buffer("weight4", torch.empty(out_features, k4 // 2, dtype=torch.uint8, device=dev))
        self.register_buffer("scale4", torch.empty(out_features, k4 // group_size, dtype=torch.float16, device=dev))
        self.register_buffer("bias", None if bias is None else bias.to(dev, dtype=torch.float16).contiguous())
        if input_permutation is not None:
            p = input_permutation.to(dev, dtype=torch.long).contiguous()
            if p.numel() != in_features or p.unique().numel() != in_features:
                raise ValueError("input_permutation must be a complete K-axis permutation")
            has_permutation = True
        else:
            p = torch.arange(in_features, dtype=torch.long, device=dev)
            has_permutation = False
        self.register_buffer("input_permutation", p)
        self.register_buffer("has_input_permutation", torch.tensor(has_permutation, device=dev))

    @classmethod
    def from_weight(cls, weight: torch.Tensor, cluster_sizes: Dict[int, int],
                    group_size: int = 128, bias: Optional[torch.Tensor] = None,
                    device: Optional[torch.device] = None,
                    input_permutation: Optional[torch.Tensor] = None) -> "SHMQKPartitionLinear":
        out_features, in_features = weight.shape
        k16, k8 = int(cluster_sizes.get(16, 0)), int(cluster_sizes.get(8, 0))
        k4 = int(cluster_sizes.get(4, in_features - k16 - k8))
        if k16 + k8 + k4 != in_features:
            raise ValueError("cluster sizes must sum to weight.in_features")
        obj = cls(in_features, out_features, k16, k8, group_size, bias,
                  device or weight.device, input_permutation)
        w = weight.detach().to(device or weight.device)
        if k16:
            obj.weight16.copy_(w[:, :k16].to(torch.float16))
        if k8:
            codes8, scales8 = _symmetric_quantize_int(w[:, k16:k16 + k8], 8, group_size)
            obj.weight8.copy_(codes8)
            obj.scale8.copy_(scales8)
        if k4:
            codes4, scales4 = _symmetric_quantize_int(w[:, k16 + k8:], 4, group_size)
            obj.weight4.copy_(pack_int4(codes4))
            obj.scale4.copy_(scales4)
        return obj

    @classmethod
    def from_quantized_linear(cls, layer: nn.Linear, cluster_sizes: Dict[int, int],
                              group_size: int = 128) -> "SHMQKPartitionLinear":
        """Build from Step 8 codes/scales without a second quantization pass."""
        input_permutation = getattr(layer, "_shmq_input_permutation", None)
        obj = cls.from_weight(
            layer.weight.data, cluster_sizes, group_size,
            layer.bias.data if layer.bias is not None else None,
            input_permutation=input_permutation,
        )
        codes = getattr(layer, "_shmq_segment_codes", None)
        scales = getattr(layer, "_shmq_segment_scales", None)
        if codes is None or scales is None:
            return obj
        if obj.k8:
            obj.weight8.copy_(codes[8].to(obj.weight8.device))
            obj.scale8.copy_(scales[8].to(obj.scale8.device))
        if obj.k4:
            obj.weight4.copy_(pack_int4(codes[4].to(obj.weight4.device)))
            obj.scale4.copy_(scales[4].to(obj.scale4.device))
        return obj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {x.shape[-1]}")
        if bool(self.has_input_permutation.item()):
            x = x.index_select(-1, self.input_permutation)
        leading = x.shape[:-1]
        xf = x.reshape(-1, self.in_features).float()
        y = torch.zeros(xf.shape[0], self.out_features, device=xf.device, dtype=torch.float32)
        if self.k16:
            y += xf[:, :self.k16] @ self.weight16.float().t()
        if self.k8:
            s8 = self.scale8.float().repeat_interleave(self.group_size, 1)
            y += xf[:, self.k16:self.k16 + self.k8] @ (self.weight8.float() * s8).t()
        if self.k4:
            codes4 = unpack_int4(self.weight4).float()
            s4 = self.scale4.float().repeat_interleave(self.group_size, 1)
            y += xf[:, self.k16 + self.k8:] @ (codes4 * s4).t()
        if self.bias is not None:
            y += self.bias.float()
        return y.reshape(*leading, self.out_features).to(x.dtype)

    @torch.no_grad()
    def dequantize_weight(self) -> torch.Tensor:
        out = torch.zeros(self.out_features, self.in_features, device=self.weight16.device, dtype=torch.float32)
        if self.k16:
            out[:, :self.k16] = self.weight16.float()
        if self.k8:
            out[:, self.k16:self.k16 + self.k8] = self.weight8.float() * self.scale8.float().repeat_interleave(self.group_size, 1)
        if self.k4:
            out[:, self.k16 + self.k8:] = unpack_int4(self.weight4).float() * self.scale4.float().repeat_interleave(self.group_size, 1)
        return out
