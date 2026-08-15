"""Model conversion for the faithful SHMQ input-axis three-level layout."""
from __future__ import annotations

from typing import Dict, List
import torch.nn as nn

from ..utils import get_module_by_name, set_module_by_name
from .k_partition_linear import SHMQKPartitionLinear


def convert_model_to_k_partition(model: nn.Module, layer_names: List[str],
                                 cluster_sizes: Dict[str, Dict[int, int]],
                                 group_size: int = 128,
                                 verbose: bool = True) -> Dict[str, Dict]:
    summary = {}
    for name in layer_names:
        mod = get_module_by_name(model, name)
        if not isinstance(mod, nn.Linear):
            continue
        cs = cluster_sizes.get(name)
        if cs is None:
            raise ValueError(f"missing Cin cluster_sizes for {name}")
        packed = SHMQKPartitionLinear.from_quantized_linear(mod, cs, group_size)
        set_module_by_name(model, name, packed)
        summary[name] = {"k16": packed.k16, "k8": packed.k8, "k4": packed.k4,
                         "avg_bits": (16 * packed.k16 + 8 * packed.k8 + 4 * packed.k4) / packed.in_features}
    if verbose:
        print(f"[k-partition] converted {len(summary)}/{len(layer_names)} layers; PyTorch reference backend")
    return summary
