# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

import os
import json
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from transformers.utils.hub import PushToHubMixin


@dataclass
class MixLLMConfig(PushToHubMixin):
    quant_method: str = field(default="mixllm")
    ratio: float = field(default=1.0)
    # Legacy MixLLM uses ratio as the INT8 fraction.  These fields are
    # optional so existing two-level checkpoints remain loadable.
    precision_percentages: Optional[Dict[str, int]] = None
    allocation_file: Optional[str] = None
    allocation_version: int = 1
    group_size: int = 128
    backend: str = "auto"
    format_version: int = 2
    model_revision: Optional[str] = None
    calibration_seed: Optional[int] = None
    dtype: str = "float16"
    kernel_capability: Optional[Tuple[int, int]] = None
    config_file_name = "config.json"
    modules_to_not_convert: Optional[List] = None

    @classmethod
    def from_dict(cls, quant_config: Dict = {}):
        if not quant_config:
            quant_config = cls()
        else:
            supported = {
                key: value for key, value in quant_config.items()
                if key in cls.__dataclass_fields__
            }
            quant_config = cls(**supported)

        if quant_config.precision_percentages is None and quant_config.ratio != 1.0:
            int8 = round(float(quant_config.ratio) * 100)
            quant_config.precision_percentages = {"4": 100 - int8, "8": int8, "16": 0}
        if quant_config.precision_percentages is not None:
            percentages = quant_config.precision_percentages
            normalized = {str(bit): int(percentages.get(str(bit), 0))
                          for bit in (4, 8, 16)}
            if any(value < 0 for value in normalized.values()) or sum(normalized.values()) != 100:
                raise ValueError("precision_percentages for 4/8/16 must sum to 100")
            quant_config.precision_percentages = normalized
        if quant_config.group_size <= 0:
            raise ValueError("group_size must be positive")
        if quant_config.format_version not in (1, 2):
            raise ValueError("unsupported MixLLM config format_version")
        if quant_config.dtype not in {"float16", "bfloat16"}:
            raise ValueError("dtype must be float16 or bfloat16")
        if quant_config.kernel_capability is not None:
            capability = tuple(int(value) for value in quant_config.kernel_capability)
            if len(capability) != 2 or any(value < 0 for value in capability):
                raise ValueError("kernel_capability must be a CUDA (major, minor) pair")
            quant_config.kernel_capability = capability

        return quant_config

    def to_dict(self):
        return {
            "ratio": self.ratio,
            "modules_to_not_convert": self.modules_to_not_convert,
            "precision_percentages": self.precision_percentages,
            "allocation_file": self.allocation_file,
            "allocation_version": self.allocation_version,
            "group_size": self.group_size,
            "backend": self.backend,
            "format_version": self.format_version,
            "model_revision": self.model_revision,
            "calibration_seed": self.calibration_seed,
            "dtype": self.dtype,
            "kernel_capability": self.kernel_capability,
        }

    def to_transformers_dict(self):
        return {
            "quant_method": self.quant_method,
            "ratio": self.ratio,
            "modules_to_not_convert": self.modules_to_not_convert,
            "precision_percentages": self.precision_percentages,
            "allocation_file": self.allocation_file,
            "allocation_version": self.allocation_version,
            "group_size": self.group_size,
            "backend": self.backend,
            "format_version": self.format_version,
            "model_revision": self.model_revision,
            "calibration_seed": self.calibration_seed,
            "dtype": self.dtype,
            "kernel_capability": self.kernel_capability,
        }

    def from_transformers_dict(self, transformers_dict: Dict):
        return {
            "quant_method":
                transformers_dict.get("quant_method"),
            "ratio":
                transformers_dict.get("ratio"),
            "modules_to_not_convert":
                transformers_dict.get("modules_to_not_convert"),
            "precision_percentages":
                transformers_dict.get("precision_percentages"),
            "allocation_file": transformers_dict.get("allocation_file"),
            "allocation_version": transformers_dict.get("allocation_version", 1),
            "group_size": transformers_dict.get("group_size", 128),
            "backend": transformers_dict.get("backend", "auto"),
            "format_version": transformers_dict.get("format_version", 1),
            "model_revision": transformers_dict.get("model_revision"),
            "calibration_seed": transformers_dict.get("calibration_seed"),
            "dtype": transformers_dict.get("dtype", "float16"),
            "kernel_capability": transformers_dict.get("kernel_capability"),
        }
