"""Version-neutral helpers used by the pinned vLLM three-level patch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

from mixllm.runtime_capability import RuntimeCapability


PINNED_VLLM_VERSION = "0.9.0"
PINNED_VLLM_COMMIT = "5fbbfe9a4c13094ad72ed3d6b4ef208a7ddc0fd7"


@dataclass(frozen=True)
class PrecisionPercentages:
    bit4: int
    bit8: int
    bit16: int

    def __post_init__(self) -> None:
        values = (self.bit4, self.bit8, self.bit16)
        if any(value < 0 for value in values) or sum(values) != 100:
            raise ValueError("4/8/16 percentages must sum to 100")


@dataclass(frozen=True)
class VLLMThreeLevelConfig:
    group_size: int
    percentages: PrecisionPercentages
    backend: str = "auto"
    allocation_version: int = 1

    @classmethod
    def from_quantization_config(cls, config: Dict) -> "VLLMThreeLevelConfig":
        if config.get("quant_method") not in {"mixllm_three_level", "mixllm"}:
            raise ValueError("checkpoint is not a MixLLM three-level checkpoint")
        raw = config.get("precision_percentages")
        if raw is None:
            raise ValueError("precision_percentages is required for three-level inference")
        budget = PrecisionPercentages(
            int(raw.get("4", raw.get(4, 0))),
            int(raw.get("8", raw.get(8, 0))),
            int(raw.get("16", raw.get(16, 0))),
        )
        group_size = int(config.get("group_size", 128))
        if group_size != 128:
            raise ValueError("current MixLLM CUDA ABI requires group_size=128")
        return cls(
            group_size=group_size,
            percentages=budget,
            backend=str(config.get("backend", "auto")),
            allocation_version=int(config.get("allocation_version", 1)),
        )

    def select_backend(self, capability: Tuple[int, int],
                       sm75_available: bool = False) -> str:
        return RuntimeCapability(*capability, self.backend, sm75_available).select_backend()


def validate_partition_indices(indices: Dict[int, Iterable[int]], output_size: int) -> None:
    values = [int(index) for bit in (4, 8, 16) for index in indices.get(bit, ())]
    if sorted(values) != list(range(output_size)):
        raise ValueError("vLLM partition indices must cover every output channel exactly once")


def remap_partition_indices_for_tp(
    indices: Dict[int, Iterable[int]],
    global_output_size: int,
    shard_start: int,
    shard_size: int,
) -> Dict[int, Tuple[int, ...]]:
    """Map checkpoint-global output indices to one tensor-parallel shard.

    vLLM loads output-parallel weights as contiguous ranges. The packed tensors
    retain their precision order, while the operator receives indices local to
    the current shard so its output remains in original local-channel order.
    """
    validate_partition_indices(indices, global_output_size)
    if shard_start < 0 or shard_size <= 0 or shard_start + shard_size > global_output_size:
        raise ValueError("invalid tensor-parallel output shard")
    shard_end = shard_start + shard_size
    local = {
        bit: tuple(int(index) - shard_start for index in indices.get(bit, ())
                   if shard_start <= int(index) < shard_end)
        for bit in (4, 8, 16)
    }
    validate_partition_indices(local, shard_size)
    return local


def restore_global_partition_indices(
    local_indices: Dict[int, Iterable[int]], shard_start: int, shard_size: int,
) -> Dict[int, Tuple[int, ...]]:
    """Restore checkpoint-global indices after a local shard round trip."""
    validate_partition_indices(local_indices, shard_size)
    if shard_start < 0:
        raise ValueError("shard_start must be non-negative")
    return {
        bit: tuple(int(index) + shard_start for index in local_indices.get(bit, ()))
        for bit in (4, 8, 16)
    }