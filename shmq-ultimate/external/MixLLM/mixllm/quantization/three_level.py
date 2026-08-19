"""Three-level allocation and reference quantization for MixLLM.

The allocator uses the value of an upgrade (L4-L8 or L8-L16), rather than
the absolute loss of the lowest precision.  This keeps the existing global
output-feature design while making the extension to FP16 unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch


LEVELS = (4, 8, 16)


@dataclass(frozen=True)
class ThreeLevelBudget:
    """Percentages of output channels assigned to each precision."""

    bit4: int
    bit8: int
    bit16: int

    def __post_init__(self) -> None:
        values = (self.bit4, self.bit8, self.bit16)
        if any(value < 0 for value in values) or sum(values) != 100:
            raise ValueError("bit4 + bit8 + bit16 must equal 100")

    def as_dict(self) -> Dict[int, int]:
        return {4: self.bit4, 8: self.bit8, 16: self.bit16}


@dataclass(frozen=True)
class ThreeLevelAllocation:
    """A complete, deterministic output-channel allocation."""

    indices: Dict[int, tuple[int, ...]]
    scores: Dict[int, tuple[float, ...]]
    budget: ThreeLevelBudget
    version: int = 1
    metadata: Dict[str, object] = field(default_factory=dict)

    def verify(self, out_features: int) -> None:
        assigned = [index for bit in LEVELS for index in self.indices[bit]]
        if sorted(assigned) != list(range(out_features)):
            raise ValueError("allocation must cover every output channel exactly once")
        if any(index < 0 or index >= out_features for index in assigned):
            raise ValueError("allocation contains an out-of-range channel")

    def to_json(self, path: str | Path) -> None:
        payload = {
            "format": "mixllm-three-level",
            "version": self.version,
            "metadata": self.metadata,
            "budget": {str(k): v for k, v in self.budget.as_dict().items()},
            "indices": {str(k): list(v) for k, v in self.indices.items()},
            "scores": {str(k): list(v) for k, v in self.scores.items()},
        }
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "ThreeLevelAllocation":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "mixllm-three-level":
            raise ValueError("unsupported allocation format")
        budget = ThreeLevelBudget(**{
            f"bit{bit}": int(value) for bit, value in payload["budget"].items()
        })
        allocation = cls(
            indices={int(bit): tuple(int(i) for i in values)
                     for bit, values in payload["indices"].items()},
            scores={int(bit): tuple(float(i) for i in values)
                    for bit, values in payload["scores"].items()},
            budget=budget,
            version=int(payload["version"]),
            metadata=dict(payload.get("metadata", {})),
        )
        if set(allocation.indices) != set(LEVELS):
            raise ValueError("allocation must contain 4, 8 and 16 bit partitions")
        return allocation


def _rounded_counts(out_features: int, budget: ThreeLevelBudget,
                    alignment: int) -> Dict[int, int]:
    """Round counts to alignment while preserving the requested total."""
    raw = {bit: out_features * pct / 100 for bit, pct in budget.as_dict().items()}
    counts = {bit: int(raw[bit] // alignment) * alignment for bit in LEVELS}
    remainder = out_features - sum(counts.values())
    order = sorted(LEVELS, key=lambda bit: (raw[bit] - counts[bit], -bit), reverse=True)
    for bit in order:
        if remainder < alignment:
            break
        counts[bit] += alignment
        remainder -= alignment
    # A tail is legal for the metadata/reference path; the backend may reject it.
    counts[order[0]] += remainder
    return counts


def allocate_channels(losses: Mapping[int, torch.Tensor | Sequence[float]],
                      budget: ThreeLevelBudget, alignment: int = 1
                      ) -> ThreeLevelAllocation:
    """Allocate channels using marginal upgrade losses.

    ``losses[bit][channel]`` is the estimated output error when that channel
    is quantized at ``bit``.  Lower is better.  The allocation is global across
    all channels and therefore matches MixLLM's output-feature abstraction.
    """
    if alignment < 1:
        raise ValueError("alignment must be positive")
    values = {bit: torch.as_tensor(losses[bit], dtype=torch.float64).flatten()
              for bit in LEVELS}
    lengths = {tensor.numel() for tensor in values.values()}
    if len(lengths) != 1 or next(iter(lengths), 0) == 0:
        raise ValueError("all loss vectors must have the same non-zero length")
    if any(not torch.isfinite(tensor).all() for tensor in values.values()):
        raise ValueError("loss vectors must be finite")
    out_features = next(iter(lengths))
    counts = _rounded_counts(out_features, budget, alignment)

    # Upgrades are ranked by the error removed by moving up one level.
    benefit8 = values[4] - values[8]
    benefit16 = values[8] - values[16]
    # Choose the final classes hierarchically.  An FP16 channel is upgraded
    # from INT8, so it must be selected before INT8 candidates are considered.
    selected: Dict[int, list[int]] = {4: [], 8: [], 16: []}
    selected[16] = [index for _, index in sorted(
        ((float(benefit16[i]), i) for i in range(out_features)),
        key=lambda item: (-item[0], item[1]))[:counts[16]]]
    used = set(selected[16])
    selected[8] = [index for _, index in sorted(
        ((float(benefit8[i]), i) for i in range(out_features) if i not in used),
        key=lambda item: (-item[0], item[1]))[:counts[8]]]
    used.update(selected[8])
    selected[4] = [index for index in range(out_features) if index not in used]
    if len(selected[4]) != counts[4]:
        raise ValueError("budget/alignment could not be represented exactly")
    result = ThreeLevelAllocation(
        indices={bit: tuple(sorted(selected[bit])) for bit in LEVELS},
        scores={4: tuple(float(x) for x in values[4]),
                8: tuple(float(x) for x in benefit8),
                16: tuple(float(x) for x in benefit16)},
        budget=budget,
    )
    result.verify(out_features)
    return result


def allocate_model_channels(
    layer_losses: Mapping[str, Mapping[int, torch.Tensor | Sequence[float]]],
    budget: ThreeLevelBudget,
    alignment: int = 1,
    layer_minimums: Mapping[str, int] | None = None,
    metadata: Dict[str, object] | None = None,
) -> Dict[str, ThreeLevelAllocation]:
    """Allocate one global bit budget across all output channels in all layers.

    The returned per-layer allocations preserve original channel indices.  A
    layer minimum is expressed as the minimum number of non-INT4 channels and
    is satisfied before global marginal-benefit ranking.  This is the clean
    replacement for the upstream searcher's per-iteration percentage split.
    """
    if not layer_losses:
        raise ValueError("layer_losses must contain at least one layer")
    minimums = dict(layer_minimums or {})
    flattened = []
    per_layer = {}
    for name, losses in layer_losses.items():
        values = {bit: torch.as_tensor(losses[bit], dtype=torch.float64).flatten()
                  for bit in LEVELS}
        lengths = {value.numel() for value in values.values()}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise ValueError(f"invalid loss vectors for layer {name}")
        n = next(iter(lengths))
        if alignment > 1 and n < alignment:
            raise ValueError(f"layer {name} is smaller than alignment")
        if minimums.get(name, 0) < 0 or minimums.get(name, 0) > n:
            raise ValueError(f"invalid layer minimum for {name}")
        per_layer[name] = values
        flattened.extend((float(values[4][i] - values[8][i]),
                          float(values[8][i] - values[16][i]), name, i)
                         for i in range(n))

    total = len(flattened)
    counts = _rounded_counts(total, budget, alignment)
    required = sum(minimums.values())
    if required > counts[8] + counts[16]:
        raise ValueError("layer minimums exceed non-INT4 global budget")
    chosen: Dict[str, Dict[int, set[int]]] = {
        name: {4: set(), 8: set(), 16: set()} for name in per_layer
    }
    ranked16 = sorted(((benefit16, name, index)
                       for _, benefit16, name, index in flattened), reverse=True)
    for _, name, index in ranked16[:counts[16]]:
        chosen[name][16].add(index)
    used = {(name, index) for name in chosen for index in chosen[name][16]}
    ranked8 = sorted(((benefit8, name, index)
                      for benefit8, _, name, index in flattened
                      if (name, index) not in used), reverse=True)
    for _, name, index in ranked8[:counts[8]]:
        chosen[name][8].add(index)
    used.update((name, index) for name in chosen for index in chosen[name][8])

    # Meet per-layer minimums without changing global precision counts.  Each
    # swap replaces the weakest selected channel from a donor layer that stays
    # above its own minimum with the strongest candidate from the deficient one.
    def selected_count(layer: str) -> int:
        return len(chosen[layer][8]) + len(chosen[layer][16])

    for name in sorted(per_layer):
        while selected_count(name) < minimums.get(name, 0):
            candidates = []
            for index in range(per_layer[name][4].numel()):
                if (name, index) in used:
                    continue
                candidates.extend((
                    (float(per_layer[name][4][index] - per_layer[name][8][index]),
                     8, index),
                    (float(per_layer[name][8][index] - per_layer[name][16][index]),
                     16, index),
                ))
            swapped = False
            for _, bit, incoming in sorted(candidates, reverse=True):
                donors = []
                for donor in per_layer:
                    if selected_count(donor) <= minimums.get(donor, 0):
                        continue
                    for outgoing in chosen[donor][bit]:
                        values = per_layer[donor]
                        benefit = (values[4][outgoing] - values[8][outgoing]
                                   if bit == 8 else
                                   values[8][outgoing] - values[16][outgoing])
                        donors.append((float(benefit), donor, outgoing))
                if not donors:
                    continue
                _, donor, outgoing = min(donors)
                chosen[donor][bit].remove(outgoing)
                chosen[name][bit].add(incoming)
                used.remove((donor, outgoing))
                used.add((name, incoming))
                swapped = True
                break
            if not swapped:
                raise ValueError("layer minimums cannot be satisfied with exact budgets")
    result = {}
    for name, values in per_layer.items():
        indices = {bit: tuple(sorted(chosen[name][bit])) for bit in LEVELS}
        indices[4] = tuple(i for i in range(values[4].numel())
                           if all(i not in chosen[name][bit] for bit in (8, 16)))
        allocation = ThreeLevelAllocation(
            indices=indices,
            scores={4: tuple(float(x) for x in values[4]),
                    8: tuple(float(x) for x in values[4] - values[8]),
                    16: tuple(float(x) for x in values[8] - values[16])},
            budget=budget,
            metadata={**(metadata or {}), "layer_name": name,
                      "global_channel_count": total},
        )
        allocation.verify(values[4].numel())
        result[name] = allocation
    return result


def allocate_model_channels_auto(
    layer_losses: Mapping[str, Mapping[int, torch.Tensor | Sequence[float]]],
    target_average_bits: float,
    alignment: int = 1,
    metadata: Dict[str, object] | None = None,
) -> tuple[Dict[str, ThreeLevelAllocation], Dict[str, object]]:
    """Allocate a global average-bit budget using marginal upgrade value.

    Alignment applies to each layer and precision transition independently: an
    upgrade moves ``alignment`` channels from one precision to the next within
    a single layer. Layer tails smaller than a block stay at their precision.
    """
    if not layer_losses:
        raise ValueError("layer_losses must contain at least one layer")
    if not math.isfinite(target_average_bits) or not 4 <= target_average_bits <= 16:
        raise ValueError("target_average_bits must be between 4 and 16")
    if alignment < 1:
        raise ValueError("alignment must be positive")

    per_layer = {}
    for name in sorted(layer_losses):
        losses = layer_losses[name]
        try:
            values = {bit: torch.as_tensor(losses[bit], dtype=torch.float64).flatten()
                      for bit in LEVELS}
        except KeyError as error:
            raise ValueError(f"missing {error.args[0]}-bit losses for layer {name}") from error
        lengths = {value.numel() for value in values.values()}
        if len(lengths) != 1 or next(iter(lengths), 0) == 0:
            raise ValueError(f"invalid loss vectors for layer {name}")
        if any(not torch.isfinite(value).all() for value in values.values()):
            raise ValueError(f"loss vectors must be finite for layer {name}")
        per_layer[name] = values

    total = sum(values[4].numel() for values in per_layer.values())
    requested_bit_budget = (math.floor(target_average_bits * total + 1e-9) -
                            4 * total)
    remaining_bits = requested_bit_budget
    selected = {name: {4: set(range(values[4].numel())), 8: set(), 16: set()}
                for name, values in per_layer.items()}

    while True:
        candidates = []
        for source, target in ((4, 8), (8, 16)):
            cost = (target - source) * alignment
            if cost > remaining_bits:
                continue
            for name in per_layer:
                ranked = [
                    (float(per_layer[name][source][index] -
                           per_layer[name][target][index]), index)
                    for index in selected[name][source]
                ]
                ranked.sort(key=lambda item: (-item[0], item[1]))
                if len(ranked) < alignment:
                    continue
                batch = ranked[:alignment]
                benefit = sum(item[0] for item in batch)
                candidates.append((-(benefit / cost), source, name,
                                   batch[0][1], target, batch))
        if not candidates:
            break
        _, source, name, _, target, batch = min(candidates)
        for _, index in batch:
            selected[name][source].remove(index)
            selected[name][target].add(index)
        remaining_bits -= (target - source) * alignment

    counts = {bit: sum(len(selected[name][bit]) for name in selected)
              for bit in LEVELS}
    achieved_bits = sum(bit * counts[bit] for bit in LEVELS) / total
    raw_percentages = {bit: 100 * counts[bit] / total for bit in LEVELS}
    achieved_bit_budget = requested_bit_budget - remaining_bits
    percentages = {bit: int(raw_percentages[bit]) for bit in LEVELS}
    remainder_order = sorted(
        LEVELS, key=lambda bit: (raw_percentages[bit] - percentages[bit], -bit),
        reverse=True)
    for bit in remainder_order[:100 - sum(percentages.values())]:
        percentages[bit] += 1
    budget = ThreeLevelBudget(percentages[4], percentages[8], percentages[16])

    summary = {
        "allocator": "marginal_benefit_bit_budget",
        "target_average_bits": float(target_average_bits),
        "achieved_average_bits": achieved_bits,
        "total_channels": total,
        "channel_counts": {str(bit): counts[bit] for bit in LEVELS},
        "requested_bit_budget": requested_bit_budget,
        "achieved_bit_budget": achieved_bit_budget,
        "achieved_channel_counts": {str(bit): counts[bit] for bit in LEVELS},
        "alignment": alignment,
        "unused_bit_budget": remaining_bits,
    }
    result = {}
    for name, values in per_layer.items():
        allocation = ThreeLevelAllocation(
            indices={bit: tuple(sorted(selected[name][bit])) for bit in LEVELS},
            scores={4: tuple(float(x) for x in values[4]),
                    8: tuple(float(x) for x in values[4] - values[8]),
                    16: tuple(float(x) for x in values[8] - values[16])},
            budget=budget,
            metadata={**(metadata or {}), "layer_name": name,
                      "global_channel_count": total,
                      "target_average_bits": float(target_average_bits),
                      "requested_bit_budget": requested_bit_budget,
                      "achieved_average_bits": achieved_bits,
                      "achieved_bit_budget": achieved_bit_budget,
                      "achieved_channel_counts": {
                          str(bit): counts[bit] for bit in LEVELS}},
        )
        allocation.verify(values[4].numel())
        result[name] = allocation
    return result, summary

@torch.no_grad()
def estimate_channel_losses(activation: torch.Tensor, weight: torch.Tensor,
                            group_size: int = 128
                            ) -> tuple[Dict[int, torch.Tensor], torch.Tensor]:
    """Estimate activation-aware output MSE for all three precision choices.

    The same activation subset and group rules are used for every precision.
    The returned AWQ statistic is kept separate and can be used as a tie-breaker
    in an ablation without silently changing the marginal-loss objective.
    """
    if activation.shape[-1] != weight.shape[1] or weight.dim() != 2:
        raise ValueError("activation and weight shapes are incompatible")
    if weight.shape[1] % group_size:
        raise ValueError("in_features must be divisible by group_size")
    x = activation.reshape(-1, activation.shape[-1]).float()
    reference = torch.nn.functional.linear(x, weight.float())
    losses = {}
    all_fp16 = ThreeLevelAllocation(
        indices={4: (), 8: (), 16: tuple(range(weight.shape[0]))},
        scores={bit: (0.0,) * weight.shape[0] for bit in LEVELS},
        budget=ThreeLevelBudget(0, 0, 100),
    )
    for bit in LEVELS:
        indices = {level: () for level in LEVELS}
        indices[bit] = tuple(range(weight.shape[0]))
        allocation = ThreeLevelAllocation(
            indices=indices, scores=all_fp16.scores,
            budget=ThreeLevelBudget(100 if bit == 4 else 0,
                                    100 if bit == 8 else 0,
                                    100 if bit == 16 else 0),
        )
        output = fake_linear(x, weight.float(), allocation, group_size)
        losses[bit] = (output - reference).pow(2).mean(dim=0).cpu()
    awq_stat = x.abs().mean(dim=0).cpu()
    return losses, awq_stat


def fake_linear(x: torch.Tensor, weight: torch.Tensor,
                allocation: ThreeLevelAllocation,
                group_size: int = 128) -> torch.Tensor:
    """Reference output for a three-level weight-only linear layer."""
    if weight.dim() != 2 or x.shape[-1] != weight.shape[1]:
        raise ValueError("incompatible x/weight shapes")
    output = torch.empty((*x.shape[:-1], weight.shape[0]), dtype=x.dtype, device=x.device)
    for bit in LEVELS:
        indices = allocation.indices[bit]
        if not indices:
            continue
        chunk = weight[list(indices)]
        if bit == 16:
            quantized = chunk
        else:
            if chunk.shape[1] % group_size:
                raise ValueError("in_features must be divisible by group_size")
            groups = chunk.reshape(chunk.shape[0], -1, group_size)
            if bit == 8:
                maximum = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
                quantized = (groups / (maximum / 127)).round().clamp(-128, 127) * (maximum / 127)
            else:
                minimum = groups.amin(dim=-1, keepdim=True)
                maximum = groups.amax(dim=-1, keepdim=True)
                scale = (maximum - minimum).clamp_min(1e-5) / 15
                zero = (-minimum / scale).round().clamp(0, 15)
                quantized = ((groups / scale).round() + zero).clamp(0, 15)
                quantized = (quantized - zero) * scale
            quantized = quantized.reshape_as(chunk)
        output[..., list(indices)] = torch.nn.functional.linear(x, quantized)
    return output