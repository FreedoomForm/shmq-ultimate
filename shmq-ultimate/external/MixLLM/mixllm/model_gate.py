"""Small-to-large model validation for the three-level MixLLM format."""

from __future__ import annotations

from collections import OrderedDict
import math
import time
from typing import Dict, Iterable, Mapping

import torch
from torch import nn

from mixllm.nn.modules.three_level_linear import ThreeLevelLinear
from mixllm.quantization.three_level import (
    ThreeLevelBudget,
    allocate_model_channels,
    allocate_model_channels_auto,
    estimate_channel_losses,
)


DEFAULT_TEXTS = (
    "Mixed precision protects channels whose quantization error matters most.",
    "A reproducible benchmark separates numerical quality from kernel speed.",
    "The quick model gate must pass before running the seven billion parameter model.",
    "CUDA events measure device work after warmup and explicit synchronization.",
)


def _transformer_linears(model: nn.Module) -> "OrderedDict[str, nn.Linear]":
    result = OrderedDict()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ".layers." in name:
            result[name] = module
    if not result:
        raise ValueError("no transformer Linear modules were found")
    return result


def _parent_and_child(model: nn.Module, name: str):
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


@torch.no_grad()
def collect_layer_losses(model: nn.Module, input_ids: torch.Tensor,
                         group_size: int = 128,
                         max_activation_rows: int = 64) -> Dict[str, Mapping[int, torch.Tensor]]:
    """Collect per-channel losses during one deterministic calibration pass."""
    linears = _transformer_linears(model)
    losses: Dict[str, Mapping[int, torch.Tensor]] = {}
    handles = []

    def make_hook(name: str):
        def hook(module, args):
            if name in losses:
                return
            activation = args[0].detach().reshape(-1, module.in_features)
            activation = activation[:max_activation_rows]
            layer_losses, _ = estimate_channel_losses(
                activation, module.weight.detach(), group_size=group_size,
            )
            losses[name] = layer_losses
        return hook

    for name, module in linears.items():
        if module.in_features % group_size:
            raise ValueError(f"{name}: in_features must be divisible by group_size")
        handles.append(module.register_forward_pre_hook(make_hook(name)))
    try:
        model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    missing = set(linears) - set(losses)
    if missing:
        raise RuntimeError(f"calibration did not execute layers: {sorted(missing)[:3]}")
    return losses


@torch.no_grad()
def pack_transformer_linears(model: nn.Module, allocations, group_size: int = 128) -> int:
    """Replace transformer linears with actual packed three-level modules."""
    linears = _transformer_linears(model)
    if set(linears) != set(allocations):
        raise ValueError("allocation names do not match model Linear names")
    for name, module in linears.items():
        packed = ThreeLevelLinear.from_weight(
            module.weight.detach(), allocations[name], group_size=group_size,
            bias=module.bias,
        )
        parent, child = _parent_and_child(model, name)
        setattr(parent, child, packed)
    return len(linears)


def _timed_forward(model, input_ids, warmup: int, iterations: int):
    for _ in range(warmup):
        model(input_ids=input_ids, use_cache=False)
    torch.cuda.synchronize(input_ids.device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        model(input_ids=input_ids, use_cache=False)
    end.record()
    torch.cuda.synchronize(input_ids.device)
    return float(start.elapsed_time(end)) / iterations


@torch.no_grad()
def run_model_gate(model_id: str, tokenizer, model, calibration_ids: torch.Tensor,
                   evaluation_ids: torch.Tensor, budget: ThreeLevelBudget | None = None,
                   target_average_bits: float | None = None,
                   group_size: int = 128, calibration_rows: int = 64,
                   timing_warmup: int = 2, timing_iterations: int = 5
                   ) -> Dict[str, object]:
    """Quantize a loaded causal LM and return measured quality evidence."""
    if (budget is None) == (target_average_bits is None):
        raise ValueError("provide exactly one of budget or target_average_bits")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    calibration_ids = calibration_ids.to(device)
    evaluation_ids = evaluation_ids.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        reference = model(input_ids=evaluation_ids, labels=evaluation_ids, use_cache=False)
        reference_loss = float(reference.loss.item())
        reference_logits = reference.logits[:, -1].float().cpu()
        reference_forward_ms = (
            _timed_forward(model, evaluation_ids, timing_warmup, timing_iterations)
            if device.type == "cuda" else None
        )

        started = time.perf_counter()
        losses = collect_layer_losses(model, calibration_ids, group_size, calibration_rows)

        allocation_metadata = {"model_id": model_id, "group_size": group_size}
        if target_average_bits is None:
            allocations = allocate_model_channels(
                losses, budget, alignment=1, metadata=allocation_metadata,
            )
            allocation_summary = {"allocator": "fixed_precision_percentages"}
        else:
            allocations, allocation_summary = allocate_model_channels_auto(
                losses, target_average_bits, metadata=allocation_metadata,
            )
        packed_layers = pack_transformer_linears(model, allocations, group_size)
        quantization_seconds = time.perf_counter() - started

        actual = model(input_ids=evaluation_ids, labels=evaluation_ids, use_cache=False)
        actual_logits = actual.logits[:, -1].float().cpu()
        quantized_loss = float(actual.loss.item())
        max_logit_error = float((actual_logits - reference_logits).abs().max().item())
        mean_logit_error = float((actual_logits - reference_logits).abs().mean().item())
        deterministic = torch.equal(
            actual.logits,
            model(input_ids=evaluation_ids, use_cache=False).logits,
        )
        quantized_forward_ms = (
            _timed_forward(model, evaluation_ids, timing_warmup, timing_iterations)
            if device.type == "cuda" else None
        )
        counts = {bit: sum(len(item.indices[bit]) for item in allocations.values())
                  for bit in (4, 8, 16)}
        total = sum(counts.values())
        return {
            "model_id": model_id,
            "backend": "packed_reference",
            "packed_layers": packed_layers,
            "channel_counts": {str(bit): count for bit, count in counts.items()},
            "average_weight_bits": sum(bit * counts[bit] for bit in counts) / total,
            "allocation_summary": allocation_summary,
            "reference_loss": reference_loss,
            "reference_perplexity": math.exp(reference_loss),
            "quantized_loss": quantized_loss,
            "quantized_perplexity": math.exp(quantized_loss),
            "loss_delta": quantized_loss - reference_loss,
            "max_last_token_logit_error": max_logit_error,
            "mean_last_token_logit_error": mean_logit_error,
            "deterministic": bool(deterministic),
            "finite": bool(torch.isfinite(actual.logits).all().item()),
            "quantization_seconds": quantization_seconds,
            "reference_forward_ms": reference_forward_ms,
            "quantized_forward_ms": quantized_forward_ms,
            "full_model_speedup_vs_reference": (
                reference_forward_ms / max(quantized_forward_ms, 1e-9)
                if reference_forward_ms is not None and quantized_forward_ms is not None
                else None
            ),
            "evaluation_tokens": int(evaluation_ids.numel()),
            "quantized_tokens_per_second": (
                (evaluation_ids.numel() / (quantized_forward_ms / 1000.0))
                if quantized_forward_ms is not None else None
            ),
            "allocation_count": len(allocations),
            "peak_cuda_vram_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda" else None
            ),
        }
    finally:
        if was_training:
            model.train()


def tokenize_texts(tokenizer, texts: Iterable[str], max_length: int = 128):
    text = "\n".join(texts)
    return tokenizer(text, return_tensors="pt", truncation=True,
                     max_length=max_length).input_ids
