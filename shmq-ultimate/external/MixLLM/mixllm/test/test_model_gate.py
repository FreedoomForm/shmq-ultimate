import importlib
import unittest
from types import SimpleNamespace

import torch
from torch import nn


class TinyBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, hidden):
        return torch.tanh(self.proj(hidden))


class TinyCausalLM(nn.Module):

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyBlock()])
        self.head = nn.Linear(4, 8, bias=False)

    def forward(self, input_ids, labels=None, use_cache=False):
        hidden = self.embed(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        logits = self.head(hidden)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return SimpleNamespace(logits=logits, loss=loss)


class ModelGateTest(unittest.TestCase):

    def test_model_gate_module_imports(self):
        module = importlib.import_module("mixllm.model_gate")
        self.assertTrue(callable(module.run_model_gate))

    def test_auto_model_gate_quantizes_and_reports_budget(self):
        from mixllm.model_gate import run_model_gate
        from mixllm.nn.modules.three_level_linear import ThreeLevelLinear

        torch.manual_seed(13)
        model = TinyCausalLM()
        ids = torch.tensor([[0, 1, 2, 3]])

        result = run_model_gate(
            "tiny", None, model, ids, ids, target_average_bits=8,
            group_size=4, calibration_rows=4)

        self.assertEqual(result["packed_layers"], 1)
        self.assertIsInstance(model.model.layers[0].proj, ThreeLevelLinear)
        self.assertEqual(result["allocation_summary"]["target_average_bits"], 8)
        self.assertLessEqual(result["average_weight_bits"], 8)
        self.assertTrue(result["deterministic"])
        self.assertTrue(result["finite"])

    def test_fixed_budget_model_gate_remains_supported(self):
        from mixllm.model_gate import run_model_gate
        from mixllm.quantization.three_level import ThreeLevelBudget

        torch.manual_seed(17)
        model = TinyCausalLM()
        ids = torch.tensor([[0, 1, 2, 3]])
        result = run_model_gate(
            "tiny", None, model, ids, ids,
            budget=ThreeLevelBudget(50, 25, 25), group_size=4,
            calibration_rows=4)

        self.assertEqual(result["channel_counts"], {"4": 2, "8": 1, "16": 1})
        self.assertEqual(
            result["allocation_summary"]["allocator"],
            "fixed_precision_percentages")

    def test_model_gate_uses_eval_and_restores_training_mode(self):
        from mixllm.model_gate import run_model_gate

        model = TinyCausalLM().train()
        ids = torch.tensor([[0, 1, 2, 3]])
        observed_modes = []
        handle = model.register_forward_pre_hook(
            lambda current, args, kwargs: observed_modes.append(current.training),
            with_kwargs=True,
        )
        try:
            run_model_gate(
                "tiny", None, model, ids, ids, target_average_bits=8,
                group_size=4, calibration_rows=4)
        finally:
            handle.remove()

        self.assertTrue(observed_modes)
        self.assertFalse(any(observed_modes))
        self.assertTrue(model.training)

    def test_model_gate_requires_exactly_one_budget_mode(self):
        from mixllm.model_gate import run_model_gate
        from mixllm.quantization.three_level import ThreeLevelBudget

        ids = torch.tensor([[0, 1]])
        with self.assertRaises(ValueError):
            run_model_gate("tiny", None, TinyCausalLM(), ids, ids, group_size=4)
        with self.assertRaises(ValueError):
            run_model_gate(
                "tiny", None, TinyCausalLM(), ids, ids,
                budget=ThreeLevelBudget(100, 0, 0), target_average_bits=8,
                group_size=4)


if __name__ == "__main__":
    unittest.main()
