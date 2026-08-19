import tempfile
import unittest
from pathlib import Path

import torch

from mixllm.quantization.three_level import (
    ThreeLevelAllocation,
    ThreeLevelBudget,
    allocate_channels,
    allocate_model_channels,
    allocate_model_channels_auto,
    estimate_channel_losses,
    fake_linear,
)
from mixllm.nn.modules.three_level_linear import ThreeLevelLinear


class ThreeLevelAllocationTest(unittest.TestCase):

    def test_uses_marginal_benefit_for_each_upgrade(self):
        losses = {
            4: [100.0, 10.0, 9.0, 8.0],
            8: [99.0, 0.0, 8.0, 7.0],
            16: [0.0, 0.0, 8.0, 7.0],
        }
        result = allocate_channels(losses, ThreeLevelBudget(50, 25, 25))

        self.assertEqual(result.indices[16], (0,))
        self.assertEqual(result.indices[8], (1,))
        self.assertEqual(result.indices[4], (2, 3))

    def test_allows_empty_partitions(self):
        losses = {4: [3.0, 2.0], 8: [1.0, 1.0], 16: [0.0, 0.0]}
        result = allocate_channels(losses, ThreeLevelBudget(0, 0, 100))

        self.assertEqual(result.indices[4], ())
        self.assertEqual(result.indices[8], ())
        self.assertEqual(result.indices[16], (0, 1))

    def test_alignment_preserves_complete_partition(self):
        losses = {bit: torch.arange(10, dtype=torch.float32) / bit for bit in (4, 8, 16)}
        result = allocate_channels(losses, ThreeLevelBudget(60, 20, 20), alignment=2)

        result.verify(10)
        self.assertEqual({bit: len(result.indices[bit]) for bit in (4, 8, 16)},
                         {4: 6, 8: 2, 16: 2})

    def test_json_round_trip(self):
        losses = {4: [3.0, 2.0], 8: [1.0, 1.0], 16: [0.0, 0.0]}
        result = allocate_channels(losses, ThreeLevelBudget(50, 0, 50))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "allocation.json"
            result.to_json(path)
            loaded = ThreeLevelAllocation.from_json(path)

        self.assertEqual(loaded, result)

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            ThreeLevelBudget(90, 20, 0)
        with self.assertRaises(ValueError):
            allocate_channels({4: [], 8: [], 16: []}, ThreeLevelBudget(100, 0, 0))
        with self.assertRaises(ValueError):
            allocate_channels({4: [1], 8: [1, 2], 16: [0]},
                              ThreeLevelBudget(100, 0, 0))

    def test_global_allocator_preserves_counts_and_layer_minimums(self):
        losses = {
            "a": {4: [10, 9, 8, 7], 8: [1, 1, 1, 1], 16: [0, 0, 0, 0]},
            "b": {4: [2, 2, 2, 2], 8: [1, 1, 1, 1], 16: [0, 0, 0, 0]},
        }
        result = allocate_model_channels(
            losses, ThreeLevelBudget(50, 25, 25), layer_minimums={"b": 2})

        counts = {bit: sum(len(value.indices[bit]) for value in result.values())
                  for bit in (4, 8, 16)}
        self.assertEqual(counts, {4: 4, 8: 2, 16: 2})
        self.assertGreaterEqual(len(result["b"].indices[8]) +
                                len(result["b"].indices[16]), 2)
        for name, allocation in result.items():
            allocation.verify(4)
            self.assertEqual(allocation.metadata["layer_name"], name)

    def test_auto_allocator_is_deterministic_across_layer_order(self):
        losses = {
            "a": {4: [10, 10], 8: [9, 9], 16: [9, 9]},
            "b": {4: [10, 10], 8: [9, 9], 16: [9, 9]},
        }

        first, first_summary = allocate_model_channels_auto(losses, 6)
        second, second_summary = allocate_model_channels_auto(
            dict(reversed(list(losses.items()))), 6)

        self.assertEqual(first, second)
        self.assertEqual(first_summary, second_summary)
        self.assertEqual(first["a"].indices[8], (0, 1))
        self.assertEqual(first["b"].indices[4], (0, 1))

    def test_auto_allocator_honors_bit_budget_and_alignment(self):
        losses = {
            "a": {4: [12, 11, 10, 9], 8: [6, 6, 6, 6], 16: [0, 0, 0, 0]},
            "b": {4: [8, 7, 6, 5], 8: [4, 4, 4, 4], 16: [0, 0, 0, 0]},
        }

        result, summary = allocate_model_channels_auto(
            losses, 10, alignment=2, metadata={"model_id": "tiny"})

        counts = {bit: sum(len(item.indices[bit]) for item in result.values())
                  for bit in (4, 8, 16)}
        self.assertEqual(counts, {int(bit): count
                                 for bit, count in summary["channel_counts"].items()})
        for allocation in result.values():
            self.assertEqual(len(allocation.indices[8]) % 2, 0)
            self.assertEqual(len(allocation.indices[16]) % 2, 0)
        self.assertLessEqual(summary["achieved_average_bits"], 10)
        self.assertEqual(summary["unused_bit_budget"], 0)
        self.assertEqual(result["a"].metadata["model_id"], "tiny")
        for allocation in result.values():
            allocation.verify(4)

    def test_auto_allocator_uses_per_layer_aligned_upgrade_blocks(self):
        losses = {
            "a": {4: [100, 0], 8: [0, 0], 16: [0, 0]},
            "b": {4: [99, 98], 8: [0, 0], 16: [0, 0]},
        }

        result, _ = allocate_model_channels_auto(losses, 6, alignment=2)

        self.assertEqual(result["a"].indices[4], (0, 1))
        self.assertEqual(result["b"].indices[8], (0, 1))

    def test_auto_allocator_allows_small_unaligned_totals(self):
        losses = {
            "small": {4: [100], 8: [0], 16: [0]},
            "full": {4: [2, 1], 8: [0, 0], 16: [0, 0]},
        }

        result, summary = allocate_model_channels_auto(losses, 8, alignment=2)

        self.assertEqual(result["small"].indices[4], (0,))
        self.assertEqual(summary["requested_bit_budget"], 12)
        self.assertEqual(summary["achieved_bit_budget"], 8)
        self.assertEqual(summary["unused_bit_budget"], 4)
        self.assertEqual(summary["achieved_channel_counts"],
                         {"4": 1, "8": 2, "16": 0})
        self.assertEqual(result["small"].metadata["requested_bit_budget"], 12)
        self.assertEqual(result["small"].metadata["achieved_channel_counts"],
                         {"4": 1, "8": 2, "16": 0})

    def test_auto_allocator_validates_target_and_alignment(self):
        losses = {"a": {4: [2, 2, 2], 8: [1, 1, 1], 16: [0, 0, 0]}}
        with self.assertRaises(ValueError):
            allocate_model_channels_auto(losses, 3.99)
        with self.assertRaises(ValueError):
            allocate_model_channels_auto(losses, 16.01)
        with self.assertRaises(ValueError):
            allocate_model_channels_auto(losses, 8, alignment=0)


class ThreeLevelReferenceTest(unittest.TestCase):

    def test_activation_aware_losses_share_reference_and_fp16_is_zero(self):
        torch.manual_seed(5)
        activation = torch.randn(2, 3, 128)
        weight = torch.randn(6, 128)
        losses, awq = estimate_channel_losses(activation, weight)

        self.assertEqual(set(losses), {4, 8, 16})
        torch.testing.assert_close(losses[16], torch.zeros(6), rtol=0, atol=0)
        self.assertEqual(awq.shape, (128,))
        self.assertTrue((losses[4] >= 0).all() and (losses[8] >= 0).all())

    def test_all_fp16_is_exact(self):
        torch.manual_seed(7)
        x = torch.randn(3, 128, dtype=torch.float32)
        weight = torch.randn(4, 128, dtype=torch.float32)
        allocation = allocate_channels(
            {4: [2.0] * 4, 8: [1.0] * 4, 16: [0.0] * 4},
            ThreeLevelBudget(0, 0, 100),
        )

        actual = fake_linear(x, weight, allocation)
        expected = torch.nn.functional.linear(x, weight)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_mixed_output_matches_per_partition_reference(self):
        torch.manual_seed(11)
        x = torch.randn(2, 128)
        weight = torch.randn(4, 128)
        allocation = ThreeLevelAllocation(
            indices={4: (0, 1), 8: (2,), 16: (3,)},
            scores={4: (0.0,) * 4, 8: (0.0,) * 4, 16: (0.0,) * 4},
            budget=ThreeLevelBudget(50, 25, 25),
        )

        actual = fake_linear(x, weight, allocation)
        self.assertEqual(actual.shape, (2, 4))
        torch.testing.assert_close(actual[:, 3], x @ weight[3], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(actual).all())

    def test_packed_module_matches_fake_reference(self):
        torch.manual_seed(19)
        x = torch.randn(3, 128)
        weight = torch.randn(8, 128)
        allocation = allocate_channels(
            {4: torch.linspace(8, 1, 8),
             8: torch.linspace(4, 0.5, 8),
             16: torch.zeros(8)},
            ThreeLevelBudget(50, 25, 25),
        )
        module = ThreeLevelLinear.from_weight(weight, allocation)

        actual = module(x)
        expected = fake_linear(x, weight, allocation)
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)

    def test_packed_module_state_dict_round_trip(self):
        weight = torch.randn(4, 128)
        allocation = allocate_channels(
            {4: [2.0] * 4, 8: [1.0] * 4, 16: [0.0] * 4},
            ThreeLevelBudget(50, 25, 25),
        )
        original = ThreeLevelLinear.from_weight(weight, allocation)
        original._sm75_int4_expanded = ("runtime-only", torch.empty(1))
        loaded = ThreeLevelLinear(128, 4)
        loaded._sm75_int4_expanded = ("stale", torch.empty(1))
        loaded.load_state_dict(original.state_dict())

        self.assertNotIn("_sm75_int4_expanded", original.state_dict())
        self.assertIsNone(loaded._sm75_int4_expanded)
        torch.testing.assert_close(loaded.dequantize_weight(), original.dequantize_weight())

    def test_runtime_caches_clear_on_apply_and_load(self):
        module = ThreeLevelLinear(128, 4)
        module._sm75_fp16_placeholders = object()
        module._sm75_int4_expanded = object()

        module.to(dtype=torch.float16)

        self.assertIsNone(module._sm75_fp16_placeholders)
        self.assertIsNone(module._sm75_int4_expanded)

    def test_backward_compatible_two_level_state_dict(self):
        weight = torch.randn(4, 128)
        allocation = allocate_channels(
            {4: [2.0] * 4, 8: [1.0] * 4, 16: [0.0] * 4},
            ThreeLevelBudget(50, 50, 0),
        )
        original = ThreeLevelLinear.from_weight(weight, allocation)
        legacy = original.state_dict()
        legacy.pop("weight_fp16")
        legacy.pop("indices_16")
        loaded = ThreeLevelLinear(128, 4)
        loaded.load_state_dict(legacy, strict=True)

        self.assertEqual(loaded.indices_16.numel(), 0)
        torch.testing.assert_close(loaded.dequantize_weight(), original.dequantize_weight())


if __name__ == "__main__":
    unittest.main()
