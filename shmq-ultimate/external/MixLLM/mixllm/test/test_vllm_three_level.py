import unittest

from mixllm.vllm_three_level import (
    PINNED_VLLM_COMMIT,
    VLLMThreeLevelConfig,
    remap_partition_indices_for_tp,
    restore_global_partition_indices,
    validate_partition_indices,
)


class VLLMThreeLevelContractTest(unittest.TestCase):

    def test_pins_real_upstream_submodule_commit(self):
        self.assertEqual(PINNED_VLLM_COMMIT,
                         "5fbbfe9a4c13094ad72ed3d6b4ef208a7ddc0fd7")

    def test_config_and_backend_gate(self):
        config = VLLMThreeLevelConfig.from_quantization_config({
            "quant_method": "mixllm_three_level",
            "precision_percentages": {"4": 75, "8": 20, "16": 5},
            "group_size": 128,
        })
        self.assertEqual(config.select_backend((7, 5)), "reference")
        self.assertEqual(config.select_backend((8, 0)), "ampere")

    def test_requires_three_level_percentages(self):
        with self.assertRaises(ValueError):
            VLLMThreeLevelConfig.from_quantization_config({"quant_method": "mixllm"})

    def test_partition_validation(self):
        validate_partition_indices({4: [2, 3], 8: [1], 16: [0]}, 4)
        with self.assertRaises(ValueError):
            validate_partition_indices({4: [1], 8: [1], 16: [0]}, 3)

    def test_tensor_parallel_index_remapping_round_trip(self):
        global_indices = {4: [0, 3, 6, 7], 8: [1, 5], 16: [2, 4]}
        local = remap_partition_indices_for_tp(global_indices, 8, 2, 4)
        self.assertEqual(local, {4: (1,), 8: (3,), 16: (0, 2)})
        restored = restore_global_partition_indices(local, 2, 4)
        self.assertEqual(restored, {4: (3,), 8: (5,), 16: (2, 4)})

    def test_tensor_parallel_rejects_invalid_shards(self):
        indices = {4: [0, 1], 8: [2], 16: [3]}
        with self.assertRaises(ValueError):
            remap_partition_indices_for_tp(indices, 4, 3, 2)


if __name__ == "__main__":
    unittest.main()