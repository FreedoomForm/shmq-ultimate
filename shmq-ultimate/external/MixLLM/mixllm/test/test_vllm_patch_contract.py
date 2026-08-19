import re
import unittest
from pathlib import Path

from mixllm.vllm_three_level import PINNED_VLLM_COMMIT


ROOT = Path(__file__).resolve().parents[2]
PATCH_DIR = ROOT / "vllm_v0.9.0_patch"
PATCH = PATCH_DIR / "0002-add-mixllm-three-level-support.patch"
MANIFEST = PATCH_DIR / "THREE_LEVEL_MANIFEST.md"
APPLY_SCRIPT = ROOT / "apply_vllm_patche.sh"


class VLLMNativePatchContractTest(unittest.TestCase):

    def test_pin_is_exact_and_consistent(self):
        script = APPLY_SCRIPT.read_text(encoding="utf-8")
        manifest = MANIFEST.read_text(encoding="utf-8")
        self.assertIn(f'VLLM_COMMIT="{PINNED_VLLM_COMMIT}"', script)
        self.assertIn(f"Required vLLM commit: `{PINNED_VLLM_COMMIT}`", manifest)
        self.assertIn('git checkout --detach "$VLLM_COMMIT"', script)
        self.assertNotIn("checkout releases/v0.9.0", script)

    def test_patch_registers_explicit_three_level_method(self):
        text = PATCH.read_text(encoding="utf-8")
        self.assertIn('"mixllm_three_level"', text)
        self.assertIn("class MixLLMThreeLevelConfig(QuantizationConfig)", text)
        self.assertIn("class MixLLMThreeLevelLinearMethod(LinearMethodBase)", text)
        self.assertIn("from mixllm.vllm_three_level import VLLMThreeLevelConfig", text)

    def test_patch_implements_packed_loader_tp_remap_and_linear(self):
        text = PATCH.read_text(encoding="utf-8")
        for name in (
            "weight_fp16", "weight_int8", "scale_int8", "weight_int4",
            "scale_int4", "zero_int4", "indices_4", "indices_8", "indices_16",
        ):
            self.assertIn(f'"{name}"', text)
        self.assertIn("get_tensor_model_parallel_rank", text)
        self.assertIn("local_index = index[keep] - output_start", text)
        self.assertLess(text.index("input_sharded = local_input != input_size"),
                        text.index("global_output_sizes ="))
        self.assertIn("device=param.device, non_blocking=True", text)
        self.assertIn("row-parallel MixLLM input shard must be divisible", text)
        self.assertIn("three_level_linear", text)
        self.assertIn("requires backend=auto or sm75", text)

    def test_patch_advertises_native_sm75_contract(self):
        text = PATCH.read_text(encoding="utf-8")
        self.assertIn("+        return 75", text)
        self.assertIn('+        if contract.backend not in {"auto", "sm75"}:', text)
        self.assertIn('+        if not hasattr(layer, "_sm75_int4_expanded"):', text)
        self.assertIn('+        if not hasattr(layer, "_sm75_fp16_placeholders"):', text)
        self.assertIn("get_device_capability", text)
        self.assertNotIn("gated to backend=reference", text)
    def test_patch_does_not_modify_cuda_sources(self):
        text = PATCH.read_text(encoding="utf-8")
        changed_paths = re.findall(r"^diff --git a/(\S+) b/(\S+)$", text,
                                   flags=re.MULTILINE)
        self.assertTrue(changed_paths)
        for old, new in changed_paths:
            self.assertNotRegex(old + new, r"\.(cu|cuh|cpp|cc)$")


if __name__ == "__main__":
    unittest.main()
