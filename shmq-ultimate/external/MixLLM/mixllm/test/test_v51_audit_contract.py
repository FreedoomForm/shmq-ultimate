from pathlib import Path
import unittest


class V51AuditContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        package_root = Path(__file__).resolve().parents[1]
        repo_candidates = [package_root.parents[3], Path.cwd()]
        builder_path = next(
            (
                candidate / "scripts" / "build_mixllm_3level_kaggle.py"
                for candidate in repo_candidates
                if (candidate / "scripts" / "build_mixllm_3level_kaggle.py").exists()
            ),
            None,
        )
        cls.backend = (package_root / "sm75_backend.py").read_text(encoding="utf-8")
        cls.cuda = (package_root / "kernels" / "three_level_sm75.cu").read_text(encoding="utf-8")
        cls.builder = builder_path.read_text(encoding="utf-8") if builder_path else ""

    def test_v51_decode_keeps_packed_int4_and_skips_expansion(self):
        self.assertIn("if x.shape[0] == 1 or not module.indices_4.numel():", self.backend)
        decode_start = self.cuda.index("__global__ void three_level_decode_kernel")
        decode_end = self.cuda.index("void check_cuda_contiguous", decode_start)
        decode = self.cuda[decode_start:decode_end]
        self.assertIn("const uint8_t* packed_int4", decode)
        self.assertIn("const uint8_t* weights = packed_int4", decode)
        launch_start = self.cuda.index("if (rows == 1)")
        launch_end = self.cuda.index("} else", launch_start)
        launch = self.cuda[launch_start:launch_end]
        self.assertIn("weight_int4.data_ptr<uint8_t>()", launch)
        self.assertNotIn("expanded_int4.data_ptr<int8_t>()", launch)

    def test_gate_executes_sm75_regressions_not_only_reference_tests(self):
        if not self.builder:
            self.skipTest("Kaggle embeds package sources, not the sandbox builder")
        self.assertIn("test_sm75_backend.py", self.builder)
        self.assertIn("test_sm75_source.py", self.builder)
        self.assertIn("test_vllm_three_level.py", self.builder)
        self.assertIn("test_runtime_capability.py", self.builder)
        self.assertIn("'mixllm.test.test_sm75_backend'", self.builder)
        self.assertIn("'mixllm.test.test_sm75_source'", self.builder)
        self.assertIn("'mixllm.test.test_v51_audit_contract'", self.builder)

    def test_sm75_build_unpacks_embedded_cutlass_vendor(self):
        self.assertIn("cutlass_sm75_vendor.b64", self.backend)
        self.assertIn("base64.b64decode", self.backend)
        self.assertIn("staging_root", self.backend)
        self.assertIn("staged_vendor_root", self.backend)
        self.assertIn("shutil.copytree(staged_vendor_root, vendor_root)", self.backend)
        self.assertIn("extra_include_paths", self.backend)
        self.assertIn("vendor_include / \"cutlass\" / \"array.h\"", self.backend)
        self.assertNotIn("archive.extractall(extraction_root)", self.backend)

    def test_benchmark_exposes_runtime_memory_telemetry(self):
        self.assertIn("peak_cuda_memory", self.backend)
        self.assertIn("expanded_int4_bytes", self.backend)
        self.assertIn("activation_quantized_bytes", self.backend)
        self.assertIn("prefill_metadata_bytes", self.backend)
        self.assertIn("output_bytes", self.backend)

    def test_v189_metadata_cache_and_v3_operator_contract(self):
        self.assertIn("_prefill_metadata_for_cutlass", self.backend)
        self.assertIn("_sm75_prefill_metadata", self.backend)
        self.assertIn("three_level_linear_v3", self.backend)
        self.assertIn('m.def("three_level_linear_v3', self.cuda)
        self.assertIn('m.impl("three_level_linear_v3"', self.cuda)
        self.assertIn("cached_scale_int4", self.cuda)
        self.assertIn("cached_zero_int4", self.cuda)
        self.assertIn("cached_scale_int8", self.cuda)

    def test_production_report_cannot_claim_go_without_model_quality(self):
        if not self.builder:
            self.skipTest("Kaggle embeds package sources, not the sandbox builder")
        self.assertIn("operator_production", self.builder)
        self.assertIn("model_vllm_production", self.builder)
        self.assertIn("full_model_qwen_quality", self.builder)
        self.assertIn("vllm_apply_path", self.builder)


if __name__ == "__main__":
    unittest.main()
