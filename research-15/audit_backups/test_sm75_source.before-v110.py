from pathlib import Path
import unittest


class SM75SourceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).parents[1]
        cls.source = (root / "kernels" / "three_level_sm75.cu").read_text(encoding="utf-8")
        cls.linear = (root / "nn" / "modules" / "three_level_linear.py").read_text(encoding="utf-8")

    def test_one_gemm_kernel_contains_integer_and_fp16_tensorcore_paths(self):
        text = self.source
        self.assertEqual(text.count("three_level_tensorcore_kernel<<<"), 1)
        self.assertEqual(text.count("three_level_decode_kernel<<<"), 1)
        self.assertIn("quantize_activation_sm75_kernel", text)
        self.assertIn("three_level_tensorcore_kernel", text)
        self.assertIn("three_level_decode_kernel", text)
        self.assertNotIn("three_level_tensorcore_reuse_kernel<<<", text)
        self.assertNotIn("three_level_partition_kernel", text)
        self.assertNotIn("packed_weight_to_fp16", text)
        self.assertIn("weight_fp16", text)
        self.assertIn("wmma::mma_sync", text)
        self.assertIn("__half22float2(__hmul2(input2[k2], weights2[k2]))", text)
        self.assertIn("expanded_int4", text)
        self.assertIn("expanded_int4.data_ptr<int8_t>()", text)
        self.assertIn("expand_int4_sm75_kernel", text)
        self.assertIn("three_level_linear_v2", text)
        self.assertIn("three_level_linear_legacy_cuda", text)
        self.assertIn("check_same_device", text)
        self.assertIn("output_width > 0", text)
        self.assertIn("channels_per_block", text)
        self.assertIn("kPrefillWarps * kWarpSize", text)
        self.assertIn("kDecodeChannelsPerWarp", text)
        self.assertIn("linear += blockDim.x", text)

    def test_m8n32_prefill_ownership_and_output_indexing_are_explicit(self):
        text = self.source
        self.assertIn("wmma::fragment<wmma::accumulator, 8, 32, kTile, int>", text)
        self.assertIn("wmma::fragment<wmma::matrix_a, 8, 32, kTile, signed char", text)
        self.assertIn("wmma::fragment<wmma::matrix_b, 8, 32, kTile, signed char", text)
        self.assertIn("a_int8[warp / 2]", text)
        self.assertIn("row_base += (warp / 2) * 8", text)
        self.assertIn("channel_base += (warp % 2) * 32", text)
        self.assertIn("linear / 32", text)
        self.assertIn("linear % 32", text)
        self.assertIn("store_matrix_sync", text)

    def test_partial_row_staging_uses_a_block_uniform_barrier(self):
        text = self.source
        self.assertIn("const bool full_rows = row_base + 8 <= rows;", text)
        branch_start = text.index("      if (full_rows) {")
        branch_end = text.index("      if (!full_rows) {", branch_start)
        prefill_fragment = text[branch_start:branch_end]
        self.assertIn("      } else {", prefill_fragment)
        self.assertIn("__syncthreads();", prefill_fragment)
        self.assertNotIn("if (!full_rows) {\n        __syncthreads();", prefill_fragment)
        self.assertGreaterEqual(text.count("__syncthreads();"), 3)

    def test_native_three_level_linear_forward_is_present(self):
        text = self.linear
        self.assertIn("get_device_capability", text)
        self.assertIn("three_level_linear(self, flat, torch)", text)
        self.assertIn("flat.reshape", text)
        self.assertIn("bias", text)


if __name__ == "__main__":
    unittest.main()