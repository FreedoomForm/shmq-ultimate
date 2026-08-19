from pathlib import Path
import unittest


class SM75SourceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).parents[1]
        cls.source = (root / "kernels" / "three_level_sm75.cu").read_text(encoding="utf-8")
        cls.linear = (root / "nn" / "modules" / "three_level_linear.py").read_text(encoding="utf-8")
        cls.cutlass_testbed = (root / "kernels" / "sm75_cutlass_testbed.h").read_text(encoding="utf-8")
        cls.cutlass_pipeline = (root / "kernels" / "cutlass_extension" / "mq_mma_pipelined_sm75.h").read_text(encoding="utf-8")

    def test_v51_prefill_and_three_level_paths(self):
        text = self.source
        self.assertIn("__global__ void three_level_tensorcore_kernel", text)
        self.assertIn("wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, signed char", text)
        self.assertIn("wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, signed char", text)
        self.assertIn("wmma::fragment<wmma::accumulator, kTile, kTile, kTile, int", text)
        self.assertGreaterEqual(text.count("__syncthreads();"), 3)
        self.assertIn("expanded_int4", text)
        self.assertIn("input_fp16.size(0) == 1", text)
        self.assertIn("three_level_tensorcore_kernel", self.source)
        self.assertNotIn("wmma::fragment<wmma::accumulator, 8, 32, kTile, int>", text)
        self.assertNotIn("linear % 32", text)

    def test_v45_decode_mapping_is_isolated(self):
        text = self.source
        self.assertIn("constexpr int kDecodeWarps = 8;", text)
        self.assertIn("constexpr int kDecodeChannelsPerWarp = 2;", text)
        self.assertIn("constexpr int kDecodeSubwarp = kWarpSize / kDecodeChannelsPerWarp;", text)
        self.assertIn("__shfl_down_sync(subwarp_mask, accumulator, delta,", text)
        self.assertIn("local_warp * kDecodeChannelsPerWarp + subwarp", text)
        self.assertNotIn("__shared__ int8_t decode_activation", text)

    def test_cutlass_sm75_port_contract(self):
        self.assertIn('#include "sm75_cutlass_testbed.h"', self.source)
        self.assertIn("run_cutlass_int_partition", self.source)
        cutlass_compact = "".join(self.cutlass_testbed.split())
        pipeline_compact = "".join(self.cutlass_pipeline.split())
        self.assertIn("DefaultMmaCore<", cutlass_compact)
        self.assertIn("GemmShape<8,8,16>", cutlass_compact)
        self.assertIn("MQMmaPipelinedSm75", cutlass_compact)
        self.assertIn("GemmShape<32,128,64>", cutlass_compact)
        self.assertIn("typenameCore::MmaPolicy,2>", cutlass_compact)
        self.assertIn("sync_copy", pipeline_compact)
        self.assertIn("usingArchTag=arch::Sm75", pipeline_compact)
        self.assertNotIn("cp_async", pipeline_compact)

    def test_v186_wide_prefill_launch_contract(self):
        text = self.source
        self.assertIn("constexpr int kPrefillWideWarps = 8;", text)
        self.assertIn("constexpr int kPrefillWideChannels = kPrefillWideWarps * kTile;", text)
        self.assertIn("template <int PrefillWarps = kPrefillWarps>", text)
        self.assertIn("three_level_tensorcore_kernel<kPrefillWideWarps>", text)
        self.assertIn("kPrefillWideChannels - 1", text)
        self.assertNotIn("three_level_tensorcore_kernel<<<grid, kPrefillWarps", text)

    def test_native_three_level_linear_forward_is_present(self):
        text = self.linear
        self.assertIn("get_device_capability", text)
        self.assertIn("three_level_linear(self, flat, torch)", text)
        self.assertIn("result.reshape", text)
        self.assertIn("bias", text)


if __name__ == "__main__":
    unittest.main()