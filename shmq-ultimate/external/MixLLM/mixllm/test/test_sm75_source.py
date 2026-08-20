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
        cls.backend = (root / "sm75_backend.py").read_text(encoding="utf-8")

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
        self.assertIn("typenameCore::MmaPolicy,Stages>", cutlass_compact)
        self.assertIn("sync_copy", pipeline_compact)
        self.assertIn("usingArchTag=arch::Sm75", pipeline_compact)
        self.assertNotIn("cp_async", pipeline_compact)

    def test_v194_measured_v188_large_mixed_prefill_restore_contract(self):
        self.assertIn("def _use_v188_mixed_prefill_path(", self.backend)
        self.assertIn("_three_level_linear_v2_unchecked(*arguments)", self.backend)
        self.assertIn("_use_v188_mixed_prefill_path(module, x, torch_module)", self.backend)

    def test_cached_v2_metadata_adapter_contract(self):
        compact = "".join(self.source.split())
        backend_compact = "".join(self.backend.split())
        self.assertIn("_three_level_linear_v2_cached_unchecked", compact)
        self.assertIn("three_level_linear_v2_cached_unchecked_cuda", self.source)
        self.assertIn("_prefill_metadata_for_cutlass(module,x,torch_module)", backend_compact)
        self.assertIn("cached_v2(*arguments,*metadata[1:])", backend_compact)
        self.assertIn("returntorch_module.ops.mixllm_sm75._three_level_linear_v2_unchecked(*arguments)", backend_compact)

    def test_v196_timing_integrity_contract(self):
        self.assertIn("timing_integrity_ratio", self.backend)
        self.assertIn("timing_integrity", self.backend)
        self.assertIn("timing_integrity", self.source)

    def test_v197_sm75_three_stage_runner_contract(self):
        self.assertIn("template <typename Core, int Stages>", self.cutlass_testbed)
        self.assertIn("MQMmaPipelinedSm75<", self.cutlass_testbed)
        self.assertIn("Stages>", self.cutlass_testbed)
        self.assertIn("DefaultMmaCore<", self.cutlass_testbed)
        self.assertIn("OpMultiplyAddSaturate>;", self.cutlass_testbed)

    def test_v198_supported_core_three_stage_pipeline_contract(self):
        cutlass_compact = "".join(self.cutlass_testbed.split())
        self.assertIn("OpClassTensorOp,2,cutlass::arch::OpMultiplyAddSaturate>", cutlass_compact)
        self.assertIn("usingInt8Runner=Runner<Core,2>", cutlass_compact)

    def test_v214_guarded_k128_contract(self):
        pipeline_compact = "".join(self.cutlass_pipeline.split())
        cutlass_compact = "".join(self.cutlass_testbed.split())
        source_compact = "".join(self.source.split())
        self.assertIn("static_assert(Shape::kK==64||Shape::kK==128)", pipeline_compact)
        self.assertIn("ifconstexpr(Shape::kK==64)", pipeline_compact)
        self.assertIn("row_groupsize64_+=2", pipeline_compact)
        self.assertIn("ifconstexpr(Shape::kK==64){if(gemm_k_iterations>=0){mac_loop_iter", pipeline_compact)
        self.assertIn("usingKWideInt8Runner=Runner<KWideCore,2>", cutlass_compact)
        self.assertIn("KWideInt8Runner::run", source_compact)
        self.assertIn("rows>=32&&channels>=128", source_compact)

    def test_v200_integer_prefill_overlap_contract(self):
        source_compact = "".join(self.source.split())
        cutlass_compact = "".join(self.cutlass_testbed.split())
        self.assertIn("structIntegerPrefillStreams", source_compact)
        self.assertIn("begin_integer_prefill_overlap", source_compact)
        self.assertIn("finish_integer_prefill_overlap", source_compact)
        self.assertIn("cudaStreamWaitEvent(streams.int4,streams.fork,0)", source_compact)
        self.assertIn("cudaStreamWaitEvent(streams.int8,streams.fork,0)", source_compact)
        self.assertIn("usingInt8Runner=Runner<Core,2>", cutlass_compact)
        self.assertIn("KWideInt8Runner::run", source_compact)
        self.assertNotIn("combined_int8", source_compact)
        self.assertNotIn("three_level_linear_cublas", source_compact)

    def test_v193_rejected_geometry_is_not_present(self):
        cutlass_compact = "".join(self.cutlass_testbed.split())
        self.assertIn("GemmShape<32,128,64>", cutlass_compact)
        self.assertIn("GemmShape<32,32,64>", cutlass_compact)
        self.assertIn("GemmShape<8,8,16>", cutlass_compact)
        self.assertIn("typenameCore::MmaPolicy,Stages>", cutlass_compact)
        self.assertNotIn("GemmShape<16,8,32>", cutlass_compact)
        self.assertNotIn("typenameCore::MmaPolicy,5>", cutlass_compact)

    def test_v188_cutlass_prefill_dispatch_contract(self):
        text = self.source
        self.assertIn("} else if (rows >= 32 && (n4 > 0 || n8 > 0)) {", text)
        self.assertIn("run_cutlass_int_partition(", text)
        self.assertIn("expanded_int4, scale_int4, zero_int4", text)
        self.assertIn("weight_int8, scale_int8", text)
        self.assertIn("matrix_zero", text)
        self.assertIn("output_width, 0, 0, n16", text)
        cutlass_branch = text.index("} else if (rows >= 32 && (n4 > 0 || n8 > 0)) {")
        fallback_branch = text.index("} else {", cutlass_branch)
        self.assertLess(cutlass_branch, fallback_branch)
        self.assertIn("three_level_tensorcore_kernel<kPrefillWarps>", text[cutlass_branch:])
        self.assertNotIn("kPrefillWideWarps", text)

    def test_native_three_level_linear_forward_is_present(self):
        text = self.linear
        self.assertIn("get_device_capability", text)
        self.assertIn("three_level_linear(self, flat, torch)", text)
        self.assertIn("result.reshape", text)
        self.assertIn("bias", text)


if __name__ == "__main__":
    unittest.main()