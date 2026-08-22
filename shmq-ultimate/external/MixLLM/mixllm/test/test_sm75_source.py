from pathlib import Path
import unittest


class SM75SourceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).parents[1]
        cls.source = (root / "kernels" / "three_level_sm75.cu").read_text(encoding="utf-8")
        cls.linear = (root / "nn" / "modules" / "three_level_linear.py").read_text(encoding="utf-8")
        cls.cutlass_testbed = (root / "kernels" / "sm75_cutlass_testbed.h").read_text(encoding="utf-8")
        cls.cutlass_sm75_mixed = (root / "kernels" / "cutlass_extension" / "mq_mma_tensor_op_sm75.h").read_text(encoding="utf-8")
        cls.cutlass_pipeline = (root / "kernels" / "cutlass_extension" / "mq_mma_pipelined_sm75.h").read_text(encoding="utf-8")
        cls.dequantizer = (root / "kernels" / "cutlass_extension" / "mq_mma_tensor_op_dequantizer.h").read_text(encoding="utf-8")
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
        self.assertIn("usingLayoutC=cutlass::layout::RowMajor", cutlass_compact)
        self.assertIn("GemmShape<32,128,64>", cutlass_compact)
        self.assertIn("typenameCore::MmaPolicy,Stages>", cutlass_compact)
        self.assertIn("sync_copy", pipeline_compact)
        self.assertIn("usingArchTag=arch::Sm75", pipeline_compact)
        self.assertNotIn("cp_async", pipeline_compact)

    def test_v194_measured_v188_large_mixed_prefill_restore_contract(self):
        self.assertIn("def _use_v188_mixed_prefill_path(", self.backend)
        self.assertIn("_three_level_linear_v2_unchecked(*arguments)", self.backend)
        self.assertIn("_use_v188_mixed_prefill_path(module, x, torch_module)", self.backend)

    def test_v263_large_m_fp16_uses_cublas_scatter_contract(self):
        source_compact = "".join(self.source.split())
        self.assertIn("voidrun_fp16_partition_cublas(", source_compact)
        self.assertIn("at::mm(input_fp16,weight_fp16.transpose(0,1))", source_compact)
        self.assertIn("scatter_fp16_partition_kernel<<<blocks,threads,0,stream>>>", source_compact)
        self.assertIn("run_fp16_partition_cublas(input_fp16,weight_fp16,indices_fp16,output,plan.rows,plan.width,caller_stream)", source_compact)
        self.assertIn("run_unified_prefill(", source_compact)

    def test_v261_mixed_prefill_uses_cached_metadata_dispatch(self):
        selector = self.backend[self.backend.index("def _use_v188_mixed_prefill_path"):]
        self.assertIn("return False", selector)
        self.assertIn("_prefill_metadata_for_cutlass(module, x, torch_module)", self.backend)
        self.assertIn("use_cached_v3", self.backend)
        self.assertIn("native_v3", self.backend)
        self.assertIn("prefill_int4", self.backend)
        self.assertIn("*metadata[1:]", self.backend)

    def test_v284_native_packed_prefill_defers_expanded_int4(self):
        self.assertIn("Do not materialize SHMQ's signed expanded copy", self.backend)
        self.assertIn("expanded_int4 = module.weight_int8[:0]", self.backend)
        self.assertIn("signed expanded copy lazy", self.linear)

    def test_v299_large_m_uses_expanded_correctness_control(self):
        backend_compact = "".join(self.backend.split())
        dispatch = backend_compact[
            backend_compact.index("expanded_int4=None"):
        ]
        self.assertIn("#v299control:usetheoriginal-safeexpandedINT4+stagedCUTLASS", dispatch)
        self.assertIn("native_v3=None", dispatch)
        self.assertIn("use_cached_v3=False", dispatch)
        self.assertNotIn("_prefill_metadata_for_cutlass(module,x,torch_module)", dispatch)
        self.assertIn("expanded_int4=_expanded_int4_for_prefill(module,x,torch_module)", dispatch)

    def test_v196_timing_integrity_contract(self):
        self.assertIn("timing_integrity_ratio", self.backend)
        self.assertIn("timing_integrity", self.backend)
        self.assertIn("timing_integrity", self.source)

    def test_v197_sm75_three_stage_runner_contract(self):
        self.assertIn("template <typename Core, int Stages, typename ElementB_ = ElementB>", self.cutlass_testbed)
        self.assertIn("MQMmaPipelinedSm75<", self.cutlass_testbed)
        self.assertIn("Stages>", self.cutlass_testbed)
        self.assertIn("DefaultMmaCore<", self.cutlass_testbed)
        self.assertIn("OpMultiplyAddSaturate>;", self.cutlass_testbed)

    def test_v198_supported_core_three_stage_pipeline_contract(self):
        cutlass_compact = "".join(self.cutlass_testbed.split())
        self.assertIn("OpClassTensorOp,2,cutlass::arch::OpMultiplyAddSaturate>", cutlass_compact)
        self.assertIn("usingInt8Runner=Runner<Core,2>", cutlass_compact)

    def test_v227_persistent_runtime_cache_contract(self):
        linear_text = self.linear
        backend_compact = "".join(self.backend.split())
        source_compact = "".join(self.source.split())
        self.assertIn("prepare_sm75_prefill_cache", linear_text)
        self.assertIn("prepare_sm75_prefill_metadata", linear_text)
        self.assertIn("prepare_sm75_packed_tensors", linear_text)
        self.assertIn("prepare_sm75_packed_tensors", backend_compact)
        self.assertIn("hashlib.sha256(source.read_bytes())", self.backend)
        self.assertIn("mixllm_sm75_backend_{source_digest}", self.backend)
        self.assertIn("is_current_stream_capturing", self.backend)

    def test_v228_epilogue_index_fragment_contract(self):
        cutlass_text = self.cutlass_testbed
        self.assertIn("using IndexFragment = cutlass::Array<int", cutlass_text)
        self.assertIn("index_fragment[fragment_index]", cutlass_text)
        self.assertIn("index_fragment[fragment_index] >= 0", cutlass_text)
        self.assertEqual(cutlass_text.count("indices[partition_channel]"), 1)
        self.assertNotIn("ptr_C[global_row * ldc + indices[partition_channel]]", cutlass_text)

    def test_v231_fused_int4_pair_probe_contract(self):
        probe = (Path(__file__).parents[1] / "kernels" / "cutlass_extension" / "mq_mma_sm75_int4_pair.h").read_text(encoding="utf-8")
        self.assertIn("using LowMma", probe)
        self.assertIn("using HighMma", probe)
        self.assertIn("cutlass::uint4b_t, LayoutA", probe)
        self.assertIn("cutlass::int4b_t, LayoutA", probe)
        self.assertIn("GemmShape<8, 8, 32>", probe)
        self.assertIn("struct InstructionPair", probe)
        self.assertIn("mq_mma_sm75_int4_pair.h", self.cutlass_testbed)

    def test_v233_mixed_stride_probe_contract(self):
        source_compact = "".join(self.source.split())
        self.assertIn("sm75_int4_pair_mixed_stride_probe_cuda", source_compact)
        self.assertIn("sm75_int4_pair_mixed_stride_probe(Tensordevice_tensor)->Tensor", source_compact)
        self.assertIn("output_width=static_cast<int>(output.size(1))", source_compact)
        self.assertIn("output[row*output_width+indices_int4[channel]]", source_compact)

    def test_v235_direct_sm75_accumulator_mapping_contract(self):
        source_compact = "".join(self.source.split())
        self.assertIn("floatpartial[kPairRowTiles][kPairNSubtiles][2]", source_compact)
        self.assertIn("for(introw_tile=0;row_tile<kPairRowTiles;++row_tile)", source_compact)
        self.assertIn("wmma::load_matrix_sync(a_low_u4,&a_low_packed[row_tile*8][0]", source_compact)
        self.assertIn("constintlocal_row=row_tile*8+(lane>>2)", source_compact)
        self.assertIn("constintlocal_channel_base=(lane&3)*2", source_compact)
        self.assertIn("low_accum[row_tile][n_tile][register_index]+16*high_accum[row_tile][n_tile][register_index]", source_compact)
        self.assertNotIn("low_tile[warp][item]", source_compact)
        self.assertNotIn("wmma::store_matrix_sync(low_tile", self.source)

    def test_v238_complete_fused_warp_tile_contract(self):
        source_compact = "".join(self.source.split())
        self.assertIn("template<intkPairWarps,intkPairChannels>", source_compact)
        self.assertIn("constexprintkPairRowTiles=4", source_compact)
        self.assertIn("floatpartial[kPairRowTiles][kPairNSubtiles][2]", source_compact)
        self.assertIn("sm75_int4_pair_gemm_kernel<8,128>", source_compact)
        self.assertIn("sm75_int4_pair_gemm_kernel<4,64>", source_compact)
        self.assertIn("constexprintkPairNSubtiles=2", source_compact)
        self.assertIn("wmma::load_matrix_sync(b_u4,&b_packed[warp*16+n_tile*8][0]", source_compact)
        self.assertEqual(source_compact.count("channel_base+warp*16+n_tile*8+local_channel_base+register_index"), 2)
        self.assertIn("output[row*output_width+indices_int4[channel]]=__float2half_rn(partial[row_tile][n_tile][register_index])", source_compact)
        self.assertEqual(source_compact.count("a_low_packed[row][pair]=low0|(low1<<4)"), 1)
        self.assertEqual(source_compact.count("b_packed[local_channel][pair]=channel<channels?weight_int4[source]:0"), 1)

    def test_v283_fused_int4_isolated_from_production_dispatch(self):
        source_compact = "".join(self.source.split())
        self.assertIn("__global__voidsm75_int4_pair_gemm_kernel", source_compact)
        self.assertIn("voidrun_int4_pair_partition(", source_compact)
        self.assertNotIn("n4>0&&n8==0&&n16==0&&!has_cached_metadata", source_compact)
        self.assertIn("begin_integer_prefill_overlap", source_compact)
        self.assertIn("cached_scale_int8, false);", self.source)
        self.assertIn("low_mma(low_accum[row_tile][n_tile],low_a,weights,low_accum[row_tile][n_tile])", source_compact)
        self.assertIn("high_mma(high_accum[row_tile][n_tile],high_a,weights,high_accum[row_tile][n_tile])", source_compact)
        self.assertIn("16*high_accum[row_tile][n_tile][register_index]-correction", source_compact)

    def test_v226_legal_sm75_candidate_tuner_contract(self):
        source_compact = "".join(self.source.split())
        cutlass_compact = "".join(self.cutlass_testbed.split())
        self.assertIn("GemmShape<32,64,64>", cutlass_compact)
        self.assertIn("usingInt8RunnerN64=Runner<CoreN64,2>", cutlass_compact)
        self.assertIn("GemmShape<128,64,64>", cutlass_compact)
        self.assertIn("usingInt8RunnerM128N64=Runner<CoreM128N64,2>", cutlass_compact)
        self.assertIn("GemmShape<64,64,64>", cutlass_compact)
        self.assertIn("usingInt8RunnerM64N64=Runner<CoreM64N64,2>", cutlass_compact)
        self.assertIn("kM128N64=2", source_compact)
        self.assertIn("kM64N64=3", source_compact)
        self.assertIn("GemmShape<128,128,64>", cutlass_compact)
        self.assertIn("usingInt8RunnerM128N128=Runner<CoreM128N128,2>", cutlass_compact)
        self.assertNotIn("GemmShape<256,64,64>", cutlass_compact)
        self.assertNotIn("usingInt8RunnerM256N64", cutlass_compact)
        self.assertIn("kCutlassTuningAbi=296", source_compact)
        self.assertIn("kM128N64", source_compact)
        self.assertIn("enumclassCutlassConfig", source_compact)
        self.assertIn("cutlass_tuning_key", source_compact)
        self.assertIn("cudaEventElapsedTime", source_compact)
        self.assertIn("SHMQ_SM75_TUNE_CACHE", source_compact)
        self.assertIn("cudaStreamIsCapturing", source_compact)

    def test_v271_boundary_corrected_quantizer_contract(self):
        source_compact = "".join(self.source.split())
        self.assertIn("constfloatinverse_scale=1.0f/scale", source_compact)
        self.assertIn("constfloatlower_boundary=static_cast<float>(fast_value)-0.5f", source_compact)
        self.assertIn("scaled=values[item]/scale", source_compact)
        self.assertIn("kCutlassTuningAbi=296", source_compact)

    def test_v286_native_int4_compile_safe_shape_contract(self):
        source_compact = "".join(self.source.split())
        cutlass_compact = "".join(self.cutlass_testbed.split())
        self.assertIn("CorePackedInt4M64N64", cutlass_compact)
        self.assertIn("PackedInt4RunnerM64N64=Runner<CorePackedInt4M64N64,2,cutlass::uint4b_t>", cutlass_compact)
        self.assertIn("PackedInt4RunnerM64N64::run", source_compact)
        self.assertNotIn("CorePackedInt4M128N64", cutlass_compact)
        self.assertNotIn("PackedInt4RunnerM128N64", source_compact)
        self.assertNotIn("if(rows>=96)", source_compact)
        self.assertIn("kCutlassTuningAbi=296", source_compact)

    def test_v291_single_partition_validation_contract(self):
        backend = "".join(self.backend.split())
        self.assertIn("partition_validated=False", backend)
        self.assertIn("ifnotpartition_validated:_validate_partition(module,x,torch_module)", backend)
        self.assertIn("_validate_partition(module,x,torch_module)ifx.shape[0]==0", backend)
        self.assertIn("packed_tensors,True", backend)

    def test_v288_cached_packed_wrapper_contract(self):
        backend = self.backend
        self.assertIn("packed_argument_tensors=packed_tensors", "".join(backend.split()))
        self.assertIn("packed_argument_tensors=tuple", "".join(backend.split()))
        self.assertIn("tensoriftensor.is_contiguous()elsetensor.contiguous()", "".join(backend.split()))
        self.assertIn("packed_argument_tensors[0]", "".join(backend.split()))
        self.assertIn("*packed_argument_tensors[1:]", "".join(backend.split()))

    def test_v287_persistent_packed_stream_record_contract(self):
        source_compact = "".join(self.source.split())
        self.assertIn("Packedweightsandcachedmetadataarepersistentmodule-ownedbuffers", source_compact)
        self.assertIn("record_tensor_stream(input_int8,stream)", source_compact)
        self.assertIn("record_tensor_stream(scale_act,stream)", source_compact)
        self.assertIn("record_tensor_stream(output,stream)", source_compact)
        packed_helper = self.source.split("void run_cutlass_packed_int4_partition(", 1)[1].split("void run_fp16_partition_cublas(", 1)[0]
        self.assertNotIn("record_tensor_stream(weight_int4_interleaved", packed_helper)
        self.assertNotIn("record_tensor_stream(matrix_scale", packed_helper)
        self.assertNotIn("record_tensor_stream(matrix_zero", packed_helper)

    def test_v282_unified_three_level_prefill_scheduler_contract(self):
        source_compact = "".join(self.source.split())
        self.assertIn("structUnifiedPrefillPlan", source_compact)
        self.assertIn("boolhas_integer()const", source_compact)
        self.assertIn("voidrun_unified_prefill(", source_compact)
        self.assertIn("run_unified_prefill(plan,input_fp16,input_int8,scale_act", source_compact)
        self.assertIn("finish_integer_prefill_overlap(plan.n4,plan.n8,caller_stream,streams)", source_compact)
        self.assertIn("run_fp16_partition_cublas(input_fp16,weight_fp16,indices_fp16,output,plan.rows,plan.width,caller_stream)", source_compact)
        self.assertIn("kCutlassTuningAbi=296", source_compact)

    def test_v275_native_packed_int4_staged_contract(self):
        source_compact = "".join(self.source.split())
        testbed_compact = "".join(self.cutlass_testbed.split())
        mixed_compact = "".join(self.cutlass_sm75_mixed.split())
        linear_compact = "".join(self.linear.split())
        backend_compact = "".join(self.backend.split())
        self.assertIn("structOpMultiplyAddSm75PackedInputUpcast", mixed_compact)
        self.assertIn("MmaTensorOpPolicyK32", mixed_compact)
        self.assertIn("MQMmaPackedInputTensorOpSm75", mixed_compact)
        self.assertIn("MatrixShape<ArchMmaOperator::Shape::kM,32>", mixed_compact)
        self.assertIn("MatrixShape<32,ArchMmaOperator::Shape::kN>", mixed_compact)
        self.assertIn("kASecondK=MmaIterations::kRow", mixed_compact)
        self.assertIn("kBSecondK=MmaIterations::kColumn", mixed_compact)
        self.assertIn("FragmentShuffler<ElementBMma,ElementB,MmaIterations::kColumn,FragmentB::kElements,2*MmaOperandB::kElements", mixed_compact)
        dequantizer_compact = "".join(self.dequantizer.split())
        self.assertIn("constexprintkKGroups", dequantizer_compact)
        self.assertIn("k_group<kKGroups", dequantizer_compact)
        self.assertIn("usingPackedInt4RunnerM64N64=Runner<CorePackedInt4M64N64,2,cutlass::uint4b_t>", testbed_compact)
        self.assertIn("reinterpret_cast<ElementB*>(matrix_B.data_ptr())", testbed_compact)
        self.assertIn("run_cutlass_packed_int4_partition", source_compact)
        self.assertIn("weight_int4_interleaved", source_compact)
        self.assertIn("Tensorweight_int4_interleaved", source_compact)
        self.assertIn("prepare_sm75_prefill_int4", linear_compact)
        self.assertIn("prefill_int4", backend_compact)
        self.assertIn("*arguments[:4],prefill_int4,*arguments[4:]", backend_compact)
        self.assertNotIn("weight_int4_interleaved.data_ptr<int8_t>()", source_compact)

    def test_v200_integer_prefill_overlap_contract(self):
        source_compact = "".join(self.source.split())
        cutlass_compact = "".join(self.cutlass_testbed.split())
        self.assertIn("structIntegerPrefillStreams", source_compact)
        self.assertIn("begin_integer_prefill_overlap", source_compact)
        self.assertIn("finish_integer_prefill_overlap", source_compact)
        self.assertIn("cudaStreamWaitEvent(streams.int4,streams.fork,0)", source_compact)
        self.assertIn("cudaStreamWaitEvent(streams.int8,streams.fork,0)", source_compact)
        self.assertIn("usingInt8Runner=Runner<Core,2>", cutlass_compact)
        self.assertNotIn("WideInt8Runner", source_compact)
        self.assertNotIn("WideCore", cutlass_compact)

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
        self.assertIn("run_fp16_partition_cublas", text)
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