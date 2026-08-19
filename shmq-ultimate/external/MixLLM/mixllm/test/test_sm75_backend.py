import os
import unittest
from unittest import mock

import torch

from mixllm.nn.modules.three_level_linear import ThreeLevelLinear
from mixllm.quantization.three_level import ThreeLevelAllocation, ThreeLevelBudget
from mixllm import sm75_backend
from mixllm.sm75_backend import (
    load_sm75_backend,
    quantize_activation,
    quantize_activation_native,
    quantized_reference,
    three_level_linear,
    three_level_linear_prequantized,
)



class SM75PythonDispatchTest(unittest.TestCase):

    def _module(self, counts):
        n4, n8, n16 = counts
        indices = {}
        start = 0
        for bit, count in ((4, n4), (8, n8), (16, n16)):
            indices[bit] = tuple(range(start, start + count))
            start += count
        allocation = ThreeLevelAllocation(indices=indices, scores={bit: (0.0,) * start for bit in (4, 8, 16)}, budget=ThreeLevelBudget(100, 0, 0))
        return ThreeLevelLinear.from_weight(torch.randn(start, 128), allocation)

    def test_empty_rows_bypass_native_operators(self):
        module = self._module((1, 1, 1))
        with mock.patch.object(sm75_backend, '_LOADED', True), mock.patch.object(sm75_backend, 'quantize_activation_native') as quantize, mock.patch.object(sm75_backend, 'three_level_linear_prequantized') as gemm:
            actual = three_level_linear(module, torch.empty(0, 128), torch)
        self.assertEqual(actual.shape, (0, 3))
        self.assertEqual(actual.dtype, torch.float32)
        quantize.assert_not_called()
        gemm.assert_not_called()

    def test_zero_row_quantizers_return_consistent_shapes(self):
        x = torch.empty(0, 256, dtype=torch.float16)
        reference = quantize_activation(x, torch)
        with mock.patch.object(sm75_backend, "_LOADED", True):
            native = quantize_activation_native(x, torch)
        for quantized, scales in (reference, native):
            self.assertEqual(quantized.shape, (0, 256))
            self.assertEqual(scales.shape, (2, 0))

    def test_prefill_int4_expansion_is_signed_cached_and_invalidated(self):
        module = self._module((2, 0, 0))
        x = torch.empty(2, 128, dtype=torch.float16)

        first = sm75_backend._expanded_int4_for_prefill(module, x, torch)
        codes = torch.empty(2, 128, dtype=torch.uint8)
        codes[:, 0::2] = module.weight_int4 & 0x0f
        codes[:, 1::2] = module.weight_int4 >> 4
        expected = (
            codes.to(torch.int16)
            - module.zero_int4.repeat_interleave(128, dim=1).to(torch.int16)
        ).to(torch.int8)
        torch.testing.assert_close(first, expected, rtol=0, atol=0)

        second = sm75_backend._expanded_int4_for_prefill(module, x, torch)
        self.assertIs(first, second)
        module.weight_int4[0, 0] ^= 0x0f
        third = sm75_backend._expanded_int4_for_prefill(module, x, torch)
        self.assertIsNot(first, third)

    def test_decode_keeps_packed_int4_and_does_not_expand(self):
        module = self._module((2, 0, 0))
        placeholder = sm75_backend._expanded_int4_for_prefill(
            module, torch.empty(1, 128, dtype=torch.float16), torch,
        )

        self.assertEqual(placeholder.shape, (0, 128))
        self.assertEqual(placeholder.data_ptr(), module.weight_int8.data_ptr())
        self.assertIsNone(module._sm75_int4_expanded)

    def test_rejects_malformed_partitions_and_invalidates_cache(self):
        module = self._module((1, 1, 1))
        x = torch.randn(2, 128, dtype=torch.float16)
        input_int8 = torch.empty_like(x, dtype=torch.int8)
        scale_act = torch.empty(1, 2, dtype=torch.float16)
        malformed = (
            ((0,), (0,), (2,)),
            ((0,), (1,), ()),
            ((0,), (1,), (3,)),
        )
        with mock.patch.object(sm75_backend, "_LOADED", True):
            for indices in malformed:
                with self.subTest(indices=indices):
                    module.indices_4 = torch.tensor(indices[0], dtype=torch.int32)
                    module.indices_8 = torch.tensor(indices[1], dtype=torch.int32)
                    module.indices_16 = torch.tensor(indices[2], dtype=torch.int32)
                    with self.assertRaisesRegex(ValueError, "complete output partition"):
                        three_level_linear_prequantized(
                            module, x, input_int8, scale_act, torch,
                        )

    def test_partition_validation_cache_tracks_in_place_mutation(self):
        module = self._module((1, 1, 1))
        x = torch.randn(2, 128, dtype=torch.float16)
        input_int8 = torch.empty_like(x, dtype=torch.int8)
        scale_act = torch.empty(1, 2, dtype=torch.float16)
        operator = mock.Mock(return_value=torch.empty(2, 3))
        with mock.patch.object(sm75_backend, "_LOADED", True), mock.patch.object(
                torch.ops.mixllm_sm75, "_three_level_linear_v2_unchecked",
                operator, create=True):
            three_level_linear_prequantized(
                module, x, input_int8, scale_act, torch,
            )
            module.indices_8.copy_(module.indices_4)
            with self.assertRaisesRegex(ValueError, "duplicates"):
                three_level_linear_prequantized(
                    module, x, input_int8, scale_act, torch,
                )

    def test_validates_prequantized_tensor_devices(self):
        module = self._module((1, 1, 1))
        x = torch.randn(2, 128, dtype=torch.float16)
        input_int8 = torch.empty(2, 128, device="meta", dtype=torch.int8)
        scale_act = torch.empty(1, 2, dtype=torch.float16)
        with mock.patch.object(sm75_backend, "_LOADED", True):
            with self.assertRaisesRegex(ValueError, "all operator tensors"):
                three_level_linear_prequantized(
                    module, x, input_int8, scale_act, torch,
                )
    def test_pure_fp16_skips_quantization_and_reuses_placeholders(self):
        module = self._module((0, 0, 3))
        x = torch.randn(2, 128, dtype=torch.float16)
        with mock.patch.object(sm75_backend, '_LOADED', True), mock.patch.object(sm75_backend, 'quantize_activation_native') as quantize, mock.patch.object(sm75_backend, 'three_level_linear_prequantized', return_value=torch.empty(2, 3)) as gemm:
            three_level_linear(module, x, torch)
            first = gemm.call_args.args[2:4]
            three_level_linear(module, x, torch)
            second = gemm.call_args.args[2:4]
        quantize.assert_not_called()
        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])

    def test_quantized_and_mixed_partitions_quantize_once(self):
        for counts in ((3, 0, 0), (0, 3, 0), (1, 1, 1), (1, 0, 2)):
            module = self._module(counts)
            x = torch.randn(2, 128, dtype=torch.float16)
            quantized = torch.empty_like(x, dtype=torch.int8)
            scales = torch.empty(1, 2, dtype=torch.float16)
            with mock.patch.object(sm75_backend, '_LOADED', True), mock.patch.object(sm75_backend, 'quantize_activation_native', return_value=(quantized, scales)) as quantize, mock.patch.object(sm75_backend, 'three_level_linear_prequantized', return_value=torch.empty(2, 3)) as gemm:
                three_level_linear(module, x, torch)
            quantize.assert_called_once_with(x, torch, 128)
            self.assertIs(gemm.call_args.args[2], quantized)
            self.assertIs(gemm.call_args.args[3], scales)

@unittest.skipUnless(
    os.environ.get("MIXLLM_TEST_SM75") == "1" and torch.cuda.is_available(),
    "requires an explicit SM75 GPU test run",
)
class SM75BackendTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if tuple(torch.cuda.get_device_capability()) != (7, 5):
            raise unittest.SkipTest("requires compute capability 7.5")
        load_sm75_backend(torch)

    def _run_case(self, counts, rows=3, width=128, seed=31):
        torch.manual_seed(seed)
        n4, n8, n16 = counts
        output_width = sum(counts)
        order = torch.randperm(output_width).tolist()
        allocation = ThreeLevelAllocation(
            indices={
                4: tuple(sorted(order[:n4])),
                8: tuple(sorted(order[n4:n4 + n8])),
                16: tuple(sorted(order[n4 + n8:])),
            },
            scores={bit: (0.0,) * output_width for bit in (4, 8, 16)},
            budget=ThreeLevelBudget(100, 0, 0),
        )
        weight = torch.randn(output_width, width, device="cuda", dtype=torch.float16)
        x = torch.randn(rows, width, device="cuda", dtype=torch.float16)
        module = ThreeLevelLinear.from_weight(weight, allocation).cuda()

        actual = three_level_linear(module, x, torch)
        expected = quantized_reference(module, x, torch)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        self.assertTrue(torch.isfinite(actual).all())

    def test_native_activation_quantizer_matches_reference(self):
        for rows, width in ((1, 128), (5, 512), (32, 1024)):
            with self.subTest(rows=rows, width=width):
                torch.manual_seed(rows * 1000 + width)
                x = torch.randn(rows, width, device="cuda", dtype=torch.float16)
                actual_q, actual_scale = quantize_activation_native(x, torch)
                expected_q, expected_scale = quantize_activation(x, torch)
                torch.testing.assert_close(actual_scale, expected_scale, rtol=2e-3,
                                           atol=2e-5)
                self.assertLessEqual(
                    int((actual_q.to(torch.int16) - expected_q.to(torch.int16))
                        .abs().max().item()),
                    1,
                )

    def test_mixed_and_empty_partitions(self):
        for counts in ((5, 3, 2), (10, 0, 0), (0, 10, 0), (0, 0, 10),
                       (0, 4, 6), (7, 0, 3)):
            with self.subTest(counts=counts):
                self._run_case(counts)

    def test_int4_tile_width_boundaries(self):
        for channels in (1, 15, 16, 17, 63, 64, 65):
            with self.subTest(channels=channels):
                self._run_case((channels, 0, 0), rows=3,
                               seed=7000 + channels)
    def test_random_rows_widths_and_determinism(self):
        for rows, width in ((1, 128), (5, 128), (3, 256), (7, 384)):
            with self.subTest(rows=rows, width=width):
                self._run_case((4, 3, 3), rows=rows, width=width,
                               seed=rows * 1000 + width)

        allocation = ThreeLevelAllocation(
            indices={4: (1,), 8: (2,), 16: (0,)},
            scores={bit: (0.0,) * 3 for bit in (4, 8, 16)},
            budget=ThreeLevelBudget(100, 0, 0),
        )
        weight = torch.randn(3, 128, device="cuda", dtype=torch.float16)
        x = torch.randn(3, 128, device="cuda", dtype=torch.float16)
        module = ThreeLevelLinear.from_weight(weight, allocation).cuda()
        first = three_level_linear(module, x, torch)
        second = three_level_linear(module, x, torch)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_non_default_stream_dependency(self):
        allocation = ThreeLevelAllocation(
            indices={4: (1,), 8: (2,), 16: (0,)},
            scores={bit: (0.0,) * 3 for bit in (4, 8, 16)},
            budget=ThreeLevelBudget(34, 33, 33),
        )
        module = ThreeLevelLinear.from_weight(
            torch.randn(3, 128, device="cuda", dtype=torch.float16), allocation,
        ).cuda()
        producer = torch.cuda.Stream()
        consumer = torch.cuda.current_stream()
        with torch.cuda.stream(producer):
            x = torch.randn(5, 128, device="cuda", dtype=torch.float16)
            ready = torch.cuda.Event()
            ready.record()
        consumer.wait_event(ready)
        actual = three_level_linear(module, x, torch)
        expected = quantized_reference(module, x, torch)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_cuda_graph_capture(self):
        allocation = ThreeLevelAllocation(
            indices={4: (0, 3), 8: (1,), 16: (2,)},
            scores={bit: (0.0,) * 4 for bit in (4, 8, 16)},
            budget=ThreeLevelBudget(50, 25, 25),
        )
        module = ThreeLevelLinear.from_weight(
            torch.randn(4, 128, device="cuda", dtype=torch.float16), allocation,
        ).cuda()
        static_input = torch.randn(2, 128, device="cuda", dtype=torch.float16)
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                three_level_linear(module, static_input, torch)
        torch.cuda.current_stream().wait_stream(side_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = three_level_linear(module, static_input, torch)
        graph.replay()
        expected = quantized_reference(module, static_input, torch)
        torch.testing.assert_close(captured, expected, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
