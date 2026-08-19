import unittest

from mixllm.runtime_capability import RuntimeCapability


class RuntimeCapabilityTest(unittest.TestCase):

    def test_t4_uses_reference_until_backend_is_validated(self):
        self.assertEqual(RuntimeCapability(7, 5).select_backend(), "reference")

    def test_auto_selects_validated_t4_backend(self):
        self.assertEqual(RuntimeCapability(7, 5, sm75_available=True).select_backend(), "sm75")

    def test_auto_selects_ampere_backend(self):
        self.assertEqual(RuntimeCapability(8, 0).select_backend(), "ampere")

    def test_unsupported_gpu_uses_reference(self):
        self.assertEqual(RuntimeCapability(7, 0).select_backend(), "reference")

    def test_rejects_forced_ampere_on_t4(self):
        with self.assertRaises(RuntimeError):
            RuntimeCapability(7, 5, "ampere").select_backend()

    def test_rejects_unavailable_sm75_backend(self):
        with self.assertRaises(RuntimeError):
            RuntimeCapability(7, 5, "sm75").select_backend()

    def test_rejects_unknown_backend(self):
        with self.assertRaises(ValueError):
            RuntimeCapability(8, 0, "unknown").select_backend()


if __name__ == "__main__":
    unittest.main()