import unittest

from h3_pipeline.preflight import cuda_environment_errors


class CudaCompatibilityTests(unittest.TestCase):
    def test_supported_runtimes_and_newer_drivers(self):
        for cuda, driver in (("12.8", "570.124.06"), ("12.9", "575.57.08"),
                             ("12.8", "595.1"), ("13.0", "580.65.06"), ("13.2", "595.1")):
            with self.subTest(cuda=cuda, driver=driver):
                self.assertEqual(cuda_environment_errors(cuda, driver), [])

    def test_rejects_old_runtime_and_cpu_build(self):
        for cuda in (None, "11.8", "12.6", "12.7"):
            with self.subTest(cuda=cuda):
                self.assertIn("12.8", cuda_environment_errors(cuda, "595.1")[0])

    def test_driver_minimum_depends_on_runtime(self):
        self.assertIn("R570", cuda_environment_errors("12.8", "565.1")[0])
        self.assertIn("R580", cuda_environment_errors("13.0", "575.1")[0])

    def test_new_major_requires_matching_stack(self):
        self.assertIn("matching PyTorch stack", cuda_environment_errors("14.0", "600.1")[0])


if __name__ == "__main__":
    unittest.main()
