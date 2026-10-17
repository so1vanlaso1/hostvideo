import os
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from h3_pipeline.config import RuntimePaths
from h3_pipeline.hardware import device_info, is_gb10, runtime_policy
from h3_pipeline.preflight import preflight
from h3_pipeline.server import command


GIB = 1024**3


class HardwareTests(unittest.TestCase):
    def test_gb10_identification_does_not_treat_every_arm_gpu_as_gb10(self):
        self.assertTrue(is_gb10("NVIDIA GB10", machine="aarch64"))
        self.assertTrue(is_gb10("Blackwell", (12, 1), machine="arm64"))
        self.assertFalse(is_gb10("RTX 5060 Ti", (12, 0), machine="x86_64"))
        self.assertFalse(is_gb10("NVIDIA GH200", (9, 0), machine="aarch64"))

    def test_startup_policy_uses_shared_memory_headroom_and_unpinned_buffers(self):
        paths = RuntimePaths(Path("/tmp/h3-test"))
        with patch.dict(os.environ, {}, clear=True):
            cmd = command(paths, hardware={"shared_memory": True})
            self.assertIn("--disable-dynamic-vram", cmd)
            self.assertNotIn("--enable-dynamic-vram", cmd)
            self.assertIn("--disable-pinned-memory", cmd)
            self.assertEqual(float(cmd[cmd.index("--reserve-vram") + 1]), 12)
            self.assertNotIn("--highvram", cmd)
            self.assertNotIn("--gpu-only", cmd)
            discrete = command(paths, hardware={"shared_memory": False})
            self.assertIn("--enable-dynamic-vram", discrete)
            self.assertNotIn("--disable-pinned-memory", discrete)
            self.assertEqual(float(discrete[discrete.index("--reserve-vram") + 1]), 1)

    def test_policy_overrides_and_invalid_values(self):
        self.assertEqual(runtime_policy({"shared_memory": True}, {"H3_MEMORY_MODE": "dynamic", "H3_RESERVE_VRAM": "16.5"}),
                         {"memory_mode": "dynamic", "reserve_vram_gib": 16.5, "disable_pinned_memory": True})
        for env in ({"H3_MEMORY_MODE": "highvram"}, {"H3_RESERVE_VRAM": "nan"},
                    {"H3_RESERVE_VRAM": "inf"}, {"H3_RESERVE_VRAM": "-1"}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                runtime_policy({}, env)

    def fake_torch(self, cuda="13.0"):
        torch = MagicMock()
        torch.__version__ = "2.13.0+cu130"
        torch.version.cuda = cuda
        torch.cuda.is_available.return_value = True
        torch.cuda.current_device.return_value = 0
        torch.cuda.get_device_name.return_value = "NVIDIA GB10"
        torch.cuda.get_device_capability.return_value = (12, 1)
        torch.cuda.get_device_properties.return_value = SimpleNamespace(total_memory=119 * GIB, is_integrated=True)
        return torch

    def run_preflight(self, *, cuda="13.0", reserve="12", ram=125 * GIB, machine="aarch64"):
        with tempfile.TemporaryDirectory() as root, \
             patch.dict("sys.modules", {"torch": self.fake_torch(cuda)}), \
             patch.dict(os.environ, {"H3_RESERVE_VRAM": reserve, "H3_MEMORY_MODE": "auto"}), \
             patch("h3_pipeline.preflight.platform.machine", return_value=machine), \
             patch("h3_pipeline.preflight.platform.system", return_value="Linux"), \
             patch("h3_pipeline.preflight.platform.platform", return_value="Linux-aarch64"), \
             patch("h3_pipeline.preflight.allocated_ram", return_value=ram), \
             patch("h3_pipeline.preflight.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("h3_pipeline.preflight.subprocess.run", return_value=SimpleNamespace(stdout="580.65.06\n")), \
             patch("h3_pipeline.preflight.kernel_smoke", return_value={"int8_convrot": "passed"}) as kernels, \
             patch("h3_pipeline.preflight.attention_smoke", return_value={"checks": ["passed"]}):
            report = preflight(root, kernels=True)
            return report, kernels.call_count

    def test_gb10_preflight_uses_one_pool_and_runs_kernel_checks(self):
        report, count = self.run_preflight()
        self.assertTrue(report["passed"], report["errors"])
        self.assertTrue(report["gb10"])
        self.assertEqual(report["shared_memory_pool_bytes"], 119 * GIB)
        self.assertEqual(report["runtime_policy"]["memory_mode"], "static")
        self.assertIn("not additive", report["memory_accounting"])
        self.assertFalse(any("5060" in message for message in report["warnings"]))
        self.assertEqual(count, 1)

    def test_invalid_gb10_runtime_or_emulation_never_runs_kernels(self):
        for extra, message in (({"cuda": "12.8"}, "CUDA 13"), ({"machine": "x86_64"}, "native Linux ARM64")):
            with self.subTest(extra=extra):
                report, count = self.run_preflight(**extra)
                self.assertFalse(report["passed"])
                self.assertTrue(any(message in error for error in report["errors"]))
                self.assertEqual(count, 0)

    def test_shared_budget_respects_container_allocation_and_headroom(self):
        report, _ = self.run_preflight(ram=80 * GIB)
        self.assertEqual(report["shared_memory_pool_bytes"], 80 * GIB)
        self.assertTrue(any("96 GiB" in message for message in report["warnings"]))
        report, count = self.run_preflight(reserve="128")
        self.assertFalse(report["passed"])
        self.assertIn("H3_RESERVE_VRAM", report["errors"][0])
        self.assertEqual(count, 0)

    def test_cuda_visible_integrated_property_is_optional(self):
        torch = self.fake_torch()
        torch.cuda.get_device_properties.return_value = SimpleNamespace(total_memory=119 * GIB)
        self.assertTrue(device_info(torch)["shared_memory"])

    def test_shared_memory_telemetry_is_serializable_and_tracks_pressure(self):
        from h3_pipeline.server_runtime import PerformanceMonitor
        torch = self.fake_torch()
        torch.cuda.max_memory_allocated.return_value = 20 * GIB
        torch.cuda.max_memory_reserved.return_value = 24 * GIB
        process = MagicMock()
        process.memory_info.return_value.rss = 8 * GIB
        process.children.return_value = []
        with tempfile.TemporaryDirectory() as root, \
             patch.dict("sys.modules", {"torch": torch}), \
             patch.dict(os.environ, {"H3_MEMORY_MODE": "auto", "H3_RESERVE_VRAM": "12"}), \
             patch("psutil.Process", return_value=process), \
             patch("psutil.virtual_memory", side_effect=[SimpleNamespace(available=60 * GIB), SimpleNamespace(available=35 * GIB)]), \
             patch("h3_pipeline.server_runtime.threading.Thread"):
            monitor = PerformanceMonitor("test", root, 0)
            report = monitor.finish()
            self.assertEqual(report["system_available_memory_min_bytes"], 35 * GIB)
            self.assertEqual(report["gpu_total_vram_bytes"], 119 * GIB)
            self.assertEqual(report["gpu_peak_reserved_bytes"], 24 * GIB)
            self.assertEqual(report["system_ram_peak_bytes"], 8 * GIB)
            self.assertEqual(report["runtime_policy"]["memory_mode"], "static")
            self.assertIn("do not add", report["memory_accounting"])
            self.assertEqual(json.loads(monitor.path.read_text()), report)


if __name__ == "__main__":
    unittest.main()
