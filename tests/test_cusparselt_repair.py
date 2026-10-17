import base64
import csv
import hashlib
import importlib.metadata
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.repair_cusparselt import repair_cusparselt


class CusparseltRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.info = self.root / "nvidia_cusparselt_cu13-0.8.1.dist-info"
        self.info.mkdir()
        (self.info / "METADATA").write_text("Name: nvidia-cusparselt-cu13\nVersion: 0.8.1\n")
        (self.info / "WHEEL").write_text("Wheel-Version: 1.0\nTag: py3-none-manylinux2014_sbsa\n")
        (self.info / "RECORD").write_text(
            f"{self.info.name}/WHEEL,sha256=old,1\n{self.info.name}/RECORD,,\n"
        )
        self.distribution = importlib.metadata.PathDistribution(self.info)
        for target, value in (("platform.system", "Linux"), ("platform.machine", "aarch64"),
                              ("importlib.metadata.distribution", self.distribution)):
            mock = patch("scripts.repair_cusparselt." + target, return_value=value)
            mock.start()
            self.addCleanup(mock.stop)
        library_patch = patch("scripts.repair_cusparselt.ctypes.CDLL")
        self.library = library_patch.start()
        self.addCleanup(library_patch.stop)

    def test_repairs_tag_and_record_only_after_library_loads(self):
        self.assertTrue(repair_cusparselt())
        wheel_bytes = (self.info / "WHEEL").read_bytes()
        self.assertIn(b"Tag: py3-none-manylinux2014_aarch64", wheel_bytes)
        with (self.info / "RECORD").open() as record:
            rows = list(csv.reader(record))
        expected_hash = base64.urlsafe_b64encode(hashlib.sha256(wheel_bytes).digest()).decode().rstrip("=")
        self.assertEqual(rows[0][1:], ["sha256=" + expected_hash, str(len(wheel_bytes))])
        self.assertEqual(rows[1], [f"{self.info.name}/RECORD", "", ""])
        self.library.assert_called_once_with(str(self.root / "nvidia/cusparselt/lib/libcusparseLt.so.0"))
        self.assertFalse(repair_cusparselt())
        self.library.assert_called_once()

    def test_library_failure_leaves_metadata_unchanged(self):
        original_wheel = (self.info / "WHEEL").read_bytes()
        original_record = (self.info / "RECORD").read_bytes()
        self.library.side_effect = OSError("Cannot load library")
        with self.assertRaises(OSError):
            repair_cusparselt()
        self.assertEqual((self.info / "WHEEL").read_bytes(), original_wheel)
        self.assertEqual((self.info / "RECORD").read_bytes(), original_record)

    def test_other_architectures_and_versions_are_unchanged(self):
        with patch("scripts.repair_cusparselt.platform.machine", return_value="x86_64"):
            self.assertFalse(repair_cusparselt())
        (self.info / "METADATA").write_text("Name: nvidia-cusparselt-cu13\nVersion: 0.9.0\n")
        self.assertFalse(repair_cusparselt())
        self.library.assert_not_called()

    def test_missing_package_is_unchanged(self):
        with patch("scripts.repair_cusparselt.importlib.metadata.distribution",
                   side_effect=importlib.metadata.PackageNotFoundError):
            self.assertFalse(repair_cusparselt())
        self.library.assert_not_called()
