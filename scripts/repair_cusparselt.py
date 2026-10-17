import base64
import csv
import ctypes
import hashlib
import importlib.metadata
import io
import platform


def repair_cusparselt():
    if platform.system() != "Linux" or platform.machine() not in {"aarch64", "arm64"}:
        return False
    try:
        distribution = importlib.metadata.distribution("nvidia-cusparselt-cu13")
    except importlib.metadata.PackageNotFoundError:
        return False
    if distribution.version != "0.8.1":
        return False
    wheel_text = distribution.read_text("WHEEL") or ""
    bad_tag = "Tag: py3-none-manylinux2014_sbsa"
    if bad_tag not in wheel_text.splitlines():
        return False
    wheel_file = next(path for path in distribution.files if path.parts[-1] == "WHEEL"
                      and path.parts[-2].endswith(".dist-info"))
    record_file = wheel_file.parent / "RECORD"
    wheel_path = distribution.locate_file(wheel_file)
    record_path = distribution.locate_file(record_file)
    wheel_bytes = wheel_text.replace(bad_tag, "Tag: py3-none-manylinux2014_aarch64").encode()
    rows = list(csv.reader(io.StringIO(record_path.read_text())))
    wheel_row = next(row for row in rows if row[0] == str(wheel_file))
    wheel_row[1] = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(wheel_bytes).digest()).decode().rstrip("=")
    wheel_row[2] = str(len(wheel_bytes))
    record_text = io.StringIO(newline="")
    csv.writer(record_text).writerows(rows)
    ctypes.CDLL(str(distribution.locate_file("nvidia/cusparselt/lib/libcusparseLt.so.0")))
    wheel_path.write_bytes(wheel_bytes)
    record_path.write_text(record_text.getvalue())
    print("Repaired nvidia-cusparselt-cu13 0.8.1 ARM64 wheel tag (sbsa -> aarch64).")
    return True


if __name__ == "__main__":
    repair_cusparselt()
