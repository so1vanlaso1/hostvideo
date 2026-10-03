from hashlib import sha256
from pathlib import Path
import shutil
import time

from .config import DEFAULT_PROFILE, MODEL_REPO, MODEL_REVISION, model_files, write_json


def digest(path):
    result = sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class ModelManager:
    def __init__(self, root, profile=DEFAULT_PROFILE):
        self.root = Path(root)
        self.profile = profile
        self.files = model_files(profile)

    def verify(self, hashes=False):
        results = []
        for item in self.files.values():
            path = self.root / item["path"]
            valid = path.is_file() and path.stat().st_size == item["size"]
            if valid and hashes:
                valid = digest(path) == item["sha256"]
            results.append({"path": str(path), "valid": valid, "hash_checked": hashes})
        return results

    def download(self, fetch=None, verify_existing=False):
        """HF's local-directory cache resumes partial transfers; verify before use."""
        from .locking import file_lock
        self.root.mkdir(parents=True, exist_ok=True)
        with file_lock(self.root / ".download.lock"):
            return self._download(fetch, verify_existing)

    def _download(self, fetch, verify_existing):
        receipt_path = self.root / ".h3-verified.json"
        import json
        receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        missing_bytes = 0
        pending = []
        for item in self.files.values():
            path = self.root / item["path"]
            if path.exists():
                stat = path.stat()
                if stat.st_size != item["size"]:
                    raise RuntimeError(f"Wrong-sized model {path}; move it aside before retrying")
                previous = receipts.get(item["path"], {})
                trusted = previous.get("sha256") == item["sha256"] and previous.get("mtime_ns") == stat.st_mtime_ns
                if verify_existing or not trusted:
                    if digest(path) != item["sha256"]:
                        raise RuntimeError(f"SHA256 mismatch for {path}; move it aside before retrying")
                    receipts[item["path"]] = {"sha256": item["sha256"], "mtime_ns": stat.st_mtime_ns}
                    write_json(receipt_path, receipts)
            else:
                pending.append(item)
                missing_bytes += item["size"]
        # Keep 10 GiB for dependency caches, video buffers and outputs.
        required = missing_bytes + 10 * 1024**3
        if shutil.disk_usage(self.root).free < required:
            raise RuntimeError(f"Need {required / 1024**3:.1f} GiB free on {self.root} for remaining models and headroom")
        if fetch is None:
            from huggingface_hub import hf_hub_download
            fetch = hf_hub_download
        for item in pending:
            print(f"Downloading {item['path']} ({item['size'] / 1024**3:.2f} GiB)", flush=True)
            path = Path(fetch(repo_id=MODEL_REPO, revision=MODEL_REVISION, filename=item["path"], local_dir=str(self.root)))
            if path.stat().st_size != item["size"] or digest(path) != item["sha256"]:
                raise RuntimeError(f"Download integrity check failed: {path}")
            receipts[item["path"]] = {"sha256": item["sha256"], "mtime_ns": path.stat().st_mtime_ns, "verified_at": time.time()}
            write_json(receipt_path, receipts)
        return self.verify()
