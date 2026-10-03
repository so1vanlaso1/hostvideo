from contextlib import contextmanager
from pathlib import Path


@contextmanager
def file_lock(path):
    """Fail quickly if another client is submitting/retrying jobs on this host."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another H3 operation holds {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
