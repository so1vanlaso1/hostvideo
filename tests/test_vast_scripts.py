"""Exercise the actual Bash setup/service flow with local substitutes for GPU tools."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]

# The substitutes never install packages or download weights. tmux launches the
# real foreground wrapper, whose mocked Python serves a real HTTP health endpoint.
TOOL = r'''import fcntl
import http.server
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

name = Path(sys.argv[0]).name
args = sys.argv[1:]
work = Path(os.environ["MOCK_WORK"])
with (work / "calls.jsonl").open("a") as output:
    output.write(json.dumps([name, *args]) + "\n")

def venv(path):
    directory = Path(path) / "bin"
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("python", "uv"):
        target = directory / name
        if not target.exists():
            target.symlink_to(work / "bin" / "driver")
    (directory / "activate").write_text("")

if name == "uname":
    print("Linux" if args == ["-s"] else "x86_64")
elif name == "nvidia-smi":
    print(os.environ.get("MOCK_DRIVER", "580.10"))
elif name == "nvcc":
    print("Cuda compilation tools, release " + os.environ.get("MOCK_CUDA", "13.0"))
elif name == "mountpoint":
    sys.exit(0 if os.environ.get("MOCK_MOUNTED_DATA") else 1)
elif name == "git":
    if args[0] == "clone":
        target = Path(args[-1])
        (target / ".git").mkdir(parents=True)
        (target / "custom_nodes").mkdir()
        (target / "requirements.txt").touch()
    elif "get-url" in args:
        print("https://github.com/Comfy-Org/ComfyUI.git")
elif name == "flock":
    unlock = args[0] == "-u"
    operand = args[1] if args[0].startswith("-") else args[0]
    descriptor = int(operand) if operand.isdigit() else os.open(operand, os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN if unlock else fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(1)
elif name == "tmux":
    state = work / (args[1] + ".session")
    action = args[2]
    if action == "has-session":
        if not state.exists():
            sys.exit(1)
        pid, marker = json.loads(state.read_text())
        sys.exit(1 if Path(marker).exists() else 0)
    elif action == "new-session":
        launcher = args[-1]
        marker = str(state) + ".done"
        Path(marker).unlink(missing_ok=True)
        wrapper = 'import pathlib, subprocess, sys; result = subprocess.run(["bash", sys.argv[1]]); pathlib.Path(sys.argv[2]).touch(); sys.exit(result.returncode)'
        process = subprocess.Popen([sys.executable, "-c", wrapper, launcher, marker],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
        state.write_text(json.dumps([process.pid, marker]))
    elif action == "kill-session":
        pid, marker = json.loads(state.read_text())
        try:
            os.killpg(pid, signal.SIGHUP)
        except ProcessLookupError:
            pass
        Path(marker).touch()
        state.unlink()
elif name == "uv":
    if args[0] == "venv":
        venv(args[-1])
elif name in ("python", "python3", "python3.11"):
    if args[:2] == ["-m", "venv"]:
        venv(args[-1])
    elif args[:2] == ["-m", "pip"]:
        if "freeze" in args:
            print("requests==2.32.5")
        if any("sageattention @" in arg for arg in args):
            (work / "sage-built").touch()
    elif args[:2] == ["-m", "h3_pipeline"]:
        if "preflight" in args and os.environ.get("MOCK_PREFLIGHT_FAIL"):
            sys.exit("Mock CUDA preflight failed")
        if "serve" in args:
            if os.environ.get("MOCK_START_FAIL"):
                sys.exit("Mock startup failed")
            port = int(args[args.index("--port") + 1])
            class Handler(http.server.BaseHTTPRequestHandler):
                def do_GET(self):
                    self.send_response(503 if os.environ.get("MOCK_UNHEALTHY") else 200)
                    self.end_headers()
                    self.wfile.write(b'{"system":{}}')
                def log_message(self, *args):
                    pass
            print("Mock ComfyUI startup", flush=True)
            http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()
    elif args and args[0] == "-":
        code = sys.stdin.read()
        if "from h3_pipeline.attention import metadata" in code:
            sys.exit(0 if (work / "sage-built").exists() else 1)
        elif "shutil.disk_usage" in code and os.environ.get("MOCK_LOW_DISK"):
            sys.exit("Need at least 75 GiB free for remaining models")
        elif "shutil.disk_usage" in code:
            pass
        else:
            sys.exit(subprocess.run([sys.executable, *args], input=code, text=True).returncode)
    elif args[:1] == ["-c"] and args[1] == "import venv, ensurepip" and os.environ.get("MOCK_MISSING_VENV"):
        sys.exit(1)
    elif args[:1] == ["-c"] and ("sys.version_info" in args[1] or "import pathlib, sys, h3_pipeline" in args[1]):
        pass
    else:
        sys.exit(subprocess.run([sys.executable, *args]).returncode)
'''


class VastScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.work = Path(self.temp.name)
        self.repo = self.work / "source with spaces"
        shutil.copytree(REPO / "scripts", self.repo / "scripts")
        shutil.copy2(REPO / "setup.sh", self.repo / "setup.sh")
        self.bin = self.work / "bin"
        self.bin.mkdir()
        driver = self.bin / "driver"
        driver.write_text(f"#!{sys.executable}\n" + TOOL)
        driver.chmod(0o755)
        for name in ("uname", "nvidia-smi", "nvcc", "mountpoint", "git", "flock", "tmux", "python3", "python3.11", "g++", "make", "ffmpeg", "ffprobe", "apt-get", "sudo"):
            (self.bin / name).symlink_to(driver)
        self.root = self.work / "runtime with spaces"
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.env = {**os.environ, "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
                    "MOCK_WORK": str(self.work), "H3_ROOT": str(self.root),
                    "H3_PORT": str(self.port), "H3_PROFILE": "int8-encoder", "H3_START_TIMEOUT": "10"}
        self.env.pop("MOCK_PREFLIGHT_FAIL", None)

    def tearDown(self):
        if (self.root / "locks").exists():
            self.run_script("scripts/vast.sh", "stop")
        self.temp.cleanup()

    def run_script(self, script, *args, env=None):
        return subprocess.run(["bash", str(self.repo / script), *args], cwd=self.repo,
                              env=env or self.env, text=True, capture_output=True, timeout=30)

    def calls(self):
        path = self.work / "calls.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def setup_ok(self, *args):
        result = self.run_script("setup.sh", *args)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_default_setup_downloads_starts_and_remembers_settings(self):
        self.setup_ok()
        self.assertTrue(any("download-models" in call for call in self.calls()))
        self.assertTrue((self.root / "logs/setup.log").is_file())
        saved = self.repo / ".h3-vast.env"
        self.assertEqual(saved.stat().st_mode & 0o777, 0o600)
        env = dict(self.env)
        for key in ("H3_ROOT", "H3_PORT", "H3_PROFILE"):
            env.pop(key)
        result = self.run_script("scripts/vast.sh", "status", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(self.port), result.stdout)
        result = self.run_script("scripts/vast.sh", "start")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(sum("new-session" in call for call in self.calls()), 1)
        # Setup must refuse to mutate a live runtime and must not retain its lock.
        result = self.run_script("setup.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Stop ComfyUI", result.stderr)
        self.assertEqual(self.run_script("scripts/vast.sh", "stop").returncode, 0)
        self.assertNotEqual(self.run_script("scripts/vast.sh", "status").returncode, 0)
        self.setup_ok("--no-start")
        builds = [call for call in self.calls() if any("sageattention @" in arg for arg in call)]
        self.assertEqual(len(builds), 1, "matching Sage build should be reused")
        self.assertEqual(self.run_script("scripts/vast.sh", "start").returncode, 0)
        self.assertEqual(self.run_script("scripts/vast.sh", "restart").returncode, 0)

    def test_skip_download_and_cli_overrides(self):
        self.setup_ok("--skip-download", "--profile", "fp8", "--port", str(self.port))
        self.assertFalse(any("download-models" in call or "new-session" in call for call in self.calls()))
        # Environment overrides must beat saved settings when launching later.
        env = {**self.env, "H3_PROFILE": "primary"}
        result = self.run_script("scripts/vast.sh", "start", env=env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        serve = next(call for call in self.calls() if "serve" in call)
        self.assertEqual(serve[serve.index("--profile") + 1], "primary")
        self.assertEqual(self.run_script("scripts/vast.sh", "stop").returncode, 0)

    def test_preflight_failure_never_downloads_or_saves_success(self):
        result = self.run_script("setup.sh", env={**self.env, "MOCK_PREFLIGHT_FAIL": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Mock CUDA preflight failed", result.stdout)
        self.assertFalse(any("download-models" in call or "new-session" in call for call in self.calls()))
        self.assertFalse((self.repo / ".h3-vast.env").exists())

    def test_startup_failure_reports_log_and_can_be_retried(self):
        result = self.run_script("setup.sh", env={**self.env, "MOCK_START_FAIL": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Mock startup failed", result.stdout)
        self.assertNotIn("Setup complete", result.stdout)
        result = self.run_script("scripts/vast.sh", "start")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_wrong_toolkit_and_insufficient_disk_fail_before_installing(self):
        for extra, message in (({"MOCK_CUDA": "12.8"}, "CUDA 13"),
                               ({"MOCK_DRIVER": "570.10"}, "R580"),
                               ({"MOCK_LOW_DISK": "1"}, "GiB free")):
            with self.subTest(extra=extra):
                result = self.run_script("setup.sh", env={**self.env, **extra})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stdout + result.stderr)
        self.assertFalse(any("pip" in call or "clone" in call for call in self.calls()))

    def test_occupied_port_does_not_start_another_server(self):
        self.setup_ok("--no-start")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", self.port))
            sock.listen()
            result = self.run_script("scripts/vast.sh", "start")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already in use", result.stdout + result.stderr)
        self.assertFalse(any("new-session" in call for call in self.calls()))

    def test_help_and_bad_arguments_do_not_run_installation(self):
        self.assertEqual(self.run_script("setup.sh", "--help").returncode, 0)
        self.assertEqual(self.run_script("setup.sh", "--root").returncode, 2)
        self.assertEqual(self.run_script("setup.sh", "--unknown").returncode, 2)
        self.assertEqual(self.calls(), [])

    def test_installs_missing_tools_and_bootstraps_python_with_uv(self):
        (self.bin / "python3.11").unlink()
        env = {**self.env, "MOCK_MISSING_VENV": "1", "PATH": str(self.bin) + ":/usr/bin:/bin"}
        result = self.run_script("setup.sh", "--skip-download", env=env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls()
        self.assertTrue(any("install" in call and "python3-venv" in call for call in calls))
        self.assertTrue(any(call[:4] == ["uv", "python", "install", "3.11"] for call in calls))
        self.assertTrue(any("venv" in call and "--seed" in call for call in calls))

    def test_storage_selection_uses_mounted_data_or_container_fallback(self):
        env = dict(self.env)
        env.pop("H3_ROOT")
        command = ['bash', '-c', 'source "$1"; printf "%s" "$H3_ROOT"',
                   'bash', str(self.repo / 'scripts/runtime_env.sh')]
        mounted = subprocess.run(command, env={**env, "MOCK_MOUNTED_DATA": "1"}, text=True, capture_output=True)
        self.assertEqual(mounted.returncode, 0, mounted.stderr)
        self.assertEqual(mounted.stdout, "/data/minimax-h3")
        fallback = subprocess.run(command, env=env, text=True, capture_output=True)
        self.assertEqual(fallback.returncode, 0, fallback.stderr)
        expected = "/workspace/minimax-h3" if Path("/workspace").is_dir() else str(self.repo / ".runtime")
        self.assertEqual(fallback.stdout, expected)

    def test_health_timeout_reports_failure_and_leaves_session_for_inspection(self):
        self.setup_ok("--no-start")
        env = {**self.env, "MOCK_UNHEALTHY": "1", "H3_START_TIMEOUT": "1"}
        result = self.run_script("scripts/vast.sh", "start", env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("session left running", result.stderr)
        self.assertTrue(list(self.work.glob("*.session")))
        result = self.run_script("scripts/vast.sh", "status")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unhealthy", result.stdout)


if __name__ == "__main__":
    unittest.main()
