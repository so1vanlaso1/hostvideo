from pathlib import Path
import time
import uuid


class ComfyExecutionError(RuntimeError):
    def __init__(self, details):
        self.details = details
        super().__init__(details.get("exception_message", str(details)))

    @property
    def is_oom(self):
        kind = self.details.get("exception_type", "")
        message = self.details.get("exception_message", "").lower()
        return "OutOfMemoryError" in kind or "cuda out of memory" in message or "cuda error: out of memory" in message


class ComfyClient:
    def __init__(self, url="http://127.0.0.1:8188", username=None, password=None):
        import requests
        self.url = url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["ngrok-skip-browser-warning"] = "1"
        if username:
            self.session.auth = (username, password)
        self.client_id = str(uuid.uuid4())

    def request(self, method, endpoint, **kwargs):
        result = self.session.request(method, self.url + endpoint, timeout=kwargs.pop("timeout", 60), **kwargs)
        if not result.ok:
            raise RuntimeError(f"ComfyUI {endpoint}: HTTP {result.status_code}: {result.text[:3000]}")
        return result

    def check(self):
        info = self.request("GET", "/object_info").json()
        required = {"H3Begin", "H3ReferenceToVideo", "H3Sampler", "H3Scheduler", "H3SaveVideo", "H3LoadReferenceVideo"}
        missing = required - info.keys()
        if missing:
            raise RuntimeError(f"H3 custom nodes not loaded: {sorted(missing)}; use scripts/start_comfy.sh")
        return info

    def upload(self, path, subfolder):
        with Path(path).open("rb") as source:
            data = self.request("POST", "/upload/image", files={"image": (Path(path).name, source)},
                                data={"type": "input", "subfolder": subfolder, "overwrite": "false"}).json()
        return f"{data['subfolder']}/{data['name']}" if data.get("subfolder") else data["name"]

    def execute(self, graph, timeout=7200, progress=None):
        submitted = self.request("POST", "/prompt", json={"prompt": graph, "client_id": self.client_id}).json()
        if submitted.get("node_errors"):
            raise RuntimeError(f"Workflow validation failed: {submitted['node_errors']}")
        prompt_id = submitted["prompt_id"]
        deadline = time.monotonic() + timeout
        last_report = 0
        while time.monotonic() < deadline:
            entry = self.request("GET", f"/history/{prompt_id}").json().get(prompt_id)
            if entry:
                for event, details in entry.get("status", {}).get("messages", []):
                    if event == "execution_error":
                        raise ComfyExecutionError(details)
                    if event == "execution_interrupted":
                        raise RuntimeError("Generation was interrupted in ComfyUI")
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI execution failed: {status}")
                if status.get("completed"):
                    return entry
            if progress and time.monotonic() - last_report > 30:
                progress(f"ComfyUI job {prompt_id} is queued or running")
                last_report = time.monotonic()
            time.sleep(2)
        # Do not interrupt another browser's work or resubmit a potentially running job.
        raise TimeoutError(f"Job {prompt_id} exceeded {timeout}s; it may still be running. Inspect ComfyUI before retrying.")

    def download(self, item, destination):
        with self.request("GET", "/view", params={"filename": item["filename"], "subfolder": item.get("subfolder", ""),
                           "type": item.get("type", "output")}, stream=True) as response:
            with Path(destination).open("wb") as output:
                for block in response.iter_content(1024 * 1024):
                    output.write(block)
        return str(destination)
