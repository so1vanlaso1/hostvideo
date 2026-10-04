"""Propagate section and executor failures to the owning dance report."""
from functools import wraps
from pathlib import Path
import json
import logging

from .config import write_json

_prompt_reports = {}


def fail_report(path, error, status="failed", section_index=None, stage=None, server=None):
    path = Path(path)
    report = json.loads(path.read_text())
    if report.get("status") in {"completed", "failed", "interrupted"}:
        return
    report.update(status=status, error=error)
    if stage:
        report["failed_stage"] = stage
    for section in report["sections"]:
        if section.get("status") == "completed":
            continue
        if section["index"] == section_index or section.get("status") == "running":
            section.update(status=status, error=error)
            report["failed_section"] = section["index"]
            if stage:
                section["failed_stage"] = stage
            if server:
                section["server"] = server
        else:
            section["status"] = "cancelled"
    write_json(path, report)


def register_report(path):
    from comfy_execution.utils import get_executing_context
    context = get_executing_context()
    if context is not None:
        _prompt_reports.setdefault(context.prompt_id, set()).add(Path(path))


def install_execution_reporting():
    """Observe the pinned executor's terminal events, including between nodes.

    Native CreateVideo and pre-node interruption can fail outside our stage
    adapters. Match by prompt ID so unrelated queued jobs are never modified.
    """
    import execution
    original = execution.PromptExecutor.add_message
    if getattr(original, "_h3_reporting", False):
        return

    @wraps(original)
    def add_message(executor, event, data, broadcast):
        if event in {"execution_error", "execution_interrupted", "execution_success"}:
            paths = _prompt_reports.pop(data["prompt_id"], set())
            if event != "execution_success":
                status = "interrupted" if event == "execution_interrupted" else "failed"
                error = data.get("exception_message") or "Execution interrupted"
                from . import server_runtime as runtime
                monitor = runtime._active
                if monitor and monitor.data.get("prompt_id") == data["prompt_id"]:
                    try:
                        monitor.data.update(error=error, failed_stage=data.get("node_type", "execution"))
                        runtime.finish(monitor.data["job_id"], status)
                    except Exception:
                        logging.exception("Could not finalize H3 telemetry")
                for path in paths:
                    try:
                        fail_report(path, error, status, stage=data.get("node_type", "execution"))
                    except Exception:
                        logging.exception("Could not update dance report %s", path)
        return original(executor, event, data, broadcast)

    add_message._h3_reporting = True
    execution.PromptExecutor.add_message = add_message
