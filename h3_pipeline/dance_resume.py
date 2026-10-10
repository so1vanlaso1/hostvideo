"""Selective rerender dependencies and explicit, artifact-bound visual review."""
import json
from pathlib import Path
import shutil
import time
import uuid

from .config import write_json
from .dance_quality import sha256, approved


def selected_sections(report, selected):
    if not isinstance(selected, list) or any(isinstance(i, bool) or not isinstance(i, int) for i in selected):
        raise ValueError("sections must be an array of integer section indices")
    entries = {s["index"]: s for s in report["sections"]}
    if set(selected) - entries.keys():
        raise ValueError("Unknown section index")
    if not selected:
        selected = [s["index"] for s in entries.values() if s.get("status") != "completed" or (s.get("quality", {}).get("status") == "flagged" and not approved(s))]
    chosen = set(selected)
    # Same-outfit chunks carry generated future-context poses. Regenerate the
    # downstream chunks of THAT outfit, not unrelated outfits.
    for s in entries.values():
        if s.get("kind") == "transition":
            if s["left_section"] in chosen or s["right_section"] in chosen:
                chosen.add(s["index"])
        elif any(entries[i].get("kind") != "transition" and entries[i]["outfit_index"] == s["outfit_index"] and i < s["index"] for i in chosen):
            chosen.add(s["index"])
        if s["index"] not in chosen and not Path(s.get("video_path", "")).is_file():
            chosen.add(s["index"])
    return chosen


def archive_section(work, index):
    work = Path(work).resolve()
    section = work / f"section-{index}"
    if not section.is_dir() or not section.resolve().is_relative_to(work):
        raise ValueError("Missing section directory")
    archive = section / "attempts" / (time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:8])
    archive.mkdir(parents=True)
    for file in section.iterdir():
        if file.name == "attempts":
            continue
        if file.is_dir():
            shutil.copytree(file, archive / file.name)
        else:
            shutil.copy2(file, archive / file.name)
    shutil.copy2(work / "report.json", archive / "job-report.json")
    return archive


def archive_job(work):
    work = Path(work)
    archive = work / "attempts" / (time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:8])
    archive.mkdir(parents=True)
    for name in ("report.json", "generated.mp4", "review.mp4", "review.html", "quality.json", "expanded-workflow-api.json", "resume-workflow-api.json"):
        if (work / name).is_file():
            shutil.copy2(work / name, archive / name)
    return archive


def approve_sections(report_path, indices, note):
    if not note.strip():
        raise ValueError("A review note is required to approve a section")
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    if report["status"] in {"planned", "running"}:
        raise ValueError("Wait until the active job finishes before reviewing")
    entries = {s["index"]: s for s in report["sections"]}
    if not indices or set(indices) - entries.keys():
        raise ValueError("Select existing section indices to approve")
    for index in indices:
        section = entries[index]
        if section.get("status") != "completed":
            raise ValueError(f"Section {index} did not finish rendering")
        section["approval"] = {"accepted": True, "note": note, "sha256": sha256(section["video_path"]),
                               "reviewed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(report_path, report)
    return report
