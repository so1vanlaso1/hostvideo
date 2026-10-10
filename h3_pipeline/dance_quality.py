"""Synchronized visual review and optional CPU pose checks.

Pose scores flag choreography mismatch, not garment/identity quality. Missing
or occluded joints are inconclusive rather than a passing result. All times
are relative to the normalized 24 fps timeline, with no temporal warping.
"""
import hashlib
import json
import math
from pathlib import Path
import urllib.request

from .media import executable, run
from .references import FPS

POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def download_pose_model(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(".download")
        try:
            urllib.request.urlretrieve(POSE_MODEL_URL, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    digest = sha256(destination)
    expected = "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"
    if digest != expected:
        raise ValueError("Pose model checksum differs from the verified version-1 download; inspect the file")
    return {"path": str(destination.resolve()), "url": POSE_MODEL_URL, "sha256": digest}


def require_pose(model):
    if not model or not Path(model).is_file():
        raise ValueError("Pose quality checks require pose_model. Run h3 dance-quality-setup --directory PATH and select its .task file.")
    try:
        import cv2
        import mediapipe
        import numpy
    except ImportError as exc:
        raise RuntimeError('Install CPU quality dependencies in the server environment: pip install -e ".[dance-quality]"') from exc


def compare_poses(source, generated, *, visibility=0.5, error_threshold=0.35,
                  velocity_threshold=0.25, min_coverage=0.65):
    """Compare normalized (x,y,visibility) joints on the EXACT same frames.

    Keep image-space translation and scale: centering each dancer separately
    would conceal wrong travel or foot placement. Distances use source torso
    length so output resolution does not change the interpretation.
    """
    import numpy as np
    a, b = np.asarray(source, dtype=float), np.asarray(generated, dtype=float)
    if a.shape != b.shape or a.ndim != 3 or a.shape[1:] != (33, 3):
        raise ValueError("Pose arrays must have identical [frames,33,3] shapes")
    if not len(a):
        return {"status": "inconclusive", "reason": "no frames"}
    joints = np.arange(11, 33)
    valid = (a[:, joints, 2] >= visibility) & (b[:, joints, 2] >= visibility)
    valid &= np.isfinite(a[:, joints, :2]).all(-1) & np.isfinite(b[:, joints, :2]).all(-1)
    torso = np.linalg.norm((a[:, 11, :2] + a[:, 12, :2] - a[:, 23, :2] - a[:, 24, :2]) / 2, axis=-1)
    torso_valid = np.isfinite(torso) & (torso > 0.025) & (a[:, [11, 12, 23, 24], 2] >= visibility).all(-1)
    valid &= torso_valid[:, None]
    errors = np.linalg.norm(a[:, joints, :2] - b[:, joints, :2], axis=-1) / np.maximum(torso[:, None], 0.025)
    # At least eight comparable body joints per frame, including torso scale.
    usable = valid.sum(-1) >= 8
    coverage = float(usable.mean())
    distances = errors[valid & usable[:, None]]
    frame_errors = [float(np.median(errors[i, valid[i]])) if usable[i] else None for i in range(len(a))]
    bad_frames = [i for i, value in enumerate(frame_errors) if value is not None and value > error_threshold]
    pair_valid = valid[1:] & valid[:-1] & usable[1:, None] & usable[:-1, None]
    velocities = np.linalg.norm(np.diff(a[:, joints, :2], axis=0) - np.diff(b[:, joints, :2], axis=0), axis=-1)
    velocities /= np.maximum(torso[1:, None], 0.025)
    speed_error = float(np.quantile(velocities[pair_valid], 0.9)) if pair_valid.any() else None
    velocity_frames = [i + 1 for i in range(len(velocities)) if pair_valid[i].sum() >= 8 and
                       float(np.median(velocities[i, pair_valid[i]])) > velocity_threshold * 3]
    result = {"status": "inconclusive", "comparable_frame_fraction": coverage,
              "median_pose_error": float(np.median(distances)) if len(distances) else None,
              "p90_velocity_error": speed_error, "bad_frames": bad_frames,
              "severe_jump_frames": velocity_frames,
              "frame_pose_error": frame_errors,
              "thresholds": {"pose_error": error_threshold, "velocity_error": velocity_threshold,
                             "min_coverage": min_coverage},
              "scope": "Motion screening only. Garment flicker, identity and transition quality need visual review."}
    if coverage < min_coverage or speed_error is None:
        result["reason"] = "Insufficient confident body joints or consecutive frames"
    else:
        result["status"] = "fail" if len(bad_frames) / max(1, int(usable.sum())) > 0.15 or speed_error > velocity_threshold or velocity_frames else "pass"
    return result


def extract_poses(video, model):
    require_pose(model)
    import cv2
    import mediapipe as mp
    import numpy as np
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=1)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not decode pose-check video: {video}")
    poses = []
    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect_for_video(image, round(len(poses) * 1000 / FPS))
                if result.pose_landmarks:
                    poses.append([[p.x, p.y, p.visibility] for p in result.pose_landmarks[0]])
                else:
                    poses.append([[math.nan, math.nan, 0]] * 33)
    finally:
        capture.release()
    return np.asarray(poses, dtype=float).reshape(-1, 33, 3)


def review_section(source, generated, folder, *, pose_model=None, mode="preview"):
    from .media import probe
    from .config import write_json
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    info = probe(generated)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    # Both clips have already been trimmed to the identical core interval.
    height = min(768, int(video["height"]))
    height -= height % 2
    preview = folder / "review.mp4"
    run([executable("ffmpeg"), "-nostdin", "-v", "error", "-y", "-i", str(source), "-i", str(generated),
         "-filter_complex", f"[0:v]scale=-2:{height},setsar=1,setpts=PTS-STARTPTS[a];[1:v]scale=-2:{height},setsar=1,setpts=PTS-STARTPTS[b];[a][b]hstack=shortest=1[v]",
         "-map", "[v]", "-map", "0:a:0?", "-c:v", "libx264", "-crf", "18", "-preset", "fast",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(preview)])
    result = {"status": "needs_review", "preview": str(preview), "layout": "source left; generated right",
              "generated_sha256": sha256(generated), "source_sha256": sha256(source),
              "visual_review_required": ["garment flicker", "identity", "hands and feet", "transition continuity"]}
    if mode.startswith("pose"):
        result["motion"] = compare_poses(extract_poses(source, pose_model), extract_poses(generated, pose_model))
        result["pose_model_sha256"] = sha256(pose_model)
        result["status"] = "flagged" if result["motion"]["status"] != "pass" else "needs_review"
    write_json(folder / "quality.json", result)
    return result


def approved(section):
    approval = section.get("approval", {})
    return approval.get("accepted") is True and Path(section["video_path"]).is_file() and approval.get("sha256") == sha256(section["video_path"])


def write_review_index(report, folder):
    from html import escape
    folder = Path(folder)
    rows = []
    for s in report["sections"]:
        index = s["index"]
        status = s.get("quality", {}).get("motion", {}).get("status", "visual review needed")
        rows.append(f'<tr><td>{index}</td><td>{escape(s.get("kind", "stable"))}</td><td>{s["start_frame"] / FPS:.3f}–{s["end_frame"] / FPS:.3f}s</td>'
                    f'<td>{escape(status)}</td><td><a href="section-{index}/review.mp4">Compare</a> · '
                    f'<a href="section-{index}/quality.json">Scores</a></td></tr>')
    html = '<!doctype html><meta charset="utf-8"><title>Dance review</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:16px}video{width:100%}td,th{padding:12px;border-bottom:1px solid #ccc;text-align:left}</style>'
    html += f'<h1>Dance review</h1><p>{escape(report["job_id"])}</p><p>Source left, generated result right. Check choreography, clothing stability, face, hands, feet and outfit changes.</p>'
    html += '<video controls src="review.mp4"></video><p><a href="generated.mp4">Generated video</a> · <a href="report.json">Report</a></p>'
    html += '<table><thead><tr><th>Section</th><th>Type</th><th>Source time</th><th>Motion screen</th><th>Review</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table>'
    path = folder / "review.html"
    path.write_text(html, encoding="utf-8")
    return str(path)
