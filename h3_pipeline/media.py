"""CPU-only media preparation. ffmpeg limits/resizes before frames become tensors."""
from pathlib import Path
import json
import os
import shutil
import subprocess

from .references import FPS, reference_canvas


def executable(name):
    result = os.environ.get(name.upper()) or shutil.which(name)
    if not result:
        raise RuntimeError(f"{name} is required; install ffmpeg or set {name.upper()}")
    return result


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{Path(command[0]).name} failed: {result.stderr[-4000:]}")
    return result.stdout


def probe(path):
    return json.loads(run([executable("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]))


def prepare_image(source, destination, width, height, mode="match"):
    from PIL import Image, ImageOps
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        size = reference_canvas(*image.size, width, height, mode)
        image.resize(size, Image.Resampling.LANCZOS).save(destination, "PNG")
    return str(destination)


def prepare_video(source, destination, duration, width, height, start=0, include_audio=True):
    if start < 0:
        raise ValueError("reference_video_start must be nonnegative")
    metadata = probe(source)
    stream = next((s for s in metadata["streams"] if s["codec_type"] == "video"), None)
    if not stream:
        raise ValueError("Reference file has no video stream")
    # Respect portrait/rotation metadata; ffmpeg applies autorotation before scaling.
    rotation = next((s.get("rotation", 0) for s in stream.get("side_data_list", []) if "rotation" in s), 0)
    iw, ih = stream["width"], stream["height"]
    if abs(rotation) % 180 == 90:
        iw, ih = ih, iw
    rw, rh = reference_canvas(iw, ih, width, height)
    audio_present = include_audio and any(s["codec_type"] == "audio" for s in metadata["streams"])
    vf = f"fps={FPS},scale={rw}:{rh}:flags=lanczos,setsar=1,setpts=PTS-STARTPTS"
    command = [executable("ffmpeg"), "-nostdin", "-v", "error", "-y", "-ss", str(start), "-i", str(source),
               "-t", str(duration), "-map", "0:v:0", "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p"]
    if audio_present:
        command += ["-map", "0:a:0", "-af", "asetpts=PTS-STARTPTS", "-c:a", "aac", "-ar", "48000", "-ac", "2"]
    else:
        command += ["-an"]
    run(command + ["-movflags", "+faststart", str(destination)])
    result = probe(destination)
    video = next(s for s in result["streams"] if s["codec_type"] == "video")
    # Crop down to 17k+5 before extracting the paired soundtrack in the server.
    count = int(video.get("nb_frames", 0))
    if count < 5:
        raise ValueError("Reference video must supply at least five frames after trimming")
    usable = count - (count - 5) % 17
    return {"source": str(source), "prepared": str(destination), "source_width": iw, "source_height": ih,
            "width": rw, "height": rh, "frames": count, "usable_frames": usable,
            "duration": usable / FPS, "paired_audio": audio_present, "start": start}


def prepare_audio(source, destination, duration):
    run([executable("ffmpeg"), "-nostdin", "-v", "error", "-y", "-i", str(source), "-t", str(duration),
         "-vn", "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(destination)])
    return str(destination)


def verify_output(path, expected_frames=None):
    result = probe(path)
    video = next((s for s in result["streams"] if s["codec_type"] == "video"), None)
    audio = next((s for s in result["streams"] if s["codec_type"] == "audio"), None)
    if not video or not audio:
        raise RuntimeError("Generated MP4 must contain video AND native generated audio")
    if video["codec_name"] != "h264" or audio["codec_name"] != "aac":
        raise RuntimeError("Expected H.264 video and AAC audio")
    numerator, denominator = map(int, video["avg_frame_rate"].split("/"))
    if abs(numerator / denominator - FPS) > 0.01:
        raise RuntimeError("Generated video must be 24 fps")
    if expected_frames and int(video.get("nb_frames", 0)) != expected_frames:
        raise RuntimeError("Generated video frame count differs from the aligned request")
    if abs(float(video["duration"]) - float(audio["duration"])) > 0.15:
        raise RuntimeError("Generated audio/video durations differ by more than 150 ms")
    return result
