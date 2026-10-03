"""Frame-indexed media operations for the reusable browser dance workflow."""
from pathlib import Path
import os

from .media import executable, run, probe, verify_output
from .references import FPS


def normalize_dance_video(source, destination, width, height, start=0, duration=None):
    """Make one 24 fps timeline; every section subsequently uses frame indices."""
    command = [executable("ffmpeg"), "-nostdin", "-v", "error", "-y", "-ss", str(start), "-i", str(source)]
    if duration is not None:
        command += ["-t", str(duration)]
    run(command + ["-map", "0:v:0", "-map", "0:a:0?", "-vf",
                   f"fps={FPS},scale={width}:{height}:flags=lanczos,setsar=1,setpts=PTS-STARTPTS",
                   "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-pix_fmt", "yuv420p",
                   "-af", "asetpts=PTS-STARTPTS,aresample=48000", "-c:a", "aac", "-ac", "2",
                   "-movflags", "+faststart", str(destination)])
    return probe(destination)


def extract_frame(source, destination, frame):
    run([executable("ffmpeg"), "-nostdin", "-v", "error", "-y", "-i", str(source),
         "-vf", f"select=eq(n\\,{frame})", "-frames:v", "1", str(destination)])
    if not Path(destination).is_file():
        raise RuntimeError(f"Could not extract frame {frame} from {source}")
    return str(destination)


def slice_dance_video(source, destination, start_frame, frames, *, pad=False):
    # Trim the normalized timeline by frame index. End padding is model context
    # only, and is removed from the assembled video.
    vf = f"trim=start_frame={start_frame},setpts=PTS-STARTPTS"
    if pad:
        vf += f",tpad=stop_mode=clone:stop_duration={frames / FPS}"
    vf += f",trim=end_frame={frames}"
    af = f"atrim=start={start_frame / FPS},asetpts=PTS-STARTPTS,apad,atrim=duration={frames / FPS}"
    run([executable("ffmpeg"), "-nostdin", "-v", "error", "-y", "-i", str(source),
         "-map", "0:v:0", "-map", "0:a:0?", "-vf", vf, "-af", af,
         "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-ar", "48000", "-ac", "2", "-movflags", "+faststart", str(destination)])
    metadata = probe(destination)
    video = next(s for s in metadata["streams"] if s["codec_type"] == "video")
    if int(video.get("nb_frames", 0)) != frames:
        raise RuntimeError(f"Section has {video.get('nb_frames')} frames, expected {frames}")
    return metadata


def assemble_dance_video(clips, destination, total_frames, source_audio=None):
    """Concatenate exact core frames without crossfades or duplicated context."""
    destination = Path(destination)
    manifest = destination.parent / "concat.txt"
    # Owned section paths are made relative; escaping also handles spaces/apostrophes.
    manifest.write_text("".join("file '" + os.path.relpath(p, destination.parent).replace("'", "'\\''") + "'\n" for p in clips))
    command = [executable("ffmpeg"), "-nostdin", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(manifest)]
    if source_audio:
        command += ["-i", str(source_audio)]
    command += ["-map", "0:v:0", "-map", "1:a:0" if source_audio else "0:a:0",
                "-vf", f"trim=end_frame={total_frames},setpts=PTS-STARTPTS",
                "-af", f"aresample=async=1:first_pts=0,apad,atrim=duration={total_frames / FPS}",
                "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-ar", "48000", "-ac", "2", "-movflags", "+faststart", str(destination)]
    run(command)
    return verify_output(destination, total_frames)
