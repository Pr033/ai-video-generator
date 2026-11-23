"""Extract first / last frames from a video file using ffmpeg."""
from __future__ import annotations
import base64
import subprocess
from pathlib import Path


def _find_ffmpeg() -> str | None:
    """Locate ffmpeg binary. Prefers system PATH, falls back to imageio-ffmpeg."""
    import shutil
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def extract_first_frame(video_path: str | Path, out_png: str | None = None) -> Path:
    """Extract frame at t=0 as PNG."""
    video_path = Path(video_path)
    out_png = Path(out_png) if out_png else video_path.with_name(video_path.stem + "_first.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg 未找到，请安装 ffmpeg 并加入 PATH")
    result = subprocess.run(
        [ffmpeg, "-y", "-ss", "0", "-i", str(video_path),
         "-frames:v", "1", "-q:v", "2", str(out_png)],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 抽帧失败: {result.stderr[-500:]}")
    if not out_png.exists():
        raise RuntimeError("抽帧后文件不存在")
    return out_png


def extract_last_frame(video_path: str | Path, out_png: str | None = None) -> Path:
    """Extract the final frame of a video as PNG."""
    video_path = Path(video_path)
    out_png = Path(out_png) if out_png else video_path.with_name(video_path.stem + "_last.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg 未找到，请安装 ffmpeg 并加入 PATH")
    result = subprocess.run(
        [ffmpeg, "-y", "-sseof", "-0.1", "-i", str(video_path),
         "-frames:v", "1", "-q:v", "2", str(out_png)],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        result = subprocess.run(
            [ffmpeg, "-y", "-i", str(video_path),
             "-vf", "select='eq(n\\,n_forced)',last",
             "-frames:v", "1", "-q:v", "2", str(out_png)],
            capture_output=True, text=True, timeout=60,
        )
    if result.returncode != 0 or not out_png.exists():
        raise RuntimeError(f"ffmpeg 抽末帧失败: {result.stderr[-500:]}")
    return out_png


def image_to_base64(image_path: str | Path) -> str:
    """Read image file and return base64 string (no data: prefix)."""
    data = Path(image_path).read_bytes()
    return base64.b64encode(data).decode()


def check_ffmpeg() -> bool:
    return _find_ffmpeg() is not None
