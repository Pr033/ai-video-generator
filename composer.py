"""ComposerAgent — concatenate shot mp4s into one final video via ffmpeg.

Usage:
    from composer import ComposerAgent
    composer = ComposerAgent()
    final_mp4 = composer.run(board_with_mp4s, output_dir="runs/demo")

Handles:
  - Sequential concat respecting each shot's duration_sec / start_s ordering.
  - Optional audio overlay (if a separate audio track is supplied later).
  - Consistent re-encoding so concat never breaks on codec mismatches.
"""
from __future__ import annotations
import json
import shutil
import subprocess
from pathlib import Path

def _find_ffmpeg() -> str:
    """Find ffmpeg binary: system PATH first, fallback to imageio-ffmpeg."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"  # hope for the best


class ComposerAgent:
    """Stitch per-shot mp4 segments into one final.mp4."""

    def __init__(self, preset: str = "veryfast", crf: int = 20):
        self.preset = preset
        self.crf = crf

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, board: dict, output_dir: str | Path,
            audio_path: str | Path | None = None) -> Path:
        """Concatenate shots[].mp4_path into final.mp4. Returns final path."""
        if board.get("status") != "video_ready":
            raise ValueError("需要 VideoSegmentAgent 跑完的 board (status=video_ready)")

        shots = board["shots"]
        segments = [Path(s["mp4_path"]) for s in shots]
        for seg in segments:
            if not seg.exists():
                raise FileNotFoundError(f"缺失片段: {seg}")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        final = out / "final.mp4"

        if len(segments) == 1:
            # Single shot — just copy to final.mp4
            import shutil
            shutil.copy2(segments[0], final)
            print(f"只有一个镜头，直接复制为 final.mp4")
        else:
            self._concat(segments, final)

        # Audio overlay (if supplied)
        if audio_path is not None:
            final_with_audio = self._mix_audio(final, Path(audio_path),
                                               out / "final_with_audio.mp4")
            final = final_with_audio

        print(f"✅ 合成完成 → {final}")
        print(f"   大小: {final.stat().st_size // 1024} KB")
        return final

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _concat(self, segments: list[Path], final: Path) -> None:
        """ffmpeg filter_complex concat — always stable across codecs."""
        n = len(segments)

        # Build filter_complex
        inputs_args: list[str] = []
        filter_parts: list[str] = []
        for i, seg in enumerate(segments):
            inputs_args += ["-i", str(seg)]
            filter_parts.append(f"[{i}:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
                                f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2[v{i}];")
            filter_parts.append(f"[{i}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a{i}];")

        concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(n))
        filter_str = "".join(filter_parts) + concat_inputs + f"concat=n={n}:v=1:a=1[v][a]"

        cmd = ([_find_ffmpeg(), "-y"] + inputs_args +
               ["-filter_complex", filter_str,
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", self.preset, "-crf", str(self.crf),
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                str(final)])

        print(f"运行 ffmpeg 合成 {n} 个镜头...")
        print(f"  {' '.join(cmd[:10])} ... (共 {len(cmd)} 个参数)")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"ffmpeg stderr:\n{result.stderr[-2000:]}")
            raise RuntimeError(f"ffmpeg 拼接失败 (exit {result.returncode})")
        if not final.exists():
            raise RuntimeError("合成后 final.mp4 不存在")

    @staticmethod
    def _mix_audio(video: Path, audio: Path, output: Path) -> Path:
        """Overlay audio track onto video; audio longer than video is truncated."""
        cmd = [_find_ffmpeg(), "-y", "-i", str(video), "-i", str(audio),
               "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
               "-map", "0:v:0", "-map", "1:a:0",
               "-shortest", "-movflags", "+faststart",
               str(output)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"混音 stderr:\n{result.stderr[-1000:]}")
            raise RuntimeError("混音失败")
        return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("board", type=Path, help="VideoSegmentAgent 输出的 board JSON")
    p.add_argument("--out", type=Path, required=True, help="final.mp4 输出目录")
    p.add_argument("--audio", type=Path, default=None, help="可选音频轨道")
    args = p.parse_args()

    board = json.loads(args.board.read_text(encoding="utf-8"))
    try:
        ComposerAgent().run(board, args.out, args.audio)
    except Exception as exc:
        print(f"失败：{exc}", file=sys.stderr)
        sys.exit(1)
