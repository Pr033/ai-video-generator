"""AudioAgent — TTS narration + ffmpeg mix into video.

Supports three sound modes from board['brief']['sound']:
  "silent"        → no audio, just passes through
  "no_narration"  → optional BGM overlay (no speech track yet)
  "narrated"      → per-beat TTS narration timed to each shot's start_s

Backends:
  - edge-tts (free, no key required) — default
  - elevenlabs (API key) — set ELEVENLABS_API_KEY env var

Usage:
    from audio_agent import AudioAgent
    agent = AudioAgent(output_dir="runs/demo")
    audio_path = agent.generate(board)
    # Then feed audio_path into ComposerAgent.run(board, out_dir, audio_path)
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


class AudioAgent:
    """Generate narration audio track + mix into final video."""

    def __init__(self, output_dir: str | Path, voice: str = "zh-CN-XiaoxiaoNeural",
                 backend: str = "auto"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.voice = voice
        self.backend = backend  # "auto" | "edge" | "elevenlabs"
        self._edge_available = self._check_edge()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate(self, board: dict) -> Optional[Path]:
        """Generate narration audio track. Returns audio path or None."""
        brief = board.get("brief", {})
        sound_mode = brief.get("sound", "no_narration")
        narration_map = board.get("narration_by_beat", {})
        total_duration = board.get("duration_s", 0)

        print(f"🎵 AudioAgent: sound={sound_mode}, backend={self.backend}")

        if sound_mode == "silent":
            print("   ↳ silent mode → 跳过音频生成")
            return None

        # Build time-aligned narration segments
        segments = self._collect_narration(board, narration_map)
        if sound_mode == "no_narration" or not segments:
            if sound_mode == "narrated":
                print("   ⚠️ narrated 模式但 narration_by_beat 为空 → 降级为无旁白")
            print("   ↳ 无旁白 → 暂跳过（BGM 后续接入）")
            return None

        # Generate per-segment audio
        audio_dir = self.output_dir / "narration"
        audio_dir.mkdir(parents=True, exist_ok=True)
        generated = []
        for seg in segments:
            audio_file = audio_dir / f"{seg['beat_id']}.mp3"
            if self._tts(seg["text"], audio_file):
                generated.append({**seg, "audio_path": str(audio_file)})

        if not generated:
            print("   ❌ 所有 TTS 片段生成失败")
            return None

        # Melt into one timeline-aligned track (silence pads between segments)
        final_audio = self.output_dir / "narration_track.mp3"
        self._assemble_timeline(generated, final_audio, total_duration)
        print(f"   ✅ 旁白音轨 → {final_audio}")
        return final_audio

    def mix_into_video(self, video_path: Path, audio_path: Path,
                       output_path: Path | None = None) -> Path:
        """Overlay audio onto video (shortest wins). Returns final path."""
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        if output_path is None:
            output_path = video_path.with_name(video_path.stem + "_with_audio.mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [ffmpeg, "-y",
               "-i", str(video_path), "-i", str(audio_path),
               "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
               "-map", "0:v:0", "-map", "1:a:0",
               "-shortest", "-movflags", "+faststart",
               str(output_path)]
        print(f"🎵 混音: {' '.join(cmd[:8])}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"混音失败: {result.stderr[-800:]}")
        print(f"   ✅ → {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Internal: narration collection
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_narration(board: dict, narration_map: dict) -> list[dict]:
        """Collect non-empty narration strings with their beat timings."""
        segments = []
        shots = board.get("shots", [])
        for shot in shots:
            bid = shot.get("beat_id", "")
            text = (narration_map.get(bid) or "").strip()
            if not text:
                # Try per-shot narration field as fallback
                text = (shot.get("narration") or "").strip()
            if text:
                segments.append({
                    "beat_id": bid,
                    "shot_id": shot["shot_id"],
                    "start_s": shot["start_s"],
                    "text": text,
                })
        return segments

    # ------------------------------------------------------------------
    # Internal: TTS backends
    # ------------------------------------------------------------------

    def _check_edge(self) -> bool:
        try:
            import edge_tts  # noqa: F401
            return True
        except ImportError:
            return False

    def _tts(self, text: str, output_path: Path) -> bool:
        """Run TTS for one text segment."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        backend = self.backend
        if backend == "auto":
            backend = "edge" if self._edge_available else "elevenlabs"

        if backend == "edge":
            return self._tts_edge(text, output_path)
        elif backend == "elevenlabs":
            return self._tts_elevenlabs(text, output_path)
        return False

    def _tts_edge(self, text: str, output_path: Path) -> bool:
        try:
            import asyncio, edge_tts
            async def _speak():
                communicate = edge_tts.Communicate(text=text, voice=self.voice)
                await communicate.save(str(output_path))
            asyncio.run(_speak())
            ok = output_path.exists() and output_path.stat().st_size > 100
            print(f"   edge-tts [{output_path.name}] → {output_path.stat().st_size // 1024}KB" if ok else f"   edge-tts 产出空文件")
            return ok
        except ImportError:
            print("   ⚠️ edge-tts 未安装。pip install edge-tts")
            return False
        except Exception as exc:
            print(f"   edge-tts 失败: {exc}")
            return False

    def _tts_elevenlabs(self, text: str, output_path: Path) -> bool:
        key = os.environ.get("ELEVENLABS_API_KEY")
        if not key:
            print("   ⚠️ 无 ELEVENLABS_API_KEY")
            return False
        try:
            import urllib.request, urllib.error
            url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"
            body = json.dumps({"text": text, "model_id": "eleven_monolingual_v1"}).encode()
            req = urllib.request.Request(url, data=body, method="POST",
                headers={"xi-api-key": key, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                output_path.write_bytes(resp.read())
            return output_path.exists() and output_path.stat().st_size > 100
        except Exception as exc:
            print(f"   elevenlabs 失败: {exc}")
            return False

    # ------------------------------------------------------------------
    # Internal: assemble timeline
    # ------------------------------------------------------------------

    def _assemble_timeline(self, segments: list[dict], output: Path,
                           total_duration: float) -> None:
        """Concat per-segment MP3s with silence padding to align with start_s."""
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        # Sort by start_s, build ffmpeg filter_complex
        segs = sorted(segments, key=lambda s: s["start_s"])

        # Build input list: silence + each narration
        inputs: list[tuple[str, str]] = []  # (type, value)
        prev_end = 0.0
        for seg in segs:
            gap = seg["start_s"] - prev_end
            if gap > 0.01:
                inputs.append(("silence", str(gap)))
            inputs.append(("audio", seg["audio_path"]))
            prev_end = seg["start_s"] + self._probe_duration(seg["audio_path"])

        tail = max(0.0, total_duration - prev_end)
        if tail > 0.01:
            inputs.append(("silence", str(tail)))

        # Build ffmpeg cmd
        args = [ffmpeg, "-y"]
        filter_parts = []
        for i, (itype, ival) in enumerate(inputs):
            if itype == "silence":
                args += ["-f", "lavfi", "-t", ival, "-i", "anullsrc=channel_layout=mono:sample_rate=44100"]
                filter_parts.append(f"[{i}:a]aresample=44100,aformat=sample_fmts=s16:sample_rates=44100[seq{i}];")
            else:
                args += ["-i", ival]
                filter_parts.append(f"[{i}:a]aresample=44100,aformat=sample_fmts=s16:sample_rates=44100[seq{i}];")

        concat = "".join(f"[seq{i}]" for i in range(len(inputs))) + f"concat=n={len(inputs)}:v=0:a=1[aout]"
        filter_str = "".join(filter_parts) + concat

        args += ["-filter_complex", filter_str, "-map", "[aout]",
                 "-c:a", "libmp3lame", "-b:a", "128k", str(output)]

        result = subprocess.run(args, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"   ⚠️ 音轨拼接失败: {result.stderr[-300:]}")
            # Fallback: just concat narration segments in order (no silence padding)
            self._fallback_concat(segs, output, ffmpeg)

    @staticmethod
    def _probe_duration(audio_path: str | Path) -> float:
        """Return duration in seconds via ffprobe (or ffmpeg)."""
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        try:
            result = subprocess.run(
                [ffmpeg.replace("ffmpeg", "ffprobe"), "-v", "error",
                 "-show_entries", "format=duration", "-of",
                 "default=noprint_wrappers=1:nokey=1", str(audio_path)],
                capture_output=True, text=True, timeout=15)
            return float(result.stdout.strip())
        except Exception:
            # fallback: approximate via ffmpeg decode
            try:
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
                result = subprocess.run(
                    [ffmpeg, "-i", str(audio_path)], capture_output=True, text=True, timeout=15)
                match = re.search(r"Duration:\s*(\d+):(\d+):(\d+)", result.stderr)
                if match:
                    h, m, s = map(int, match.groups())
                    return h * 3600 + m * 60 + s
            except Exception:
                pass
            return 2.0  # rough fallback

    @staticmethod
    def _fallback_concat(segs: list[dict], output: Path, ffmpeg: str) -> None:
        print("   尝试直接拼接...")
        concat_list = output.with_suffix(".txt")
        lines = [f"file '{Path(s['audio_path']).resolve()}'" for s in segs]
        concat_list.write_text("\n".join(lines), encoding="utf-8")
        subprocess.run(
            [ffmpeg, "-y", "-f", "concat", "-safe", "0",
             "-i", str(concat_list), "-c", "copy", str(output)],
            capture_output=True, text=True, timeout=60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("board", type=Path, help="board_with_mp4s.json (or prompted_board.json)")
    p.add_argument("--video", type=Path, help="如果提供则混音到视频上")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--voice", default="zh-CN-XiaoxiaoNeural")
    args = p.parse_args()

    board = json.loads(args.board.read_text(encoding="utf-8"))
    agent = AudioAgent(args.out, args.voice)
    audio = agent.generate(board)
    if audio and args.video:
        agent.mix_into_video(args.video, audio)
    print(f"\n完成。音频 → {audio or '(无)'}")
