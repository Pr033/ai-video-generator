"""VideoSegmentAgent — generate one mp4 per shot in a storyboard.

Usage:
    from video_segment import VideoSegmentAgent
    from video import SeedanceProvider

    agent = VideoSegmentAgent(SeedanceProvider(), output_dir="runs/demo")
    board = agent.run(prompted_board_json)

The agent respects transition=="continue" by uploading the predecessor's
last frame as first-frame reference (image-to-video) when available.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from copy import deepcopy

from video import (
    VideoProvider, VideoSubmissionError, download_file,
    extract_last_frame, image_to_base64, check_ffmpeg,
)


class VideoSegmentAgent:
    """Turn a prompted storyboard into one mp4 per shot."""

    def __init__(self, provider: VideoProvider, output_dir: str | Path,
                 poll_timeout: float = 300, poll_interval: float = 10):
        self.provider = provider
        self.output_dir = Path(output_dir)
        self.poll_timeout = poll_timeout
        self.poll_interval = poll_interval
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, board: dict) -> dict:
        """Generate video for every shot. Returns updated board with mp4_path filled in."""
        if board.get("status") != "ready" or not board.get("shots"):
            raise ValueError("需要 status='ready' 且含 shots 的分镜")

        shots = board["shots"]
        shot_by_id = {s["shot_id"]: s for s in shots}
        results: list[dict] = []

        prev_mp4_path: Path | None = None

        for i, shot in enumerate(shots):
            sid = shot["shot_id"]
            print(f"▶ [{i+1}/{len(shots)}] {sid} "
                  f"({shot.get('transition','?')}, {shot['duration_sec']}s) "
                  f"— {shot.get('action','')[:50]}...")

            # 1. Prepare reference image for "continue" shots
            ref_b64 = None
            if shot.get("transition") == "continue" and prev_mp4_path is not None:
                if not check_ffmpeg():
                    raise RuntimeError("continue 镜头需要 ffmpeg 抽参考图")
                png = extract_last_frame(prev_mp4_path)
                ref_b64 = image_to_base64(png)
                print(f"   + 用上一个镜头末帧作参考图 ({prev_mp4_path.name})")

            # 2. Submit + poll + download
            shot_dir = self.output_dir / sid
            mp4_path = self._generate_one(shot, ref_b64, shot_dir)

            # 3. Record
            shot["mp4_path"] = str(mp4_path)
            shot["mp4_status"] = "generated"
            shot["used_reference_image"] = ref_b64 is not None
            prev_mp4_path = mp4_path

            results.append({
                "shot_id": sid,
                "mp4_path": str(mp4_path),
                "duration_sec": shot["duration_sec"],
                "used_reference_image": ref_b64 is not None,
            })

        # Save results manifest
        (self.output_dir / "segment_results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

        return {**deepcopy(board), "status": "video_ready", "segments": results}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_one(self, shot: dict, ref_b64: str | None,
                      shot_dir: Path) -> Path:
        prompt = shot["video_prompt"]
        duration = shot["duration_sec"]

        # Clamp to provider's supported range
        lo = getattr(self.provider, "min_duration", 2)
        hi = getattr(self.provider, "max_duration", 15)
        submit_dur = max(lo, min(duration, hi))
        if submit_dur != duration:
            print(f"   (provider 支持 {lo}-{hi}s, 提交 {submit_dur}s)")

        aspect = self._detect_aspect(shot)

        shot_dir.mkdir(parents=True, exist_ok=True)

        # Submit
        print(f"   submitting (duration={submit_dur}s, aspect={aspect}, "
              f"ref={'yes' if ref_b64 else 'no'})...")
        task_id = self.provider.submit(prompt, submit_dur, aspect, ref_b64)
        (shot_dir / "task_id.txt").write_text(str(task_id), encoding="utf-8")

        # Poll
        print(f"   polling...", end="", flush=True)
        start = time.monotonic()
        poll_result = self.provider.poll(task_id, self.poll_timeout, self.poll_interval)
        elapsed = time.monotonic() - start
        print(f" {poll_result['status']} ({elapsed:.0f}s)")

        if poll_result["status"] != "succeeded":
            raise VideoSubmissionError(
                f"任务失败: {poll_result['status']}, error={poll_result.get('error')}")

        # Save raw response for debugging
        (shot_dir / "poll_response.json").write_text(
            json.dumps(poll_result.get("raw", {}), ensure_ascii=False, indent=2),
            encoding="utf-8")

        # Download
        video_url = poll_result.get("video_url")
        if not video_url:
            raise VideoSubmissionError(f"任务成功但无 video_url: {poll_result}")

        mp4 = download_file(video_url, shot_dir / f"{shot['shot_id']}.mp4")
        size_kb = mp4.stat().st_size // 1024
        print(f"   ✅ downloaded → {mp4} ({size_kb} KB)")
        return mp4

    @staticmethod
    def _detect_aspect(shot: dict) -> str:
        """Try to infer aspect ratio from brief; default 16:9."""
        brief_style = shot.get("brief", {}).get("style", "")
        for ratio in ("9:16", "1:1", "16:9"):
            if ratio in brief_style:
                return ratio
        return "16:9"


# ---------------------------------------------------------------------------
# CLI — run directly with a prompted board JSON
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("board", type=Path, help="prompted_board.json")
    p.add_argument("--provider", choices=["seedance", "minimax"], default="seedance")
    p.add_argument("--out", type=Path, required=True, help="输出目录")
    args = p.parse_args()

    board = json.loads(args.board.read_text(encoding="utf-8"))

    # Lazy import so one can switch providers at CLI time
    from video import SeedanceProvider, MiniMaxProvider
    provider = SeedanceProvider() if args.provider == "seedance" else MiniMaxProvider()

    try:
        agent = VideoSegmentAgent(provider, args.out)
        result = agent.run(board)
        # Save updated board
        out_board = args.out / "board_with_mp4s.json"
        out_board.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"\n完成。分镜视频汇总 → {out_board}")
    except Exception as exc:
        print(f"失败：{exc}", file=sys.stderr)
        sys.exit(1)
