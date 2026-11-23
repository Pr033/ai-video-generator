"""OrchestratorAgent — full pipeline scheduler with state tracking + dry-run mode.

Stages (in order):
  1. script       → script.json
  2. storyboard   → board.json
  3. prompts      → prompted_board.json
  4. reference    → board_with_refs.json  (skip-able)
  5. video        → board_with_mp4s.json
  6. audio        → narration_track.mp3   (skip-able if silent)
  7. compose      → final.mp4
  8. qa           → qa_report.json        (skip-able)

Key features:
  - run(idea, dry_run=True) — full fake pipeline, no API keys needed
  - run(idea, dry_run=False) — real API calls, fails gracefully on missing keys
  - resume_from(stage) — skip already-done stages
  - manifest.json — every artifact path + status

Usage:
    from orchestrator import OrchestratorAgent
    orch = OrchestratorAgent("runs/my_run")
    manifest = orch.run("雨停后的校园，15秒，电影感，无旁白")
    print(manifest["stages"]["compose"]["artifact"])  # final.mp4 path
"""
from __future__ import annotations
import copy
import json
import os
import random
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


# Stage ordering
STAGES = ["script", "storyboard", "prompts", "reference", "video", "audio", "compose", "qa"]
SKIP_STAGES = {"reference", "qa"}  # no keys → auto skip


class OrchestratorAgent:
    """Coordinator that chains every video-production agent."""

    def __init__(self, run_dir: str | Path, dry_run: bool = False,
                 video_provider: str = "seedance",
                 script_backend: str = "auto", script_budget: str = "standard"):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        self.video_provider = video_provider
        self.script_backend = script_backend
        self.script_budget = script_budget

        self.manifest = {
            "run_name": self.run_dir.name,
            "started_at": datetime.now().isoformat(),
            "dry_run": dry_run,
            "video_provider": video_provider,
            "script_backend": script_backend,
            "stages": {s: {"status": "pending", "artifact": None, "error": None}
                       for s in STAGES},
        }

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def run(self, idea: str, slots: list[int] | None = None) -> dict:
        """Run full pipeline idea → final.mp4. Returns manifest."""
        print(f"\n{'='*60}")
        mode = "🎭 DRY-RUN" if self.dry_run else "🚀 LIVE"
        print(f"{mode}  |  run_dir={self.run_dir}")
        print(f"idea: {idea[:60]}")
        print(f"{'='*60}\n")

        board = {}  # accumulates state across stages

        for stage in STAGES:
            status = self.manifest["stages"][stage]
            if status["status"] in ("done", "skipped"):
                continue  # already done from prior run

            # Decide skip rules
            if stage in SKIP_STAGES and not self._has_stage_backend(stage):
                status["status"] = "skipped"
                status["error"] = "no backend configured"
                print(f"  ⏭  {stage} — 跳过（无后端）")
                continue

            try:
                board = self._run_stage(stage, board, idea, slots)
                status["status"] = "done"
            except Exception as exc:
                status["status"] = "failed"
                status["error"] = str(exc)
                print(f"  ❌ {stage} 失败: {exc}")
                # Continue or stop? We let downstream stages handle missing inputs.
                if stage in ("script", "storyboard", "prompts"):
                    # Core stages must succeed
                    print("  💀 核心阶段失败，终止流水线")
                    break

        self.manifest["finished_at"] = datetime.now().isoformat()
        self._save_manifest()
        self._print_summary()
        return self.manifest

    def resume_from(self, stage: str, idea: str,
                    slots: list[int] | None = None) -> dict:
        """Mark all stages before `stage` as already done, then run from `stage`."""
        if stage not in STAGES:
            raise ValueError(f"未知阶段: {stage}. 可选: {STAGES}")
        idx = STAGES.index(stage)
        for s in STAGES[:idx]:
            self.manifest["stages"][s]["status"] = "done"
        return self.run(idea, slots)

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _run_stage(self, stage: str, board: dict,
                   idea: str, slots: list[int] | None) -> dict:
        status = self.manifest["stages"][stage]
        stage_dir = self.run_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)

        if stage == "script":
            print("📖 Stage 1: ScriptAgent")
            if self.dry_run:
                board = self._dry_run_script(idea, slots)
            else:
                from main import run_agent, load_config
                load_config()
                result = run_agent(idea, slots=slots, backend=self.script_backend,
                                   budget=self.script_budget)
                # run_agent returns the full script dict; convert to our shape
                board = self._script_to_board(result)
            self._save(stage_dir / "script.json", board)
            status["artifact"] = str(stage_dir / "script.json")

        elif stage == "storyboard":
            print("🎬 Stage 2: StoryboardAgent")
            if self.dry_run:
                board = self._dry_run_storyboard(board)
            else:
                from storyboard import build_storyboard
                board = build_storyboard(board, backend=self.script_backend,
                                         budget=self.script_budget)
            self._save(stage_dir / "board.json", board)
            status["artifact"] = str(stage_dir / "board.json")

        elif stage == "prompts":
            print("✍️ Stage 3: PromptEngineerAgent")
            if self.dry_run:
                board = self._dry_run_prompts(board)
            else:
                from prompt_engineer import enhance_board
                board = enhance_board(board, budget=self.script_budget)
            self._save(stage_dir / "prompted_board.json", board)
            status["artifact"] = str(stage_dir / "prompted_board.json")

        elif stage == "reference":
            print("🖼️ Stage 4: ReferenceImageAgent")
            if self.dry_run:
                board = self._dry_run_reference(board)
            else:
                from reference_agent import ReferenceImageAgent
                board = ReferenceImageAgent(stage_dir).run(board)
            self._save(stage_dir / "board_with_refs.json", board)
            status["artifact"] = str(stage_dir / "board_with_refs.json")

        elif stage == "video":
            print("🎥 Stage 5: VideoSegmentAgent")
            if self.dry_run:
                board = self._dry_run_video(board, stage_dir)
            else:
                from video_segment import VideoSegmentAgent
                from video import SeedanceProvider, MiniMaxProvider
                Prov = SeedanceProvider if self.video_provider == "seedance" else MiniMaxProvider
                board = VideoSegmentAgent(Prov(), stage_dir).run(board)
            self._save(stage_dir / "board_with_mp4s.json", board)
            status["artifact"] = str(stage_dir / "board_with_mp4s.json")

        elif stage == "audio":
            print("🎵 Stage 6: AudioAgent")
            if self.dry_run:
                audio_path = self._dry_run_audio(board, stage_dir)
            else:
                from audio_agent import AudioAgent
                audio_path = AudioAgent(stage_dir).generate(board)
            if audio_path:
                board["audio_track"] = str(audio_path)
                status["artifact"] = str(audio_path)
            else:
                status["artifact"] = None
                print("   ↳ 音频阶段无产出（可能 silent/no_narration 模式）")

        elif stage == "compose":
            print("🎞️ Stage 7: ComposerAgent")
            audio = board.get("audio_track")
            if self.dry_run:
                final = self._dry_run_compose(board, self.run_dir / "final.mp4")
            else:
                from composer import ComposerAgent
                final = ComposerAgent().run(board, self.run_dir, audio)
            board["final_video"] = str(final)
            status["artifact"] = str(final)

        elif stage == "qa":
            print("🔍 Stage 8: QA Review")
            if self.dry_run:
                qa = {"status": "dry_run", "visual_score": 0.8, "consistency_score": 0.85,
                      "notes": "dry-run 未做实际 VLM 审查"}
            else:
                from qa_agent import run_all_reviews
                qa = run_all_reviews(board)
            self._save(stage_dir / "qa_report.json", qa)
            status["artifact"] = str(stage_dir / "qa_report.json")

        return board

    # ------------------------------------------------------------------
    # Dry-run fabricators (fake outputs without any API call)
    # ------------------------------------------------------------------

    @staticmethod
    def _dry_run_script(idea: str, slots: list[int] | None = None) -> dict:
        slots = slots or [5, 5, 5]
        total = sum(slots)
        beats = []
        for i, dur in enumerate(slots, 1):
            beats.append({
                "beat_id": f"B{i:02d}",
                "action": f"[dry-run] beat {i} 动作描述",
                "start_state": f"beat {i} 起始状态",
                "end_state": f"beat {i} 结束状态",
                "camera": {"shot_size": "中景", "movement": "固定", "angle": "平拍"},
                "duration_s": dur,
                "narration": "",
            })
        return {
            "schema_version": 1,
            "status": "ready",
            "source_idea": idea,
            "brief": {"theme": idea[:40], "style": "cinematic", "sound": "silent",
                      "must": [], "avoid": [], "seconds": total},
            "title": idea[:20] or "Untitled",
            "duration_s": total,
            "narration_by_beat": {b["beat_id"]: b["narration"] for b in beats},
            "beats": beats,
        }

    @staticmethod
    def _script_to_board(script: dict) -> dict:
        """ScriptAgent returns a script; convert to our universal board shape."""
        beats = script.get("beats", [])
        shots = []
        cursor = 0
        for i, beat in enumerate(beats, 1):
            dur = beat.get("duration_s", 5)
            shots.append({
                "beat_id": beat.get("beat_id", f"B{i:02d}"),
                "shot_id": f"shot_{i:03d}",
                "order": i,
                "start_s": cursor,
                "end_s": cursor + dur,
                "duration_sec": dur,
                "visual_description": beat.get("start_state", ""),
                "action": beat.get("action", ""),
                "start_state": beat.get("start_state", ""),
                "end_state": beat.get("end_state", ""),
                "camera": beat.get("camera", {}),
                "transition": "opening" if i == 1 else "cut",
                "continuity_requirements": [],
                "dialogue": "",
                "characters": [],
                "scene_id": "S001",
            })
            cursor += dur
        return {
            "schema_version": script.get("schema_version", 1),
            "status": "ready",
            "title": script.get("title", "Untitled"),
            "brief": script.get("brief", {}),
            "duration_s": script.get("duration_s", cursor),
            "narration_by_beat": script.get("narration_by_beat", {}),
            "shots": shots,
        }

    def _dry_run_storyboard(self, board: dict) -> dict:
        # ScriptAgent (live) produces a "script" with beats → needs _script_to_board
        # Our _dry_run_script also produces beats → same conversion
        if "shots" not in board and "beats" in board:
            board = self._script_to_board(board)
        # Add transition / continuity requirements
        for i, shot in enumerate(board.get("shots", [])):
            shot.setdefault("transition", "opening" if i == 0 else "cut")
            shot.setdefault("continuity_requirements", [])
        return board

    @staticmethod
    def _dry_run_prompts(board: dict) -> dict:
        board = copy.deepcopy(board)
        for shot in board["shots"]:
            desc = shot.get("visual_description", shot.get("action", "a scene"))
            cam = shot.get("camera", {}).get("movement", "fixed")
            shot["video_prompt"] = (
                f"{shot['duration_sec']}-second shot, {cam} camera, "
                f"{desc}. Cinematic lighting, 4K, professional color grading."
            )
            shot["prompt_mode"] = "dry_run"
        board["status"] = "ready"
        return board

    @staticmethod
    def _dry_run_reference(board: dict) -> dict:
        board = copy.deepcopy(board)
        board["status"] = "prompted"
        return board

    def _dry_run_video(self, board: dict, stage_dir: Path) -> dict:
        import subprocess, sys
        board = copy.deepcopy(board)

        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            print("   ⚠️ 没 ffmpeg，dry-run 生成空占位 mp4")
            for shot in board["shots"]:
                shot_dir = stage_dir / shot["shot_id"]
                shot_dir.mkdir(parents=True, exist_ok=True)
                dummy = shot_dir / f"{shot['shot_id']}.mp4"
                dummy.write_bytes(b"\x00" * 1024)
                shot["mp4_path"] = str(dummy)
                shot["mp4_status"] = "generated_dummy"
                shot["used_reference_image"] = False
            board["status"] = "video_ready"
            return board

        for shot in board["shots"]:
            shot_dir = stage_dir / shot["shot_id"]
            shot_dir.mkdir(parents=True, exist_ok=True)
            dur = max(1, int(shot["duration_sec"]))
            out = shot_dir / f"{shot['shot_id']}.mp4"
            # Generate a trivial solid-color video with correct duration
            subprocess.run(
                [ffmpeg, "-y", "-f", "lavfi", "-t", str(dur),
                 "-i", f"color=c=0x{random.randint(0x101010, 0xFFFFFF):06x}:s=640x360:r=30",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
                capture_output=True, timeout=60)
            shot["mp4_path"] = str(out)
            shot["mp4_status"] = "generated_dry_run"
            shot["used_reference_image"] = False
        board["status"] = "video_ready"
        return board

    @staticmethod
    def _dry_run_audio(board: dict, stage_dir: Path) -> Optional[Path]:
        if board.get("brief", {}).get("sound") != "narrated":
            return None
        # No fake audio for dry-run — just skip
        return None

    @staticmethod
    def _dry_run_compose(board: dict, final_path: Path) -> Path:
        """Dry-run compose: copy first shot's mp4 as 'final' (cheat)."""
        shots = board.get("shots", [])
        if not shots or not shots[0].get("mp4_path"):
            # No segments — create an empty placeholder
            final_path.parent.mkdir(parents=True, exist_ok=True)
            final_path.write_bytes(b"\x00" * 1024)
            print(f"   ⚠️ dry-run: 无视频片段，占位 final.mp4")
            return final_path
        first = Path(shots[0]["mp4_path"])
        final_path.parent.mkdir(parents=True, exist_ok=True)
        # Copy first shot as "final" to satisfy downstream
        if first.exists():
            shutil.copy2(first, final_path)
            print(f"   dry-run: 复制第一个镜头作为 final.mp4 ({final_path.stat().st_size // 1024} KB)")
        else:
            final_path.write_bytes(b"\x00" * 1024)
        return final_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _has_stage_backend(self, stage: str) -> bool:
        """Does this stage have the env/config it needs?"""
        if stage == "reference":
            return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("REPLICATE_API_TOKEN"))
        if stage == "qa":
            return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("CHAT_API_KEY"))
        return True  # script/storyboard/prompts have Chat() fallback; video has provider config

    def _save(self, path: Path, data) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_manifest(self) -> None:
        self._save(self.run_dir / "manifest.json", self.manifest)

    def _print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"📋 Pipeline Summary: {self.run_dir.name}")
        print(f"{'='*60}")
        for s in STAGES:
            st = self.manifest["stages"][s]
            icon = {"done": "✅", "skipped": "⏭", "failed": "❌", "pending": "⏳",
                    "running": "🔄"}.get(st["status"], "?")
            art = (st["artifact"] or "")
            if art:
                art = Path(art).name
            print(f"  {icon} {s:12s}  {st['status']:10s}  → {art}")
        final = self.manifest["stages"]["compose"]["artifact"]
        if final:
            print(f"\n🏁 最终视频: {final}")
        print(f"📁 全部产物: {self.run_dir}")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("idea", help="视频灵感")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true", help="不调任何 API，跑通流程验证")
    p.add_argument("--provider", choices=["seedance", "minimax"], default="seedance")
    p.add_argument("--slots", type=int, nargs="+")
    p.add_argument("--resume", choices=STAGES, help="从某阶段恢复")
    args = p.parse_args()

    orch = OrchestratorAgent(args.run_dir, args.dry_run, args.provider)
    if args.resume:
        manifest = orch.resume_from(args.resume, args.idea, args.slots)
    else:
        manifest = orch.run(args.idea, args.slots)
    print(f"\n清单: {args.run_dir / 'manifest.json'}")
    sys.exit(0 if manifest["stages"]["compose"]["status"] in ("done", "skipped") else 1)
