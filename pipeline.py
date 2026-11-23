"""Full-end-to-end pipeline: idea → script → storyboard → prompts → video segments → final.mp4.

Stage flow:
    idea → produce_prompts(board) → produce_video(board) → compose(board_with_mp4s)

Run the whole chain with:
    python pipeline.py "雨停后的校园，18秒，电影感"
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

from main import run_agent, load_config
from storyboard import build_storyboard
from prompt_engineer import enhance_board
from video_segment import VideoSegmentAgent
from composer import ComposerAgent


# ---------------------------------------------------------------------------
# Stage 1–3: idea → prompted board (existing)
# ---------------------------------------------------------------------------

def produce_prompts(idea, slots=None, backend='auto', budget='standard'):
    """idea → script → board → prompted board."""
    script = run_agent(idea, slots=slots, backend=backend, budget=budget)
    board = build_storyboard(script, backend=backend, budget=budget)
    return enhance_board(board, budget=budget)


# ---------------------------------------------------------------------------
# Stage 4: prompted board → per-shot mp4s
# ---------------------------------------------------------------------------

def produce_video(board, provider='seedance', output_dir=None):
    """Generate one mp4 per shot in a prompted board."""
    load_config()
    from video import SeedanceProvider, MiniMaxProvider
    ProvClass = SeedanceProvider if provider == 'seedance' else MiniMaxProvider
    agent = VideoSegmentAgent(ProvClass(), output_dir=output_dir)
    return agent.run(board)


# ---------------------------------------------------------------------------
# Stage 5: per-shot mp4s → final.mp4
# ---------------------------------------------------------------------------

def compose(board_with_mp4s, output_dir, audio_path=None):
    return ComposerAgent().run(board_with_mp4s, output_dir, audio_path)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_full(idea, slots=None, backend='auto', budget='standard',
             provider='seedance', run_name=None):
    """Run idea → final.mp4. Returns dict with every stage's artifact path."""
    load_config()
    run_name = run_name or uuid4().hex[:12]
    root = Path(__file__).resolve().parent / "runs" / run_name
    root.mkdir(parents=True, exist_ok=True)

    manifest = {"run_name": run_name, "stages": {}}

    # Stage 1-3: prompts
    print("\n=== Stage 1-3: idea → prompted board ===")
    board = produce_prompts(idea, slots=slots, backend=backend, budget=budget)
    board_path = root / "prompted_board.json"
    board_path.write_text(json.dumps(board, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["stages"]["prompts"] = str(board_path)
    print(f"   ✅ → {board_path} ({len(board['shots'])} shots)")

    if board.get("status") == "needs_input":
        print(f"   ⚠️  需要更多信息: {board.get('question')}")
        return manifest

    # Stage 4: video
    print("\n=== Stage 4: prompted board → per-shot mp4s ===")
    board = produce_video(board, provider=provider, output_dir=root / "segments")
    board_with_mp4s = root / "board_with_mp4s.json"
    board_with_mp4s.write_text(json.dumps(board, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["stages"]["segments"] = str(root / "segments")
    print(f"   ✅ → {len(board.get('segments', []))} segments in {root / 'segments'}")

    # Stage 5: compose
    print("\n=== Stage 5: compose → final.mp4 ===")
    final = compose(board, output_dir=root)
    manifest["stages"]["final"] = str(final)
    print(f"   ✅ → {final}")

    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n🏁 全流程完成！清单 → {manifest_path}")
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("idea", nargs="?", help="视频灵感")
    p.add_argument("--backend", choices=["auto", "baseline", "vimax"], default="auto")
    p.add_argument("--budget", choices=["economy", "standard", "ample"], default="standard")
    p.add_argument("--slots", type=int, nargs="+", help="每个 beat 的秒数")
    p.add_argument("--provider", choices=["seedance", "minimax"], default="minimax")
    p.add_argument("--run-name", help="runs/{name}/ 下的输出目录名")
    p.add_argument("--stage", choices=["prompts", "video", "compose", "full"], default="full",
                   help="只跑某一阶段或完整流水线")
    p.add_argument("--board", type=Path, help="video / compose 阶段输入的 board JSON 路径")
    p.add_argument("--audio", type=Path, help="compose 阶段叠加的音频轨道")
    args = p.parse_args()

    try:
        if args.stage == "prompts":
            board = produce_prompts(args.idea or input("灵感："), args.slots, args.backend, args.budget)
            out = Path("runs") / (args.run_name or uuid4().hex[:12])
            out.mkdir(parents=True, exist_ok=True)
            path = out / "prompted_board.json"
            path.write_text(json.dumps(board, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"✅ prompts → {path}")

        elif args.stage == "video":
            if not args.board:
                print("--stage video 需要 --board", file=sys.stderr); return 1
            board = json.loads(args.board.read_text(encoding="utf-8"))
            out = Path("runs") / (args.run_name or uuid4().hex[:12])
            out.mkdir(parents=True, exist_ok=True)
            produce_video(board, args.provider, out / "segments")

        elif args.stage == "compose":
            if not args.board:
                print("--stage compose 需要 --board", file=sys.stderr); return 1
            board = json.loads(args.board.read_text(encoding="utf-8"))
            out = Path("runs") / (args.run_name or uuid4().hex[:12])
            out.mkdir(parents=True, exist_ok=True)
            compose(board, out, args.audio)

        else:  # full
            manifest = run_full(
                args.idea or input("请输入视频灵感："),
                args.slots, args.backend, args.budget,
                args.provider, args.run_name,
            )
            final = manifest["stages"].get("final", "(未产出最终视频)")
            print(f"\n📼 最终视频: {final}")
            print(f"📁 全部产物: runs/{manifest['run_name']}/")

    except Exception as exc:
        print(f"\n❌ 失败：{exc}", file=sys.stderr)
        import traceback; traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
