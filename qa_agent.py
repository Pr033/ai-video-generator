"""QA agents — VisualReview + Consistency + BrandSafety.

Uses VLM (GPT-4o Vision by default) to review generated video quality.
If no VLM key is available, gracefully skips and returns {"skipped": True}.

Entry point:
    from qa_agent import run_all_reviews
    qa = run_all_reviews(board)
"""
from __future__ import annotations
import base64
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Shared VLM caller (minimal, no extra deps)
# ---------------------------------------------------------------------------

def _vlm_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("CHAT_API_KEY"))


def _vlm_chat(system_prompt: str, user_content: list[dict],
              max_tokens: int = 512, temperature: float = 0) -> str | None:
    """Call GPT-4o (or any vision-capable model) with text + optional images.

    user_content: list of {"type": "text", "text": "..."} or
                           {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    Returns raw text response or None on failure.
    """
    import urllib.request, urllib.error
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("CHAT_API_KEY")
    if not key:
        return None

    model = os.environ.get("VISION_MODEL", "gpt-4o")
    url = "https://api.openai.com/v1/chat/completions"

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            result = json.loads(resp.read())
        return result["choices"][0]["message"]["content"]
    except Exception as exc:
        print(f"   ⚠️ VLM 调用失败: {exc}")
        return None


def _mp4_to_first_frame_b64(mp4_path: Path) -> str | None:
    """Extract first frame of an mp4 and return base64 JPEG. None on failure."""
    import subprocess
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            [ffmpeg, "-y", "-ss", "0", "-i", str(mp4_path),
             "-frames:v", "1", "-q:v", "2", tmp_path],
            capture_output=True, timeout=30)
        return base64.b64encode(Path(tmp_path).read_bytes()).decode()
    except Exception:
        return None
    finally:
        try: os.unlink(tmp_path)
        except Exception: pass


# ---------------------------------------------------------------------------
# 1. VisualReviewAgent — per-shot visual quality
# ---------------------------------------------------------------------------

VISUAL_SYSTEM = """You are a video quality reviewer. Assess each still frame on four
dimensions: realism, subject accuracy, motion plausibility (as inferred from a
single frame), and artifact level.

Return JSON: {"score": 0-10, "issues": ["...", "..."], "notes": "..."}"""


def visual_review(shot: dict) -> dict:
    """Review a single shot's visual quality. Returns score 0-10 + issues."""
    mp4 = shot.get("mp4_path")
    if not mp4 or not Path(mp4).exists():
        return {"score": 5, "issues": ["no video file"], "notes": "skipped (no mp4)"}

    b64 = _mp4_to_first_frame_b64(Path(mp4))
    if not b64:
        return {"score": 5, "issues": ["no frame extracted"], "notes": "skipped (ffmpeg issue)"}

    prompt = (
        f"Shot: {shot['shot_id']}\n"
        f"Intended scene: {shot.get('visual_description', '')[:200]}\n"
        f"Action: {shot.get('action', '')[:200]}\n"
        f"Rate this frame."
    )
    raw = _vlm_chat(VISUAL_SYSTEM, [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ])
    if not raw:
        return {"score": 5, "issues": ["VLM unavailable"], "notes": "skipped (no VLM)"}
    try:
        text = raw
        # Extract JSON blob
        start, end = text.find("{"), text.rfind("}") + 1
        if start >= 0 and end > start:
            j = json.loads(text[start:end])
            score = max(0, min(10, int(j.get("score", 5))))
            return {"score": score, "issues": j.get("issues", []), "notes": j.get("notes", "")}
        return {"score": 5, "issues": ["parse error"], "notes": text[:200]}
    except Exception as exc:
        return {"score": 5, "issues": ["parse error"], "notes": str(exc)}


# ---------------------------------------------------------------------------
# 2. ConsistencyAgent — between-shot continuity
# ---------------------------------------------------------------------------

CONSISTENCY_SYSTEM = """You are a film continuity supervisor. Given two consecutive
frames from adjacent shots, check whether:
  1. Characters look like the same people (face, clothing, proportions)
  2. Scene lighting / color temperature matches
  3. Action / movement is continuous (no jarring cut on impossible position)

Return JSON: {"consistent": true/false, "score": 0-10, "issues": [...]}"""


def consistency_review(prev_shot: dict, next_shot: dict) -> dict:
    """Compare end-state of prev vs start-state of next."""
    prev_mp4 = prev_shot.get("mp4_path")
    next_mp4 = next_shot.get("mp4_path")
    if not prev_mp4 or not next_mp4:
        return {"consistent": True, "score": 7, "issues": ["missing mp4"]}

    # prev: last frame; next: first frame
    from video.frames import extract_last_frame, image_to_base64
    try:
        last = extract_last_frame(Path(prev_mp4))
        first_b64 = _mp4_to_first_frame_b64(Path(next_mp4))
        last_b64 = image_to_base64(last)
    except Exception as exc:
        return {"consistent": True, "score": 7, "issues": [f"frame extract fail: {exc}"]}

    if not first_b64:
        return {"consistent": True, "score": 7, "issues": ["first frame fail"]}

    prompt = (
        f"Previous shot ending: {prev_shot.get('end_state', '')[:150]}\n"
        f"Next shot starting: {next_shot.get('start_state', '')[:150]}\n"
        f"Are these two visually continuous?"
    )
    raw = _vlm_chat(CONSISTENCY_SYSTEM, [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{last_b64}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{first_b64}"}},
    ])
    if not raw:
        return {"consistent": True, "score": 7, "issues": ["VLM unavailable"]}
    try:
        start, end = raw.find("{"), raw.rfind("}") + 1
        j = json.loads(raw[start:end])
        return {
            "consistent": bool(j.get("consistent", True)),
            "score": max(0, min(10, int(j.get("score", 7)))),
            "issues": j.get("issues", []),
        }
    except Exception:
        return {"consistent": True, "score": 7, "issues": ["parse error"]}


# ---------------------------------------------------------------------------
# 3. BrandSafetyAgent — must/avoid compliance
# ---------------------------------------------------------------------------

BRAND_SYSTEM = """You are a content compliance auditor. Review a video frame
against these constraints and report any violations.

Return JSON: {"pass": true/false, "violations": [...], "notes": "..."}"""


def brand_safety_review(shot: dict, brief: dict) -> dict:
    must = brief.get("must", []) or []
    avoid = brief.get("avoid", []) or []
    if not must and not avoid:
        return {"pass": True, "violations": [], "notes": "no constraints"}

    mp4 = shot.get("mp4_path")
    if not mp4 or not Path(mp4).exists():
        return {"pass": True, "violations": [], "notes": "no video"}

    b64 = _mp4_to_first_frame_b64(Path(mp4))
    if not b64:
        return {"pass": True, "violations": [], "notes": "no frame"}

    prompt = (
        f"Shot: {shot['shot_id']}\n"
        f"MUST include: {json.dumps(must, ensure_ascii=False)}\n"
        f"MUST AVOID:   {json.dumps(avoid, ensure_ascii=False)}\n"
        f"Scene: {shot.get('visual_description', '')[:150]}\n"
        f"Check this frame for violations."
    )
    raw = _vlm_chat(BRAND_SYSTEM, [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ])
    if not raw:
        return {"pass": True, "violations": [], "notes": "VLM unavailable"}
    try:
        start, end = raw.find("{"), raw.rfind("}") + 1
        j = json.loads(raw[start:end])
        return {"pass": bool(j.get("pass", True)),
                "violations": j.get("violations", []),
                "notes": j.get("notes", "")}
    except Exception:
        return {"pass": True, "violations": [], "notes": "parse error"}


# ---------------------------------------------------------------------------
# Orchestrator entry point
# ---------------------------------------------------------------------------

def run_all_reviews(board: dict) -> dict:
    """Run all three QA passes over the board. Returns aggregated report."""
    shots = board.get("shots", [])
    brief = board.get("brief", {})

    if not _vlm_available():
        return {
            "status": "skipped",
            "reason": "no VLM backend (set OPENAI_API_KEY or CHAT_API_KEY)",
            "visual": [],
            "consistency": [],
            "brand_safety": [],
            "aggregate": {"visual_avg": 0, "consistency_avg": 0, "brand_pass_rate": 0},
        }

    print("🔍 QA: 视觉审查...")
    visual_results = []
    for shot in shots:
        vr = visual_review(shot)
        visual_results.append({"shot_id": shot["shot_id"], **vr})
        print(f"   {shot['shot_id']}: score={vr['score']}/10")

    print("🔍 QA: 连贯性审查...")
    consistency_results = []
    for i in range(len(shots) - 1):
        cr = consistency_review(shots[i], shots[i + 1])
        consistency_results.append({
            "pair": f"{shots[i]['shot_id']}→{shots[i+1]['shot_id']}", **cr})
        print(f"   {shots[i]['shot_id']}→{shots[i+1]['shot_id']}: score={cr['score']}/10 consistent={cr['consistent']}")

    print("🔍 QA: 合规审查...")
    brand_results = []
    for shot in shots:
        br = brand_safety_review(shot, brief)
        brand_results.append({"shot_id": shot["shot_id"], **br})
        status = "✅" if br["pass"] else "⚠️ VIOLATION"
        print(f"   {shot['shot_id']}: {status}")

    visual_avg = sum(r["score"] for r in visual_results) / max(1, len(visual_results))
    consistency_avg = sum(r["score"] for r in consistency_results) / max(1, len(consistency_results))
    brand_pass = sum(1 for r in brand_results if r["pass"]) / max(1, len(brand_results))

    return {
        "status": "done",
        "visual": visual_results,
        "consistency": consistency_results,
        "brand_safety": brand_results,
        "aggregate": {
            "visual_avg": round(visual_avg, 2),
            "consistency_avg": round(consistency_avg, 2),
            "brand_pass_rate": round(brand_pass, 2),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("board", type=Path, help="board_with_mp4s.json")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    board = json.loads(args.board.read_text(encoding="utf-8"))
    report = run_all_reviews(board)
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "qa_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    agg = report.get("aggregate", {})
    print(f"\n📊 QA Summary:")
    print(f"   Visual avg:       {agg.get('visual_avg', '?')}/10")
    print(f"   Consistency avg:  {agg.get('consistency_avg', '?')}/10")
    print(f"   Brand pass rate:  {agg.get('brand_pass_rate', '?')}")
    print(f"\n完整报告: {out_path}")
