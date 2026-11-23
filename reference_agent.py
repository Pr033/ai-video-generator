"""ReferenceImageAgent — generate keyframe reference images for image-to-video.

For opening / first shots, generates a visual anchor so the downstream
VideoSegmentAgent can submit a consistent first-frame reference.
"continue" shots still use the predecessor's last frame (handled by VideoSegmentAgent).

Supports:
  - DALL·E 3 (OpenAI key → OPENAI_API_KEY env)
  - FLUX / SDXL via Replicate (REPLICATE_API_TOKEN env)

If no image backend is configured, the agent gracefully skips — downstream
video generation falls back to pure text-to-video.

Usage:
    from reference_agent import ReferenceImageAgent
    agent = ReferenceImageAgent(output_dir="runs/demo")
    board = agent.run(prompted_board_json)
    # board['shots'][i]['reference_image_path'] set for opening shots
"""
from __future__ import annotations
import json
import os
import urllib.error
import urllib.request
from copy import deepcopy
from pathlib import Path


class ReferenceImageAgent:
    """Generate a keyframe image per opening / first shot."""

    def __init__(self, output_dir: str | Path, backend: str = "auto",
                 size: str = "1024x576"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend  # "auto" | "dalle" | "replicate"
        self.size = size
        self._resolved = self._resolve_backend()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, board: dict) -> dict:
        """Generate reference images; returns updated board with reference_image_path filled."""
        if self._resolved is None:
            print("🖼️ ReferenceImageAgent: 未配置图像生成后端，跳过（纯文生视频）")
            return board

        if board.get("status") != "ready" or not board.get("shots"):
            raise ValueError("需要 ready 状态的 prompted board")

        shots = board["shots"]
        new_board = deepcopy(board)
        generated = 0

        for i, shot in enumerate(new_board["shots"]):
            # Only generate for opening / first shot — "continue" shots
            # use predecessor's last frame (VideoSegmentAgent handles that)
            is_opening = (i == 0) or (shot.get("transition") == "opening")
            if not is_opening:
                continue

            prompt = shot.get("video_prompt") or shot.get("visual_description", "")
            brief = new_board.get("brief", {})
            style = brief.get("style", "")
            full_prompt = self._build_image_prompt(prompt, style)

            img_path = self._generate_one(shot["shot_id"], full_prompt)
            if img_path:
                shot["reference_image_path"] = str(img_path)
                shot["reference_image_size"] = self.size
                generated += 1
                print(f"   ✅ {shot['shot_id']} → {img_path.name}")

        new_board["status"] = "prompted"
        print(f"🖼️ ReferenceImageAgent: 生成 {generated} 张参考图")
        return new_board

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_backend(self) -> str | None:
        if self.backend != "auto":
            return self.backend if self.backend in ("dalle", "replicate") else None
        if os.environ.get("OPENAI_API_KEY"):
            return "dalle"
        if os.environ.get("REPLICATE_API_TOKEN"):
            return "replicate"
        return None

    @staticmethod
    def _build_image_prompt(video_prompt: str, style: str) -> str:
        """Clean video prompt into a strong image-generation prompt."""
        # Remove motion / camera / duration keywords irrelevant for a still
        stop = {
            "5-second", "10-second", "medium shot", "close up", "close-up",
            "opening state", "end state", "final shot",
            "slow push", "push-in", "horizontal tracking", "fixed flat angle",
            "slow horizontal tracking movement",
            "no dialogue", "no narration", "no dialogue no narration",
            "camera movement",
        }
        cleaned = video_prompt
        for term in sorted(stop, key=len, reverse=True):
            cleaned = cleaned.replace(term, "")
        # Trim stray punctuation / whitespace
        cleaned = " ".join(cleaned.split())
        if style and style != "unspecified":
            cleaned = f"{cleaned}, {style} style"
        # Image models respond better to comma-separated descriptors
        return cleaned[:400]  # truncate for safety

    def _generate_one(self, shot_id: str, prompt: str) -> Path | None:
        out_dir = self.output_dir / "references"
        out_dir.mkdir(parents=True, exist_ok=True)
        img = out_dir / f"{shot_id}_ref.jpg"

        if self._resolved == "dalle":
            return self._dalle(prompt, img)
        elif self._resolved == "replicate":
            return self._replicate(prompt, img)
        return None

    # ---- DALL·E 3 ----

    def _dalle(self, prompt: str, out: Path) -> Path | None:
        key = os.environ["OPENAI_API_KEY"]
        body = json.dumps({
            "model": "dall-e-3",
            "prompt": prompt[:4000],
            "n": 1, "size": "1024x576", "quality": "standard",
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/images/generations", data=body, method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            url = data["data"][0]["url"]
            with urllib.request.urlopen(url, timeout=60) as dl, out.open("wb") as f:
                f.write(dl.read())
            return out if out.exists() else None
        except Exception as exc:
            print(f"   ⚠️ DALL·E 失败: {exc}")
            return None

    # ---- Replicate FLUX (optional) ----

    def _replicate(self, prompt: str, out: Path) -> Path | None:
        key = os.environ["REPLICATE_API_TOKEN"]
        # FLUX.1-schnell
        body = json.dumps({
            "model": "black-forest-labs/flux-schnell",
            "input": {"prompt": prompt, "width": 1024, "height": 576, "num_inference_steps": 4},
        }).encode()
        req = urllib.request.Request(
            "https://api.replicate.com/v1/predictions", data=body, method="POST",
            headers={"Authorization": f"Token {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            pred_id = data.get("id")
            # Poll
            import time
            for _ in range(30):
                time.sleep(2)
                qreq = urllib.request.Request(
                    f"https://api.replicate.com/v1/predictions/{pred_id}",
                    headers={"Authorization": f"Token {key}"})
                with urllib.request.urlopen(qreq, timeout=30) as qresp:
                    qdata = json.loads(qresp.read())
                status = qdata.get("status")
                if status == "succeeded":
                    url = qdata["output"][0]
                    with urllib.request.urlopen(url, timeout=60) as dl, out.open("wb") as f:
                        f.write(dl.read())
                    return out
                if status in ("failed", "canceled"):
                    break
            print(f"   ⚠️ Replicate 超时或失败")
        except Exception as exc:
            print(f"   ⚠️ Replicate 失败: {exc}")
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("board", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--backend", choices=["auto", "dalle", "replicate"], default="auto")
    args = p.parse_args()

    board = json.loads(args.board.read_text(encoding="utf-8"))
    agent = ReferenceImageAgent(args.out, args.backend)
    result = agent.run(board)
    (args.out / "board_with_refs.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
