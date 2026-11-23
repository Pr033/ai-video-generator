"""FastAPI backend for AI Video Generator.

Usage:
    cd d:\\Trae_python\\TraeScriptStoryboard\\TraeScriptAgent\\web
    python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /                     → index.html
    POST /api/generate         → submit a generation job (async, returns task_id)
    GET  /api/status/{task_id} → poll progress + manifest
    GET  /api/video/{task_id}  → download final.mp4
    GET  /api/artifacts/{task_id}/{path} → view/download any artifact
"""
from __future__ import annotations
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Make parent importable
PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT))
os.chdir(PARENT)

from orchestrator import OrchestratorAgent  # noqa: E402

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="AI Video Generator", version="1.0")
WEB_DIR = Path(__file__).resolve().parent

# 持久化存储路径：优先用环境变量（Render Disk 挂载点），否则用本地 runs/
RUNS_DIR = Path(os.environ.get("RUNS_DIR", str(PARENT / "runs")))
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory task store (simple, fine for single-user demo)
tasks: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    idea: str = Field(..., min_length=3, max_length=500,
                      description="视频灵感，例如 '雨停后的校园，15秒，电影感'")
    slots: Optional[list[int]] = Field(default=None, description="每个 beat 的秒数")
    dry_run: bool = Field(default=False, description="不调任何 API，验证流程")
    video_provider: str = Field(default="minimax", pattern="^(seedance|minimax)$")
    script_backend: str = Field(default="auto", pattern="^(auto|baseline|vimax)$")

    # API credentials — user fills in per-request, NEVER persisted on disk
    chat_api_key: Optional[str] = None
    chat_model: Optional[str] = None
    chat_base_url: Optional[str] = None
    seedream_model_id: Optional[str] = None  # Seedance endpoint ID
    minimax_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    return {"ok": True, "ffmpeg": __import__("video").check_ffmpeg()}


@app.post("/api/generate")
def generate(req: GenerateRequest):
    """Submit a job. Returns task_id immediately; poll /api/status/{id} for progress."""
    task_id = uuid.uuid4().hex[:12]
    run_dir = RUNS_DIR / f"web_{task_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot current env, inject user-provided keys (for this thread only)
    keys_to_set = {}
    if req.chat_api_key:
        keys_to_set["CHAT_API_KEY"] = req.chat_api_key
    if req.chat_model:
        keys_to_set["CHAT_MODEL"] = req.chat_model
    if req.chat_base_url:
        keys_to_set["CHAT_BASE_URL"] = req.chat_base_url
    if req.seedream_model_id:
        keys_to_set["SEEDREAM_MODEL_ID"] = req.seedream_model_id
        keys_to_set["SEED_MODEL_ID"] = req.seedream_model_id
    if req.minimax_api_key:
        keys_to_set["MINIMAX_API_KEY"] = req.minimax_api_key
    if req.openai_api_key:
        keys_to_set["OPENAI_API_KEY"] = req.openai_api_key
    if req.elevenlabs_api_key:
        keys_to_set["ELEVENLABS_API_KEY"] = req.elevenlabs_api_key

    tasks[task_id] = {
        "task_id": task_id,
        "idea": req.idea[:80],
        "status": "queued",
        "stages": {},
        "final_video": None,
        "run_dir": str(run_dir),
        "started_at": time.time(),
    }

    def _worker():
        # Apply env vars (Windows process-wide, fine for demo)
        for k, v in keys_to_set.items():
            os.environ[k] = v
        try:
            orch = OrchestratorAgent(run_dir, dry_run=req.dry_run,
                                     video_provider=req.video_provider,
                                     script_backend=req.script_backend)
            tasks[task_id]["status"] = "running"
            # OrchestratorAgent.run() writes manifest.json with per-stage status
            manifest = orch.run(req.idea, req.slots)
            final = manifest["stages"].get("compose", {}).get("artifact")
            tasks[task_id]["final_video"] = final
            # Final status: done if compose succeeded, failed otherwise
            if final and Path(final).exists():
                tasks[task_id]["status"] = "done"
            else:
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["error"] = "compose did not produce final video"
        except Exception as exc:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = str(exc)
        finally:
            tasks[task_id]["finished_at"] = time.time()

    threading.Thread(target=_worker, daemon=True).start()
    return JSONResponse({"task_id": task_id, "status": "queued",
                          "message": "提交成功，轮询 /api/status/{task_id}"})


@app.get("/api/status/{task_id}")
def status(task_id: str):
    t = tasks.get(task_id)
    if not t:
        raise HTTPException(404, "task not found")
    # Reload manifest.json for freshest stage data
    run_dir = Path(t["run_dir"])
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            mf = json.loads(manifest_path.read_text(encoding="utf-8"))
            t["manifest"] = mf
            # Sync status from manifest
            if mf["stages"].get("compose", {}).get("artifact"):
                t["status"] = "done"
        except Exception:
            pass
    # Don't leak API keys
    safe = {k: v for k, v in t.items() if k not in ("chat_api_key", "ark_api_key")}
    return safe


@app.get("/api/video/{task_id}")
def download_video(task_id: str):
    t = tasks.get(task_id)
    if not t:
        raise HTTPException(404, "task not found")
    final = t.get("final_video")
    if not final or not Path(final).exists():
        raise HTTPException(404, "video not ready")
    return FileResponse(final, media_type="video/mp4", filename=f"{task_id}_final.mp4")


@app.get("/api/artifacts/{task_id}/{path:path}")
def artifact(task_id: str, path: str):
    """Stream any artifact (script.json, board.json, etc.) for preview."""
    t = tasks.get(task_id)
    if not t:
        raise HTTPException(404, "task not found")
    target = Path(t["run_dir"]) / path
    target = target.resolve()
    # Prevent path traversal
    if not str(target).startswith(str(Path(t["run_dir"]).resolve())):
        raise HTTPException(400, "invalid path")
    if not target.exists():
        raise HTTPException(404, "not found")
    if target.suffix == ".json":
        return JSONResponse(json.loads(target.read_text(encoding="utf-8")))
    return FileResponse(target)


# ---------------------------------------------------------------------------
# Progress-aware runner — wraps OrchestratorAgent.run to record stage-by-stage
# ---------------------------------------------------------------------------

def _run_with_progress(orch: OrchestratorAgent, idea: str, slots, task_id: str) -> dict:
    """Run orchestrator; after each stage, persist stage status into tasks[task_id]."""
    from orchestrator import STAGES

    board = {}
    for stage in STAGES:
        tasks[task_id]["stages"][stage] = "running"
        try:
            # Reuse _run_stage logic directly
            board = orch._run_stage(stage, board, idea, slots)
            status = orch.manifest["stages"].get(stage, {})
            tasks[task_id]["stages"][stage] = status.get("status", "done")
        except Exception as exc:
            tasks[task_id]["stages"][stage] = f"failed: {exc}"
            # Store manifest as-is so caller can inspect which stage failed
            orch._save_manifest()
            raise
    orch._save_manifest()
    return orch.manifest


# ---------------------------------------------------------------------------
# Static files (JS/CSS under web/static/)
# ---------------------------------------------------------------------------
static_dir = WEB_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ---------------------------------------------------------------------------
# Dev entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
