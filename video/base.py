"""Video generation providers and shared submission/poll/download logic.

Adapted from AIE1901_Three_Agents; decoupled from video_agents.common.AgentError.
"""
from __future__ import annotations
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol


class VideoSubmissionError(Exception):
    """Raised when video task creation / polling / download fails."""


class VideoProvider(Protocol):
    """Interface that any video generation backend must implement."""
    name: str
    min_duration: int
    max_duration: int

    def submit(self, prompt: str, duration_seconds: int,
               aspect_ratio: str, first_frame_image_b64: str | None = None) -> str:
        """Submit one video task. Returns a task_id string. No blocking poll."""
        ...

    def poll(self, task_id: str, timeout: float = 300, interval: float = 10) -> dict:
        """Poll until terminal state or timeout.
        Returns {'status': ..., 'video_url': ..., 'error': ..., 'raw': ...}.
        """
        ...


def api_call(method: str, url: str, key: str,
             body: dict | None = None, timeout: float = 90) -> dict:
    """Simple JSON POST/GET with Bearer auth; no auto retry."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        return json.loads(raw)
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        exc.close()
        hint = {400: "检查请求体格式和模型ID", 401: "检查API Key",
                403: "检查项目权限/模型是否已开通", 429: "限流或额度不足"}.get(exc.code, "服务端错误")
        raise VideoSubmissionError(f"HTTP {exc.code} ({hint}): {detail[:300]}") from None
    except Exception as exc:
        raise VideoSubmissionError(f"网络或解析错误: {exc}") from None


def download_file(url: str, dest: Path, timeout: float = 120) -> Path:
    """Download HTTPS media URL to dest. Supports .mp4 / .mov."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, tmp.open("wb") as f:
            while chunk := resp.read(65536):
                f.write(chunk)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def fingerprint(body: dict) -> str:
    """Stable hash for request deduplication / caching."""
    raw = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()
