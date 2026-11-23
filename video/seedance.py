"""Seedance (火山方舟) + MiniMax video providers.

Both implement the VideoProvider Protocol:
  submit(prompt, duration_seconds, aspect_ratio, first_frame_image_b64) -> task_id
  poll(task_id, timeout, interval) -> {'status':..., 'video_url':..., 'error':..., 'raw':...}
"""
from __future__ import annotations
import os
import time
import urllib.parse

from .base import VideoProvider, VideoSubmissionError, api_call


# ---------------------------------------------------------------------------
# Seedance (火山方舟)
# ---------------------------------------------------------------------------
SUBMIT_URL = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
QUERY_URL  = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks/{task_id}"


class SeedanceProvider:
    name = "seedance"
    min_duration = 2
    max_duration = 10

    def __init__(self, api_key: str | None = None, model_id: str | None = None):
        self.api_key = api_key or os.environ.get("ARK_API_KEY", "")
        self.model_id = model_id or os.environ.get("SEEDREAM_MODEL_ID", "")
        if not self.api_key or not self.model_id:
            raise VideoSubmissionError("需要 ARK_API_KEY + SEEDREAM_MODEL_ID 环境变量")

    def submit(self, prompt: str, duration_seconds: int = 5,
               aspect_ratio: str = "16:9",
               first_frame_image_b64: str | None = None) -> str:
        content = [{"type": "text", "text": prompt}]
        if first_frame_image_b64:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{first_frame_image_b64}"}})
        body = {"model": self.model_id, "content": content,
                "duration": duration_seconds, "ratio": aspect_ratio}
        resp = api_call("POST", SUBMIT_URL, self.api_key, body)
        task_id = (resp.get("id") or resp.get("task_id")
                   or resp.get("task", {}).get("id"))
        if not task_id:
            raise VideoSubmissionError(f"Seedance 未返回 task id: {resp}")
        return str(task_id)

    def poll(self, task_id: str, timeout: float = 300, interval: float = 10) -> dict:
        url = QUERY_URL.format(task_id=task_id)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            resp = api_call("GET", url, self.api_key)
            status = (resp.get("status") or resp.get("state")
                      or resp.get("task", {}).get("status", "")).lower()
            if status in ("succeeded", "success", "completed", "done"):
                video_url = self._extract_video_url(resp)
                return {"status": "succeeded", "video_url": video_url, "error": None, "raw": resp}
            if status in ("failed", "cancelled", "canceled", "error"):
                err = resp.get("error") or resp.get("message") or str(resp)[:300]
                return {"status": "failed", "video_url": None, "error": err, "raw": resp}
            time.sleep(min(interval, max(0, deadline - time.monotonic())))
        return {"status": "timeout", "video_url": None,
                "error": f"超过 {timeout}s 未完成", "raw": resp}

    @staticmethod
    def _extract_video_url(resp: dict) -> str | None:
        if isinstance(resp.get("video_url"), str):
            return resp["video_url"]
        if isinstance(resp.get("output_url"), str):
            return resp["output_url"]
        data = resp.get("data") or resp.get("task") or {}
        if isinstance(data.get("video_url"), str):
            return data["video_url"]
        if isinstance(data.get("output_url"), str):
            return data["output_url"]
        for item in (resp.get("content") or resp.get("output") or []):
            url = item.get("video_url") or item.get("url")
            if isinstance(url, str) and url.startswith("http"):
                return url
        return None


# ---------------------------------------------------------------------------
# MiniMax H3
# ---------------------------------------------------------------------------
MINIMAX_SUBMIT = "https://api.minimax.cn/v2/video_generation"
MINIMAX_QUERY  = "https://api.minimax.cn/v2/query/video_generation/"


class MiniMaxProvider:
    name = "minimax"
    min_duration = 4
    max_duration = 15

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        self.model = model or os.environ.get("MINIMAX_VIDEO_MODEL_ID", "minimax-h3")
        if not self.api_key:
            raise VideoSubmissionError("需要 MINIMAX_API_KEY 环境变量")

    def submit(self, prompt: str, duration_seconds: int = 5,
               aspect_ratio: str = "16:9",
               first_frame_image_b64: str | None = None) -> str:
        body = {"model": self.model,
                "content": [{"type": "text", "text": prompt}],
                "duration": duration_seconds, "resolution": "768P",
                "ratio": aspect_ratio}
        if first_frame_image_b64:
            body["content"].append({"type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{first_frame_image_b64}"}})
        resp = api_call("POST", MINIMAX_SUBMIT, self.api_key, body)
        task_id = resp.get("task_id") or resp.get("task", {}).get("id")
        if not task_id:
            raise VideoSubmissionError(f"MiniMax 未返回 task_id: {resp}")
        return str(task_id)

    def poll(self, task_id: str, timeout: float = 300, interval: float = 10) -> dict:
        url = MINIMAX_QUERY + urllib.parse.quote(task_id, safe="")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            resp = api_call("GET", url, self.api_key)
            task = resp.get("task", {})
            status = task.get("status", "").lower()
            if status == "succeeded":
                video_url = None
                content = task.get("content")
                if isinstance(content, dict):
                    video_url = content.get("url")
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("url"):
                            video_url = item["url"]
                            break
                return {"status": "succeeded", "video_url": video_url, "error": None, "raw": resp}
            if status in ("failed", "cancelled", "canceled"):
                return {"status": "failed", "video_url": None,
                        "error": task.get("error", status), "raw": resp}
            time.sleep(min(interval, max(0, deadline - time.monotonic())))
        return {"status": "timeout", "video_url": None,
                "error": f"超过 {timeout}s 未完成", "raw": resp}
