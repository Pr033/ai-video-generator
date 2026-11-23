# ===================================================================
# AI Video Generator — Dockerfile
# 基础镜像 + 系统 ffmpeg + Python 依赖 + FastAPI 服务
#
# 持久化说明：
#   Render Disk 挂载到 /data，应用通过 RUNS_DIR=/data/runs 写入产物。
#   只有 /data 下的文件跨部署/重启保留，其他目录仍是临时文件系统。
# ===================================================================
FROM python:3.11-slim

# 系统依赖：ffmpeg (composer.py / audio_agent.py 直接调用系统 ffmpeg)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 工作目录
WORKDIR /app

# 先复制 requirements 利用 Docker 缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制全部源码
COPY . .

# 创建默认 runs 目录（本地开发 / 无 disk 时用）
# /data/runs 是 Render Disk 挂载点（容器启动时自动存在）
RUN mkdir -p runs /data/runs

# FastAPI 端口
EXPOSE 8000

# Zeabur / Render 会注入 PORT 环境变量；默认 8000
ENV PORT=8000
ENV RUNS_DIR=/app/runs

# 启动命令 —— 用 $PORT 适配平台动态端口分配
CMD uvicorn web.app:app --host 0.0.0.0 --port ${PORT}
