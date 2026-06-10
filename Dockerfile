FROM python:3.11-slim

LABEL maintainer="YT-Short-Clipper Contributors"
LABEL description="YT-Short-Clipper v2 — AI-powered YouTube short-form clip generator"
LABEL version="2.0.0"

# ── System dependencies ──────────────────────────────────────────────────
# ffmpeg:         video processing (cut, convert, burn captions)
# git:            yt-dlp auto-update, pip VCS installs
# libgl1-mesa-glx + libglib2.0-0: OpenCV headless runtime deps
# curl:           healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0t64 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user ────────────────────────────────────────────────────────
RUN groupadd -r clipper && useradd -r -g clipper -d /app -s /sbin/nologin clipper

# ── Application setup ────────────────────────────────────────────────────
WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# ── Runtime directories ──────────────────────────────────────────────────
RUN mkdir -p /app/output /app/logs \
    && chown -R clipper:clipper /app

# ── Environment defaults ─────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# ── Ports ────────────────────────────────────────────────────────────────
# 7860: Gradio web UI (RFN-33)
EXPOSE 7860

# ── Health check ─────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# ── Entrypoint ───────────────────────────────────────────────────────────
USER clipper

# server.py (RFN-33) is the Gradio entrypoint.
# Until server.py exists, this image supports CLI/test invocations:
#   docker run --rm yt-clipper python -m pytest tests/
#   docker run --rm yt-clipper python pipeline/downloader.py
ENTRYPOINT ["python"]
CMD ["server.py"]
