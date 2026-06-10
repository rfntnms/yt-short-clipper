FROM python:3.11-slim

LABEL org.opencontainers.image.title="YT-Short-Clipper v2"
LABEL org.opencontainers.image.description="Self-hosted Gradio runtime for AI podcast/video clipping"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    APP_HOME=/app

WORKDIR /app

# Runtime dependencies:
# - ffmpeg: video/audio processing
# - curl: healthcheck and simple diagnostics
# - libgl1/libglib2.0-0: OpenCV runtime libraries
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN groupadd --system clipper \
    && useradd --system --gid clipper --home-dir /app --shell /usr/sbin/nologin clipper \
    && mkdir -p /app/output /app/tmp \
    && chown -R clipper:clipper /app

USER clipper

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:7860/ >/dev/null || exit 1

CMD ["python", "server.py"]
