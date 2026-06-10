FROM python:3.11-slim

# Install system dependencies needed for FFmpeg and OpenCV
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create volume mount points for runtime data
RUN mkdir -p /app/output /app/models /app/.tmp

# Copy the rest of the application
COPY . .

# Expose Gradio port
EXPOSE 7860

# Define volumes
VOLUME ["/app/output", "/app/models", "/app/.tmp", "/app/config.json"]

# Run the Gradio server
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "7860"]