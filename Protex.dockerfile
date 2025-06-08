FROM python:3.9-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire video_pipeline package and config
COPY video_pipeline/ ./video_pipeline/
COPY default_model_config.yaml .

EXPOSE 8000

ENTRYPOINT ["python", "-m", "video_pipeline.pipeline"]

