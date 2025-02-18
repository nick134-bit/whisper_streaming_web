# Use NVIDIA CUDA 12.3 with cuDNN 9
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install PyTorch first
RUN pip3 install --no-cache-dir torch>=2.1.0

# Install other Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir --force-reinstall "ctranslate2>=4.4.0,<5.0.0"

# Pre-download and cache the Silero VAD model
RUN python3 -c 'import torch; torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True, force_reload=True)'

# Copy the rest of the application
COPY . .

# Expose the port the app runs on (both HTTP and WebSocket)
EXPOSE 8000/tcp
EXPOSE 8000/udp

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["python3", "whisper_fastapi_online_server.py", "--host", "0.0.0.0", "--port", "8000"] 