FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

RUN pip3 install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip3 install --no-cache-dir -U openmim && \
    mim install mmengine && \
    mim install "mmcv==2.0.1" && \
    mim install "mmdet==3.1.0" && \
    pip3 install --no-cache-dir --no-build-isolation "mmpose==1.1.0"

COPY pyproject.toml .

RUN pip3 install --no-cache-dir \
    diffusers==0.30.2 \
    accelerate==0.28.0 \
    transformers==4.39.2 \
    "huggingface_hub>=0.20.0" \
    numpy==1.23.5 \
    "scipy>=1.10.0" \
    "librosa>=0.10.0" \
    "soundfile>=0.12.0" \
    "opencv-python>=4.8.0" \
    "imageio[ffmpeg]" \
    ffmpeg-python \
    moviepy \
    "gfpgan>=1.3.8" \
    "facexlib>=0.3.0" \
    "basicsr>=1.4.2" \
    "realesrgan>=0.3.0" \
    "fastapi>=0.100.0" \
    "uvicorn[standard]>=0.22.0" \
    "python-multipart>=0.0.6" \
    omegaconf \
    "einops>=0.7.0" \
    "pyyaml>=6.0" \
    tqdm \
    gdown \
    requests \
    "gradio>=4.0.0" \
    "filterpy>=1.4.5" \
    "lmdb>=1.4.0" \
    "yapf>=0.40.0"

ENV PYTHONPATH=/app:$PYTHONPATH

COPY api/ ./api/
COPY musetalk/ ./musetalk/
COPY configs/ ./configs/
COPY run_musetalk.py .
COPY app.py .
COPY download_models.py .
COPY download_weights.sh .

RUN mkdir -p /app/models /app/results /app/data

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
