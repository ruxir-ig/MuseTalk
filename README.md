# MuseTalk API

Production-ready REST API for real-time lip-synced video generation. Fork of [TMElyralab/MuseTalk](https://github.com/TMElyralab/MuseTalk) with Docker support, optimized GFPGAN enhancement (~1.8x faster), and a clean HTTP interface.

## Features

- **REST API** - Simple HTTP endpoints for video generation
- **Docker-first** - Production-ready container with GPU support
- **Optimized GFPGAN** - Face enhancement runs ~1.8x faster (skips redundant face detection)
- **Streaming downloads** - Large video files served via chunked transfer
- **Multi-language audio** - Supports Chinese, English, Japanese, and more

## Quick Start

### Docker (Recommended)

```bash
# Build the image
docker build -t musetalk-api .

# Run with GPU support (models downloaded on first run)
docker run --gpus all -p 8000:8000 \
  -v ./models:/app/models \
  -v ./results:/app/results \
  musetalk-api

# Check health
curl http://localhost:8000/health
```

### Local Development

```bash
# Install dependencies
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Download model weights
sh download_weights.sh

# Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/health` | GET | Health check with GPU info |
| `/generate` | POST | Generate video (file upload) |
| `/generate/json` | POST | Generate video (path-based) |
| `/download/{filename}` | GET | Download generated video |

### Generate Video (File Upload)

Upload source image/video and audio files directly:

```bash
curl -X POST http://localhost:8000/generate \
  -F "source=@photo.jpg" \
  -F "audio=@speech.mp3" \
  -F "enhance=true" \
  -F "output_name=my_video"
```

**Response:**
```json
{
  "status": "success",
  "filename": "my_video.mp4",
  "download_url": "/download/my_video.mp4",
  "file_size_bytes": 12345678,
  "processing_time_seconds": 45.2
}
```

**Download the result:**
```bash
curl http://localhost:8000/download/my_video.mp4 --output my_video.mp4
```

### Generate Video (Path-Based)

For files already on the server:

```bash
curl -X POST http://localhost:8000/generate/json \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/app/data/photo.jpg",
    "audio_path": "/app/data/speech.mp3",
    "enhance": true,
    "gfpgan_weight": 0.5,
    "output_name": "my_video"
  }'
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` / `video_path` | file/string | required | Source image or video |
| `audio` / `audio_path` | file/string | required | Driving audio file |
| `enhance` | bool | `false` | Apply GFPGAN face enhancement |
| `gfpgan_weight` | float | `0.5` | Enhancement blend (0=original, 1=full) |
| `bbox_shift` | int | `0` | Face bbox vertical shift (affects mouth openness) |
| `extra_margin` | int | `10` | Extra margin for jaw movement |
| `parsing_mode` | string | `"jaw"` | Face blending mode: `"jaw"` or `"raw"` |
| `fps` | int | `25` | Output video FPS |
| `batch_size` | int | `8` | Inference batch size |
| `output_name` | string | auto | Custom output filename (without extension) |

### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_name": "NVIDIA L4",
  "gpu_memory_gb": 22.5,
  "models_loaded": true,
  "gfpgan_available": true
}
```

## Examples

### Basic lip-sync (no enhancement)

```bash
curl -X POST http://localhost:8000/generate \
  -F "source=@portrait.jpg" \
  -F "audio=@narration.wav" \
  -o response.json

# Get download URL from response and download
curl http://localhost:8000/download/portrait_narration.mp4 --output result.mp4
```

### With GFPGAN face enhancement

```bash
curl -X POST http://localhost:8000/generate \
  -F "source=@portrait.jpg" \
  -F "audio=@narration.wav" \
  -F "enhance=true" \
  -F "gfpgan_weight=0.7" \
  -F "output_name=enhanced_result"
```

### Adjust mouth openness

Positive `bbox_shift` = more open mouth, negative = more closed:

```bash
curl -X POST http://localhost:8000/generate \
  -F "source=@portrait.jpg" \
  -F "audio=@narration.wav" \
  -F "bbox_shift=-5" \
  -F "output_name=subtle_mouth"
```

### Using video as source

Works with video files too (uses first frame as reference):

```bash
curl -X POST http://localhost:8000/generate \
  -F "source=@input_video.mp4" \
  -F "audio=@new_audio.mp3" \
  -F "enhance=true"
```

## Docker Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | all | GPU device(s) to use |

### Volume Mounts

| Path | Description |
|------|-------------|
| `/app/models` | Model weights (persisted) |
| `/app/results` | Generated videos |
| `/app/data` | Input files (for path-based API) |

### Production Deployment

```bash
docker run -d --gpus all \
  --name musetalk \
  --restart unless-stopped \
  -p 8000:8000 \
  -v /path/to/models:/app/models \
  -v /path/to/results:/app/results \
  -v /path/to/inputs:/app/data \
  musetalk-api
```

## Performance

### GFPGAN Optimization

This fork includes an optimized GFPGAN pipeline that's ~1.8x faster than the original:

| GPU | Original | Optimized | Speedup |
|-----|----------|-----------|---------|
| RTX 4060 Laptop | 25-28 min | 15 min | ~1.8x |
| NVIDIA L4 (24GB) | 38 min | 21 min | ~1.8x |

*Benchmarks on ~4800 frames (3-minute video) with enhancement enabled.*

**How it works:** The original pipeline runs RetinaFace detection on every frame. Since MuseTalk already extracts the face region, we use GFPGAN's `has_aligned=True` mode to skip redundant detection.

### Recommended Settings

| Use Case | Settings |
|----------|----------|
| Fast preview | `enhance=false`, `batch_size=16` |
| Quality output | `enhance=true`, `gfpgan_weight=0.5` |
| Maximum quality | `enhance=true`, `gfpgan_weight=0.7`, `fps=30` |

## Model Weights

Models are downloaded automatically on first run. For manual download:

```bash
sh download_weights.sh
```

Or download from [HuggingFace](https://huggingface.co/TMElyralab/MuseTalk):

```
./models/
├── musetalk/           # MuseTalk 1.0
├── musetalkV15/        # MuseTalk 1.5 (recommended)
├── dwpose/
├── face-parse-bisent/
├── sd-vae/
├── syncnet/
└── whisper/
```

## Differences from Upstream

This fork differs from [TMElyralab/MuseTalk](https://github.com/TMElyralab/MuseTalk):

| Feature | Upstream | This Fork |
|---------|----------|-----------|
| Primary interface | CLI / Gradio | REST API |
| Deployment | Manual setup | Docker-first |
| GFPGAN speed | Baseline | ~1.8x faster |
| Large file handling | N/A | Streaming response |
| Output naming | Auto-generated | Configurable |

## Upstream Documentation

For model architecture, training, and research details, see the [original repository](https://github.com/TMElyralab/MuseTalk).

## License

- **Code**: MIT License
- **Models**: Available for commercial use
- **Dependencies**: Subject to their respective licenses (whisper, GFPGAN, etc.)

## Citation

```bibtex
@article{musetalk,
  title={MuseTalk: Real-Time High-Fidelity Video Dubbing via Spatio-Temporal Sampling},
  author={Zhang, Yue and Zhong, Zhizhou and Liu, Minhao and Chen, Zhaokang and Wu, Bin and Zeng, Yubin and Zhan, Chao and He, Yingjie and Huang, Junxin and Zhou, Wenjiang},
  journal={arxiv},
  year={2025}
}
```
