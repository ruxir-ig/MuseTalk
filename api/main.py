import os
import time
import tempfile
import shutil
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import InferenceRequest, HealthResponse, InferenceResponse
from api.inference_service import MuseTalkInference

inference_engine: Optional[MuseTalkInference] = None


def check_gfpgan_available() -> bool:
    try:
        import gfpgan

        return True
    except ImportError:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_engine
    print("Starting MuseTalk API...")

    if not check_gfpgan_available():
        raise RuntimeError("GFPGAN not installed. Run: pip install gfpgan")

    inference_engine = MuseTalkInference(use_float16=True)
    inference_engine.load_models()
    print("MuseTalk API ready!")

    yield

    print("Shutting down MuseTalk API...")


app = FastAPI(
    title="MuseTalk API",
    description="Real-Time High-Fidelity Video Dubbing API - Generate lip-synced videos from image/video and audio",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    gpu_info = inference_engine.get_gpu_info()
    gpu_memory_gb = None
    if gpu_info.get("memory_total"):
        gpu_memory_gb = round(gpu_info["memory_total"] / (1024**3), 2)

    return HealthResponse(
        status="healthy",
        gpu_available=gpu_info["gpu_available"],
        gpu_name=gpu_info.get("gpu_name"),
        gpu_memory_gb=gpu_memory_gb,
        models_loaded=inference_engine.models_loaded,
        gfpgan_available=check_gfpgan_available(),
    )


@app.get("/")
async def root():
    return {
        "service": "MuseTalk API",
        "version": "1.0.0",
        "description": "Generate lip-synced videos from image/video and audio",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "generate": "/generate (POST - file upload)",
            "generate_json": "/generate/json (POST - path-based)",
        },
    }


@app.post("/generate")
async def generate_video(
    source: UploadFile = File(..., description="Source image or video file"),
    audio: UploadFile = File(..., description="Driving audio file"),
    enhance: bool = Form(False, description="Apply GFPGAN face enhancement"),
    bbox_shift: int = Form(0),
    extra_margin: int = Form(10),
    parsing_mode: str = Form("jaw"),
    left_cheek_width: int = Form(90),
    right_cheek_width: int = Form(90),
    fps: int = Form(25),
    batch_size: int = Form(8),
    output_name: Optional[str] = Form(None),
):
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    temp_dir = tempfile.mkdtemp()
    try:
        start_time = time.time()

        audio_filename = audio.filename or "audio.wav"
        source_filename = source.filename or "source.png"
        audio_ext = Path(audio_filename).suffix or ".wav"
        source_ext = Path(source_filename).suffix or ".png"

        audio_path = os.path.join(temp_dir, f"audio{audio_ext}")
        source_path = os.path.join(temp_dir, f"source{source_ext}")

        with open(audio_path, "wb") as f:
            content = await audio.read()
            f.write(content)

        with open(source_path, "wb") as f:
            content = await source.read()
            f.write(content)

        output_video_path = inference_engine.generate(
            audio_path=audio_path,
            video_path=source_path,
            enhance=enhance,
            bbox_shift=bbox_shift,
            extra_margin=extra_margin,
            parsing_mode=parsing_mode,
            left_cheek_width=left_cheek_width,
            right_cheek_width=right_cheek_width,
            fps=fps,
            batch_size=batch_size,
            output_name=output_name,
        )

        shutil.rmtree(temp_dir)

        processing_time = round(time.time() - start_time, 2)
        print(f"Video generated in {processing_time}s: {output_video_path}")

        return FileResponse(
            path=output_video_path,
            media_type="video/mp4",
            filename=os.path.basename(output_video_path),
        )

    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/json", response_model=InferenceResponse)
async def generate_video_json(request: InferenceRequest):
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        start_time = time.time()

        if not os.path.exists(request.audio_path):
            raise HTTPException(
                status_code=404, detail=f"Audio file not found: {request.audio_path}"
            )

        if not os.path.exists(request.video_path):
            raise HTTPException(
                status_code=404, detail=f"Source file not found: {request.video_path}"
            )

        output_video_path = inference_engine.generate(
            audio_path=request.audio_path,
            video_path=request.video_path,
            enhance=request.enhance,
            bbox_shift=request.bbox_shift,
            extra_margin=request.extra_margin,
            parsing_mode=request.parsing_mode,
            left_cheek_width=request.left_cheek_width,
            right_cheek_width=request.right_cheek_width,
            fps=request.fps,
            batch_size=request.batch_size,
            output_name=request.output_name,
        )

        processing_time = round(time.time() - start_time, 2)

        return InferenceResponse(
            status="success",
            output_video_path=output_video_path,
            message="Video generated successfully",
            processing_time_seconds=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{filename}")
async def download_result(filename: str):
    results_dir = "./results"
    file_path = os.path.join(results_dir, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path=file_path, media_type="video/mp4", filename=filename)
