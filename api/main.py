import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import InferenceRequest, HealthResponse, InferenceResponse
from api.inference_service import MuseTalkInference

app = FastAPI(
    title="MuseTalk API",
    description="Real-Time High-Fidelity Video Dubbing API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

inference_engine = None

@app.on_event("startup")
async def startup_event():
    global inference_engine
    print("Starting MuseTalk API...")
    inference_engine = MuseTalkInference(use_float16=True)
    inference_engine.load_models()
    print("MuseTalk API is ready!")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    gpu_info = inference_engine.get_gpu_info()
    return HealthResponse(
        status="healthy",
        gpu_available=gpu_info["gpu_available"],
        gpu_name=gpu_info["gpu_name"],
        models_loaded=inference_engine.models_loaded
    )

@app.get("/")
async def root():
    return {
        "message": "MuseTalk API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.post("/generate")
async def generate_video(
    audio: UploadFile = File(...),
    video: UploadFile = File(...),
    bbox_shift: int = Form(0),
    extra_margin: int = Form(10),
    parsing_mode: str = Form("jaw"),
    left_cheek_width: int = Form(90),
    right_cheek_width: int = Form(90),
    fps: int = Form(25),
    batch_size: int = Form(8),
    output_name: Optional[str] = Form(None)
):
    try:
        temp_dir = tempfile.mkdtemp()
        audio_ext = Path(audio.filename).suffix
        video_ext = Path(video.filename).suffix

        audio_path = os.path.join(temp_dir, f"audio{audio_ext}")
        video_path = os.path.join(temp_dir, f"video{video_ext}")

        with open(audio_path, "wb") as f:
            f.write(await audio.read())

        with open(video_path, "wb") as f:
            f.write(await video.read())

        output_video_path = inference_engine.generate(
            audio_path=audio_path,
            video_path=video_path,
            bbox_shift=bbox_shift,
            extra_margin=extra_margin,
            parsing_mode=parsing_mode,
            left_cheek_width=left_cheek_width,
            right_cheek_width=right_cheek_width,
            fps=fps,
            batch_size=batch_size,
            output_name=output_name
        )

        shutil.rmtree(temp_dir)

        return FileResponse(
            path=output_video_path,
            media_type="video/mp4",
            filename=os.path.basename(output_video_path)
        )

    except Exception as e:
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/json", response_model=InferenceResponse)
async def generate_video_json(request: InferenceRequest):
    try:
        if not os.path.exists(request.audio_path):
            raise HTTPException(status_code=404, detail=f"Audio file not found: {request.audio_path}")

        if not os.path.exists(request.video_path):
            raise HTTPException(status_code=404, detail=f"Video file not found: {request.video_path}")

        output_video_path = inference_engine.generate(
            audio_path=request.audio_path,
            video_path=request.video_path,
            bbox_shift=request.bbox_shift,
            extra_margin=request.extra_margin,
            parsing_mode=request.parsing_mode,
            left_cheek_width=request.left_cheek_width,
            right_cheek_width=request.right_cheek_width,
            fps=request.fps,
            batch_size=request.batch_size,
            output_name=request.output_name
        )

        return InferenceResponse(
            status="success",
            output_video_path=output_video_path,
            message="Video generated successfully"
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

    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=filename
    )
