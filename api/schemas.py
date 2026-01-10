from pydantic import BaseModel, Field
from typing import Optional


class InferenceRequest(BaseModel):
    audio_path: str = Field(..., description="Path to the driving audio file")
    video_path: str = Field(..., description="Path to the reference image/video file")
    enhance: bool = Field(default=False, description="Apply GFPGAN face enhancement")
    bbox_shift: int = Field(default=0, description="Face bounding box shift in pixels")
    extra_margin: int = Field(default=10, ge=0, le=40, description="Extra margin for jaw movement")
    parsing_mode: str = Field(default="jaw", description="Face blending mode: 'jaw' or 'raw'")
    left_cheek_width: int = Field(default=90, ge=20, le=160, description="Left cheek region width")
    right_cheek_width: int = Field(
        default=90, ge=20, le=160, description="Right cheek region width"
    )
    fps: int = Field(default=25, ge=1, le=60, description="Video frames per second")
    batch_size: int = Field(default=8, ge=1, le=32, description="Inference batch size")
    output_name: Optional[str] = Field(default=None, description="Custom output filename")
    gfpgan_weight: float = Field(
        default=0.5, ge=0.0, le=1.0, description="GFPGAN blend weight (0=original, 1=enhanced)"
    )
    gfpgan_batch_size: int = Field(
        default=8, ge=1, le=32, description="GFPGAN batch size for processing"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "audio_path": "/app/data/audio.wav",
                "video_path": "/app/data/image.png",
                "enhance": True,
                "bbox_shift": 0,
                "extra_margin": 10,
                "parsing_mode": "jaw",
                "fps": 25,
                "batch_size": 8,
            }
        }


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_name: Optional[str]
    gpu_memory_gb: Optional[float]
    models_loaded: bool
    gfpgan_available: bool


class InferenceResponse(BaseModel):
    status: str
    output_video_path: str
    message: str
    processing_time_seconds: Optional[float] = None


class ErrorResponse(BaseModel):
    status: str = "error"
    detail: str
    code: Optional[str] = None
