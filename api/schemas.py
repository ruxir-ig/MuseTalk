from pydantic import BaseModel, Field
from typing import Optional

class InferenceRequest(BaseModel):
    audio_path: str = Field(..., description="Path to the driving audio file")
    video_path: str = Field(..., description="Path to the reference image/video file")
    bbox_shift: int = Field(default=0, description="Face bounding box shift in pixels")
    extra_margin: int = Field(default=10, ge=0, le=40, description="Extra margin for jaw movement")
    parsing_mode: str = Field(default="jaw", description="Face blending parsing mode: 'jaw' or 'raw'")
    left_cheek_width: int = Field(default=90, ge=20, le=160, description="Width of left cheek region")
    right_cheek_width: int = Field(default=90, ge=20, le=160, description="Width of right cheek region")
    fps: int = Field(default=25, description="Video frames per second")
    batch_size: int = Field(default=8, description="Inference batch size")
    output_name: Optional[str] = Field(default=None, description="Optional custom output filename")

    class Config:
        json_schema_extra = {
            "example": {
                "audio_path": "/path/to/audio.wav",
                "video_path": "/path/to/video.mp4",
                "bbox_shift": 0,
                "extra_margin": 10,
                "parsing_mode": "jaw",
                "left_cheek_width": 90,
                "right_cheek_width": 90,
                "fps": 25,
                "batch_size": 8
            }
        }

class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_name: Optional[str]
    models_loaded: bool

class InferenceResponse(BaseModel):
    status: str
    output_video_path: str
    message: str
