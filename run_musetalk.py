#!/usr/bin/env python3
"""
MuseTalk CLI: Generate lip-synced videos from image/video and audio.

Usage:
    python run_musetalk.py <input_image_or_video> <audio_file> <output_dir> [options]

Examples:
    python run_musetalk.py ./face.png ./audio.wav ./output/
    python run_musetalk.py ./face.png ./audio.wav ./output/ --enhance
"""

import os
import sys
import cv2
import copy
import glob
import shutil
import pickle
import argparse
import subprocess
import numpy as np
from tqdm import tqdm

import torch
from torch.amp.autocast_mode import autocast
from transformers import WhisperModel
from omegaconf import OmegaConf

from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import (
    get_landmark_and_bbox,
    read_imgs,
    coord_placeholder,
)


def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def ensure_gfpgan_installed():
    """Ensure GFPGAN is installed, auto-install if not."""
    try:
        import gfpgan

        return True
    except ImportError:
        print("GFPGAN not found. Installing (this may take a while)...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "gfpgan"],
                timeout=300,  # 5 minute timeout
            )
            print("GFPGAN installed successfully.")
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print("Failed to install GFPGAN. Try: pip install gfpgan manually.")
            return False


def apply_gfpgan_enhancement(input_dir, output_dir, upscale=1, weight=0.5, use_fp16=False):
    """
    Apply GFPGAN deep learning enhancement to all images in a directory.
    Requires GFPGAN to be installed (pip install gfpgan).

    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save enhanced images
        upscale: Upscale factor (1 = no upscaling)
        weight: Blending weight (0 = original, 1 = enhanced)
        use_fp16: Use half precision for faster processing
    """
    from gfpgan import GFPGANer

    os.makedirs(output_dir, exist_ok=True)

    print("Initializing GFPGAN (downloading model if needed)...")
    restorer = GFPGANer(
        model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
        upscale=upscale,
        arch="clean",
        channel_multiplier=2,
    )

    if use_fp16:
        print("GFPGAN will run in FP16 mode...")
        use_fp16_context = True
    else:
        use_fp16_context = False

    image_files = sorted(glob.glob(os.path.join(input_dir, "*.[jpJP][pnPN]*[gG]")))

    print(f"Applying GFPGAN enhancement to {len(image_files)} frames...")
    for img_path in tqdm(image_files, desc="GFPGAN"):
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)

        if use_fp16_context:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, _, enhanced_img = restorer.enhance(
                    img,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                    weight=weight,
                )
        else:
            _, _, enhanced_img = restorer.enhance(
                img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=weight,
            )

        cv2.imwrite(os.path.join(output_dir, img_name), enhanced_img)

    print(f"Enhanced frames saved to {output_dir}")


@torch.no_grad()
def run_inference(args):
    """Main inference function adapted from scripts/inference.py."""

    # Configure ffmpeg path
    if not check_ffmpeg():
        print("Warning: ffmpeg not found. Please install ffmpeg for video processing.")

    # Set computing device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model weights
    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device,
    )
    timesteps = torch.tensor([0], device=device)

    # Convert models to half precision if float16 is enabled
    if args.use_float16:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()

    # Move models to specified device
    pe = pe.to(device)
    vae.vae = vae.vae.to(device)
    unet.model = unet.model.to(device)

    # Initialize audio processor and Whisper model
    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained(args.whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    # Initialize face parser with configurable parameters based on version
    if args.version == "v15":
        fp = FaceParsing(
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width,
        )
    else:  # v1
        fp = FaceParsing()

    # Get input paths
    video_path = args.input
    audio_path = args.audio

    # Validate input files
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input file not found: {video_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set output paths
    input_basename = os.path.basename(video_path).split(".")[0]
    audio_basename = os.path.basename(audio_path).split(".")[0]
    output_basename = f"{input_basename}_{audio_basename}"

    # Temporary directories
    temp_dir = os.path.join(args.output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    result_img_save_path = os.path.join(temp_dir, output_basename)
    os.makedirs(result_img_save_path, exist_ok=True)

    crop_coord_save_path = os.path.join(args.output_dir, f"{input_basename}.pkl")

    if args.output_vid_name is None:
        output_vid_name = os.path.join(args.output_dir, f"{output_basename}.mp4")
    else:
        output_vid_name = os.path.join(args.output_dir, args.output_vid_name)

    # Extract frames from source video or use image directly
    save_dir_full = None
    if get_file_type(video_path) == "video":
        save_dir_full = os.path.join(temp_dir, input_basename)
        os.makedirs(save_dir_full, exist_ok=True)
        cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
        os.system(cmd)
        input_img_list = sorted(glob.glob(os.path.join(save_dir_full, "*.[jpJP][pnPN]*[gG]")))
        fps = get_video_fps(video_path)
    elif get_file_type(video_path) == "image":
        input_img_list = [video_path]
        fps = args.fps
    elif os.path.isdir(video_path):
        input_img_list = glob.glob(os.path.join(video_path, "*.[jpJP][pnPN]*[gG]"))
        input_img_list = sorted(
            input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )
        fps = args.fps
    else:
        raise ValueError(
            f"{video_path} should be a video file, an image file or a directory of images"
        )

    print(f"Processing {len(input_img_list)} frames at {fps} FPS")

    # Extract audio features
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features,
        device,
        weight_dtype,
        whisper,
        librosa_length,
        fps=fps,
        audio_padding_length_left=args.audio_padding_length_left,
        audio_padding_length_right=args.audio_padding_length_right,
    )

    # Preprocess input images
    if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
        print("Using saved coordinates")
        with open(crop_coord_save_path, "rb") as f:
            coord_list = pickle.load(f)
        frame_list = read_imgs(input_img_list)
    else:
        print("Extracting landmarks... (time-consuming operation)")
        coord_list, frame_list = get_landmark_and_bbox(
            input_img_list, args.bbox_shift if args.version == "v1" else 0
        )
        with open(crop_coord_save_path, "wb") as f:
            pickle.dump(coord_list, f)

    print(f"Number of frames: {len(frame_list)}")

    # Process each frame
    input_latent_list = []
    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        if args.version == "v15":
            y2 = y2 + args.extra_margin
            y2 = min(y2, frame.shape[0])
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)

    # Smooth first and last frames
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

    # Batch inference
    print("Starting MuseTalk inference...")
    video_num = len(whisper_chunks)
    batch_size = args.batch_size
    gen = datagen(
        whisper_chunks=whisper_chunks,
        vae_encode_latents=input_latent_list_cycle,
        batch_size=batch_size,
        delay_frame=0,
        device=device,
    )

    res_frame_list = []
    total = int(np.ceil(float(video_num) / batch_size))

    # Execute inference
    for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total)):
        audio_feature_batch = pe(whisper_batch)
        latent_batch = latent_batch.to(dtype=unet.model.dtype)

        pred_latents = unet.model(
            latent_batch, timesteps, encoder_hidden_states=audio_feature_batch
        ).sample
        recon = vae.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)

    # Pad generated images to original video size
    print("Processing generated frames...")
    for i, res_frame in enumerate(tqdm(res_frame_list)):
        bbox = coord_list_cycle[i % (len(coord_list_cycle))]
        ori_frame = copy.deepcopy(frame_list_cycle[i % (len(frame_list_cycle))])
        x1, y1, x2, y2 = bbox
        if args.version == "v15":
            y2 = y2 + args.extra_margin
            y2 = min(y2, frame.shape[0])
        try:
            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        except:
            continue

        # Merge results with version-specific parameters
        if args.version == "v15":
            combine_frame = get_image(
                ori_frame, res_frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp
            )
        else:
            combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], fp=fp)
        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)

    # Apply enhancement if requested
    if args.enhance:
        enhanced_img_path = os.path.join(temp_dir, f"{output_basename}_enhanced")

        print("Applying GFPGAN enhancement...")
        apply_gfpgan_enhancement(
            result_img_save_path,
            enhanced_img_path,
            upscale=1,
            weight=0.5,
            use_fp16=args.use_float16,
        )

        result_img_save_path = enhanced_img_path

    # Save prediction results
    temp_vid_path = os.path.join(temp_dir, f"temp_{output_basename}.mp4")
    cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 {temp_vid_path}"
    print("Encoding video...")
    os.system(cmd_img2video)

    # Combine with audio
    cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {temp_vid_path} {output_vid_name}"
    print("Combining with audio...")
    os.system(cmd_combine_audio)

    # Clean up temporary files
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir, ignore_errors=True)
    if not args.use_saved_coord and os.path.exists(crop_coord_save_path):
        os.remove(crop_coord_save_path)

    print(f"Output saved to: {output_vid_name}")
    return output_vid_name


def main():
    parser = argparse.ArgumentParser(
        description="MuseTalk: Generate lip-synced videos from image/video and audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_musetalk.py ./face.png ./audio.wav ./output/
  python run_musetalk.py ./face.png ./audio.wav ./output/ --enhance
  python run_musetalk.py ./video.mp4 ./audio.wav ./output/ --version v1 --use_float16
        """,
    )

    parser.add_argument("input", type=str, help="Path to input image or video file")
    parser.add_argument("audio", type=str, help="Path to input audio file")
    parser.add_argument("output_dir", type=str, help="Directory to save output video")

    parser.add_argument(
        "--enhance",
        action="store_true",
        help="Apply GFPGAN face enhancement (better quality)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v15",
        choices=["v1", "v15"],
        help="Model version (default: v15)",
    )
    parser.add_argument(
        "--use_float16",
        action="store_true",
        help="Use float16 for faster inference (lower VRAM usage)",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use (default: 0)")
    parser.add_argument("--fps", type=int, default=25, help="Output video FPS (default: 25)")
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Inference batch size (default: 8)"
    )
    parser.add_argument(
        "--bbox_shift",
        type=int,
        default=0,
        help="Bounding box shift for mouth openness (v1 only, default: 0)",
    )
    parser.add_argument(
        "--extra_margin",
        type=int,
        default=10,
        help="Extra margin for face cropping (v15 only, default: 10)",
    )
    parser.add_argument(
        "--parsing_mode",
        type=str,
        default="jaw",
        choices=["jaw", "raw"],
        help="Face blending parsing mode (default: jaw)",
    )
    parser.add_argument(
        "--left_cheek_width",
        type=int,
        default=90,
        help="Width of left cheek region (v15 only, default: 90)",
    )
    parser.add_argument(
        "--right_cheek_width",
        type=int,
        default=90,
        help="Width of right cheek region (v15 only, default: 90)",
    )
    parser.add_argument(
        "--audio_padding_length_left",
        type=int,
        default=2,
        help="Left padding length for audio (default: 2)",
    )
    parser.add_argument(
        "--audio_padding_length_right",
        type=int,
        default=2,
        help="Right padding length for audio (default: 2)",
    )
    parser.add_argument(
        "--output_vid_name",
        type=str,
        default=None,
        help="Custom output video filename (default: input_audio.mp4)",
    )
    parser.add_argument(
        "--use_saved_coord",
        action="store_true",
        help="Use saved coordinates to speed up processing",
    )
    parser.add_argument(
        "--saved_coord", action="store_true", help="Save coordinates for future use"
    )
    parser.add_argument(
        "--vae_type",
        type=str,
        default="sd-vae",
        help="Type of VAE model (default: sd-vae)",
    )

    args = parser.parse_args()

    # Check for GPU
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU (will be slow)")

    # Set model paths based on version
    if args.version == "v15":
        args.unet_model_path = "./models/musetalkV15/unet.pth"
        args.unet_config = "./models/musetalkV15/musetalk.json"
    else:  # v1
        args.unet_model_path = "./models/musetalk/pytorch_model.bin"
        args.unet_config = "./models/musetalk/config.json"

    args.whisper_dir = "./models/whisper"

    if args.enhance:
        if not ensure_gfpgan_installed():
            print("Error: GFPGAN required for --enhance but installation failed.")
            print("Install manually: pip install gfpgan")
            sys.exit(1)

    # Run inference
    try:
        output_path = run_inference(args)
        print(f"\nDone! Output saved to: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
