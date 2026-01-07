import os
import shutil
import copy
import glob
from typing import Dict, Any, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import WhisperModel

from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, coord_placeholder


class MuseTalkInference:
    def __init__(self, use_float16: bool = True, gpu_id: int = 0):
        self.use_float16 = use_float16
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.models_loaded = False
        self.gfpgan_restorer = None

        self.vae = None
        self.unet = None
        self.pe = None
        self.timesteps = None
        self.weight_dtype = torch.float32
        self.audio_processor = None
        self.whisper = None

    def load_models(self) -> None:
        if self.models_loaded:
            return

        print(f"Loading models on device: {self.device}")

        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path="./models/musetalkV15/unet.pth",
            vae_type="sd-vae",
            unet_config="./models/musetalkV15/musetalk.json",
            device=self.device,
        )

        self.timesteps = torch.tensor([0], device=self.device)

        if self.use_float16:
            self.pe = self.pe.half()
            self.vae.vae = self.vae.vae.half()
            self.unet.model = self.unet.model.half()
            self.weight_dtype = torch.float16
        else:
            self.weight_dtype = torch.float32

        self.pe = self.pe.to(self.device)
        self.vae.vae = self.vae.vae.to(self.device)
        self.unet.model = self.unet.model.to(self.device)

        self.audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
        self.whisper = WhisperModel.from_pretrained("./models/whisper")
        self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)

        self.models_loaded = True
        print("Models loaded successfully!")

    def _load_gfpgan(self) -> None:
        if self.gfpgan_restorer is not None:
            return

        from gfpgan import GFPGANer

        print("Loading GFPGAN model...")
        self.gfpgan_restorer = GFPGANer(
            model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
            upscale=1,
            arch="clean",
            channel_multiplier=2,
        )
        print("GFPGAN loaded!")

    def _apply_gfpgan(self, input_dir: str, output_dir: str, weight: float = 0.5) -> None:
        self._load_gfpgan()

        os.makedirs(output_dir, exist_ok=True)
        image_files = sorted(glob.glob(os.path.join(input_dir, "*.[jpJP][pnPN]*[gG]")))

        print(f"Applying GFPGAN enhancement to {len(image_files)} frames...")
        for img_path in tqdm(image_files, desc="GFPGAN"):
            img_name = os.path.basename(img_path)
            img = cv2.imread(img_path)

            if self.use_float16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, _, enhanced_img = self.gfpgan_restorer.enhance(
                        img,
                        has_aligned=False,
                        only_center_face=False,
                        paste_back=True,
                        weight=weight,
                    )
            else:
                _, _, enhanced_img = self.gfpgan_restorer.enhance(
                    img,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                    weight=weight,
                )

            cv2.imwrite(os.path.join(output_dir, img_name), enhanced_img)

        print(f"Enhanced frames saved to {output_dir}")

    @torch.no_grad()
    def generate(
        self,
        audio_path: str,
        video_path: str,
        enhance: bool = False,
        bbox_shift: int = 0,
        extra_margin: int = 10,
        parsing_mode: str = "jaw",
        left_cheek_width: int = 90,
        right_cheek_width: int = 90,
        fps: int = 25,
        batch_size: int = 8,
        output_name: Optional[str] = None,
        result_dir: str = "./results",
    ) -> str:
        if not self.models_loaded:
            self.load_models()

        os.makedirs(result_dir, exist_ok=True)

        input_basename = os.path.basename(video_path).split(".")[0]
        audio_basename = os.path.basename(audio_path).split(".")[0]

        if output_name:
            output_name = (
                os.path.splitext(output_name)[0] if output_name.endswith(".mp4") else output_name
            )
            output_vid_name = os.path.join(result_dir, f"{output_name}.mp4")
        else:
            output_vid_name = os.path.join(result_dir, f"{input_basename}_{audio_basename}.mp4")

        temp_dir = os.path.join(result_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        result_img_save_path = os.path.join(temp_dir, f"{input_basename}_{audio_basename}")
        os.makedirs(result_img_save_path, exist_ok=True)

        save_dir_full = None
        if get_file_type(video_path) == "video":
            save_dir_full = os.path.join(temp_dir, input_basename)
            os.makedirs(save_dir_full, exist_ok=True)

            import imageio

            reader = imageio.get_reader(video_path)
            for i, im in enumerate(reader):
                imageio.imwrite(f"{save_dir_full}/{i:08d}.png", im)
            reader.close()

            input_img_list = sorted(glob.glob(os.path.join(save_dir_full, "*.[jpJP][pnPN]*[gG]")))
            fps = int(get_video_fps(video_path))
        elif get_file_type(video_path) == "image":
            input_img_list = [video_path]
        else:
            raise ValueError(f"{video_path} should be a video file or an image file")

        whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(audio_path)
        whisper_chunks = self.audio_processor.get_whisper_chunk(
            whisper_input_features,
            self.device,
            self.weight_dtype,
            self.whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=2,
            audio_padding_length_right=2,
        )

        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)

        fp = FaceParsing(left_cheek_width=left_cheek_width, right_cheek_width=right_cheek_width)

        input_latent_list = []
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            y2 = y2 + extra_margin
            y2 = min(y2, frame.shape[0])
            crop_frame = frame[y1:y2, x1:x2]
            crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = self.vae.get_latents_for_unet(crop_frame)
            input_latent_list.append(latents)

        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

        print("Starting inference...")
        video_num = len(whisper_chunks)
        device_str = str(self.device)
        gen = datagen(
            whisper_chunks=whisper_chunks,
            vae_encode_latents=input_latent_list_cycle,
            batch_size=batch_size,
            delay_frame=0,
            device=device_str,
        )

        res_frame_list = []
        total = int(np.ceil(float(video_num) / batch_size))

        for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total)):
            audio_feature_batch = self.pe(whisper_batch)
            latent_batch = latent_batch.to(dtype=self.weight_dtype)

            pred_latents = self.unet.model(
                latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch
            ).sample
            recon = self.vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_list.append(res_frame)

        print("Padding generated images to original video size")
        for i, res_frame in enumerate(tqdm(res_frame_list)):
            bbox = coord_list_cycle[i % len(coord_list_cycle)]
            ori_frame = copy.deepcopy(frame_list_cycle[i % len(frame_list_cycle)])
            x1, y1, x2, y2 = bbox
            y2 = y2 + extra_margin
            y2 = min(y2, ori_frame.shape[0])
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except Exception:
                continue

            combine_frame = get_image(
                ori_frame, res_frame, [x1, y1, x2, y2], mode=parsing_mode, fp=fp
            )
            cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)

        final_frames_path = result_img_save_path
        if enhance:
            enhanced_path = os.path.join(temp_dir, f"{input_basename}_{audio_basename}_enhanced")
            self._apply_gfpgan(result_img_save_path, enhanced_path)
            final_frames_path = enhanced_path

        temp_vid_path = os.path.join(temp_dir, f"temp_{input_basename}_{audio_basename}.mp4")
        cmd_img2video = (
            f"ffmpeg -y -v warning -r {fps} -f image2 "
            f"-i {final_frames_path}/%08d.png -vcodec libx264 "
            f"-vf format=yuv420p -crf 18 {temp_vid_path}"
        )
        os.system(cmd_img2video)

        cmd_combine_audio = (
            f"ffmpeg -y -v warning -i {audio_path} -i {temp_vid_path} {output_vid_name}"
        )
        os.system(cmd_combine_audio)

        shutil.rmtree(temp_dir)

        print(f"Results saved to {output_vid_name}")
        return output_vid_name

    def get_gpu_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": None,
            "memory_allocated": None,
            "memory_reserved": None,
            "memory_total": None,
        }
        if info["gpu_available"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["memory_allocated"] = torch.cuda.memory_allocated(0)
            info["memory_reserved"] = torch.cuda.memory_reserved(0)
            props = torch.cuda.get_device_properties(0)
            info["memory_total"] = props.total_memory
        return info
