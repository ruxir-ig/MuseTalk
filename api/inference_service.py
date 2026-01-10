import os
import shutil
import glob
from typing import Dict, Any, Optional, List

import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import WhisperModel
from basicsr.utils import img2tensor, tensor2img
from torchvision.transforms.functional import normalize

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
            device=self.device,
        )
        print("GFPGAN loaded!")

    def _enhance_batch_direct(
        self, face_crops: List[np.ndarray], weight: float = 0.5
    ) -> List[np.ndarray]:
        """
        BATCHED GFPGAN enhancement - process multiple faces in one GPU call.

        This bypasses GFPGAN's face detection entirely and directly calls the
        neural network on pre-cropped faces. Much faster than calling enhance()
        per frame because:
        1. No face detection overhead (RetinaFace skipped)
        2. Batched GPU inference (one kernel launch for N faces)
        3. No face_helper state management overhead
        """
        if not face_crops or self.gfpgan_restorer is None:
            return face_crops

        batch_size = len(face_crops)
        original_sizes = [(crop.shape[1], crop.shape[0]) for crop in face_crops]

        faces_512 = []
        for crop in face_crops:
            face_512 = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LANCZOS4)
            faces_512.append(face_512)

        face_tensors = []
        for face in faces_512:
            face_t = img2tensor(face / 255.0, bgr2rgb=True, float32=True)
            normalize(face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            face_tensors.append(face_t)

        batch_tensor = torch.stack(face_tensors, dim=0).to(self.device)

        try:
            with torch.no_grad():
                output_batch = self.gfpgan_restorer.gfpgan(
                    batch_tensor, return_rgb=False, weight=weight
                )[0]

            enhanced_crops = []
            for i in range(batch_size):
                restored = tensor2img(output_batch[i], rgb2bgr=True, min_max=(-1, 1))
                restored = restored.astype("uint8")
                resized_back = cv2.resize(
                    restored, original_sizes[i], interpolation=cv2.INTER_LANCZOS4
                )
                enhanced_crops.append(resized_back)

            return enhanced_crops

        except Exception as e:
            print(f"Batched GFPGAN failed: {e}, falling back to sequential")
            return face_crops

    def _enhance_face_aligned(self, face_crop: np.ndarray, weight: float = 0.5) -> np.ndarray:
        """Single-frame enhancement for compatibility."""
        result = self._enhance_batch_direct([face_crop], weight)
        return result[0] if result else face_crop

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
        gfpgan_weight: float = 0.5,
        gfpgan_batch_size: int = 8,
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

        print("Starting MuseTalk inference...")
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

        for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total, desc="MuseTalk")):
            audio_feature_batch = self.pe(whisper_batch)
            latent_batch = latent_batch.to(dtype=self.weight_dtype)

            pred_latents = self.unet.model(
                latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch
            ).sample
            recon = self.vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_list.append(res_frame)

        if enhance:
            self._load_gfpgan()
            print(f"Blending frames with BATCHED GFPGAN (batch_size={gfpgan_batch_size})")
        else:
            print("Blending frames")

        total_frames = len(res_frame_list)
        frame_idx = 0

        for chunk_start in range(0, total_frames, gfpgan_batch_size):
            chunk_end = min(chunk_start + gfpgan_batch_size, total_frames)
            chunk_indices = list(range(chunk_start, chunk_end))

            face_crops = []
            frame_data = []

            for i in chunk_indices:
                res_frame = res_frame_list[i]
                bbox = coord_list_cycle[i % len(coord_list_cycle)]
                ori_frame = frame_list_cycle[i % len(frame_list_cycle)].copy()
                x1, y1, x2, y2 = bbox
                y2 = y2 + extra_margin
                y2 = min(y2, ori_frame.shape[0])

                face_crop = res_frame.astype(np.uint8)
                face_crops.append(face_crop)
                frame_data.append((ori_frame, x1, y1, x2, y2))

            if enhance and face_crops:
                enhanced_crops = self._enhance_batch_direct(face_crops, gfpgan_weight)
            else:
                enhanced_crops = face_crops

            for idx, (enhanced_crop, (ori_frame, x1, y1, x2, y2)) in enumerate(
                zip(enhanced_crops, frame_data)
            ):
                try:
                    face_resized = cv2.resize(enhanced_crop, (x2 - x1, y2 - y1))
                except Exception:
                    continue

                combine_frame = get_image(
                    ori_frame, face_resized, [x1, y1, x2, y2], mode=parsing_mode, fp=fp
                )
                cv2.imwrite(f"{result_img_save_path}/{str(frame_idx).zfill(8)}.png", combine_frame)
                frame_idx += 1

        print(f"Saved {frame_idx} frames")

        temp_vid_path = os.path.join(temp_dir, f"temp_{input_basename}_{audio_basename}.mp4")
        cmd_img2video = (
            f"ffmpeg -y -v warning -r {fps} -f image2 "
            f"-i {result_img_save_path}/%08d.png -vcodec libx264 "
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
