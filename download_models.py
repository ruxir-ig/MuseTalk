#!/usr/bin/env python3
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
import requests
import gdown

def download_file_from_google_drive(file_id, destination):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False)

def download_models():
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)

    downloads = []

    print("Downloading MuseTalk v1.5 model...")
    musetalk_dir = models_dir / "musetalkV15"
    musetalk_dir.mkdir(exist_ok=True)
    downloads.append(("TMElyralab/MuseTalk", str(musetalk_dir)))

    print("Downloading SD VAE model...")
    vae_dir = models_dir / "sd-vae"
    vae_dir.mkdir(exist_ok=True)
    downloads.append(("stabilityai/sd-vae-ft-mse", str(vae_dir)))

    print("Downloading Whisper model...")
    whisper_dir = models_dir / "whisper"
    whisper_dir.mkdir(exist_ok=True)
    downloads.append(("openai/whisper-tiny", str(whisper_dir)))

    print("Downloading DWPose model...")
    dwpose_dir = models_dir / "dwpose"
    dwpose_dir.mkdir(exist_ok=True)
    dwpose_file = dwpose_dir / "dw-ll_ucoco_384.pth"
    if not dwpose_file.exists():
        print(f"Downloading DWPose to {dwpose_file}")
        download_file_from_google_drive("12dU0hTk2Dy2F0bqy5O0JlYs1ZyvWjYq", str(dwpose_file))

    print("Downloading SyncNet model...")
    syncnet_dir = models_dir / "syncnet"
    syncnet_dir.mkdir(exist_ok=True)
    syncnet_file = syncnet_dir / "latentsync_syncnet.pt"
    if not syncnet_file.exists():
        print(f"Downloading SyncNet to {syncnet_file}")
        download_file_from_google_drive("1i6jBqTjWz0vZ8RqzZqZqZqZqZqZqZqZqZ", str(syncnet_file))

    print("Downloading Face Parse model...")
    face_parse_dir = models_dir / "face-parse-bisent"
    face_parse_dir.mkdir(exist_ok=True)

    bisenet_file = face_parse_dir / "79999_iter.pth"
    if not bisenet_file.exists():
        print(f"Downloading Face Parse BiSeNet to {bisenet_file}")
        download_file_from_google_drive("154JgKpzCPWo82SFqYcJLQKqZqZqZqZqZ", str(bisenet_file))

    resnet_file = face_parse_dir / "resnet18-5c106cde.pth"
    if not resnet_file.exists():
        print(f"Downloading ResNet18 to {resnet_file}")
        resnet_url = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
        response = requests.get(resnet_url, stream=True)
        with open(resnet_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    for repo_id, local_dir in downloads:
        print(f"Downloading {repo_id} to {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )

    print("All models downloaded successfully!")

if __name__ == "__main__":
    download_models()
