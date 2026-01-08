#!/usr/bin/env python3
"""Download all required model weights for MuseTalk."""

from pathlib import Path

import gdown
import requests
from huggingface_hub import hf_hub_download


def download_models(models_dir: str = "./models") -> None:
    models_path = Path(models_dir)
    models_path.mkdir(exist_ok=True)

    subdirs = [
        "musetalk",
        "musetalkV15",
        "syncnet",
        "dwpose",
        "face-parse-bisent",
        "sd-vae",
        "whisper",
    ]
    for subdir in subdirs:
        (models_path / subdir).mkdir(exist_ok=True)

    print("Downloading MuseTalk V1.0 weights...")
    for filename in ["musetalk/musetalk.json", "musetalk/pytorch_model.bin"]:
        dest = models_path / filename
        if not dest.exists():
            hf_hub_download(
                repo_id="TMElyralab/MuseTalk",
                filename=filename,
                local_dir=str(models_path),
            )
            print(f"  Downloaded: {filename}")
        else:
            print(f"  Exists: {filename}")

    print("Downloading MuseTalk V1.5 weights...")
    for filename in ["musetalkV15/musetalk.json", "musetalkV15/unet.pth"]:
        dest = models_path / filename
        if not dest.exists():
            hf_hub_download(
                repo_id="TMElyralab/MuseTalk",
                filename=filename,
                local_dir=str(models_path),
            )
            print(f"  Downloaded: {filename}")
        else:
            print(f"  Exists: {filename}")

    print("Downloading SD VAE weights...")
    vae_dir = models_path / "sd-vae"
    for filename in ["config.json", "diffusion_pytorch_model.bin"]:
        dest = vae_dir / filename
        if not dest.exists():
            hf_hub_download(
                repo_id="stabilityai/sd-vae-ft-mse",
                filename=filename,
                local_dir=str(vae_dir),
            )
            print(f"  Downloaded: {filename}")
        else:
            print(f"  Exists: {filename}")

    print("Downloading Whisper weights...")
    whisper_dir = models_path / "whisper"
    for filename in ["config.json", "pytorch_model.bin", "preprocessor_config.json"]:
        dest = whisper_dir / filename
        if not dest.exists():
            hf_hub_download(
                repo_id="openai/whisper-tiny",
                filename=filename,
                local_dir=str(whisper_dir),
            )
            print(f"  Downloaded: {filename}")
        else:
            print(f"  Exists: {filename}")

    # Source: HuggingFace yzd-v/DWPose (NOT Google Drive - previous version had wrong ID)
    print("Downloading DWPose weights...")
    dwpose_file = models_path / "dwpose" / "dw-ll_ucoco_384.pth"
    if not dwpose_file.exists():
        hf_hub_download(
            repo_id="yzd-v/DWPose",
            filename="dw-ll_ucoco_384.pth",
            local_dir=str(models_path / "dwpose"),
        )
        print("  Downloaded: dw-ll_ucoco_384.pth")
    else:
        print("  Exists: dw-ll_ucoco_384.pth")

    # Source: HuggingFace ByteDance/LatentSync (NOT Google Drive - previous version had placeholder ID)
    print("Downloading SyncNet weights...")
    syncnet_file = models_path / "syncnet" / "latentsync_syncnet.pt"
    if not syncnet_file.exists():
        hf_hub_download(
            repo_id="ByteDance/LatentSync",
            filename="latentsync_syncnet.pt",
            local_dir=str(models_path / "syncnet"),
        )
        print("  Downloaded: latentsync_syncnet.pt")
    else:
        print("  Exists: latentsync_syncnet.pt")

    # Source: Google Drive ID 154JgKpzCPW82qINcVieuPH3fZ2e0P812 (matches download_weights.sh)
    print("Downloading Face Parse weights...")
    bisenet_file = models_path / "face-parse-bisent" / "79999_iter.pth"
    if not bisenet_file.exists():
        gdown.download(
            id="154JgKpzCPW82qINcVieuPH3fZ2e0P812",
            output=str(bisenet_file),
            quiet=False,
        )
        print("  Downloaded: 79999_iter.pth")
    else:
        print("  Exists: 79999_iter.pth")

    print("Downloading ResNet18 backbone...")
    resnet_file = models_path / "face-parse-bisent" / "resnet18-5c106cde.pth"
    if not resnet_file.exists():
        url = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(resnet_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("  Downloaded: resnet18-5c106cde.pth")
    else:
        print("  Exists: resnet18-5c106cde.pth")

    print("\n✓ All models downloaded successfully!")
    print(f"  Location: {models_path.resolve()}")


if __name__ == "__main__":
    download_models()
