"""
precompute.py

Precomputes VAE latents for CelebA images and saves them to disk.

Usage:
    python precompute.py --split train
    python precompute.py --split val
    python precompute.py --split train --split val  # both splits
"""

import argparse
import os
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/data_new/manu/celebA/data/celeba-subset/200k")
VAE_MODEL_ID = "stabilityai/sd-vae-ft-mse"
LATENT_SCALE = 0.18215
IMAGE_SIZE = 64


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------
def build_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.CenterCrop(min(image_size, image_size)),  # square-safe no-op at 64
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def precompute_split(split: str, vae: AutoencoderKL, device: torch.device, transform: transforms.Compose) -> None:
    image_dir = DATA_ROOT / split / "images"
    latent_dir = DATA_ROOT / split / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )

    if len(image_paths) == 0:
        print(f"[WARNING] No images found in {image_dir}. Skipping split '{split}'.")
        return

    print(f"[{split}] Found {len(image_paths)} images. Saving latents to {latent_dir}")

    with torch.no_grad():
        for img_path in tqdm(image_paths, desc=f"Encoding [{split}]", dynamic_ncols=True):
            latent_path = latent_dir / (img_path.stem + ".pt")
            if latent_path.exists():
                # Skip already-computed latents to allow resuming
                continue

            # Load and transform image
            image = Image.open(img_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, H, W)

            # Encode through VAE
            posterior = vae.encode(tensor).latent_dist
            latent = posterior.sample()                         # (1, 4, h, w)
            latent = latent * LATENT_SCALE

            # Save to disk (remove batch dim for storage)
            torch.save(latent.squeeze(0).cpu(), latent_path)   # (4, h, w)

    print(f"[{split}] Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute VAE latents for CelebA images.")
    parser.add_argument(
        "--split",
        nargs="+",
        choices=["train", "val"],
        default=["train", "val"],
        help="Dataset split(s) to precompute (default: both train and val).",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp32", "fp16"],
        default="fp32",
        help="VAE dtype — use fp16 to halve memory usage (default: fp32).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    print(f"Using device: {device}  |  dtype: {torch_dtype}")

    # Load VAE
    print(f"Loading VAE from '{VAE_MODEL_ID}' ...")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(VAE_MODEL_ID, torch_dtype=torch_dtype)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    vae.to(device)
    print("VAE loaded successfully.")

    transform = build_transform(IMAGE_SIZE)

    for split in args.split:
        precompute_split(split, vae, device, transform)

    print("All splits complete.")


if __name__ == "__main__":
    main()
