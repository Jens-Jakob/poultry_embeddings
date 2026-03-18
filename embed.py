"""
Embed all images in a directory using DINOv2 ViT-B/14.
Saves embeddings.npy and filenames.npy to an output directory.

Usage:
    python embed.py --data-dir "/path/to/Poultry Defects.coco/train"
    python embed.py --data-dir "/path/to/images" --output-dir ./embeddings --batch-size 32
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def make_transform(resize=256, crop=224):
    return transforms.Compose([
        transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def load_images(data_dir):
    exts = {".bmp", ".jpg", ".jpeg", ".png", ".tiff", ".tif"}
    paths = sorted(
        p for p in Path(data_dir).rglob("*") if p.suffix.lower() in exts
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {data_dir}")
    print(f"Found {len(paths)} images in {data_dir}")
    return paths


def embed(data_dir, output_dir, model_name, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model: {model_name}")
    model = torch.hub.load("facebookresearch/dinov2", model_name).to(device).eval()
    transform = make_transform()

    image_paths = load_images(data_dir)

    all_embeddings = []
    all_filenames = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
        batch = torch.stack(images).to(device)

        with torch.inference_mode():
            embeddings = model(batch).cpu().numpy()

        all_embeddings.append(embeddings)
        all_filenames.extend([str(p) for p in batch_paths])

        print(f"  Embedded {min(i + batch_size, len(image_paths))}/{len(image_paths)}")

    all_embeddings = np.concatenate(all_embeddings, axis=0)

    # L2-normalize so cosine similarity = dot product
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    all_embeddings = all_embeddings / norms

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "embeddings.npy"), all_embeddings)
    np.save(os.path.join(output_dir, "filenames.npy"), np.array(all_filenames))

    print(f"Saved {all_embeddings.shape[0]} embeddings ({all_embeddings.shape[1]}-dim) to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory containing images")
    parser.add_argument("--output-dir", default="./embeddings", help="Where to save .npy files")
    parser.add_argument("--model", default="dinov2_vitb14", help="DINOv2 model name (dinov2_vits14, dinov2_vitb14, dinov2_vitl14)")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    embed(args.data_dir, args.output_dir, args.model, args.batch_size)
