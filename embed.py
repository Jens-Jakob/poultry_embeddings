"""
Embed all images in a directory using DINOv3 ViT-B/16.
Saves patch-level embeddings for ROI-based similarity search.

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
from transformers import AutoImageProcessor, AutoModel


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
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    patch_size = model.config.patch_size
    # Processor resize target — get from a dummy forward pass
    dummy = Image.new("RGB", (100, 100))
    dummy_inputs = processor(images=[dummy], return_tensors="pt")
    proc_h, proc_w = dummy_inputs["pixel_values"].shape[2], dummy_inputs["pixel_values"].shape[3]
    grid_h, grid_w = proc_h // patch_size, proc_w // patch_size
    num_patches = grid_h * grid_w
    print(f"Processor: {proc_h}x{proc_w}, patch_size: {patch_size}, grid: {grid_h}x{grid_w} ({num_patches} patches)")

    image_paths = load_images(data_dir)

    all_patches = []
    all_filenames = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)

        with torch.inference_mode():
            outputs = model(**inputs)

        # Patch tokens are the last num_patches tokens in the sequence
        # (skips CLS and any register tokens)
        patch_tokens = outputs.last_hidden_state[:, -num_patches:, :].cpu().numpy()
        all_patches.append(patch_tokens)
        all_filenames.extend([str(p) for p in batch_paths])

        print(f"  Embedded {min(i + batch_size, len(image_paths))}/{len(image_paths)}")

    all_patches = np.concatenate(all_patches, axis=0)  # (N, num_patches, hidden_dim)

    # L2-normalize each patch vector
    norms = np.linalg.norm(all_patches, axis=2, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    all_patches = all_patches / norms

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "patch_embeddings.npy"), all_patches.astype(np.float16))
    np.save(os.path.join(output_dir, "filenames.npy"), np.array(all_filenames))
    np.savez(os.path.join(output_dir, "meta.npz"),
             grid_h=grid_h, grid_w=grid_w, patch_size=patch_size,
             proc_h=proc_h, proc_w=proc_w)

    size_mb = all_patches.nbytes / 1e6
    print(f"Saved {all_patches.shape[0]} images x {num_patches} patches x {all_patches.shape[2]}-dim ({size_mb:.0f} MB) to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory containing images")
    parser.add_argument("--output-dir", default="./embeddings", help="Where to save .npy files")
    parser.add_argument("--model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    embed(args.data_dir, args.output_dir, args.model, args.batch_size)
