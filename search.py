"""
Given a reference image, find the most similar images from pre-computed embeddings.
Displays a matplotlib grid: reference on the left, top-K matches on the right.

Usage:
    python search.py --reference "/path/to/image.bmp"
    python search.py --reference "/path/to/image.bmp" --topk 10 --embeddings-dir ./embeddings
"""

import argparse
import os

import matplotlib.pyplot as plt
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


def load_index(embeddings_dir):
    embeddings = np.load(os.path.join(embeddings_dir, "embeddings.npy"))
    filenames = np.load(os.path.join(embeddings_dir, "filenames.npy"))
    return embeddings, filenames


def embed_single(image_path, model, transform, device):
    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        emb = model(image).cpu().numpy()[0]
    emb = emb / np.linalg.norm(emb)
    return emb


def search(reference_path, embeddings_dir, model_name, topk):
    embeddings, filenames = load_index(embeddings_dir)

    # Check if reference is already in the index
    ref_abs = os.path.abspath(reference_path)
    match_idx = None
    for i, fn in enumerate(filenames):
        if os.path.abspath(fn) == ref_abs:
            match_idx = i
            break

    if match_idx is not None:
        print(f"Reference found in index at position {match_idx}, using cached embedding")
        query = embeddings[match_idx]
    else:
        print(f"Reference not in index, embedding on-the-fly...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.hub.load("facebookresearch/dinov2", model_name).to(device).eval()
        transform = make_transform()
        query = embed_single(reference_path, model, transform, device)

    # Cosine similarity (embeddings are already L2-normalized)
    scores = embeddings @ query
    ranked = np.argsort(scores)[::-1]

    # Show results
    print(f"\nTop {topk} matches for: {os.path.basename(reference_path)}")
    print("-" * 60)
    for rank, idx in enumerate(ranked[:topk], 1):
        print(f"  #{rank:2d}  score={scores[idx]:.4f}  {os.path.basename(filenames[idx])}")

    # Plot
    show = ranked[:topk]
    cols = min(topk, 5)
    rows = (topk + cols - 1) // cols
    fig, axes = plt.subplots(rows + 1, cols, figsize=(3 * cols, 3 * (rows + 1)))

    # Make axes always 2D
    if rows + 1 == 1:
        axes = axes[np.newaxis, :]
    if cols == 1:
        axes = axes[:, np.newaxis]

    # Reference image spans first row
    for c in range(cols):
        axes[0, c].axis("off")
    ref_img = Image.open(reference_path).convert("RGB")
    axes[0, 0].imshow(ref_img)
    axes[0, 0].set_title("REFERENCE", fontsize=10, fontweight="bold", color="red")

    # Results
    for i, idx in enumerate(show):
        r = 1 + i // cols
        c = i % cols
        img = Image.open(filenames[idx]).convert("RGB")
        axes[r, c].imshow(img)
        axes[r, c].set_title(f"#{i+1} ({scores[idx]:.3f})", fontsize=8)
        axes[r, c].axis("off")

    # Clear unused cells
    for i in range(len(show), rows * cols):
        r = 1 + i // cols
        c = i % cols
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="Path to reference image")
    parser.add_argument("--embeddings-dir", default="./embeddings", help="Directory with .npy files")
    parser.add_argument("--model", default="dinov2_vitb14", help="DINOv2 model name")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    search(args.reference, args.embeddings_dir, args.model, args.topk)
