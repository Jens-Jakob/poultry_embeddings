"""
ROI-based similarity search using DINOv3 patch embeddings.

Draw a bounding box on a reference image to find images with similar
local textures in that region.

Usage:
    # Interactive: opens image, draw a box, close window to search
    python search.py --reference "/path/to/image.bmp"

    # With explicit bbox (pixels on original image): x1,y1,x2,y2
    python search.py --reference "/path/to/image.bmp" --bbox 100,200,250,400

    # Detail mode: side-by-side slideshow, arrow keys to navigate
    python search.py --reference "/path/to/image.bmp" --detail

    # Change number of results
    python search.py --reference "/path/to/image.bmp" --topk 10
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from matplotlib.widgets import RectangleSelector
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def load_index(embeddings_dir):
    patches = np.load(os.path.join(embeddings_dir, "patch_embeddings.npy"))
    filenames = np.load(os.path.join(embeddings_dir, "filenames.npy"))
    meta = np.load(os.path.join(embeddings_dir, "meta.npz"))
    return patches, filenames, meta


def draw_bbox(image_path):
    """Open image in matplotlib, let user draw a rectangle. Returns (x1, y1, x2, y2) in pixels."""
    img = Image.open(image_path).convert("RGB")
    coords = {}

    def on_select(eclick, erelease):
        coords["x1"] = int(min(eclick.xdata, erelease.xdata))
        coords["y1"] = int(min(eclick.ydata, erelease.ydata))
        coords["x2"] = int(max(eclick.xdata, erelease.xdata))
        coords["y2"] = int(max(eclick.ydata, erelease.ydata))

    fig, ax = plt.subplots(1, figsize=(10, 14))
    ax.imshow(img)
    ax.set_title("Draw a rectangle on the defect, then close this window", fontsize=12)
    ax.axis("off")

    selector = RectangleSelector(
        ax, on_select,
        useblit=True,
        button=[1],
        interactive=True,
        props=dict(edgecolor="red", linewidth=2, fill=False), # No fill, solid edge
    )

    plt.tight_layout()
    plt.show()

    if not coords:
        raise RuntimeError("No bounding box drawn. Please draw a rectangle before closing.")

    print(f"Selected bbox: ({coords['x1']}, {coords['y1']}) -> ({coords['x2']}, {coords['y2']})")
    return coords["x1"], coords["y1"], coords["x2"], coords["y2"]


def bbox_to_patch_indices(x1, y1, x2, y2, img_w, img_h, grid_w, grid_h):
    """Map pixel bbox on original image to patch grid indices."""
    col1 = int(x1 / img_w * grid_w)
    row1 = int(y1 / img_h * grid_h)
    col2 = min(int(np.ceil(x2 / img_w * grid_w)), grid_w)
    row2 = min(int(np.ceil(y2 / img_h * grid_h)), grid_h)

    indices = []
    for r in range(row1, row2):
        for c in range(col1, col2):
            indices.append(r * grid_w + c)

    return indices, row1, col1, row2, col2


def score_candidates(query_patches, all_patches):
    """
    For each query patch, find best match in each candidate image.
    Score = mean of best matches.

    query_patches: (num_query, dim)
    all_patches: (N, num_patches, dim)
    Returns: (N,) scores, (N, num_query) matched patch indices per candidate
    """
    N = all_patches.shape[0]
    num_patches = all_patches.shape[1]
    sims = query_patches @ all_patches.reshape(-1, all_patches.shape[2]).T
    sims = sims.reshape(query_patches.shape[0], N, num_patches)
    # Best match per query patch per candidate
    best_per_query = sims.max(axis=2)       # (num_query, N)
    match_indices = sims.argmax(axis=2)     # (num_query, N) — which patch matched
    scores = best_per_query.mean(axis=0)    # (N,)
    # Transpose match_indices to (N, num_query) for convenience
    return scores, match_indices.T


def matched_patches_to_bbox(match_indices, grid_w, grid_h, img_w, img_h):
    """Convert a set of matched patch indices to a pixel bounding box on the original image."""
    rows = match_indices // grid_w
    cols = match_indices % grid_w
    # Patch grid coords to pixel coords
    px_x1 = int(cols.min() / grid_w * img_w)
    px_y1 = int(rows.min() / grid_h * img_h)
    px_x2 = int((cols.max() + 1) / grid_w * img_w)
    px_y2 = int((rows.max() + 1) / grid_h * img_h)
    return px_x1, px_y1, px_x2, px_y2


def embed_patches_single(image_path, model_name):
    """Embed a single image and return its normalized patch tokens."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    patch_size = model.config.patch_size
    dummy = processor(images=[Image.new("RGB", (100, 100))], return_tensors="pt")
    proc_h, proc_w = dummy["pixel_values"].shape[2], dummy["pixel_values"].shape[3]
    num_patches = (proc_h // patch_size) * (proc_w // patch_size)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=[image], return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)

    patches = outputs.last_hidden_state[:, -num_patches:, :].cpu().numpy()[0]
    norms = np.linalg.norm(patches, axis=1, keepdims=True)
    patches = patches / np.maximum(norms, 1e-8)
    return patches


def show_detail(ref_img, x1, y1, x2, y2, ranked, scores, filenames, topk,
                match_bboxes=None):
    """Side-by-side slideshow: reference (left) + one result at a time (right). Arrow keys to navigate."""
    show = ranked[:topk]
    state = {"idx": 0}

    fig, (ax_ref, ax_result) = plt.subplots(1, 2, figsize=(14, 8))

    # Reference (stays fixed)
    ax_ref.imshow(ref_img)
    rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                               linewidth=2, edgecolor="red", facecolor="red", alpha=0.3)
    ax_ref.add_patch(rect)
    ax_ref.set_title("REFERENCE (ROI)", fontsize=12, fontweight="bold", color="red")
    ax_ref.axis("off")

    def render(i):
        ax_result.clear()
        idx = show[i]
        img = Image.open(filenames[idx]).convert("RGB")
        ax_result.imshow(img)

        # Draw matched region on candidate
        if match_bboxes is not None:
            mx1, my1, mx2, my2 = match_bboxes[idx]
            match_rect = mpatches.Rectangle(
                (mx1, my1), mx2 - mx1, my2 - my1,
                linewidth=0.5, edgecolor="red", fill=False)
            ax_result.add_patch(match_rect)

        img_id = os.path.basename(filenames[idx]).split("_")[0]
        ax_result.set_title(f"#{i+1}/{topk}  img:{img_id}  score:{scores[idx]:.3f}", fontsize=12)
        ax_result.axis("off")
        fig.suptitle("← → arrow keys to navigate, Q to quit", fontsize=10, color="gray")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "right" and state["idx"] < topk - 1:
            state["idx"] += 1
            render(state["idx"])
        elif event.key == "left" and state["idx"] > 0:
            state["idx"] -= 1
            render(state["idx"])
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    render(0)
    plt.tight_layout()
    plt.show()


def show_grid(ref_img, x1, y1, x2, y2, ranked, scores, filenames, topk,
              match_bboxes=None):
    """Grid view: reference + all top-K results."""
    show = ranked[:topk]
    cols = min(topk, 5)
    rows = (topk + cols - 1) // cols
    fig, axes = plt.subplots(rows + 1, cols, figsize=(3 * cols, 3 * (rows + 1)))

    if rows + 1 == 1:
        axes = axes[np.newaxis, :]
    if cols == 1:
        axes = axes[:, np.newaxis]

    # Reference with bbox overlay
    for c in range(cols):
        axes[0, c].axis("off")
    axes[0, 0].imshow(ref_img)
    rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                               linewidth=2, edgecolor="red", facecolor="red", alpha=0.3)
    axes[0, 0].add_patch(rect)
    axes[0, 0].set_title("REFERENCE (ROI)", fontsize=10, fontweight="bold", color="red")

    # Results
    for i, idx in enumerate(show):
        r = 1 + i // cols
        c = i % cols
        img = Image.open(filenames[idx]).convert("RGB")
        axes[r, c].imshow(img)

        if match_bboxes is not None:
            mx1, my1, mx2, my2 = match_bboxes[idx]
            match_rect = mpatches.Rectangle(
                (mx1, my1), mx2 - mx1, my2 - my1,
                linewidth=2, edgecolor="cyan", fill = False)
            axes[r, c].add_patch(match_rect)

        img_id = os.path.basename(filenames[idx]).split("_")[0]
        axes[r, c].set_title(f"#{i+1} img:{img_id} ({scores[idx]:.3f})", fontsize=8)
        axes[r, c].axis("off")

    for i in range(len(show), rows * cols):
        r = 1 + i // cols
        c = i % cols
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.show()


def search(reference_path, embeddings_dir, model_name, topk, bbox, detail=False):
    all_patches, filenames, meta = load_index(embeddings_dir)
    grid_h, grid_w = int(meta["grid_h"]), int(meta["grid_w"])

    # Get reference patches
    ref_abs = os.path.abspath(reference_path)
    match_idx = None
    for i, fn in enumerate(filenames):
        if os.path.abspath(fn) == ref_abs:
            match_idx = i
            break

    if match_idx is not None:
        print(f"Reference found in index at position {match_idx}")
        ref_patches = all_patches[match_idx]
    else:
        print(f"Reference not in index, embedding on-the-fly...")
        ref_patches = embed_patches_single(reference_path, model_name)

    # Get bbox
    ref_img = Image.open(reference_path).convert("RGB")
    img_w, img_h = ref_img.size

    if bbox is None:
        x1, y1, x2, y2 = draw_bbox(reference_path)
    else:
        x1, y1, x2, y2 = bbox

    # Map to patch grid
    patch_indices, row1, col1, row2, col2 = bbox_to_patch_indices(
        x1, y1, x2, y2, img_w, img_h, grid_w, grid_h
    )
    print(f"Bbox maps to patch grid [{row1}:{row2}, {col1}:{col2}] ({len(patch_indices)} patches)")

    if not patch_indices:
        print("ERROR: bbox too small, maps to zero patches. Draw a larger box.")
        return

    # Extract query patches
    query_patches = ref_patches[patch_indices]  # (num_query, dim)

    # Score all candidates
    scores, match_indices = score_candidates(query_patches, all_patches)
    ranked = np.argsort(scores)[::-1]

    # Compute matched-region bboxes for each candidate image
    # match_indices is (N, num_query) — for each image, which patches matched the query
    match_bboxes = {}
    for idx in ranked[:topk]:
        candidate_img = Image.open(filenames[idx]).convert("RGB")
        cw, ch = candidate_img.size
        mx1, my1, mx2, my2 = matched_patches_to_bbox(
            match_indices[idx], grid_w, grid_h, cw, ch
        )
        match_bboxes[idx] = (mx1, my1, mx2, my2)

    # Print results
    print(f"\nTop {topk} matches for ROI on: {os.path.abspath(reference_path)}")
    print("-" * 80)
    for rank, idx in enumerate(ranked[:topk], 1):
        # We use os.path.abspath to ensure it shows the full drive path (e.g., C:\Users\...)
        full_path = os.path.abspath(filenames[idx])
        print(f"  #{rank:2d}  score={scores[idx]:.4f}  {full_path}")

    # Display results
    if detail:
        show_detail(ref_img, x1, y1, x2, y2, ranked, scores, filenames, topk, match_bboxes)
    else:
        show_grid(ref_img, x1, y1, x2, y2, ranked, scores, filenames, topk, match_bboxes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="Path to reference image")
    parser.add_argument("--embeddings-dir", default="./embeddings", help="Directory with .npy files")
    parser.add_argument("--model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--topk", type=int, default=40)
    parser.add_argument("--bbox", type=str, default=None,
                        help="Bounding box as x1,y1,x2,y2 in pixels. If omitted, draw interactively.")
    parser.add_argument("--detail", action="store_true",
                        help="Side-by-side slideshow mode. Arrow keys to navigate results.")
    args = parser.parse_args()

    bbox = None
    if args.bbox:
        bbox = tuple(int(v) for v in args.bbox.split(","))

    search(args.reference, args.embeddings_dir, args.model, args.topk, bbox, detail=args.detail)
