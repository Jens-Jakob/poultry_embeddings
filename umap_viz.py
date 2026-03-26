"""
Interactive UMAP visualization of image embeddings.

Loads patch embeddings, averages per image, projects to 2D with UMAP,
and displays an interactive scatter plot. Hover over any point to see
the actual image and its file path.

Usage:
    python umap_viz.py
    python umap_viz.py --embeddings-dir ./embeddings --n-neighbors 15 --min-dist 0.1
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
import umap


def load_embeddings(embeddings_dir):
    patches = np.load(os.path.join(embeddings_dir, "patch_embeddings.npy")).astype(np.float32)
    filenames = np.load(os.path.join(embeddings_dir, "filenames.npy"))
    # Average patch tokens per image to get a single vector per image
    cls_approx = patches.mean(axis=1)  # (N, dim)
    return cls_approx, filenames


def run_umap(vectors, n_neighbors, min_dist):
    print(f"Running UMAP on {vectors.shape[0]} images ({vectors.shape[1]}-dim) ...")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine", random_state=42)
    coords = reducer.fit_transform(vectors)
    print("UMAP done.")
    return coords


def make_thumbnail(path, size=200):
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size))
    return np.array(img)


def plot(coords, filenames):
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        s=12, c=np.linalg.norm(coords, axis=1),
        cmap="plasma", alpha=0.7, edgecolors="none",
    )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("UMAP — Image Embeddings", fontsize=16, color="white", pad=15)

    # Hover annotation — image thumbnail above, filepath text below
    annot_img = OffsetImage(np.zeros((200, 200, 3), dtype=np.uint8), zoom=1.0)
    annot_box = AnnotationBbox(
        annot_img, (0, 0),
        xybox=(110, 120), xycoords="data", boxcoords="offset points",
        pad=0.4,
        bboxprops=dict(boxstyle="round,pad=0.4", fc="#16213e", ec="white", lw=1),
        arrowprops=dict(arrowstyle="->", color="white", lw=1),
    )
    annot_box.set_visible(False)
    ax.add_artist(annot_box)

    # Text sits below the image box, not overlapping it
    text_annot = ax.annotate(
        "", xy=(0, 0), xytext=(110, 10), textcoords="offset points",
        color="#aaaaaa", fontsize=7, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#16213e", ec="#444444", lw=0.5),
    )
    text_annot.set_visible(False)

    last_idx = [None]

    def on_hover(event):
        if event.inaxes != ax:
            if annot_box.get_visible():
                annot_box.set_visible(False)
                text_annot.set_visible(False)
                fig.canvas.draw_idle()
            return

        cont, ind = scatter.contains(event)
        if not cont:
            if annot_box.get_visible():
                annot_box.set_visible(False)
                text_annot.set_visible(False)
                fig.canvas.draw_idle()
            return

        idx = ind["ind"][0]
        if idx == last_idx[0]:
            return
        last_idx[0] = idx

        pos = (coords[idx, 0], coords[idx, 1])
        path = str(filenames[idx])

        try:
            thumb = make_thumbnail(path)
            annot_img.set_data(thumb)
        except Exception:
            annot_img.set_data(np.zeros((200, 200, 3), dtype=np.uint8))

        annot_box.xy = pos
        annot_box.set_visible(True)

        basename = os.path.basename(path)
        text_annot.xy = pos
        text_annot.set_text(basename)
        text_annot.set_visible(True)

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_hover)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", default="./embeddings")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    args = parser.parse_args()

    vectors, filenames = load_embeddings(args.embeddings_dir)
    coords = run_umap(vectors, args.n_neighbors, args.min_dist)
    plot(coords, filenames)
