import json
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image

REFERENCE_DIR = r"\\ihf\Pipelines\GCI_ShackleCarcassBack\Embeddings\reference_image_left"

image_paths = sorted(glob.glob(os.path.join(REFERENCE_DIR, "*.png")))

if not image_paths:
    print("No item*.jpg images found.")
    exit()

for image_path in image_paths:
    base = os.path.splitext(image_path)[0]
    output_json = base + ".json"

    # Skip if JSON already exists for this image
    if os.path.exists(output_json):
        print(f"Skipping {os.path.basename(image_path)} — bbox already exists.")
        continue

    img = Image.open(image_path).convert("RGB")
    coords = {}

    def on_select(eclick, erelease):
        coords["x1"] = int(min(eclick.xdata, erelease.xdata))
        coords["y1"] = int(min(eclick.ydata, erelease.ydata))
        coords["x2"] = int(max(eclick.xdata, erelease.xdata))
        coords["y2"] = int(max(eclick.ydata, erelease.ydata))

    fig, ax = plt.subplots(1, figsize=(10, 14))
    ax.imshow(img)
    ax.set_title(f"[{os.path.basename(image_path)}]  Draw a box, then close window", fontsize=12)
    ax.axis("off")

    selector = RectangleSelector(
        ax, on_select, useblit=True, button=[1], interactive=True,
        props=dict(edgecolor="red", linewidth=2, fill=False),
    )

    plt.tight_layout()
    plt.show()

    if not coords:
        print(f"No box drawn for {os.path.basename(image_path)}, skipping.")
    else:
        with open(output_json, "w") as f:
            json.dump(coords, f)
        print(f"Wrote {coords} to {output_json}")

