# Poultry Embeddings — Re-annotation Search Tool

## What This Is

An MVP similarity search tool for poultry quality control. The core idea: given a reference image of a defective bird, find the most visually similar birds in a large image pool. The primary use case is **re-annotation assistance** — not replacing an object detector, but helping an annotator quickly surface unlabeled images that likely contain the same defect type.

This is built for internal demonstration to the company. It is not a research paper. It does not need to beat a supervised detector.

## Approach

Use DINOv2 (self-supervised ViT) to embed images. Compare embeddings via cosine similarity. Return ranked results. DINOv3 access is pending approval — switch to DINOv3 when available (code change is minimal).

**Phase 1 (current):** Global `[CLS]` token embeddings via DINOv2 ViT-B/14. No bounding boxes. Full-image similarity. Evaluate by eyeball — do the top results look like the reference?

**Phase 2 (later):** ROI-based querying. User draws a bounding box on the reference defect region. Search is against patch-level embeddings cropped to that region. This is required for the production tool but not for the first smoke test.

## Dataset

- ~1000 dead broiler chickens hanging from shackles, front-facing camera
- Source: Roboflow
- Bounding box annotations for defects included (multi-class)
- Some swing/orientation variation present
- Defects confirmed visually present in the first 10% of images inspected

## First Test — Eyeball Similarity Search

No bounding boxes. No metrics. Just:

1. Pick 3 reference images: one with an obvious bruise, one with discoloration, one clean bird (sanity check)
2. Embed all ~1000 images with DINOv2 global CLS embedding
3. Rank by cosine similarity for each reference
4. Inspect top 20 results per reference

**Pass signal:** Defect references return more defective birds in top-20 than the clean reference does.
**Fail signal:** All three references return nearly identical ranked lists — embeddings are dominated by pose/lighting, not defect texture. If this happens, move to patch-level embeddings before anything else.

## How to Run (PC with RTX 3060)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Embed all images (runs once, ~2-3 min on 3060)
python embed.py --data-dir "/path/to/Poultry Defects.coco/train"

# 3. Search with a reference image
python search.py --reference "/path/to/some_image.bmp"

# If the reference is from train/, it will use the cached embedding (instant).
# If it's a new image, it loads the model and embeds on-the-fly.
# When using a train image as reference, ignore rank #1 (it's itself).
```

## Later Work (Not Now)

- ROI/patch-level querying
- Formal metrics (Precision@K) using Roboflow bounding box labels
- UI for the reannotation workflow (import pool, draw reference box, view ranked results)
- Scaling to full production dataset (less swing variation than smoke test set)
