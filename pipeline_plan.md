# Pipeline Migration Plan

Move similarity search from interactive demo (matplotlib, local GPU) to headless batch execution on Azure Pipelines with a 4090. Core approach: embed everything once, score each candidate against 10 reference images (one per severity grade), keep only the best-matching grade per candidate, output an HTML report.

## Work Items

| # | Item | Effort | Notes |
|---|---|---|---|
| 1 | Defect config JSON | Trivial | One file per defect type (or one file, multiple entries). Each entry: defect name, bbox in pixels, list of reference image paths with grade labels. Bbox is static — same camera, same shackle position. |
| 2 | Rework search.py to headless | Medium | Strip all matplotlib. Read bbox + references from config instead of interactive drawing. Score each reference separately, keep `(best_score, matched_grade)` per candidate. Exclude reference images from results. |
| 3 | Memory-mapped embedding loading | Trivial | `np.load(..., mmap_mode="r")`. Required because embeddings live on a network share — avoids copying the full file into RAM. Fallback: explicit chunked reads if network mmap is too slow. |
| 4 | Annotation CSV filter | Trivial | Load CSV of already-annotated filenames, build a set, mask scores before ranking (`scores[annotated] = -1`). Needs alignment on the key format — filename, full path, or image ID. Runs at search time, not embed time. |
| 5 | HTML report generator | Small | Single self-contained .html file with thumbnail grid. Reference image + ROI at top, results sorted by score. Each result shows: thumbnail, score, matched grade, filename. ~50-80 lines of string templating, no framework. |
| 6 | CSV manifest output | Trivial | Structured output for downstream tools: `image_path, score, matched_grade, matched_region_bbox`. One row per result. |
| 7 | Save CLS embeddings in embed.py | Small | Save `last_hidden_state[:, 0, :]` alongside patch embeddings. Near-zero cost since the model already computes it. Insurance for two-stage coarse filtering if brute-force becomes too slow at scale. |
| 8 | pytorch_runner.py integration | Depends | Add steps: run embed.py if embeddings missing or stale, then run headless search per defect config. Needs access to defect configs, reference images, annotation CSV, and the network-mounted image pool. |
| 9 | Reference image storage | Decision | 10 images per defect type need to be accessible to the pipeline. Options: checked into repo, Azure Blob Storage, or a folder on the same network share as the pool. |
| 10 | Embedding cache strategy | Small | At minimum: skip embed.py if `patch_embeddings.npy` exists and image count matches. Could add filename hash check for robustness. |
| 11 | Network path handling | Trivial | Normalize all paths with `pathlib.Path` consistently. Mixed slashes and different mount points between embed time and search time will break filename matching silently. |

## Open Questions

- **Annotation CSV key format**: filename, full path, or image ID? Determines how we match against embedded filenames.
- **Output destination**: where does the HTML report go after the pipeline runs? Pipeline artifact, blob storage, shared folder?
- **Reference image source**: repo, blob storage, or network share?
- **CLS coarse filtering**: worth validating on the existing 1K test set before building the full pipeline. If CLS can't separate defective from clean birds in the top 20%, two-stage search won't work and we skip item 7.

## Not Doing

- No FAISS — brute-force matmul is sufficient at 30K images, and chunked scoring handles 100K.
- No fine-tuning DINOv3 — zero-shot patch matching already works.
- No database — memory-mapped numpy + CSV manifest is enough at this scale.
- No REST API — this is batch processing, not a live service.
