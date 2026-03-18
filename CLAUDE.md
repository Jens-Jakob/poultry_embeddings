# CLAUDE.md — Agent Instructions

## Project Context

Read project.md before doing anything. This is an MVP. It will be demonstrated internally. It is not a research project.

## Scope

Build only what is needed for the current phase. Do not design for Phase 2 while implementing Phase 1. Do not add abstractions, config files, or "extensible" architecture unless explicitly asked. Three lines of direct code beats a premature class hierarchy.

## Honesty

If an idea is unlikely to work, say so clearly and with reasoning. Do not soften this to spare feelings. A wasted experiment is a wasted week. Specifically, push back when:

- There is no theoretical reason the proposed change should help.
- The added complexity does not map to a testable, measurable improvement.
- The hypothesis is unfalsifiable or too vague to evaluate.

## Technical Defaults

- Model: DINOv3 (self-supervised ViT). Do not substitute DINOv2 or CLIP without asking.
- Similarity: cosine similarity unless there is a specific reason not to.
- Embeddings: `[CLS]` token for global search. Patch tokens for ROI-based search. Do not mix without reason.
- Hardware target: RTX 3060 for smoke tests. Do not write code that assumes more VRAM than that.

## Code Style

- Python. Keep scripts simple and runnable top-to-bottom.
- No unnecessary CLI frameworks, config parsers, or logging boilerplate for one-off scripts.
- If a script is exploratory, it stays a script. Do not refactor it into a module unless the project grows to need one.
- Prefer explicit over clever. This code will be read by people who may not know the ML stack deeply.

## What Not To Do

- Do not add docstrings or type annotations to code that was not touched.
- Do not add error handling for scenarios that cannot happen in this controlled dataset context.
- Do not suggest switching models, frameworks, or approaches mid-task without a concrete reason.
- Do not pad responses with summaries of what was just done.
