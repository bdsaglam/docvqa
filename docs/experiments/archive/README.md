# Archived experiments

Experiments that are NOT part of the paper's evidence chain under the
current D-006 framing. Kept for reproducibility and historical record.

Three reasons to land here:

1. **Process narratives** that won't appear in the paper (e.g., the
   v1/v2 prompt scrub history — D-006 excludes prompt-iteration narrative
   from the paper).
2. **Shelved approaches** that didn't pan out (e.g., the pydantic-ai
   port, the multi-image VLM extension).
3. **Superseded experiments** whose evidence was rolled into a newer
   experiment file.

If you need the experimental data, it's still in `output/runs/` under
the original run IDs; this folder is the writeup, not the data.

## Index

| File | Why archived |
|---|---|
| [scrub-audit.md](scrub-audit.md) | v1/v2 prompt-scrub process — won't appear in paper per D-006. Outcome (39.0% test SC-8 for the OCR-free configuration) is captured in the current `paper/README.md` headline table. |
| [pyai-leanest-solo-da.md](pyai-leanest-solo-da.md) | pydantic-ai-rlm port — single trial underperformed dspy baseline by 8.8pp. Shelved per D-006. |
| [flat-solo-da-multi-image.md](flat-solo-da-multi-image.md) | Multi-image VLM extension — single-trial regression with no clear category lift. Shelved per D-006. |
