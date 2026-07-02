# Handoff: Perceive-Reason-Code blog polish

In-progress polishing of the *Perceive-Reason-Code* blog post (the ICDAR 2026
DocVQA competition writeup: active perception for document VQA). Publishes to
`barisdeniz.is-a.dev`. This doc lets another session pick up cleanly.

## Where things live

- **Blog repo:** `/home/baris/repos/bdsaglam.github.io`
  - Post: `posts/perceive-reason-code/index.qmd` (the source of truth)
  - Styles: `theme.scss`; figures: `posts/perceive-reason-code/f-*.png`, `trajectory.png`
  - Publishes via GitHub Action on **push to `main`** (`_site/` is gitignored, rebuilt by CI)
- **Research repo:** `/home/baris/repos/docvqa`
  - `docs/blog/draft.md` — body-only synced copy of the post (regenerate after every edit)
  - `docs/blog/make_figures.py` — all figure generators (entry point `if __name__`)
  - `workspace/writing-reflection/writing-learnings.md` — reusable writing principles
  - `docs/blog/HANDOFF.md` — this file
  - Results source of truth: `docs/results.md`, `docs/pass-at-k.md`
- **Live preview:** `http://localhost:4444/posts/perceive-reason-code/` (tmux session
  `blog-preview`, host 144.122.52.7). Auto-rebuilds on `.qmd` save, **but the
  watcher wedges silently and often** (serves stale HTML; changed PNGs are never
  re-copied). After any edit, verify with
  `curl -s localhost:4444/... | grep "<new phrase>"`; if stale, restart:
  `tmux kill-session -t blog-preview; rm -rf _site/posts/perceive-reason-code;`
  then relaunch `quarto preview` in a fresh `blog-preview` tmux session.

## Working conventions (hard rules)

- **No em-dashes.** `grep -c "—" index.qmd` must return **0** before any commit. Use
  colons, periods, parentheses, semicolons. ⚠ `grep -c` **exits 1 when the count
  is 0**, so never put it mid-`&&`-chain before a commit/push (it silently drops
  the rest of the chain); run it as its own command.
- **No AI-marker phrases** in post or poster: "load-bearing", "delve". Grep before
  committing.
- **No per-cell (n=k) annotations in figures**; state trial counts in the caption
  (e.g. "4–8 trials per cell").
- **Artifacts are syntheses, not changelogs** — no "previously/now", no "X instead of Y"
  against a Y the reader never saw, no discovery-order narration.
- **Scope claims.** OCR results are "our docling + granite-vision pipeline", never "OCR
  is useless". Same discipline everywhere.
- **Competition win stated once in the body opening** (the TL;DR summarizes it);
  never re-litigated later in the body.
- **Label val vs test** on every number. Everything is validation except the one
  held-out-test competition table.
- **After every edit:** re-sync `draft.md`, grep em-dashes (want 0), glance at preview.
  Sync one-liner (run from the blog repo):
  ```
  python3 -c "import re; src=open('posts/perceive-reason-code/index.qmd').read(); body=re.sub(r'^---\n.*?\n---\n','',src,count=1,flags=re.DOTALL); open('/home/baris/repos/docvqa/docs/blog/draft.md','w').write(body)"
  ```
- More detail in `writing-learnings.md` (esp. §2 voice/register, §4 captions).

## Current state

- Post is polished end to end; em-dash count 0.
- **Framing:** *perception is the constraint, the reasoner is the lever*
  (TL;DR, intro, "Better eyes, or a better director?", Limitations
  cheaper-eyes lever), grounded in the 3×3 matrix
  (`docs/experiments/rvlm-reasoner-perceiver-3x3.md`): 27B-reasoner/4B-VLM
  corner = 32.8 ± 3.1 (n=4) beats 4B/27B = 21.1 and ReAct-27B = 27.2;
  reasoner axis +20.8pp vs VLM axis +9.1pp. Prompt-era + n=4 asymmetry is
  disclosed in the `[^corner]` footnote. The 27B/9B cell is still running;
  the post does not cite it yet.
- **Naming closed:** the full method is **`RLM`** everywhere (tables, figures,
  prose); `CodeAct` is the append-only twin; a naming paragraph at the top of
  Ablations defines RLM / CodeAct / ReAct before any table uses them.
  "Active perception" is used only as the concept, never as a system name.
- **Section order:** intro → result → task → recipe → **Ablations** →
  model-size axis → document-length axis → Limitations → substrate. Tables
  1–6 and Figures 1–7 are numbered in reading order.
- **Figures** regenerated from `docs/blog/make_figures.py`: `f5-vlm-swap.png`
  (three reasoner groups incl. 27B), `f3-tiers.png` + `f-lengthaxis.png`
  (RLM/CodeAct labels).
- **Limitations section** covers: cost/latency (detailed lead), the
  test-below-val gap (harder-split reading first, never "overfitting"), the
  capacity gate, and the aiming-vs-reasoning open question. Deliberately
  excludes: n=3 / validation-only (user rejected), model-coverage (rejected),
  the "unharnessed baselines" comparison claim (factually wrong).

## Open decisions / pending

1. **3×3 matrix watch (active).** The 27B/9B cell is done and published
   (37.2 ± 6.2, n=4). **Figure 5 is now the full matrix heatmap**
   (`fig5_matrix` in `make_figures.py`; blog uses `f5-matrix.png`, the poster
   a wide variant `f5-matrix-poster.png` via
   `fig5_matrix(fname=..., figsize=(8.2,4.6), fscale=1.25, aspect='auto')`).
   The poster carries the same framing (panel "The constraint and the lever",
   Fig. 3; compile with `latexmk -xelatex`, must stay **1 page**). The 4B/9B
   cell is done and published (17.31 ± 1.57, n=4, current rvlm). Only
   **9B/4B** remains; when it lands: score, fill the matrix doc, regenerate
   both matrix pngs (the grey "not run" cell becomes a value; drop the
   caption's grey-cell sentence; reword its prompt-provenance sentence:
   minimal-prompt cells are 4B/4B, 4B/27B, 9B/9B, 9B/27B, all others
   current), re-sync `draft.md`, commit + push both repos.
2. **Baseline column for the homogeneous table** — parked, awaiting Qwen
   **4B/9B** no-scaffold baseline re-runs (rawvlm + official, homogeneous;
   they do not exist yet). When available: compute avg@1 (+ pass@8/SC@8 via
   `scripts/pass_at_k.py`), add the column, and record the numbers in
   `docs/results.md` + `docs/pass-at-k.md`.
3. **Parameter-efficiency framing** (pair results with cost) — user parked it
   ("revisit later"). We are compute-expensive but parameter-cheap.
4. **Vault note** — offered to add a polished `writing-learnings` to the
   user's Obsidian vault (`~/obsidian`). Awaiting go.

## Recent user corrections (internalize these)

- **Rejecting a limitation means remove it** — including pre-existing text that states it,
  not just "don't add new wording."
- **Verify factual claims before writing; do not undersell.** We compared against ReAct
  (a scaffolded baseline) across the ablations and **won the 8–35B tier**. Do not imply
  the comparison was only against weak/unharnessed baselines.
- **Do not lead with "overfitting"** for the test < validation gap. It could be a harder
  test split; the general method strips most DocVQA-specific prompting, which makes
  overfitting the weaker explanation. (Both the in-context "The result" paragraph and the
  Limitations recap now say this.)
- **Captions describe; analysis lives in the body + claim-headings.**
- **Hunt conversational filler / throat-clearing / meta-positioning** as its own pass —
  it survives structure/citation/caption passes (see `writing-learnings.md` §2, §11).

## How to continue

1. `git pull --rebase` in both repos; confirm the preview is up (tmux `blog-preview`).
2. Edit `posts/perceive-reason-code/index.qmd`. After each change: run the sync one-liner,
   `grep -c "—"` (want 0), check the preview.
3. When satisfied, commit both repos and push (blog push auto-deploys):
   ```
   cd /home/baris/repos/bdsaglam.github.io && git add -u && git commit && git push
   cd /home/baris/repos/docvqa && git add -u && git add docs/blog/HANDOFF.md workspace/writing-reflection && git commit && git push
   ```
   (Use `git add -u` in docvqa to avoid the untracked `docs/poster/*.aux/.log` build junk.)
