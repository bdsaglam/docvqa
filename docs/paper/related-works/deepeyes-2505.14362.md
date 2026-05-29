---
title: "DeepEyes: Incentivizing 'Thinking with Images' via Reinforcement Learning"
shorthand: deepeyes
arxiv: "2505.14362"
authors: "Zheng et al."
date: "2025-05"
venue: ""
section: "Closest prior work — Think-with-images trained VLMs"
verification: "verified 2026-05-30 — arXiv title matches; abstract + repo confirm think-with-images RL training"
date_added: 2026-05-30
---

# DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning

**Position in our paper:** Closest prior work — the **RL-trained analog of
our `direct_vlm` alt-angle method.** Same mechanism (single multimodal
model that interleaves visual-tool actions with its own reasoning, with
image crops re-entering the conversation), arrived at by RL training
instead of by prompted scaffolding.

**Connection (added 2026-05-30):**

DeepEyes proposes an "interleaved Multimodal Chain-of-Thought" (iMCoT):
the VLM (initialized from Qwen2.5-VL-7B, scaled to 32B) autonomously
decides after each step of textual CoT whether to emit a final answer or
issue an **image zoom-in** — generated as bbox coordinates over the
current image. The cropped region is appended to the trajectory and the
model continues reasoning. Trained end-to-end via outcome-reward RL
(R_acc + R_format + R_tool); no SFT on intermediate reasoning data and no
auxiliary specialized vision tools — the VLM's own grounding ability is
the tool.

This is the same closed loop our `direct_vlm` solver runs
(`docs/solvers/direct-vlm.md`): single multimodal LLM in a REPL,
in-band perception via `display(image)` of pages and PIL crops chosen by
the agent's own Python — no separate VLM sub-call. Where DeepEyes
*trains* the model to interleave zoom-in actions, `direct_vlm` *prompts*
the model to do so within an REPL scaffold.

**Delta we own (vs DeepEyes), and the framing:**

- **Training-free vs RL-trained.** Our `direct_vlm` is a prompted
  scaffold over an off-the-shelf multimodal LLM (e.g. Gemini 3 Pro,
  Qwen2.5-VL). DeepEyes requires RL training on iMCoT trajectories.
  That sharpens our paper's "scaffold lifts off-the-shelf models" claim:
  DeepEyes is *evidence the mechanism works when trained-in*; we show
  it works without training.
- **Generality of the image action.** DeepEyes's tool surface is a
  single bbox-driven zoom-in over the original image. `direct_vlm`'s
  surface is arbitrary PIL operations through Python (any page, any
  crop, any composition) — broader because the agent writes code.
- **Recursive sub-call (the proposed method, `rvlm`).** Our paper's
  *proposed* method is **not** `direct_vlm` — it's `rvlm`, which adds a
  separate VLM sub-call (LLM → VLM) so the lift survives even when the
  main model is text-only. DeepEyes does not have an analog of this and
  cannot, by construction: there is no separate perceiver.
- **Document setting.** DeepEyes evaluates on visual-reasoning /
  perception benchmarks (V*, HR-Bench, MME-RealWorld, etc.), not
  multi-page document VQA. Our setting (DocVQA-2026, MP-DocVQA,
  MMLongBench-Doc) adds multi-page navigation as an axis their setting
  does not exercise.

**How to use in the paper:**

- Cite DeepEyes alongside `direct_vlm` as the closest prior instance of
  the "main multimodal model thinks with image crops" mechanism, with
  RL training as the alternative realization.
- Mention DeepEyes when framing why `direct_vlm` is the *alt-angle* cell
  rather than the proposed method: a trained equivalent of `direct_vlm`
  already exists; the paper's contribution is the **recursive sub-call
  specialization** (`rvlm`), which DeepEyes does not provide.
- If we run a `direct_vlm` vs DeepEyes head-to-head, prefer the same
  backbone family (Qwen2.5-VL) to isolate scaffold-vs-RL as the
  intervention.

**Open questions to settle from the PDF:**

1. Exact tool-surface ablation in DeepEyes — bbox-only zoom-in vs
   freeform image edits? (Determines how literal the `direct_vlm`
   parallel is.)
2. Does iMCoT survive on document images / multi-page documents? Their
   benchmarks are mostly natural images.
3. The 7B → 32B scaling story — does the RL'd think-with-images
   advantage compound with scale, or saturate?

**Code:** repo cloned at `repo/` (https://github.com/Visual-Agent/DeepEyesV2 —
the v2 codebase; includes both stages).

**Related solver docs in this repo:**

- `docs/solvers/direct-vlm.md` — our scaffolded analog.
- `docs/solvers/rvlm.md` — our proposed method (recursive VLM sub-call).
