# Brief — DocVQA REPL-agent blog post

**Status:** awaiting gate (level 0)

## Medium
Technical blog post. Personal blog. Long, definitive piece (~4–5k words),
full figures (real plots/tables generated from run data). Less formal than a
paper but keeps math/formalization where it earns its place, plus figures and
tables. Tone decided at draft stage.

## Why a blog and not a paper
No paper planned — the team judged the core idea insufficiently novel (it is a
focused application of existing paradigms: RLM-style sub-calls + CodeAct-style
REPL agents). That frees the post from a novelty-defense posture: it can be
honest about what is borrowed, what is specialized, and what simply *worked*.

## Audience
ML/AI practitioners and researchers fluent in LLMs, VLMs, and agents, but **not
assumed to know** RLM (Recursive Language Models), CodeAct, ReAct-vs-REPL
distinctions, or the DocVQA task. Those three need brief, self-contained
scaffolding. Math and tables are welcome; jargon is not free.

## What this post is (and isn't)
A **technical writeup of our method and findings** from building the ICDAR 2026
DocVQA entry. It is *not* a novelty pitch: that agentic harnesses lift model
performance, and that REPL/coding agents (RLM, CodeAct) outperform prior-
generation ReAct-style tool-calling, is already well established across many
domains. We don't sell that. The competition win is the project's **origin and a
credibility data point**, not the headline. What the post offers a practitioner
is a **clean decomposition of *why* the harness helps on this task** — and a
closing hint at a bigger idea (below).

## Payload (core message — to be sharpened into a throughline next)
The RLM harness we adopted bundles three things: **recursion** (the model calls
itself or another LM as a sub-call), **symbolic access to the prompt/context**
through a code REPL, and a **partially-observable** trajectory exposed as REPL
history. Our ablations pull these apart, and the clean finding is that **two
halves are jointly load-bearing — the REPL *and* one level of recursion — while
recursion *depth/generality* and the observability structure are not**:

- **Both halves are required; neither alone suffices.** REPL + delegated VLM call
  (`rvlm`) = 42%. Remove the REPL (`react`, delegated call only) → 27%. Remove the
  delegation (`direct_vlm`, REPL but pixels in its own context) → 22%. Neither
  (`raw_vlm_multi`) → 21%. The lift needs the code REPL **and** a focused VLM
  sub-call that returns text — a model calling a model (genuine recursion when
  agent and VLM are the same model; they need not be).
- **What's dispensable is recursion *depth/generality*, not the delegation.**
  Generalizing the single-level VLM call to arbitrary subagent delegation
  (`rvlm_subagent`) is **harmless but unused** — used as a plain perception tool
  ~99% of the time. One level of recursion is enough; deeper/general delegation
  buys nothing. So we use minimal recursion + REPL, and don't claim deep recursion.
- **The observability structure also doesn't matter.** `codeact_chat` — a fully-
  observable, append-only chat MDP — ties the partially-observable RLM form
  (`rvlm`) within ~2pp. Two trajectory formulations, same result; the common
  factor is REPL + delegated perception, not how the trajectory is exposed.

Net: **a code-capable model + a REPL + one delegated VLM perception call** is the
recipe. The REPL is the symbolic substrate; the single VLM sub-call is what feeds
it. Both are load-bearing; depth and observability are not.

## Closing idea (the post's final turn — open question, not a claim)
The REPL is, in effect, a **symbolic substrate for a neural model** — code as the
medium in which the model explores, composes, and reasons over context it can't
hold directly. That this helps at **test time** is now broadly shown (here and
elsewhere). The open question the post ends on: would the *same symbolic
substrate* help a model during **learning / training**, not just inference? The
dominant paradigm uses code at inference; using it as part of how a model *learns*
is a less-explored, potentially bigger direction — beyond RLVR-for-coding-agents.
We do **not** answer this; we pose it as a thread worth pulling. (Honest framing:
this idea isn't novel in isolation either; the *learning-time* angle is the part
worth flagging.)

## Raw material — results (verified against docs/results.md, docs/experiment-status.md)

### The competition win (credibility — the FACT, not the old numbers)
- **Joint winner, ICDAR 2026 DocVQA challenge, 8–35B parameter tier.** This fact
  is provenance-independent and always true — it's the credibility stamp.
- **NUMBER POLICY (user directive):** use the **latest current-code numbers**
  everywhere; the competition submission gets a **short historical note only**. We
  changed agents and prompts substantially since the submission (partly to shrink
  the val↔test gap), so the old test figure does **not** represent the current
  method. Do not headline it.
- Short note (all the post says about the submission): our challenge entry used a
  now-superseded solver — the REPL + active-perception agent **with OCR page text
  + a BM25 search tool**, SC-8 voting, Qwen 3.6 27B — and was a joint tier winner.
  We've since simplified the agent and prompts; numbers throughout the post are
  current code.
- **Narrative thread (keep — it's about the method, not the old score):** the
  clean ablations later showed the OCR + search were **not load-bearing** — the
  OCR-free method matches the OCR+search variant on these docs (within std). We won
  with the extra tooling, then found the minimal harness suffices. Ties the win to
  the teardown's "OCR-on-top ≈ 0" beat and the "minimal harness" takeaway.
- **Frontier comparison — use current-code VAL, with the caveat.** Official val
  baselines (external, for context): **Gemini 3 Pro 37.5%**, **Gemini 3 Flash
  33.75%** (GPT-5.2 35.0% is **test-only**, no val — don't use it in a val
  comparison). Our current OCR-free method is ~42% on our val subset. ⚠ **Subset
  caveat:** our number is the 25-doc/80-Q ablation subset; the official baselines
  are full-set — so frame as "in the range of / not behind" the closed frontier,
  not a strict beat, unless we run current code on full val. (Possible production
  task: full-val run of the current method for a clean head-to-head.)

### The clean explanatory matrix (current `main`, Qwen 3.5 27B homog, val 25-doc/80-Q, **n=8**, mean ± std)
Source of the *why*. All from `docs/results.md`.

| Solver | What it is | Val n=8 |
|---|---|---|
| `rvlm` | REPL + delegated VLM perception (`batch_look`), OCR-free | **41.88% ± 5.79** |
| `codeact_chat` | append-only chat-MDP twin; same tools, REPL | 39.53% ± 2.83 |
| `rvlm_subagent` | sub-call generalized to arbitrary delegation | 36.72% ± 2.75 (≈ rvlm, within std) |
| `rvlm_ocr` | + OCR text + BM25 search | 36.56% ± 2.89 (OCR adds ~0 on moderate docs) |
| `rvlm_nocrop` | no crop/zoom (whole pages only) | 35.78% ± 2.31 |
| `react_baseline` | same VLM tools, **no REPL** | 27.19% ± 3.19 (−14.7pp: REPL load-bearing) |
| `direct_vlm` | REPL but pixels into own context, **no sub-call** | 22.34% ± 2.79 (−17pp: delegation load-bearing) |
| `raw_vlm_multi` | raw multi-image, no scaffold | 20.94% ± 1.60 |
| `official_baseline` | competition MASTER_PROMPT, no scaffold | 18.91% ± 1.94 |
| `rlm_ocr` | REPL + OCR text, **no vision** | 14.69% ± 2.19 (−27.2pp: OCR≠vision floor) |

Three clean tiers, every cross-tier gap ≫ the std: REPL+delegated-perception
(36–42%) ≫ no-REPL-or-no-delegation (21–27%) ≫ OCR-only-no-vision (15%).

### Supporting axes (mechanism evidence)
- **Efficiency (iters/Q):** `rvlm` ~13 steps, ~never caps; `direct_vlm` 30.4,
  median = the 40-cap, 59% pin the cap (pixels-in-context → grind, no
  convergence); `react` 5.1 (too shallow without a REPL to compose steps).
- **Model-size / VLM-quality axis (n=8):** fix the reasoner, swap **only** the
  VLM → 27B: +7.87pp at 9B, +8.60pp at 4B (both significant). Gemma confirms a
  *capacity gate*: +21.4pp harness lift at 31B, ~0 at E4B. Signature of a
  **perception (not reasoning) bottleneck** — this is the mechanism beneath the
  REPL story and worth including.
- **Document-length axis (IN PROGRESS, escalating n=1→n=3):** recursive-
  perception advantage scales with doc length — gap ~2–6pp on moderate docs
  (MP-DocVQA ≤20pg) vs ~16–42pp on long docs (MMLongBench ~47pg); raw-VLM
  Unknown-rate climbs 8/22% → 36/87% as the fixed page budget misses evidence.
  Numbers may firm up before draft; treat n=1 reads as provisional.
- **pass@k / oracle headroom:** strong scaffolds have large oracle headroom
  (`codeact_chat` avg@1 39.5 → pass@8 63.8) → ~25pp recoverable by a verifier /
  best-of-n / RL reward model. Candidate "what's next" closer.
- **CodeAct = MDP twin:** `codeact_chat`'s append-only transcript is a clean
  fully-observable MDP → a natural RL fine-tuning target. Ties `rvlm` at no cost.

### Lineage / citations (no method novelty — wear it openly)
Source: `docs/paper/related-works.md` + `docs/paper/related-works/*.md` (citations
verified-to-exist 2026-05-27; load-bearing *claims* still need a PDF check before
quoting numbers). For the post's short "where this sits" beat:
- **RLM** — Zhang, Kraska, Khattab, *Recursive Language Models*, arXiv:2512.24601
  (Dec 2025). The paradigm we instantiate; already showed (text) REPL-alone lifts
  + sub-call adds 10–59%. Cite as source of the idea.
- **CodeAct** — Wang et al., *Executable Code Actions Elicit Better LLM Agents*
  (ICML 2024; verify arXiv id at draft). Code-as-action lineage.
- **VisProg / ViperGPT / Chameleon** — code-as-reasoning vision agents (CVPR/ICCV
  2023; NeurIPS 2023). Lineage for code-orchestrated vision.
- **RVLM** — arXiv:2603.24224 (concurrent; recursive vision-language, medical
  single-image). Different domain.
- **MADQA** — Borchmann et al., arXiv:2603.12180 (doc-collection agents; critiques
  RLM effort overhead). Pre-empts "constraining RLM helps."
- **DeepEyes** — Zheng et al., arXiv:2505.14362 (RL-trained think-with-images;
  the trained twin of `direct_vlm`).
- **Claims to avoid** (from related-works.md): not "first to use code/OCR/visual
  tools on DocVQA," not "first multi-page," not "recursive-visual is unexplored."

## Constraints / honesty rules
- Numbers cite `docs/results.md` / `docs/experiments/*.md`; never drift or invent.
- Two number systems kept distinct: the **Qwen 3.6 / SC-8 / test** win vs the
  **Qwen 3.5 / n=8 / val** clean matrix. Never blend them into one table.
- No "recursive" overclaim; no v1/v2/prompt-iteration history (synthesis, not
  changelog). Archived/pre-2026-06-01 numbers are off-limits as current.
- Reference RLM as *inspiration*; state plainly we use a **single level** of
  recursion (model→VLM), not deep/arbitrary delegation.
- **No legacy engineering solver names** (`flat_solo`, `leanest_solo`,
  `no_loop_multi`, …) anywhere in the prose. Describe each configuration by what
  it does (e.g., "the REPL agent with a delegated VLM call," "the ReAct
  baseline," "the raw-VLM baseline"). A repo/name mapping, if needed, goes in a
  footnote — not the body.

## Success criteria
A practitioner who builds with LLM agents finishes the post able to (a) explain
why a REPL + delegated VLM perception beats raw VLM and ReAct on documents, (b)
recognize document VQA as perception-budget-bound and know the diagnostic
(Unknown-rate / page-budget), and (c) reproduce the recipe (it's an
architecture choice, not training). Plus: they remember a 27B open model beat
the closed frontier, and trust the numbers because the honest caveats are stated.

## Resolved at the gate
- **Win is origin + credibility, not the hook.** The post is a method/findings
  technical doc. The *fact* of the joint tier win is stated; old competition
  numbers are a short note only — **latest current-code numbers are canonical
  throughout** (user directive). Not promoted as novel.
- **Doc-length axis:** build the structure now; slot final n=3 numbers at
  production. Treat current n=1 reads as provisional placeholders.
- **Two number systems kept distinct** (3.6/SC-8/test win vs 3.5/n=8/val matrix).
- **End on the neuro-symbolic open question** (symbolic substrate at training
  time) — posed, not answered.
