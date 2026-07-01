# Outline — DocVQA REPL-agent blog post

**Status:** awaiting gate (level 2). Units = post sections, stable IDs. Each unit:
its one-line job in the throughline + the concrete artifact (figure/table/code/
number) that carries it. Ordered by value density; TL;DR drafted last.

Figures/tables are listed per unit; a consolidated asset list is at the bottom.
Provisional numbers (doc-length axis, n=3 running) are flagged `[prov]`.

> **TERMINOLOGY (locked):** primary term **active perception** (reasoner directs a
> **depth-1 VLM-tool call**: crop/zoom/look). Avoid "recursive"/"delegation" as
> framing; mention recursion once, only to set aside. "ReAct" = named baseline.

---

### 01 · `tldr` — TL;DR box (top of post; **drafted LAST**)
- **Job:** Let a skimmer leave in 30s with the payload, the one counterintuitive
  finding, the win, and the recipe. Pure function of the finished post.
- **Carries:** 4–5 bullets — (1) general off-the-shelf 27B + tiny REPL harness →
  joint winner ICDAR'26 8–35B tier, beats Gemini 3 Pro / GPT-5.2 on test; (2) the
  load-bearing parts are the REPL *and* one delegated VLM call — both, jointly;
  (3) the surprise: delegating perception beats holding pixels yourself; (4) it's
  a perception-budget problem, not reasoning; (5) no fine-tuning, no method
  novelty — link to repo. No new info; distilled from below.

### 02 · `intro` — Hook + nut graf
- **Job (moves 1–2):** Open the gap — an open 27B beats the closed frontier using
  *all-borrowed parts*, so the real question is *which* parts carry the win; and
  tell the reader why this matters to them now (which agent knobs are worth
  paying for; maybe you don't need fine-tuning).
- **Carries:** the headline contrast (open 27B vs Gemini/GPT on test); the honest
  framing (no new method); one-sentence promise of a counterintuitive result.
  No figure — prose. ~3 short paragraphs.

### 03 · `task` — What document VQA is, and why it's hard for VLMs
- **Job (move 3):** Give an outsider just enough: many-page, dense, fine-grained
  documents (tables, figures, engineering drawings, maps, infographics); a single
  raw multi-image pass misses evidence. Sets up "perception is the hard part."
- **Carries:** **F1** — the **NVIDIA annual-review chart page** (`br1_page76`
  full + crop, public financial disclosure): a dense bar+line chart with ~20 tiny
  overlapping labels where a whole-page read returns the wrong number ($978.42)
  and a crop recovers the right one (NVIDIA TSR $2,287.07 vs Nasdaq-100 $238.19).
  Ties directly to the C1 trajectory. 1 short para + figure.

### 04 · `harness` — The recipe, and where it comes from
- **Job (move 4 + the woven "where this sits" beat):** Define the harness — a
  code-capable model in a REPL whose perception is a delegated VLM call
  (`batch_look`: look at page/crop → get text). Name the lineage (RLM, CodeAct,
  VisProg/ViperGPT) in a few honest sentences; state plainly **no new method**;
  note RLM already showed (for *text*) REPL-lifts + sub-call-adds, which frames
  the post's question: does it hold when the sub-call is *visual*, and which parts
  carry it?
- **Carries:** **F2** — architecture diagram, **adapt from the companion report's
  two-panel `fig-architecture`** (Panel A active-perception: Reasoner→REPL→frozen
  VLM→pages, edges "writes code"/"batch_look crop-zoom"/dashed "obs"; Panel B
  ReAct: Reasoner→VLM→pages, "no crop/zoom, perception fixed at page granularity").
  **C1** — short real trajectory excerpt on the **same NVIDIA page as F1** (from
  report `sections/03` §3.5 / `fig-trajectory`): survey → locate → read → *distrust
  & crop-verify* (catches the wrong $978.42) → compute in Python
  (`2287.07 − 238.19 ⇒ 2048.88` ✓). Prose + fenced Python, not the TikZ swimlane.
  Alt if a cleaner compute example is wanted: the science_poster pair (RLM crops +
  computes 30.2% ✓ in 6 iters vs ReAct whole-page 0.48% ✗). Light formalism OK: name the loop as
  state/action(code)/observation (the MDP framing). **Thinking-off defense** (1–2
  sentences, since our matrix is `enable_thinking=false`): disabling the native
  thinking channel *relocates* reasoning into the turn body, it doesn't remove it.

### 05 · `ablations` — What's actually carrying the lift (THE CORE / engine)
- **Job (move 5, the spine):** Knock out one component at a time and report which
  are load-bearing. Resolve the gap opened in the intro.
- **Beats (each re-opens/pays a question):**
  1. Remove the **REPL** (ReAct, same VLM tools) → collapse (~27%).
  2. Remove the **active-perception call** (REPL agent loads pixels into own
     context, `direct_vlm`) → collapse (~22%). **The surprise:** a focused VLM
     perception call > holding pixels in your own context. (DeepEyes/think-with-
     images = the RL-trained twin.)
  3. ⇒ **Both halves jointly load-bearing; neither alone suffices.**
  4. **Sub-call generality doesn't matter:** generalize the focused perception
     call into an arbitrary subagent → unused (~99% plain perception). A single
     focused call suffices. (Depth-1 throughout; no depth claim.)
  5. **Observability doesn't matter — for inference:** append-only CodeAct ties
     compacted RLM (~2pp, solid at 27B & 4B; page-count×(RLM−CodeAct) corr ≈ 0).
     Forward-looking aside (borrow report §4): it *does* matter for **training** —
     RLM compaction rewrites the trajectory, breaking the growing-prefix structure
     RL/distillation assume; append-only CodeAct is the more **trainable** target
     (MDP-like vs POMDP-like, caveat: never a true MDP — doc unobserved). Cite
     **FoldAct** (2512.22733). Seeds the `close`. No own-training mention.
  6. **OCR-free is fine:** swap vision→OCR text = the true floor; OCR *on top* of
     vision ≈ 0 on moderate docs (extension for long docs). Note OCR-only sits
     *below* the no-scaffold competition prompt.
- **Carries:** **F3** — three-tier bar chart, all solvers, val n=8, error bars
  (the headline figure). **F4** — 2×2 (REPL on/off × delegation on/off), the
  cleanest "both halves matter" visual — **must be within one number-batch** (see
  data note). **T1** — full solver matrix (role + val n=8 mean±std). Optional
  callout for `direct_vlm` as the clean context-rot illustration (not "a
  surprise"). (DeepEyes = the RL-trained twin of
  `direct_vlm`; not in the report's bib — add arXiv:2505.14362 if cited.)

### 06 · `mechanism` — Why it works: perception budget, paced by visual density
- **Job (move 6):** Explain the *why* beneath the ablations. **Primary, robust:**
  the active-perception advantage tracks **visual density** — biggest where
  cropping recovers fine structure (engineering_drawing +30, infographics +18.8),
  smallest on text-linear pages (science_paper +4.9). It's perception, not
  reasoning: fix the reasoner, swap **only** the VLM → +~8pp (stats attached);
  capacity gate across a family (sharp at 31B, none at 4B). **Secondary,
  provisional:** many pages can *also* overflow a fixed budget (Unknown-rate
  climbs) — but state plainly that **within DocVQA-val, length is a density
  confound, not the driver** (the "advantage grows with pages" hypothesis is
  false there); the cross-benchmark length result is provisional (n=3 running).
- **Carries:** **F-cat** — RLM−ReAct gap by category (adapt the report's
  `fig-category` xbar; "ranking, not exact value, is the point"). **F5** — VLM-swap
  lift (+7.87pp @9B [Welch t=3.54, CI +3.4/+12.3], +8.60pp @4B [t=4.96, CI
  +5.2/+12.0]; reasoner fixed) + Gemma capacity gate (31B RLM 32.5≫ReAct 18.4 vs
  E4B all ≈ floor). **F6** `[prov]` — cross-benchmark Unknown-rate vs length
  (recursive ~flat ~0% Unk vs raw-VLM 8/22%→36/87%), clearly flagged provisional.
  Light formalism: "perception budget" = visual tokens/page × pages vs usable
  context. Short bridge beat: **pass@k oracle headroom** (`codeact_chat` avg@1 39.5
  → pass@8 63.8 ⇒ ~25pp recoverable) — sets up the close. Honest beat:
  **effort ≠ accuracy** (more turns mark a hard doc, corr ≈ −0.31), can move to
  limitations.

### 07 · `win` — The competition result, told honestly
- **Job (move 7):** Validation + the "you may not need fine-tuning" takeaway. A
  *general off-the-shelf* model + a tiny training-free harness won the tier and
  beat the closed frontier — where the DocVQA SOTA class fine-tunes or builds
  doc-specific pipelines.
- **Carries:** **T2** — test-set results (ours 43.75% vs Gemini 3 Pro 37.5 / GPT-5.2
  35.0 / Flash 33.75 / GPT-5 Mini 22.5). Honesty beat: two number systems kept
  distinct (the 3.6/SC-8/test win vs the 3.5/n=8/val clean matrix); SC-vote noted
  as a competition-only add-on. Contrast bullet: ARIAL fine-tunes Gemma-27B/70k.

### 08 · `limitations` — The cost of generality: it's slow
- **Job:** Honest trade-off. The training-free, minimal-tool, general-model recipe
  buys generality at the price of **efficiency**: perception is many sequential
  VLM calls (~13 iters/Q), so latency/cost are high; heavy docs can even exceed
  the model's context budget; SC-vote (competition) multiplies cost ~8×.
- **Beats:** (1) the symptom + our numbers; (2) the external cautionary point —
  MADQA: RLM is flexible but incurs *catastrophic effort overhead* (~270M tokens /
  ~$850, losing to a BM25 agent); (3) levers we did **not** pull but that are
  available — high-quality OCR preprocessing + a searchable index to cut VLM
  calls, or a **smaller/faster domain-specific VLM** for the sub-call (possible
  because agent ≠ VLM); (4) reframe: the OCR extension's real payoff is likely
  **efficiency, not accuracy** (it bought ≈0 accuracy on moderate docs). We
  optimized for accuracy/generality, not throughput — naming the open
  efficiency work, not claiming it.
- **Carries:** **T3** — efficiency table (iters/Q per solver: `rvlm` ~13, `react`
  5.1, `direct_vlm` 30.4 @ cap). One external number from MADQA (cite).

### 09 · `close` — Code as a symbolic substrate; the training-time question
- **Job (move + the post's final turn):** Zoom out. The REPL is a *symbolic
  substrate* for a neural model — code as the medium it explores/reasons in.
  That it helps at *test time* is now broadly shown. The thread to end on (the
  *fundamental* version, kept narrow so it's honestly open): could exercising this
  substrate during **learning** yield a *better base model* — transferable
  capability beyond any one harness — not just better deployment inside the
  scaffold? Posed as a direction worth pulling, not a claim. Two concrete hooks:
  the **append-only form is already the trainable target** (from the `ablations`
  aside; FoldAct etc. = early steps) and the **pass@k headroom** (large recoverable
  gap → a learning signal exists).
- **Carries:** no new data — synthesis + the open question. Honest "the idea isn't
  novel in isolation; the *learning-time / better-base-model* angle is the part
  worth flagging." **No reference to any fine-tuning / companion training work.**

---

## Coverage check (every throughline element lands somewhere; nothing orphaned)
- Payload (REPL + 1 perception call beats raw/ReAct) → `ablations` (F3/F4/T1).
- Both-halves-jointly-load-bearing → `ablations` beat 3 (F4).
- Sub-call generality irrelevant → beat 4. Observability irrelevant → beat 5.
- Perception-budget mechanism, **density-led** → `mechanism` (F-cat/F5; F6 prov).
- Training-free / off-the-shelf / minimal-tool strength → `win` + reader-change goals.
- No-novelty / lineage → `harness` (woven beat) + `intro`.
- Competition win (credibility) → `intro` (hook ref) + `win` (T2).
- Closing symbolic-substrate / better-base-model question → `close`.
- Efficiency cost + MADQA overhead + OCR-as-efficiency reframe → `limitations`.
- Audience scaffolding (RLM/CodeAct/DocVQA unknown) → `task` + `harness`.

## Consolidated asset list (★ = adapt from companion report, no mention of it)
- **F1** ★ NVIDIA chart page (`br1_page76` full+crop) — whole-page-misreads /
  crop-recovers motivator. Public disclosure, safe to reuse.
- **F2** ★ harness architecture — two-panel (active-perception vs ReAct), from
  `fig-architecture`.
- **F3** three-tier bar chart, val n=8, error bars — headline (generate from runs).
- **F4** 2×2 REPL×delegation — "both halves" visual (generate; within one batch).
- **F-cat** ★ RLM−ReAct gap by category (visual-density), from `fig-category`.
- **F5** perception-budget: VLM-swap lift (+8pp, stats) + Gemma capacity gate.
- **F6** `[prov]` cross-benchmark Unknown-rate vs length (recursive vs baseline).
- **T1** full solver matrix (role, val n=8 mean±std).
- **T2** current-code VAL vs official val baselines (Gemini 3 Pro 37.5 / Flash
  33.75), with subset caveat. (NOT the old competition test table.)
- **T3** efficiency table (iters/Q per solver).
- **C1** ★ NVIDIA trajectory excerpt (prose + fenced Python), from report §3.5.
- Light equations: (PO)MDP framing of the loop (`harness`); perception-budget
  definition (`mechanism`).
- Repo / frozen-branch **links** only — in `tldr` and `win`.

## Citations to reuse (verified arXiv IDs, from report `references.bib`)
`rlm` 2512.24601 · `codeact` 2402.01030 · `react` 2210.03629 · `rvlm` 2603.24224 ·
`madqa` 2603.12180 · `docvqa` 2007.00398 · `stvqa`(ANLS) 1905.13648 · `foldact`
2512.22733 (Shao et al., *Efficient and Stable Context Folding for Long-Horizon
Search Agents* — cited in `ablations` training aside + `close`). Add manually if
kept: `deepeyes` 2505.14362 (not in report bib). Optional close-cluster: `resum`
2509.13313, `contextcurator` 2604.11462. VisProg/ViperGPT/Chameleon = code-as-vision
lineage.

## Data-integrity notes for production (do not skip)
- **Pick ONE number-batch and stay in it.** `rvlm` is 41.88 (re-run, `CLAUDE.md`
  headline) vs 39.38 (original batch, report-canonical); `react` 27.19 vs 25.16;
  `raw_vlm` 20.94 vs 20.47; `rlm_ocr` 14.69 vs 13.91. `codeact_chat` 39.53 and
  `direct_vlm` 22.34 match across both. The **F4 2×2 must be within one batch** or
  it silently compares re-run vs original. Decide at production from actual run
  data; footnote the choice.
- **Observability tie:** lean only on the **27B and 4B** RLM≈CodeAct cells (other
  CodeAct cells are the old impl, pending re-run). Don't cite an RLM>CodeAct gap
  (the +4.4pp in old mining is an artifact the corrected solver erases).
- **Doc-length:** "advantage grows with pages" is **false within DocVQA-val**;
  only the cross-benchmark (MP-DocVQA vs MMLongBench) length result stands, and it
  is provisional (n=3 running). Lead density, flag length.
- **Number policy (user directive):** current-code numbers are canonical
  EVERYWHERE; the competition submission is a one-sentence historical note, not a
  headline. We changed agents/prompts since (shrinking the val↔test gap), so the
  old test 43.75% is unrepresentative — do not lead with it. The win is carried by
  the *fact* of the joint tier win + current-code val numbers.
- **Frontier comparison is val-vs-val with a subset caveat:** our current method
  ~42% on the 25-doc/80-Q subset vs official **val** Gemini 3 Pro 37.5 / Flash
  33.75 (GPT-5.2 is test-only — exclude). Subset ≠ full val → say "in the range
  of / not behind," not a strict beat, unless a full-val current-code run is done
  (optional production task).

## Gate decisions (locked)
- `ablations` = ONE section (ladder momentum kept).
- No `reproduce` section — repo links in `tldr` + `win`.
