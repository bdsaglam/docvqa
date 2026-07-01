# Sketch — DocVQA active-perception blog post

**Status:** awaiting gate (level 3). Telegraphic bullets per section: claim →
concrete artifact/number → transition. Wording disposable; content decided.
Terminology locked: **active perception / depth-1 VLM-tool call** (avoid
"recursive"/"delegation"; "ReAct" = named baseline). Numbers = project headline
set unless flagged; F4 must be within-batch (see outline data notes).

---

## 01 · `tldr` (box, top; DRAFT LAST)
- One off-the-shelf open 27B + a tiny code harness → **joint winner, ICDAR 2026
  DocVQA 8–35B tier**, landing with the closed frontier (Gemini 3 Pro / Flash).
  (All numbers below are current code; the challenge submission is a short note.)
- We took the harness apart. **Two parts carry it, jointly:** a Python **REPL**
  and a single **active-perception call** (the model asks a VLM to look at a
  region it chose). Drop either → collapse.
- Both confirmed by knockout: drop the perception call and let the model load
  pixels into its own context instead → it collapses (context-rot, as the RLM
  line predicts — clean delegation, not raw pixels).
- The useful corrective — **the load-bearing core is smaller than what people
  build.** Don't-bother list: making the perception call a general agent, the
  trajectory format (append-only vs compacted), bolting on OCR — all ≈0 here.
- It's a **perception-budget** problem (paced by visual density), not reasoning —
  and it's **training-free**: reach for harness design before fine-tuning.
- No new method (RLM + CodeAct + code-as-action, reused). Repo: [link].

## 02 · `intro` — hook + nut graf
- Hook: **joint winners of the ICDAR 2026 DocVQA 8–35B tier** — an **open 27B**
  landing with the **closed frontier** (Gemini 3 Pro / Flash) on a hard document
  benchmark. The parts are all **borrowed** — a REPL, a VLM, a tool call. So the
  interesting question isn't "what new trick."
- The real question: **which borrowed parts actually carry the win, and which are
  along for the ride?** The payoff is a useful corrective — the minimal core is
  smaller than what most builders assemble (no general sub-agent, no clever
  trajectory format, no OCR pipeline needed).
- Nut graf (why you, now): you've stacked REPL + tools + big context + sub-agents
  and hoped. Here's a controlled ablation of which knobs matter on a real task —
  so you stop paying for the ones that don't. (And you might not need fine-tuning.)
- Honesty up front: **no new method.** A clean question, a clean answer, stated
  with the caveats.
- Transition → first, the task: why documents break VLMs.

## 03 · `task` — what DocVQA is, why it's hard for VLMs
- DocVQA in one line: answer a natural-language question about a document
  (could be 1–280+ pages: tables, figures, engineering drawings, maps, slides,
  infographics, comics). [adapt report's "finding before reading" framing:
  *locating the right page and region comes before any reading*.]
- Why a single raw multi-image VLM pass struggles: fixed page budget + dense
  fine-grained content; tiny labels in a busy chart get misread.
- **F1 (NVIDIA chart page):** a whole-page read of this dense bar+line chart
  returns the wrong number ($978.42); the answer (NVIDIA TSR **$2,287.07** vs
  Nasdaq-100 **$238.19**) is one of ~20 tiny overlapping labels. Crop → correct.
- Claim seeded: **perception, not reasoning, is the hard part here.** (Pay off in
  mechanism.)
- Transition → so what if the model could *choose where to look*?

## 04 · `harness` — the recipe + where it comes from
- Recipe in one sentence [adapt report]: give a code-capable reasoner a persistent
  **Python REPL** and one **active-perception primitive** — an on-demand call to a
  (frozen) VLM against an arbitrary image region — and it can **direct** perception
  instead of consuming it whole: crop to the evidence, zoom for acuity, composite
  regions, and do the coordinate/numeric arithmetic in code that the VLM can't.
- **F2 (two-panel architecture):** A = active perception (Reasoner→REPL→VLM→pages,
  edges "writes code"/"look: crop-zoom"/dashed "obs"); B = ReAct (Reasoner→VLM→
  pages, "whole page, perception fixed at page granularity"). Caption: *the REPL is
  the sole structural difference, and it's what converts reasoning into targeted
  perception.*
- **C1 (NVIDIA trajectory, prose + fenced Python):** survey pages → locate the
  chart → read (VLM returns a misread $978.42) → **distrust & crop-verify** →
  adjudicate → compute `2287.07 − 238.19 ⇒ 2048.88` ✓ submit. Shows all moves in
  one trace; the crop-verify catches a wrong number a single read submits.
- Light formalism: frame the loop as **state / action (code) / observation
  (REPL+VLM output)** — sets up the observability point later.
- **Where this sits (woven, short, honest):** the scaffold is **RLM** (prompt-as-
  REPL + a sub-call; Zhang et al. 2025) crossed with **CodeAct** (code-as-action)
  and the code-as-vision lineage (VisProg/ViperGPT/Chameleon). **No new method.**
  RLM already showed *for text* that the REPL lifts the baseline and the sub-call
  adds more → our question: does it hold when the call is **visual**, and which
  parts carry it?
- **Thinking-off defense (1–2 sentences):** our runs use `enable_thinking=false`;
  disabling the native thinking channel **relocates** reasoning into the turn body,
  it doesn't remove it — thinking-off ≠ answering without reasoning.
- Transition → to find out which parts carry it, knock them out one at a time.

## 05 · `ablations` — what actually carries the lift (THE CORE)
- Setup: same model (Qwen 3.5 27B, val 25-doc/80-Q, n=8, mean±std); change one
  structural thing at a time. **F3** (three-tier bar, error bars) anchors the
  section; **T1** (full matrix) for reference.
- **Ladder:**
  1. Full harness (REPL + active perception): **~42%** (top tier).
  2. **Remove the REPL** → ReAct with the *same* VLM tools: **~27%** (−15pp). Code
     REPL is load-bearing: without it the agent can't crop/compose/compute across
     steps; it terminates shallow (~5 steps/Q).
  3. **Remove the active-perception call** → REPL agent that loads page pixels into
     its *own* context (`direct_vlm`): **~22%** (−17–20pp). **Confirms the
     context-rot mechanism** (RLM line): a focused VLM call returning compact text
     keeps the reasoner's context clean; raw pixels pollute it. Expected, not
     shocking — but it's what isolates *clean delegation* (not just "has a REPL")
     as load-bearing. It also grinds: 30+ steps/Q, pins the cap, never converges.
  4. **Neither** (raw multi-image, no scaffold): **~21%**.
  - ⇒ **F4 (2×2 REPL × active-perception):** ~42 / 27 / 22 / 21. **Both halves
     jointly load-bearing; neither alone suffices.** [within-batch numbers]
- **Now the don't-bothers** (each a violated expectation, paid off):
  5. **Generality of the call doesn't matter:** swap the focused perception call
     for a general subagent (any subtask, image optional) → **≈42%, within std**;
     the agent uses it as plain perception **~99%** of the time. A single focused
     call suffices. (Depth-1 throughout; no claim about deeper recursion.)
  6. **Trajectory observability doesn't matter (for inference):** append-only
     fully-observable **CodeAct ties** the compacted REPL-history form within ~2pp
     (**~40 vs ~42**, solid at 27B & 4B). And append-only doesn't even hurt on long
     docs: page-count × (RLM−CodeAct) corr ≈ 0. The format isn't the inference lever.
     - **But it matters for *training* (forward-looking aside; borrow report §4):**
       RLM's context **compaction rewrites the trajectory between turns**, so the
       observation stream isn't a clean growing prefix — which is the structure
       policy-gradient RL and per-token distillation assume. CodeAct's append-only
       transcript preserves it, so it's the more **trainable** target — and
       choosing it costs ~0 accuracy. Loose analogy: append-only is more **MDP-like**
       for the learner, compaction more **POMDP-like** — *with the honest caveat*
       that neither is a true MDP, since the document is never fully observed (the
       agent only sees what it chose to look at). Cite **FoldAct** (Shao et al.,
       *Efficient and Stable Context Folding for Long-Horizon Search Agents*,
       arXiv:2512.22733) as an early attempt to make compacted/folded trajectories
       trainable. (Optional related: ReSum 2509.13313, ContextCurator 2604.11462.)
       This seeds the close. **No mention of any of our own training work.**
  7. **OCR-free is fine:** add OCR text + BM25 search on top → **≈37%, ≈0 gain** on
     moderate docs. (Foreshadow/callback: our *competition* entry actually carried
     exactly this OCR+search machinery — these ablations are why we know it wasn't
     load-bearing; full payoff in the win section.) And swapping vision *for* OCR
     text (no vision) is the **true floor, ~15%** — below even the no-scaffold
     competition prompt (~19%). Active *visual* perception does work OCR can't
     replace (engineering_drawing & maps: 0/10 in all 8 trials without vision).
- One-line synthesis: **the load-bearing core is REPL + one active-perception
  call; generality, observability, and OCR-on-top are not.**
- Transition → *why* does directing a VLM beat feeding it everything?

## 06 · `mechanism` — why: perception budget, paced by visual density
- Thesis: documents are **perception-budget-bound**, not reasoning-bound. A fixed
  whole-page read spends the budget badly; active perception **rations** it —
  crop where the detail is.
- **Primary, robust — visual density. F-cat:** the active-perception advantage
  (RLM − ReAct, by category) tracks how much fine structure a page packs:
  engineering_drawing **+30**, maps/infographics **+18.8**, comics +13.8, …,
  science_paper **+4.9** (text-linear). *Ranking, not exact value, is the point.*
  Biggest where cropping recovers fine structure; near-zero where a whole-page read
  already suffices.
- **Perception, not reasoning. F5:** fix the reasoner, swap **only** the VLM →27B:
  **+7.87pp @9B** (Welch t=3.54, CI [+3.4,+12.3]), **+8.60pp @4B** (t=4.96, CI
  [+5.2,+12.0]). Same reasoner, better perception → big lift = a perception
  bottleneck. [adapt report: *a stronger reasoner produces better-targeted
  perception queries and extracts more even from a weaker VLM; ReAct has no such
  actuator — its ceiling is the VLM's whole-page acuity.*]
- **Capacity gate (same F5 / companion bars):** RLM ≥ ReAct at every Qwen3.5 scale
  (27B 39.4≫25.2; 9B 24.5>21.0; 4B 21.1>15.7); Gemma-4-31B RLM **32.5 ≫ ReAct
  18.4** (+14pp, mirrors Qwen); but **Gemma-E4B**: all harnesses 6–8 ≈ baseline —
  too weak to drive the REPL. The lift is a **capacity gate**: real once the model
  can code, absent below it. (Also: older Qwen3-8B *flips* — weak coder shortcuts
  the loop.)
- **Secondary, provisional — length. F6 [prov]:** across benchmarks, many pages can
  *also* overflow the budget: recursive methods stay flat (~0% "Unknown"), raw-VLM
  Unknown-rate climbs 8/22% → 36/87% as evidence falls off the page budget. **But
  state plainly:** *within* DocVQA-val, length is a **density confound**, not the
  driver ("advantage grows with pages" is false there). n=3 running.
- Formalism (light): perception budget ≈ (visual tokens/page × pages) vs usable
  context; active perception keeps the *useful* fraction high.
- Bridge to close — **oracle headroom:** pass@8 ≈ **64%** vs avg@1 ≈ 40%
  (`codeact_chat`): the right answer is reachable in *some* trial ~25pp more often
  than it's landed → a strong learning/verification signal exists.
- Transition → so it works, and we know why. Did it actually win?

## 07 · `win` — the competition result, honestly
- The project started as a competition entry, and the headline fact is simple:
  **joint winner of the ICDAR 2026 DocVQA 8–35B tier** — a general open 27B.
- **Numbers are current-code (latest), not competition-time.** Frontier context on
  **val** (**T2**): our current OCR-free method ~**42%** vs official **Gemini 3 Pro
  37.5 / Flash 33.75**. ⚠ Honest caveat stated inline: our ~42% is the 25-doc/80-Q
  ablation subset, the baselines are full-set → frame as **"in the range of / not
  behind the closed frontier,"** not a hard beat. (GPT-5.2 is test-only — omit from
  the val comparison.)
- **Short historical note (all the post says about the submission):** the challenge
  entry was a now-superseded solver — the REPL + active-perception agent **with OCR
  text + BM25 search**, SC-8 voting, Qwen 3.6 — scored on the hidden test set. We
  changed agents and prompts substantially since (partly to shrink the val↔test
  gap), so we report current code throughout. One sentence; move on.
- **The tools we shed (ties back to the ablations):** that entry carried **more**
  than the minimal recipe (the OCR + search). The ablations later showed those
  extras were **not load-bearing** — OCR-free matches OCR+search on these docs.
  We won with the extra machinery, then learned the minimal harness suffices:
  concrete payoff of the "OCR-on-top ≈ 0" finding.
- **The real takeaway isn't the leaderboard — it's *how*:** a **general,
  off-the-shelf** model + a **tiny, training-free** harness (no domain tooling, no
  fine-tuning) gets there, where the DocVQA SOTA class **fine-tunes** (e.g. ARIAL
  fine-tunes Gemma-27B on 70k pairs) or builds specialized OCR/encoder pipelines.
- Takeaway line: **reach for harness design before fine-tuning.**
- Transition → the cost of that generality.

## 08 · `limitations` — the cost of generality: it's slow
- Honest trade-off: training-free generality is bought with **efficiency**.
  Active perception = many **sequential VLM calls** (~13 steps/Q; **T3**: rvlm ~13,
  react 5.1, direct_vlm 30.4 @ cap). Latency/cost are high; heavy docs can exceed
  the model's context budget; the competition SC-vote multiplies cost ~8×.
- External cautionary point — **MADQA:** RLM is flexible but incurs *catastrophic
  effort overhead* (an unconstrained RLM burned ~270M input tokens / ~$850 and
  **lost** to a cheaper BM25 agent). Our experience rhymes.
- **Effort ≠ accuracy** [adapt report]: extra turns mark a **hard** document, not a
  path to the answer (corr ≈ −0.31); the lever is the *quality* of the perception
  loop, not its length.
- Levers we did **not** pull (all available, future work, not claimed):
  high-quality **OCR preprocessing + a searchable index** to cut VLM calls; a
  **smaller/faster domain-specific VLM** for the perception call (possible because
  the reasoner and the VLM need not be the same model).
- **Reframe of the OCR null:** OCR bought ≈0 *accuracy* on moderate docs — but its
  real payoff is probably **efficiency**, not accuracy (fewer, cheaper looks).
- Other honest hedges: val-only ablations; some CodeAct cells provisional (lean on
  27B/4B); doc-length is a confound, not yet a clean cross-benchmark axis (n=3
  pending).
- Transition → step back: what is the REPL really doing for the model?

## 09 · `close` — code as a symbolic substrate; the question to end on
- Reframe: the REPL is a **symbolic substrate** for a neural model — code is the
  medium it explores, composes, and reasons in, over content it can't hold whole.
  Active perception is one instance: the model writes code to *aim* its eyes.
- What's settled: at **test time**, a code substrate clearly helps (here, and
  across the RLM/CodeAct/code-as-action line). Not novel — broadly shown.
- The thread to pull (kept narrow so it's honestly open): could exercising this
  substrate during **learning** produce a **better base model** — transferable
  capability beyond any single harness — rather than only better deployment inside
  a scaffold? Two things make it concrete: (a) the **append-only form is already
  the trainable target** (from the ablations aside; FoldAct etc. are early steps on
  the trajectory-stability problem), and (b) the **oracle-headroom** gap (pass@8 ≈
  64% vs avg@1 ≈ 40%) says a strong learning signal is sitting there unused.
- Tone: posed as a direction worth pulling, not a claim; "the idea isn't novel in
  isolation — the *learning-time, better-base-model* angle is the part I keep
  coming back to." End. (No mention of any fine-tuning/companion work.)

---

## Cut/flag check (blog module: a section with no concrete artifact is suspect)
- Every section carries a figure/table/number/trace except `close` (deliberate —
  it's the synthesis/vision turn) and `intro`/`tldr` (framing). OK.
- `04 harness` carries the most (F2 + C1 + lineage + thinking-off) — watch length;
  if heavy, the thinking-off defense can drop to a footnote.
- `06 mechanism` carries 3 figures (F-cat/F5/F6) — make sure each earns its place;
  F6 is provisional and could be a single sentence + small inset if space is tight.