# Throughline — DocVQA REPL-agent blog post

**Status:** awaiting gate (level 1). Arc chosen: **Arc 1 — decomposition / surprise-led.**

## (a) Payload — the one sentence
On document VQA, a code-capable model given a **REPL and a single active-perception
call** (a depth-1 call to a VLM used as a perception tool) beats both raw
multi-image VLMs and tool-only ReAct agents; clean ablations show the lift comes
**jointly from the REPL (a code/symbolic substrate) and the active-perception
call**, while **the perception call's *generality* and the trajectory's
*observability* don't matter** — and underneath, the bottleneck is **perception
budget, paced by visual density, not reasoning**.

> **TERMINOLOGY (governs the whole draft).** Primary term: **active perception** —
> the reasoner directs a **depth-1 VLM-tool call** (crop/zoom/look) rather than
> consuming pixels whole. **Avoid "recursive"/"recursion" and "delegation" as the
> framing** (overlaps the RVLM paper; our depth never varied). Mention recursion
> at most once, only to set it aside (it's a depth-1 call, recursive merely in the
> degenerate case where agent and VLM are the same model). "ReAct" stays as the
> named baseline.

## (a′) Honest positioning — what's known vs. what this post adds
**This post claims no method novelty, and says so plainly.** The components are
all established: RLM (Zhang et al., 2025) introduced the prompt-as-REPL +
recursive sub-call paradigm and *already showed, for text,* that the REPL alone
lifts the baseline and the sub-call adds more; CodeAct / VisProg / ViperGPT /
Chameleon established code-as-action for (vision) agents; RVLM and DeepEyes cover
recursive / think-with-images vision models; MADQA covers document-collection
agents (and critiques RLM's effort overhead). We borrow all of it.

**What the post offers a reader is therefore not a new method but:**
1. A **controlled decomposition in the *visual document* setting** — does RLM's
   text-only "both halves matter" still hold when the sub-call is a VLM? We run
   the ablations and answer it.
2. **One genuinely counterintuitive *visual* result**: delegating perception to a
   focused VLM sub-call beats the *same model* holding the page pixels in its own
   context (`direct_vlm`). Not present in text-only RLM.
3. A **practitioner diagnostic** — perception-budget-bound, read off the
   Unknown-rate vs. page-count curve.
4. A **reproducible, competition-validated recipe** that is **training-free and
   minimal-tool** — a *general, off-the-shelf* LLM/VLM with a small harness (a
   REPL + one VLM call, no domain-specific tooling, no fine-tuning) reaches the
   tier-winning result. Direct contrast with the DocVQA SOTA class that fine-tunes
   (e.g. ARIAL fine-tunes Gemma-27B on 70k pairs) or uses specialized document
   encoders/OCR pipelines. The actionable message: **reach for harness design
   before fine-tuning.**
5. An **honest synthesis** that ends on a real open question (training-time
   symbolic substrate).

"Novel for *this* reader" (the blog-module test): the audience isn't assumed to
know RLM, CodeAct, or DocVQA — so the synthesis and the findings are new *to
them*, and the post must earn its keep on clarity + usefulness + honesty, not on
a novelty claim. The post wears the lineage openly (a short "where this sits"
move), which builds trust rather than competing for credit it doesn't have.

## (b) Audience: current state → intended change

**Who:** ML/AI practitioners and researchers fluent in LLMs, VLMs, and agents
(ReAct, tool-calling, RAG, context windows), but **not** assumed to know RLM,
CodeAct, or the DocVQA task.

**What they currently believe (the starting beliefs the arc must move):**
- "Agentic scaffolds help" — but vaguely; they can't say *which* component does.
- When a multimodal agent fails on big/dense inputs → reach for a **bigger or
  smarter model**, or **stuff more into the context window**.
- The "fancy" parts (recursion, subagents, large context, clever trajectory
  management) are assumed to be where the magic is.

**The change the piece should cause** — by the end the reader can:
1. Explain *why* REPL + delegated perception beats raw VLM and ReAct on documents.
2. Recognize multimodal document QA as **perception-budget-bound**, paced by
   **visual density**, and name the diagnostic (per-category gap; Unknown-rate).
3. Identify the **minimal load-bearing core** (REPL + one VLM perception call) and
   what's **dispensable** (sub-call generality, trajectory observability, OCR-on-top).
4. Reproduce the recipe — it's an architecture choice, not training: a general
   off-the-shelf model + a small harness, no fine-tuning or domain tooling. Know
   to reach for harness design before fine-tuning.
5. Carry away the open question: code as a **symbolic substrate** for neural
   models, and whether it could help at **training time**, not just inference.

## (c) The arc — entry angle, ordered moves, why it lands

**Entry angle (hook):** Honest origin order — **competition first, understanding
after.** We entered the ICDAR 2026 DocVQA challenge, built an agent (off-the-shelf
27B + a thin code harness) to compete, and it was a **joint winner of the 8–35B
tier**, landing with the closed frontier. *Then* we took our own entry apart,
because the harness is all **borrowed** parts (a REPL, a VLM, a tool call) and we
didn't know which one was carrying the win. So the question driving the post is
**which of those borrowed parts is actually load-bearing, and which are along for
the ride.** *(Opens a specific information gap; the payoff is that the minimal core
is smaller than what most builders assemble — generality, trajectory format, and
OCR all turn out dispensable.)* Honesty is part of the hook: this is a
retrospective teardown of a winning entry, no new method, findings consistent with
the RLM/CodeAct literature, all numbers current-code.

**Nut graf (why-you-should-care, right after the hook):** If you build LLM
agents, you've stacked a REPL, tools, recursion/subagents, and a big context
window — and hoped. This is a clean decomposition of which of those knobs
actually matter, measured on a real task, so you stop paying for the ones that
don't.

**Ordered moves to the payload** (each crosses one inferential step from the
reader's state; each re-opens and pays off a question — the engine of the post):

1. **The task.** What document VQA is and why it's hard for VLMs: many pages,
   dense fine-grained content (tables, figures, engineering drawings, maps).
   Just enough for an outsider.
2. **The harness — the recipe, and where it comes from.** A code-capable model in
   a REPL whose *perception* is a delegated VLM call (look at page/crop, get text
   back). Scaffold RLM (prompt-as-REPL + recursive sub-call) and CodeAct
   (code-as-action) in a few sentences — name them as the lineages we stand on,
   and state plainly: **no new method here.** Note RLM already showed, for *text*,
   that the REPL lifts the baseline and the sub-call adds more — which sets up the
   honest question driving the post: does that hold when the sub-call is *visual*,
   and which parts actually carry it? (This is the "where this sits" beat — kept
   short; it earns trust and frames the contribution as a careful study, not a
   claim.)
3. **The ablations (the core).** Knock out one piece at a time:
   - Remove the **REPL** (ReAct with the same VLM tools) → collapses (~27%).
   - Remove the **active-perception call** (REPL agent that loads page *pixels into
     its own context* instead) → collapses (~22%). This **confirms RLM's
     context-rot mechanism in the visual setting**: a focused VLM call that returns
     compact text keeps the reasoner's context clean; loading raw pixels pollutes
     it. (Frame as expected/confirmatory, *not* a surprise — it validates "don't
     pollute the context / delegation helps.")
   - ⇒ **Both halves are load-bearing; neither alone suffices.**
   - **The sub-call's generality doesn't matter:** generalize the focused VLM
     perception call into an arbitrary subagent (any subtask, image optional) →
     unused (~99% used as plain perception). A single focused perception call
     suffices. (Depth-1 throughout; we never tested deeper recursion, so we make
     no depth claim.)
   - **Observability doesn't matter:** an append-only, fully-observable CodeAct
     trajectory ties the partially-observable RLM/REPL-history form (within ~2pp).
     The trajectory representation / compaction is not the lever.
   - **OCR-free is fine:** swapping vision for OCR text is the floor; adding OCR
     *on top* of vision buys ≈0 on moderate docs (it's an extension for long docs).
4. **The mechanism — why it works.** Perception budget, not reasoning, paced by
   **visual density**: the active-perception advantage tracks how much fine
   structure a page packs (largest on engineering drawings / infographics,
   smallest on text-linear pages — `fig-category`). Fix the reasoner, swap **only**
   the VLM → +~8pp (9B and 4B) — perception, not reasoning; a clean capacity gate
   across a model family (sharp lift at 31B, none at 4B). Secondary, **provisional**
   note: across benchmarks, many pages can also overflow a fixed budget
   (Unknown-rate climbs) — but *within* DocVQA-val, length is a density confound,
   not the driver. Lead with density; flag length as provisional.
5. **The win — validation, told honestly.** The project began as a competition
   entry; an earlier version of this same agent (with self-consistency voting)
   was a joint winner of the ICDAR 2026 8–35B tier, beating the closed frontier
   on test. State the caveat (different model/code from the clean matrix) plainly;
   keep the two number systems separate. The point isn't the leaderboard — it's
   that a **general off-the-shelf model with a tiny, training-free harness** got
   there, where the SOTA class fine-tunes or builds doc-specific pipelines. The
   credibility stamp *and* the "you may not need fine-tuning" takeaway.
6. **The close — the bigger question.** The REPL is a **symbolic substrate** for a
   neural model: code as the medium it explores and reasons in. That this helps
   at **test time** is now broadly shown. The open thread the post ends on: would
   the same substrate help a model **learn** — during training, not just
   inference — beyond RLVR-for-coding-agents? Posed, not answered.

7. **TL;DR** — drafted last; the payload distilled to a few lines for skimmers.

**Why this lands on this audience:** the "which part did the work?" gap is
reopened and paid off at each rung of the ablations (honestly resolved, no
manufactured surprise); the useful corrective is that the **minimal load-bearing
core is smaller than what builders assemble** (generality, trajectory format, and
OCR are dispensable) and that the bottleneck is **perception allocation, not
reasoning** — both actionable. The honest caveats on the win build trust rather
than hype; and it ends on an idea larger than the result, giving a reason to
share it.

## Notes carried from the brief (constraints on the arc)
- Win = origin + credibility, not the hook. No legacy engineering names in prose.
- Two number systems kept distinct (3.6/SC-8/test win vs 3.5/n=8/val matrix).
- "Recursion" = one level only; no deep-recursion overclaim.
- Doc-length axis numbers are provisional (n=3 running); structure now, numbers
  at production.
- Synthesis, not changelog — no v1/v2/prompt-iteration history; reader never saw
  our back-and-forth.
