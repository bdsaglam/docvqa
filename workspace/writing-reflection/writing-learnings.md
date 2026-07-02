# Technical-writing learnings

Reusable principles for technical posts and papers, distilled from polishing the
*Perceive-Reason-Code* blog post. Written as a checklist with concrete before/after
examples, not a diary. The organizing insight is at the bottom (§11): most defects
survive editing because each pass has only one lens.

## 1. Hard rules (non-negotiable)

- **No em-dashes (—).** The single strongest tell of machine-written prose. Use
  colons, periods, parentheses, or semicolons. Grep for `—` before every commit.
- **Artifacts are syntheses, not changelogs.** Write as if you only ever knew the
  final decisions. No process or chronology leak: no "previously/now/updated", no
  "X instead of Y" against a Y the reader never saw, no narrating the order in
  which you discovered or corrected things. Fixing a wrong-order implication by
  narrating the correct order is the same mistake.
- **Scope claims; never universalize.** "Our OCR pipeline (docling + granite-vision)
  did not lift accuracy on these documents" — not "OCR is useless." Name what you
  tested and concede a stronger version might differ.
- **State a headline achievement once.** Mention the win once, up front, with
  details; never re-litigate it. Repetition reads as insecurity.
- **Label which condition every number is on** (validation vs test, n, metric). A
  caption that says "ANLS" with no split invites doubt.

## 2. Voice and register (the class that survives every other pass)

This is tone, so structure/citation/caption passes sail past it. Hunt it
separately, reading as a discerning author: *would a serious paper write this?*

- **Meta-positioning / reassurance-seeking** — cut it. "We're not the first to hit
  this." No serious post says this; just cite the prior work.
- **Throat-clearing openers** — cut, lead with substance. "It helps to be clear
  about what the extra steps buy, though." → open with the finding. "The answer is
  useful in a specific way:" → delete, state the answer. "The reason is simple." →
  cut; the next sentence carries it.
- **Empty framing / filler evaluatives** — "a nice anchor", "worth keeping/doing",
  "the point is", "the real story", "the more useful half".
- **Filler transitions and hedges** — "of course", "that said", "to be fair",
  "interestingly", "it turns out", "in a way", "arguably", "somewhat".
- **Repetitive verbal tics** — any construction used often enough to become a tell:
  "worth ___" (fired 4×), "the obvious ___" (4×), overused "just" / "simply". Vary
  or cut two of three.

## 3. Structure and headings

- **Claim-as-heading.** Findings sections *assert* their conclusion ("The advantage
  grows with document length"); setup/method sections keep plain labels ("The
  recipe", "The result", "Ablations"). The table of contents then reads as a
  skimmable argument.
- **Result-first.** Front-load the landscape/overview table, then walk the
  mechanism. Skimmers get the answer; close readers get the derivation.
- **Do not promise completeness you do not deliver.** "Here is the whole landscape.
  Each row is one configuration." is a trap if a configuration is missing. Say "the
  main configurations."
- **Watch heading/content mismatch.** A "this DOES matter" result nested under
  "Three things that turn out not to matter" confuses a heading-skimmer. Give it
  its own heading.

## 4. Captions: describe, do not interpret

- Captions state **what is plotted** and the conditions (metric, n, axes, what the
  colors mean). Interpretation and analysis go in the body and the claim-heading.
- With claim-headings carrying the takeaway, an interpretive caption is redundant.
  "Every configuration sorts into three *clean* tiers... every gap is much larger
  than the per-cell spread" → move to the body; the caption just describes.
- "Describe" is not "bare label" — the caption must still be self-sufficient.
- If you strip analysis from a caption, confirm the body carries it (or add it).

## 5. Analogy (make the abstract tangible)

- **Embodied analogy before the mechanism, then bridge back explicitly.** "A person
  answering a question from a long report flips to the section, leans in on the one
  figure, re-reads a caption... Active perception gives a model the same habit."
- Ground it in real behavior, and cover the axes that matter (across-page: a long
  report; within-page: navigating a large dense map region by region).
- A forced analogy is worse than none.

## 6. Landing sentences

- Short declarative resets after dense passages ("The crop caught it.").
- **Rate-limit them** so each still lands; overuse dilutes.
- Terminate with a period, never a dash.

## 7. Numbers and claims

- **Precise, unrounded numbers with relative framing** ("+16.6pp", "one-tenth the
  cost"), and pair a result with its cost.
- **One clean limitation as a full declarative sentence**, not scattered
  "may/possibly" hedges: "The data does not separate sharper aiming from sharper
  reasoning; that decomposition stays open."
- **Match claim strength to evidence.** Ration hedging to genuinely uncertain
  points; be plainly confident where the data is clear.
- **Verify numbers against the source of truth.** A "(roughly 13 to 42)" range was
  impossible given the reported stds (the metric never dropped that low); it had
  gone stale. Recompute before publishing.
- **State an ablation as held-fixed vs changed, precisely.** Before framing a
  knockout, list what it holds fixed and what it varies; the interesting claim is
  usually "everything fixed except X." Two failure modes: a *false contrast*
  (claiming the baseline lacks something the full method lacks too — "the OCR agent
  can't see pixels," but the full method's reasoner never sees pixels either; only
  the *origin* of its text differs), and a *forced ladder* (lining up configs as one
  monotone axis when the rungs vary different things at once). When a comparison is
  confounded (weaker perceiver AND passive), look for another cell in the study that
  unconfounds it (an actively driven weak perceiver) instead of hedging.
- **Reconcile apparent contradictions at the point of tension.** "Perception-bound,
  not reasoning" vs "the reasoner is the bigger lever" reads as a contradiction
  until you say the reasoner's leverage is *unlocked by the perception loop*.
  Distinguish mechanisms a single word conflates (a "gap" driven by page *length*
  vs by page *density*).

## 8. Citations

- Cite every method and dataset at **first mention**, inline. Consistent style:
  `(Name et al., year)` for 3+ authors, `(Name & Name, year)` for two.
- **Verify each reference against arXiv / Semantic Scholar**: confirm the paper
  exists and that the arXiv ID resolves to the claimed title and first author.
  Fabrication is the expensive mistake; papers past your knowledge cutoff need a
  live lookup, not memory. (A first-author initial was wrong until checked.)
- No orphan references: every list entry is cited in the body, and every cited work
  is in the list.

## 9. Abbreviations

- Define once at first use, then use the short form. Put the canonical definition
  where it is guaranteed to be read (the body), not only in a collapsed TL;DR that
  many skip.
- Do not let one label mean two things. If two systems both have a property
  ("active perception"), do not name one of them after that property — it is
  ambiguous. Name them by what actually distinguishes them.

## 10. Figures as reproducible artifacts

- Keep every figure's generator in one script, called from a single entry point.
  An orphaned figure (image with no generator) cannot be corrected or re-themed
  later; we lost one and had to reconstruct it from the reported means and stds.

## 11. Process: why issues keep surfacing, and how to stop it

- **Each pass has one lens** (structure, citations, captions, seams, register). A
  defect outside the current lens survives every pass until you aim one at it. This
  is why "we did several passes but I keep finding issues" happens.
- **Heavy piecemeal editing introduces new defects** (filler, broken seams) that
  were not in the original draft. Budget a consolidation read after big surgery.
- **Run a dedicated pass per defect class**, then a **final cold read** as a fresh
  skeptical reader — that is what catches seams: a moved section that no longer
  flows, a claim set up and never paid off, a contradiction like "modest edge"
  against a 21pp headline.
- **When delegating a pass to a sub-agent, give a precise single-class mandate and
  steer hard against false positives.** A short high-signal list beats an
  over-flagged dump; author trust erodes fast on false positives.
