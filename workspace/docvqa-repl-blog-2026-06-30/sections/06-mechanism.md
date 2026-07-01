## Why it works: a perception budget, not a reasoning one

Here's the knockout that says what kind of problem this is. Take the exact same
REPL agent and swap its eyes for a text channel: instead of an active-perception
call, give it the OCR transcript of every page plus a search tool, and **no vision
at all.** Same scaffold, same reasoner, same loop — only the perception modality
changes.

It falls to **14.7%**, the lowest score in the whole study, below even the
no-scaffold competition prompt. On the categories where the answer lives in the
layout rather than the text — engineering drawings, maps — it scores **zero out of
ten in all eight trials.** For these layout-bound questions, on this model, OCR text
can't stand in for looking. Whatever the scaffold is buying, it is buying
*perception*, and the bottleneck it relieves is a visual one.

That reframes the whole result. Qwen 3.5 27B is not short on reasoning — give it the
right view of the evidence and it answers fine. What it
can't do is *afford* the right view in one shot. A document page can carry far
more fine-grained visual information than a single VLM read can resolve: twenty
overlapping labels on a chart, a value in a merged table cell, a dimension on a
drawing. A whole-page read spends a fixed perception budget badly. Active
perception rations that budget — crop to the evidence, zoom for acuity, read the
small thing at the scale it needs.

If that's the real story, the advantage should be largest where a page packs the
most fine detail. It broadly is. The per-category gap between the active-perception
agent and the ReAct baseline tracks visual density:

![Figure (category): active-perception advantage by document category](figures/f-category.png)
**Figure 4.** The active-perception advantage over ReAct, by document category
(Qwen 3.5 27B, eight trials). Biggest on dense, structured pages where cropping
recovers fine detail — engineering drawings (+36), business reports (+30),
infographics (+19) — and smallest on text-linear pages like science papers (+4) and
slides (+1), where a single read already gets most of the page. (Maps are a hard
case for every configuration, so the *advantage* there is modest even though the
pages are busy.) The ranking is the point, not the exact values.

### It's perception, not reasoning

The cleanest test holds the reasoner fixed and changes only the eyes. Take a
smaller model as the reasoner and feed its perception calls to progressively
better VLMs, ending at the 27B. The reasoner never changes; only the quality of
what it can see does. Accuracy climbs about **eight points** at both sizes we
tried — +7.9 with a 9B reasoner, +8.6 with a 4B one, both well outside the
noise.[^stats] Same brain, better eyes, large lift: that's the signature of a
perception bottleneck, not a reasoning one. (And it cuts the other way too: a
stronger reasoner writes better-targeted perception queries and gets more out of
even a weak VLM. ReAct has no such actuator — its ceiling is whatever a whole-page
read resolves, and a smarter reasoner can't aim it.)

![Figure 5: fixing the reasoner and improving only the VLM lifts accuracy ~8pp](figures/f5-vlm-swap.png)
**Figure 5.** Hold the reasoner fixed and swap in a stronger (27B) perception
backend, and accuracy jumps about eight points at both reasoner sizes. Same brain,
better eyes — the signature of a perception bottleneck.

The lift is a **capacity gate**, though, not a free lunch — the model has to be a
good enough coder to drive the REPL at all. You can watch the gate switch on and off
inside a single model family: a 31B Gemma clears its ReAct baseline by fourteen
points, but a 4B Gemma collapses — every configuration lands in the same low single
digits, the model too weak to write the code, so the scaffold has nothing to stand
on. The same gate shows up in Qwen: holding perception fixed at the 27B backend and
varying only the reasoner, the active-perception agent beats ReAct at every size we
tried (27B 42 vs 27, 9B 25 vs 21, 4B 21 vs 16), and the margin is widest for the
strongest reasoner. The harness amplifies a capable model; it can't rescue one that
can't code.

None of this is special to Qwen 3.5 27B. The lift shows up across Qwen sizes and in
a second family (Gemma), for any model that's a strong enough multimodal,
code-writing reasoner — Qwen 3.5 27B is simply the one we entered in the challenge.
The recipe is about the harness, not the checkpoint.

A note on what this is *not*. We did find a length effect across *benchmarks* —
on much longer documents, a fixed-page baseline starts answering "unknown" as the
evidence falls off the end of its budget, while the active-perception agent stays
flat. But within DocVQA-2026, length is a red herring: the advantage tracks
density, and page count is mostly a proxy for it. We're treating the
cross-benchmark length result as provisional until it's nailed down with more
trials; the robust story is density.[^length]

### A signal worth flagging

One last number points forward. Take the append-only variant from the last section
and, over its eight trials per question, ask how often *any* one of them got the
answer right — the oracle, pass@8. It's about **64%**, versus the **40%** that
variant actually lands on a single try. The right answer is reachable far more often
than it's reliably produced; somewhere in that ~24-point gap is a learning signal
nobody has spent yet. We'll come back to it.

First, though: did any of this actually win anything?

[^stats]: +7.87pp at 9B (Welch *t* = 3.54, 95% CI [+3.4, +12.3]) and +8.60pp at 4B
(*t* = 4.96, 95% CI [+5.2, +12.0]), eight trials per arm.

[^length]: Within-set, the "advantage grows with page count" hypothesis doesn't
hold — on the longest documents with a strong VLM the gap is flat. Across
benchmarks of very different length, a budget effect does appear; we report it
cautiously.
