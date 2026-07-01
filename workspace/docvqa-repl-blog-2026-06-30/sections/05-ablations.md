## What's actually carrying the lift

The harness has a few moving parts: a Python REPL, a VLM that the agent calls as a
perception tool, and the loop that ties them together. Which of those is doing the
work? The clean way to find out is to remove one part at a time and watch the
score move. Every run below is the same model (Qwen 3.5 27B as both reasoner and
VLM), on the same DocVQA-2026 validation set (25 documents, 80 questions), with the
same answer-formatting rules — eight trials each, scored by ANLS, the fuzzy
string-match metric DocVQA uses, and reported as mean ± standard deviation. Only the
structure changes.

Start at the top and take away the REPL. What's left is a **ReAct agent**: the
same VLM perception tool, but called through plain tool-use instead of from inside
a code environment. The score falls from **41.9%** to **27.2%** — about fifteen
points. Without a REPL the agent can't crop a region by arithmetic, can't tile a
page, can't subtract two numbers it just read; it asks for whole pages and stops
early (around five steps per question). The REPL isn't a convenience. It's the
thing that lets reasoning turn into *targeted* perception.

Now do the opposite: keep the REPL, take away the perception *call*. Give the
agent a `display()` that loads the page pixels straight into its own context, so
it looks at the document itself instead of asking a focused VLM call to look and
report back. This collapses too — down to **22.3%**. The agent also thrashes: it
runs 30+ steps per question and pins the iteration cap on most of them, grinding
without converging.

That second result is exactly what the Recursive Language Models line would
predict. Stuffing raw content into the reasoner's own context degrades it — the
familiar context-rot effect, where a model handles a long or noisy context worse
than a clean one — whereas a focused sub-call that returns *compact text* keeps the
context clean. RLM tells this story for long text; here it shows up for pixels — and
it's what pins down *which* half does the work. Having a REPL isn't enough on its
own; the perception has to go through **a call that returns text**,
not be poured into the reasoner's own window. (The RL-trained version of "let the
model look at its own image crops" is DeepEyes;[^deepeyes] the point here is only
that a *prompted* REPL agent is better off not.)

Put the two knockouts together and you get a clean 2×2:

| | **with active-perception call** | **without (pixels in-context / none)** |
|---|---|---|
| **with REPL** | **41.9%** (full method) | 22.3% (`display()` only) |
| **without REPL** | 27.2% (ReAct) | 20.9% (raw multi-image, no scaffold) |

You need both halves; neither alone gets you far. Drop either and you
land in the low-to-mid 20s, near the no-scaffold baseline. The lift lives in the
combination: a code REPL **and** a single call to a VLM used as a perception tool.

![Figure 3: accuracy by configuration, three clean tiers with error bars](figures/f3-tiers.png)
**Figure 3.** The full configuration space, eight trials each. Three tiers
separate cleanly: REPL + active perception (~36–42%), missing one of the two
(no REPL, or no perception call — 21–27%), and an OCR-only floor (next section).
Every cross-tier gap is much larger than the per-cell spread.

### Three things that turn out *not* to matter

The core is small, and the obvious ways to enrich it don't make it any bigger —
which is the more useful half of the story, because it's what tells you what you
*don't* need to build.

**Generalizing the call buys nothing.** We replaced the focused "look at this
region" call with a general sub-agent that could take on any subtask (image
optional). Accuracy didn't move — **36.7%**, inside the noise of the full method.
And when we logged what the agent actually asked the sub-call to do, about **99%**
of the calls were still plain perception. One focused perception primitive already captures the
benefit; the extra generality just sits there. (We use a single call throughout; we
never tried stacking them deeper.)

**The trajectory format doesn't matter — for inference.** Our agent compacts its
history as it goes (the RLM style). Its twin keeps an **append-only** transcript
instead, never compacting — the CodeAct style, a fully-observable log of every
turn. The two tie: **39.5%** vs 41.9%, within a couple of points, and the
append-only version doesn't even lose ground on longer documents (the per-document
gap is uncorrelated with page count). For getting the answer, how you represent
the trajectory is a wash.

It does, though, matter for something else — **training**. If you ever want to
*train* the agent (with reinforcement learning, or by distilling a stronger one),
the methods assume the model's output grows as a clean prefix — turn *t* is just
turn *t−1* with more appended. Compaction breaks that: it rewrites the history
between turns, so the sequence is no longer a growing prefix. The append-only
transcript keeps it, which makes it the more *trainable* of the two — and, as we
just saw, choosing it costs nothing in accuracy. (Making compacted trajectories
trainable anyway is its own open problem; FoldAct is an early attempt.[^foldact])
We'll come back to this at the end.

**Adding OCR on top buys nothing here.** Bolt OCR page text and a BM25 search tool
onto the full method and the score is **36.6%** — flat, within the noise. On these
moderate-length documents, text retrieval adds nothing the active-perception call
isn't already getting from the pixels.

So the core that matters is small: **a REPL plus one active-perception call.**
Generality, trajectory format, and OCR-on-top are all dispensable. If you were
going to build this, you'd build less than you think.

That leaves one more knockout — the one that says what kind of problem this is.

[^deepeyes]: Zheng et al., *DeepEyes: Incentivizing "Thinking with Images" via
Reinforcement Learning*, arXiv:2505.14362.

[^foldact]: Shao et al., *FoldAct: Efficient and Stable Context Folding for
Long-Horizon Search Agents*, arXiv:2512.22733.
