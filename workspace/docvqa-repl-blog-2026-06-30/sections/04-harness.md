## The recipe

The whole method fits in one sentence. Give a code-capable model a persistent
Python REPL and a single perception primitive — an on-demand call to a VLM,
pointed at any region of any page — and let it *direct* perception instead of
swallowing the document whole. It can crop to the evidence, zoom for acuity, sweep
pages in a loop, composite regions, and do in code the coordinate math and
arithmetic a VLM is bad at. Perception becomes something the model spends
deliberately, a region at a time, rather than a single fixed gulp of pixels. Three
moves give the system its name — **Perceive-Reason-Code**: perceive through a VLM
call, reason in language, act by writing code.

![Figure 2: two-panel architecture, active perception vs ReAct](figures/f2-architecture.png)
**Figure 2.** Left: the active-perception loop — the reasoner writes code, the
code calls a (frozen) VLM against a chosen crop, the text it returns flows back
into the REPL as the next observation. Right: a ReAct agent with the same VLM but
no REPL — it can call the tool, but only on whole pages, with no way to crop,
compose, or compute. The REPL is the only structural difference between them, and
it's what converts reasoning into targeted perception.

Concretely, the loop is the familiar agent shape: a **state** (the transcript so
far), an **action** (a block of Python the model writes), and an **observation**
(whatever that code prints — including the text a perception call returns). Hold
onto that framing; how the state is represented turns out to matter later, in a
way that doesn't affect accuracy at all.

Here's a representative trajectory, lightly trimmed. The question asks for the gap
between two figures on the dense financial chart from Figure 1 — buried deep in a
181-page report.

```python
# survey the document, find the page with the relevant chart
pages = search_pages("total shareholder return")        # -> page 76
look(page=76)
# VLM (whole page): "NVIDIA TSR is $978.42; Nasdaq-100 TSR is $190.57"
```

The numbers look off — the labels are tiny and overlapping, and a whole-page read
is exactly where a VLM misreads. So the agent distrusts itself and crops in:

```python
crop = region(page=76, box=(0.55, 0.18, 0.95, 0.42))    # the top chart band
look(crop, zoom=2)
# VLM (cropped): "NVIDIA TSR $2,287.07; Nasdaq-100 Index TSR $238.19"
answer = 2287.07 - 238.19
submit(round(answer, 2))                                # -> 2048.88  ✓
```

The crop-and-verify catches a wrong number a single read would have submitted, and
the subtraction happens in Python where it's exact. Survey, locate, read, distrust,
re-read at the right scale, compute — every move that makes the method work shows
up in one short trace.

One setup detail, since a careful reader will ask: these runs disable the model's
native "thinking" channel.[^think] That doesn't make the agent reason less — it
relocates the reasoning into the visible body of each turn, where the code and the
comments are. Thinking-off is not answering-without-reasoning; it just moves the
reasoning somewhere we can see it.

### Where this comes from

Perceive-Reason-Code stands on a few well-tested ideas. The REPL-with-a-sub-call
shape comes from **Recursive Language Models** (Zhang et al., 2025): the model works
inside a code environment where the document is just a variable it can slice and
inspect with code, and it can fire off a sub-call to a model when it needs one.
Writing actions *as code* rather than as JSON tool calls is **CodeAct**.
Orchestrating vision modules with a program goes back to **VisProg** and
**ViperGPT**. The move here is to put them together for documents, with the sub-call
specialized as visual perception. (When the same model serves as both reasoner and
VLM, that perception call is the model calling itself — but nothing here turns on
that; it's a single call used as a perception tool, and that's how we'll treat it.)

RLM had already shown, for *text*, that the REPL alone lifts a baseline and a
sub-call lifts it further. The question this post answers is whether that holds when
the sub-call is a *VLM* over a stack of document images — and, more usefully, which
piece is actually responsible. The rest of the post is the controlled answer.

[^think]: We run with `enable_thinking=false` for cost and reproducibility.
Re-enabling it doesn't change the picture (a separate ablation moves it less than
the trial-to-trial noise).
