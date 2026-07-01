## A 27B model, a Python REPL, and one question

We entered the ICDAR 2026 DocVQA challenge with Qwen 3.5 27B, an open model, and
almost no machinery around it: a Python REPL and a single call to a vision model,
used as a tool. It was a joint winner of the 8–35B tier, landing ahead of
the closed-frontier baselines — Gemini 3 Pro, GPT-5.2 — on a genuinely hard
document benchmark.

Winning is a nice anchor, but the sharper question is where the lift comes from.
That a code harness helps a model is by now well established; what's less clear is
which of its pieces — the REPL, the VLM tool, the agent loop — is actually carrying
the result. So this post takes the thing apart, one piece at a time: **which
components carry the win, and which are just along for the ride?**

The answer is useful in a specific way: the core that does the work is smaller than
what most people build. Two parts matter; the rest — a general sub-agent, clever
trajectory management, an OCR pipeline — barely move accuracy. And underneath sits
a reframe worth keeping if you build multimodal agents: on documents the bottleneck
is **perception budget, not reasoning.** The model usually isn't too weak for the
page; it just can't afford to see all of it at once.

The design builds on a few well-tested ideas — Recursive Language Models, CodeAct,
and the code-as-vision line — put to work on document QA. What this post adds is a
clean, controlled read on which of those ingredients actually matters, a mechanism
for *why*, and a competition result that shows how far it gets.

Let's start with why documents are hard.
