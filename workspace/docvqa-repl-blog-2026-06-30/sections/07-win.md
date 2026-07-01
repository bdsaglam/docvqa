## The competitive result

We were a **joint winner of the ICDAR 2026 DocVQA challenge in the 8–35B parameter
tier** — Qwen 3.5 27B, an open model, with no document-specific training and no
specialized encoder.

The challenge scores a held-out test set, with self-consistency voting over a
handful of samples (which the rules allow). The streamlined, general method this
post describes scores **39.4%** there — ahead of the official closed-frontier
baselines:

| System (held-out test set) | Score |
|---|---|
| **Active-perception agent (ours, general)** | **39.4%** |
| Gemini 3 Pro | 37.5% |
| GPT-5.2 | 35.0% |
| Gemini 3 Flash | 33.75% |
| GPT-5 Mini | 22.5% |

Two honest wrinkles, so nothing here is misread. First, the entry that actually
topped the tier scored *higher* — **43.75%** — because it was tuned for this
benchmark: DocVQA-specific prompts, plus the OCR and search we've spent the post
stripping away. Specializing buys a few points of peak score; generalizing gives
them back. The method here is the general one, and it still clears the frontier.

Second, these test numbers sit below our validation numbers. We read that gap as
mostly the **test set being genuinely harder** — its documents run longer, with more
pages to navigate, exactly the regime where a fixed budget hurts.
But we won't pretend it's all difficulty: we developed and tuned against the
validation set, so some fit to it is unavoidable, and we don't claim the validation
figures transfer untouched to test.

Either way, the takeaway isn't the leaderboard position — it's *how* it was reached.
A lot of strong document-QA systems get there by fine-tuning on tens of thousands of
question–answer pairs, or by building a specialized OCR-and-encoder pipeline. We did
neither. The model is stock Qwen 3.5 27B; the "system" is a REPL and one perception
call.
That's the part worth keeping:

> **On this task, harness design substituted for fine-tuning.** Before you reach
> for training data or a specialized pipeline, it's worth seeing how far a general
> model gets when you let it direct its own perception.

That generality isn't free, though.
