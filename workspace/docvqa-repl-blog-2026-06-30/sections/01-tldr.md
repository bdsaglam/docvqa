<!-- TL;DR box, top of post -->

> **TL;DR**
>
> - We gave Qwen 3.5 27B a Python REPL and a single VLM
>   perception call — so it can crop, zoom, and compute over document pages instead
>   of reading them whole — and it was a joint winner of the ICDAR 2026 DocVQA
>   challenge (8–35B tier), ahead of the closed frontier on the held-out test set.
>   No fine-tuning, no document pipeline.
> - Removing one piece at a time shows **two parts carry it, together**: the REPL
>   and the perception call. Drop either and it falls to the no-scaffold baseline.
>   **Three things don't matter** — making the call a general sub-agent, the
>   trajectory format, and adding OCR.
> - It's a **perception-budget** problem, not a reasoning one. The piece is built
>   on proven ideas — Recursive Language Models, CodeAct, code-as-vision — put to
>   work on documents and taken apart to see what carries the win. Code:
>   https://github.com/bdsaglam/docvqa
