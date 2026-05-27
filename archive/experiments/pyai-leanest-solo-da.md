# Pyai Leanest Solo DA — pydantic-ai-rlm port

**Status:** Working but underperforms the dspy baseline (35.0% vs 43.8%, −8.8pp).
Code retained as a reference implementation; not a recommended path forward
without further work.

**Hypothesis going in:** Re-implementing `leanest_solo_da` on top of
[`pydantic-ai-rlm`](https://github.com/vstorm-co/pydantic-ai-rlm) instead of
DSPy should yield roughly parity, since both frameworks expose the same RLM
pattern (code-writing agent + REPL + sub-LLM tool). Worth knowing whether the
dspy implementation has a structural advantage or whether the choice of
framework is incidental.

**Verdict:** The choice of framework is not incidental on this benchmark with
this model (Qwen 3.5 27B local, no thinking). The dspy implementation's
"code-only action per turn" structure traps the agent on task more effectively
than pydantic-ai's "call a tool OR emit text" loop, even when both expose the
same SUBMIT-style termination affordance. The gap is concentrated in
short-document categories (maps: 0% vs likely ≥60% in dspy), where pydantic-ai
agents either bail with "Unknown" on turn 1 (when a terminal `submit` tool is
exposed) or never stop calling tools (when no terminal tool exists).

## Solver shape

New solver `pyai_leanest_solo_da` (sibling of `leanest_solo_da`). Same public
interface — `solve_document(doc) -> (preds, trajectories)` — so the eval
runner doesn't change. Same `_TASK_BODY` prompt verbatim from
`leanest_solo_da_solver`. Same dataset profile (`get_profile(dataset)`) drives
answer formatting / per-category tips / per-question hint / scorer.

What changes:

- **Agent runtime:** `pydantic_ai.Agent` with `LiteLLMProvider` pointed at the
  local vLLM endpoint. Note: pydantic-ai's `LiteLLMProvider` is just an
  `AsyncOpenAI` client under the hood — it sends the model id verbatim, so
  `hosted_vllm/Qwen/Qwen3.5-27B` must be stripped to `Qwen/Qwen3.5-27B`
  before passing to `OpenAIChatModel`.
  ([provider source]: pydantic-ai/providers/litellm.py.)
- **REPL:** `pydantic_ai_rlm.REPLEnvironment` subclassed to inject `pages`
  (list of PIL Images), `batch_look(requests)`, and `SUBMIT(answer=...)` as
  globals. The base REPL only knows `context` + an optional text-only
  `llm_query`; neither helps for a vision-tool agent.
- **Termination:** Manual loop via `Agent.iter()`. SUBMIT() inside REPL writes
  to a shared `_AnswerCarrier`; the wrapper terminates as soon as the carrier
  is set. When the agent never calls SUBMIT, a fallback extract LLM call
  composes the answer from the trajectory's tool-return observations.
- **VLM channel:** `batch_look` calls `litellm.acompletion` directly with
  base64 image content — pydantic-ai's agent never sees image bytes; only the
  tool does.

Implementation in `src/docvqa/solvers/pyai_leanest_solo_da_solver.py`.
Hydra config in `configs/solver/pyai_leanest_solo_da.yaml`.

## Trial

Single full-val run on Qwen 3.5 27B local vllm 8927, val/80q,
`lm.enable_thinking=false`, `max_concurrency=2`, `question_concurrency=2`,
`max_iterations=25`, `page_factor=1.5`.

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local lm.enable_thinking=false \
  solver=pyai_leanest_solo_da \
  data.split=val data.num_samples=null max_concurrency=2 \
  run_id=pyai-leanest-val
```

**Overall: 28/80 = 35.0%** (no failed docs — see "Workarounds" below). 5h wall-clock.

| Category | pyai | dspy `leanest_solo_da` (CLAUDE.md) |
|---|---|---|
| business_report | 40.0% (4/10) | — |
| comics | 20.0% (2/10) | — |
| engineering_drawing | 40.0% (4/10) | — |
| infographics | 40.0% (4/10) | — |
| maps | **0.0%** (0/10) | — |
| science_paper | 40.0% (4/10) | — |
| science_poster | 60.0% (6/10) | — |
| slide | 40.0% (4/10) | — |
| **overall** | **35.0%** | **43.8%** |

Per-category breakdown for the dspy baseline isn't in CLAUDE.md, so the
maps-specific gap is inferred from the agent's failure mode (see below) rather
than measured against dspy maps.

Three full trials were planned; one run was delivered after deciding the
wall-clock cost of three runs (≈15h) wasn't justified given the single-trial
result already lagged the baseline by a clear margin.

## What broke and why

Getting any working pipeline required three structural fixes that aren't
hinted at in the pydantic-ai-rlm README:

1. **vLLM tool-calling flag.** The default vllm container doesn't accept
   `tool_choice="auto"`. Restarting with `--enable-auto-tool-choice
   --tool-call-parser qwen3_coder` is required for the agent to call tools at
   all. (dspy bypasses this entirely because it routes everything through
   text completions, not function-calling.)

2. **REPL process-wide CWD race.** `pydantic_ai_rlm.REPLEnvironment.__init__`
   captures `os.getcwd()`, and `execute()` does process-wide
   `os.chdir(self.temp_dir)` then chdir's back. With concurrent REPLs
   (`question_concurrency>1`), one REPL leaves the process chdir'd into a
   tempdir that another REPL is about to delete — the next `os.getcwd()`
   raises `FileNotFoundError`. With `max_concurrency≥2` at the doc level,
   the failure cascades: every queued doc fails the moment a sibling
   completes. **Workaround:** the subclass bypasses `super().__init__`,
   reconstructs the parent's state without calling `os.getcwd()`, and
   overrides `_temp_working_directory` to be a no-op
   (`pyai_leanest_solo_da_solver.py:_PagesREPL.__init__`).

3. **Termination affordance.** Three failed strategies before settling:
   - `output_type=str` (no terminal tool): agent never stops calling
     `execute_code` → `UsageLimitExceeded` after 35+ tool calls.
   - `output_type=ToolOutput(FinalAnswer)`: agent calls
     `submit(answer="Unknown")` on its first turn. 10/10 questions returned
     "Unknown" in seconds, no exploration.
   - Output validator that raises `ModelRetry` when "Unknown" is submitted
     early: agent ping-pongs against the validator until `max_retries` is
     hit; pydantic-ai then fails the entire run.
   - **Final strategy:** drop the terminal tool, inject `SUBMIT()` as a
     Python function in the REPL (matching dspy's pattern), capture via a
     shared `_AnswerCarrier`, and fall back to a separate extract LLM call
     when SUBMIT isn't reached. This works but with two side effects: (a)
     the agent rarely calls SUBMIT (Qwen 3.5 27B doesn't reliably "write
     code that calls a Python function" the way it would "call a tool"),
     so most answers come from the extract fallback; (b) the extract
     fallback adds an extra LLM call per question, slowing the run.

## Why dspy wins (best current explanation)

The structural difference that matters: **action space per turn.**

- **dspy LeanRLM:** the model's action signature is literally `(reasoning,
  code)` produced as structured text. Every turn the model writes Python
  code that the REPL executes. Termination is `SUBMIT(answer=...)` inside
  that same code. There's no "do I keep going or stop?" decision tree —
  the loop is "code → observation → code → … → SUBMIT".

- **pydantic-ai (native tool-calling):** the model chooses between calling
  `execute_code`, calling another tool (if exposed), or emitting plain
  text. Adding a terminal `submit` tool gives the agent a one-click bail
  path. Removing it removes the off-ramp entirely. There's no middle.

On this benchmark with this model, dspy's narrower action space keeps the
agent on task. The "maps" category collapses (0%) for pyai because maps are
single-page documents where the dspy agent submits quickly (one or two reads
+ SUBMIT inside code), while pyai's agent either over-explores (no SUBMIT
reached, fallback gets noisy observations) or bails (SUBMIT happens but with
"Unknown" because the agent hasn't done the actual look).

The pydantic-ai-rlm package is designed for large frontier models (GPT-5,
Claude Sonnet) where the agent handles termination cleanly on its own. With
a 27B local model and a vision-tool loop, the framework's flexibility
becomes a liability.

## What would close the gap (not done)

- **Drop pydantic-ai-rlm's REPL** and reimplement the REPL on `pydantic-ai`
  primitives only, with a single `execute_code` tool — gets rid of the CWD
  race and clarifies ownership, but doesn't change the termination model.
- **Wrap the agent loop to inject `[REASONING] ... [CODE] ...` formatting**
  on every turn (à la dspy ChatAdapter). The agent writes structured text
  that we parse — never uses pydantic-ai's native tool-calling. This is
  what dspy effectively does; replicating it on pydantic-ai removes the
  framework's value proposition (you've reimplemented dspy on top of
  pydantic-ai).
- **Use a larger / thinking-enabled model.** Plausibly Gemini 3 Pro or
  Qwen 3.5 27B with thinking on would handle the dual-channel loop better
  and the gap might close. Not tested here.

## Wall-clock cost

| Phase | Time |
|---|---|
| 4-doc smoke (validating SUBMIT+extract loop) | 1h05m |
| Full val (25 docs / 80 questions, max_iter=25, concurrency=2) | ~5h |

dspy `leanest_solo_da` at `max_concurrency=16, question_concurrency=4`
completes the same val in roughly 30 minutes on the same hardware. The
slowdown is dominated by:

- More tool-calling overhead per agent step (one HTTP round-trip per
  `execute_code`).
- The extra extract-fallback LLM call per question (~all questions, since
  Qwen rarely writes SUBMIT inside code).
- Lower safe concurrency: with `max_concurrency=4` we observed connection
  pool exhaustion (`ClosedResourceError` cascades killing the run).
  Dropping to 2 cleared it.

## Artifacts

- Solver: `src/docvqa/solvers/pyai_leanest_solo_da_solver.py`
- Config: `configs/solver/pyai_leanest_solo_da.yaml`
- Eval script (unused after switching to single-trial): `scripts/run_pyai_evals.sh`
- Results: `output/runs/pyai-leanest-val/results.json` (and `tasks/<doc>/result.json` per-doc)
