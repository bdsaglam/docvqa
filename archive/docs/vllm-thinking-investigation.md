# vLLM Thinking Token Investigation

**Date**: 2026-04-03

## Problem

Qwen3.5-27B on vLLM generates ~250-380 thinking tokens per call even with `enable_thinking: false`. These tokens are stripped from the response content but still consume compute and slow down every LLM/VLM call.

## Evidence

```bash
# With enable_thinking: false
curl http://localhost:8928/v1/chat/completions -d '{
  "model":"Qwen/Qwen3.5-27B",
  "messages":[{"role":"user","content":"Say just the word hello"}],
  "max_tokens":500,
  "extra_body":{"chat_template_kwargs":{"enable_thinking":false}}
}'
# Result: content="hello" but completion_tokens=381
# Without extra_body: completion_tokens=255

# Response content starts with "Thinking Process:" when max_tokens is low
# Thinking is always generated, just hidden in the final content
```

## What we tried (none worked)

- `extra_body.chat_template_kwargs.enable_thinking: false` — tokens still generated
- `extra_body.thinking.type: "disabled"` — no effect
- `/no_think` prefix in user message — no effect

## Root cause

The vLLM server was launched without thinking control. The `enable_thinking` chat template kwarg only works if the server's Jinja chat template has a conditional for it. This server's template either ignores it or always enables thinking.

## Fix options (requires server restart)

1. **`--thinking-budget 0`** — vLLM flag to cap thinking tokens (if supported in this vLLM version)
2. **`--override-chat-template`** — provide a custom Jinja template that skips the `<think>` block entirely
3. **Use a non-thinking model variant** — if available (e.g., Qwen3.5-27B-Instruct without thinking)
4. **`--chat-template-kwargs '{"enable_thinking": false}'`** — server-level default (may need vLLM >= 0.8)

## Impact

- Each VLM call wastes ~300 tokens of thinking (~10-20% of generation time)
- At ~50 VLM calls per doc, that's ~15K wasted tokens per doc
- Fixing this could speed up eval by 10-20% with no quality loss (thinking output is discarded anyway)

## Current workaround

We set `temperature=0.6` for both LLM and VLM to reduce variance. This doesn't fix the thinking overhead but improves result consistency.
