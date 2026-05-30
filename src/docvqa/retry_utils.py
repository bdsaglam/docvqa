"""Retry predicates shared across solvers."""

from __future__ import annotations


def is_retryable_lm_error(e: BaseException) -> bool:
    """Return True for LM/VLM call failures that should be retried.

    Catches:
    - Rate limits (429, RateLimit, RESOURCE_EXHAUSTED).
    - litellm/vllm timeouts (`litellm.Timeout`, "timed out" in message).
    - vllm transient server failures (InternalServerError, "Server
      disconnected" — happens under KV-cache preemption pressure).

    Per 2026-05-30 discussion: under heavy concurrent load, vllm
    preempts requests whose KV cache won't fit. Preempted requests can
    sit waiting longer than the per-request `lm.timeout` (default
    600s), at which point litellm raises `Timeout`. Restarting the
    agent loop on a fresh attempt is cheaper than raising
    `lm.timeout` — the new attempt rejoins vllm's scheduler from
    scratch, while raising the timeout would let preempted requests
    hold KV cache longer.
    """
    msg = str(e)
    name = type(e).__name__
    if "429" in msg or "RateLimit" in name or "RESOURCE_EXHAUSTED" in msg:
        return True
    if "Timeout" in name or "timed out" in msg.lower():
        return True
    if "InternalServerError" in name or "Server disconnected" in msg:
        return True
    return False
