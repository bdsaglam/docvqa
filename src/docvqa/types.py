"""Core types for the docvqa package."""

from dataclasses import dataclass
from typing import Any, Optional

import dspy


@dataclass
class LMConfig:
    """Configuration for a language model endpoint.

    Used for both the main agent LM and the VLM tool backend.
    """

    model: str = "vertex_ai/gemini-3-flash-preview"
    temperature: float = 1.0
    max_tokens: Optional[int] = None
    cache: bool = False
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    vertex_location: Optional[str] = None
    reasoning_effort: Optional[str] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    presence_penalty: Optional[float] = None
    enable_thinking: Optional[bool] = None  # None = don't set, True/False = explicit

    def to_dspy_lm(self) -> dspy.LM:
        """Create a DSPy LM from this config."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "cache": self.cache,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.vertex_location:
            kwargs["vertex_location"] = self.vertex_location
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.presence_penalty is not None:
            kwargs["presence_penalty"] = self.presence_penalty
        extra_body: dict[str, Any] = {}
        if "hosted_vllm" in self.model or "openai/" in self.model:
            kwargs.setdefault("api_key", "dummy")
            # Control thinking mode: None = don't set, True/False = explicit
            if self.enable_thinking is not None and "mistral" not in self.model.lower():
                extra_body["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        # top_k is not a standard OpenAI param — pass via extra_body for vLLM
        if self.top_k is not None:
            extra_body["top_k"] = self.top_k
        if extra_body:
            kwargs["extra_body"] = extra_body
        kwargs["timeout"] = 600  # 10min timeout to prevent hanging
        return dspy.LM(**kwargs)
