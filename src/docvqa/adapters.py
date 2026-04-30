"""Custom DSPy adapters."""

from __future__ import annotations

import logging
from typing import Any

from dspy.adapters.json_adapter import JSONAdapter
from dspy.clients.lm import LM
from dspy.signatures.signature import Signature
from dspy.utils.exceptions import AdapterParseError

logger = logging.getLogger(__name__)


class RetryJSONAdapter(JSONAdapter):
    """JSONAdapter that retries on parse errors up to max_retries times."""

    def __init__(self, max_retries: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.max_retries = max_retries

    def __call__(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        for attempt in range(self.max_retries + 1):
            try:
                return super().__call__(lm, lm_kwargs, signature, demos, inputs)
            except AdapterParseError:
                if attempt < self.max_retries:
                    logger.warning("Parse error on attempt %d/%d, retrying...", attempt + 1, self.max_retries + 1)
                else:
                    raise
