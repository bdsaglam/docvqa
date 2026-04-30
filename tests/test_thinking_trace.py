"""Quick test to verify thinking shows up in Logfire traces."""

from dotenv import load_dotenv
load_dotenv()

from docvqa.obs import setup_observability
setup_observability()

import dspy
import logfire

lm = dspy.LM(
    model="hosted_vllm/Qwen/Qwen3.5-27B",
    api_base="http://localhost:8927/v1",
    api_key="dummy",
    temperature=0.6,
    top_p=0.95,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}, "top_k": 20},
    timeout=600,
)

dspy.configure(lm=lm)

with logfire.span("test_thinking"):
    result = lm("How many tennis balls can fit in 1 m^3?")
    print(result)

print("\nDone — check Logfire for the trace.")
