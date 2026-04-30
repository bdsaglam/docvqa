"""Quick test for ThinkingRLM — verifies reasoning_content extraction from thinking tokens."""

import dspy
from docvqa.thinking_rlm import ThinkingRLM, _extract_reasoning_content


def test_thinking_rlm():
    lm = dspy.LM(
        "hosted_vllm/Qwen/Qwen3.5-27B",
        api_base="http://localhost:8927/v1",
        api_key="dummy",
        max_tokens=2000,
        temperature=0.7,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    dspy.configure(lm=lm)

    # 1. Test raw reasoning_content extraction
    print("=== Test 1: raw litellm reasoning_content ===")
    import litellm
    resp = litellm.completion(
        model="hosted_vllm/Qwen/Qwen3.5-27B",
        api_base="http://localhost:8927/v1",
        api_key="dummy",
        messages=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        max_tokens=200,
        temperature=0.7,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    msg = resp.choices[0].message
    print(f"content: {msg.content!r}")
    rc = getattr(msg, "reasoning_content", None)
    print(f"reasoning_content: {rc!r}")
    assert rc, "No reasoning_content found — is enable_thinking working?"
    print("PASS\n")

    # 2. Test that _extract_reasoning_content works after a dspy.Predict call
    print("=== Test 2: _extract_reasoning_content after Predict ===")
    sig = dspy.Signature({}, "Given a math problem, write Python code to solve it.")
    sig = sig.append("problem", dspy.InputField(), type_=str)
    sig = sig.append("code", dspy.OutputField(desc="Python code"), type_=str)

    predict = dspy.Predict(sig)
    result = predict(problem="Calculate the factorial of 5")
    print(f"code: {result.code[:100]!r}...")
    reasoning = _extract_reasoning_content(lm)
    print(f"reasoning_content ({len(reasoning)} chars): {reasoning[:200]!r}...")
    assert reasoning, "No reasoning_content extracted from LM history"
    print("PASS\n")

    # 3. Test full ThinkingRLM
    print("=== Test 3: ThinkingRLM forward ===")
    rlm = ThinkingRLM(
        signature="question -> answer: str",
        max_iterations=5,
        verbose=True,
    )
    pred = rlm(question="What is the sum of the first 10 prime numbers?")
    print(f"answer: {pred.answer!r}")
    print(f"trajectory entries: {len(pred.trajectory)}")
    print(f"final_reasoning: {pred.final_reasoning[:200]!r}..." if pred.final_reasoning else "final_reasoning: (empty)")

    # Check trajectory has reasoning populated from thinking tokens
    for i, entry in enumerate(pred.trajectory):
        r = entry.get("reasoning", "")
        print(f"  step {i+1}: reasoning={len(r)} chars, code={len(entry.get('code', ''))} chars")
        if r:
            print(f"    reasoning preview: {r[:100]!r}...")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_thinking_rlm()
