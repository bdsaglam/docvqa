# Structured VLM Output

## Idea
Let the agent pass a schema/Pydantic model to `batch_look` so VLM returns structured data instead of free-form text. This makes results directly usable in code without parsing.

## Motivation
- Agent wastes iterations parsing VLM text responses (extracting numbers, splitting lists, handling inconsistent formats)
- Structured output from VLM would be directly usable: `result.value`, `result.rows[0].qty`
- VLM providers (Gemini, vLLM) support structured output / JSON mode
- Reduces errors from text parsing (e.g., "1,234" vs "1234", "Item 5: qty 2" vs just "2")

## Design
```python
# Agent defines Pydantic models with mandatory explanation field
from pydantic import BaseModel

class TableRow(BaseModel):
    item_number: int
    part_id: str
    description: str
    quantity: int

class BOMReading(BaseModel):
    explanation: str  # always include — VLM reasoning for main agent to inspect
    rows: list[TableRow]

class NumberReading(BaseModel):
    explanation: str
    value: float

# Pass schema to batch_look
results = batch_look([
    (crop, "Read the BOM table rows", BOMReading),
    (crop2, "What number is shown?", NumberReading),
])
# results[0].rows[0].quantity — structured, no parsing
# results[0].explanation — VLM reasoning, inspectable if needed
# results[1].value — float, directly usable in math

# Majority voting becomes trivial
values = [r.value for r in number_results]
from collections import Counter
answer = Counter(values).most_common(1)[0][0]
```

Always use Pydantic models, never bare types (int, str). The `explanation` field lets the
main agent inspect VLM reasoning when results are ambiguous, without polluting the structured data.

## Implementation Notes
- `batch_look` signature: `list[(image, query)] | list[(image, query, type)]`
- When type is provided, use dspy structured output (response_format / JSON schema)
- When type is omitted, return str as before (backward compatible)
- Schema classes need to be defined in sandbox code — can't pass across IPC
- IPC protocol: sandbox sends schema as JSON schema dict, host uses it with dspy.Predict
- For Qwen via vLLM: use `response_format={"type": "json_schema", "json_schema": ...}`
- For Gemini Flash: use `response_schema` parameter

## Benefits
- Majority voting becomes trivial: compare structured fields directly
- Table extraction: get list[Row] directly, sum quantities in code
- Number reading: get float directly, no parsing "1,234" vs "1234"
- Multi-value extraction: get all values in one call
- `explanation` field preserves VLM reasoning for agent inspection without parsing overhead

## Risks
- Schema definition overhead in sandbox code (agent must define model before calling)
- Not all VLMs support structured output equally well
- May increase token usage (schema in prompt)
- VLM may hallucinate to fill required fields — `explanation` field helps detect this
