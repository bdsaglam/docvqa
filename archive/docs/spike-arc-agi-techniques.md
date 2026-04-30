# Spike: ARC-AGI Techniques for DocVQA

A team achieved **92% on DocVQA val** (with Opus) and **76% with Gemini Flash**. They previously achieved SOTA on ARC-AGI 2 and 3. Their DocVQA method isn't public yet, but their ARC repos reveal patterns that likely transfer.

Our current best: **55% with Pro/Flash flat batch solver**.

## Their ARC-AGI Repos (in `./tmp/`)

| Repo | Pattern | Key Technique |
|------|---------|---------------|
| ARC-AGI-3-Agents | Action loop + frame history | State accumulation across steps |
| arcgentica | Sub-agent spawning | `call_agent()` for parallel hypothesis exploration |
| poetiq-arc-agi-solver | Multi-expert voting + iterative refinement | N experts × M iterations, vote on best |

## Predicted DocVQA Method (Reverse-Engineered)

Given their ARC patterns and the 92%/76% results, they likely:

### 1. Multi-Path Answering (from Poetiq)
- Run 3-5 independent attempts per question with different strategies/seeds
- Vote on the answer — same answer from 2+ paths = high confidence
- **Why it works**: Our 0-100% variance on comics_1 shows single runs are unreliable. Majority voting would stabilize.
- **Cost**: 3-5x more LLM/VLM calls, but with Opus they can afford it.

### 2. Iterative Refinement with Feedback (from Poetiq)
- Attempt → evaluate → show failed attempt as context → retry
- "You tried X and got Y, but the answer format should be Z. Try again."
- **Why it works**: Our agent often gets close but makes formatting/precision errors.
- **Cost**: 2-3 iterations per question, doubles time but catches errors.

### 3. Sub-Agent Specialization (from ARCgentica)
- Spawn specialized sub-agents: text reader, table extractor, spatial reasoner, counter
- Each sub-agent has a narrow scope and returns structured data
- Main agent synthesizes sub-agent outputs into final answer
- **Why it works**: Our flat solver asks the VLM compound questions. Specialized agents do one thing well.

### 4. State Accumulation (from ARC-AGI-3-Agents)
- Full history of pages visited, searches done, VLM responses received
- Agent sees what it already tried and doesn't repeat
- Previous observations inform current strategy
- **Why it works**: Our batch solver loses context between iterations; accumulated state prevents redundant exploration.

### 5. Direct Multimodal (likely addition for DocVQA)
- Pass document pages directly into Opus/Gemini context window (not through a tool)
- The model sees the image natively — no lossy VLM intermediary
- For 76% with Flash: Flash is multimodal and can read documents well
- **Why 92% with Opus**: Opus has much better visual understanding + reasoning

## Gap Analysis: Us vs. Predicted Method

| Technique | Them (predicted) | Us (current) | Gap |
|-----------|-------------------|--------------|-----|
| LLM quality | Opus / Flash | Pro / Flash | Opus >> Pro for reasoning |
| VLM approach | Native multimodal | Tool-based (look + crop) | They skip the tool layer |
| Multi-path | 3-5 attempts, vote | Single attempt | High variance kills us |
| Feedback loop | Failed attempt → retry | None | We never self-correct |
| Sub-agents | Specialized per task | Flat (one VLM tool) | We ask VLM too many things at once |
| State tracking | Full history | Per-iteration only | We lose context |

## Actionable Improvements (Priority Order)

### P0: Multi-Path Voting (biggest impact, moderate effort)
- Run each question 3 times with temperature=1.0
- Take majority vote on the answer
- Expected: +5-10pp just from variance reduction
- Implementation: wrap `solve_document()` with a voting layer

### P1: Specialized Sub-Agents (moderate impact, high effort)
- TextReader: extract specific text from a region
- TableExtractor: extract table cells into structured data
- Counter: count visual elements systematically
- SpatialReasoner: locate objects and compute positions
- Each returns structured data, main agent synthesizes
- Expected: +3-5pp on categories where compound queries fail (comics, maps)

### P2: Diversity Techniques (low effort, moderate impact)
- Vary prompt temperature per attempt (0.7, 1.0, 1.3)
- Shuffle page order in context
- Use different prompt variants per attempt
- Expected: +2-3pp from broader exploration

## P1 Revised: Skilled Solver (Specialized Skills, Not Sub-Agents)

Instead of spawning specialized sub-agents, inject specialized **skill instructions** into VLM calls.

**Design:**
- Skills are string variables with detailed instructions for specific tasks
- Main agent selects which skill to use based on the question/doc type
- `look(image, query, skill=None)` — skill gets prepended to the VLM prompt
- Agent can also write custom instructions on the fly

**Example skills:**
- `TABLE_EXTRACTION`: "Extract all rows from this table. Read each cell left to right, top to bottom. Return as JSON array of rows."
- `LABEL_READING`: "Find and transcribe all text labels in this image. Return as a list with approximate positions."
- `COUNTING`: "Count every distinct instance of the described object. Mark each one. Return the total count."
- `SPATIAL`: "Describe the spatial layout. List all elements with their positions as (x%, y%) from top-left."

**Implementation:** Copy flat_batch_solver.py → skilled_solver.py, extend `look` tool with skill parameter.

**Why this over sub-agents:**
- No subprocess overhead
- Single VLM call with better instructions vs multi-step sub-agent
- Main agent stays in control of what to ask
- Skills can be composed (e.g., TABLE_EXTRACTION + SPATIAL)

## Quick Wins We Can Do Now

1. **Majority voting** (P0): Easiest to implement, biggest bang for buck
2. **Temperature sweep**: Run at temp 0.7 and 1.3 in addition to 1.0, vote
3. **Prompt variants**: 2-3 different strategy prompts per attempt for diversity

## Notes
- Self-correction is NOT possible in DocVQA — no ground truth during inference (unlike ARC training examples)
- They likely use RLM approach for DocVQA too (their core technique), not direct multimodal
- The 92% with Opus is partly model quality (Opus >> Pro for vision + reasoning)

## Agentica Server Analysis

The team's core framework (`tmp/agentica-server/`) is a multi-agent REPL orchestration server.

### Architecture
- **Agent class** with system prompt + REPL sandbox + inference system
- **Three inference backends**: `ResponsesSystem` (OpenAI Responses API), `ChatCompletionsSystem` (OpenAI-compatible), `MessagesSystem` (Anthropic)
- **All routed through OpenRouter** — models specified as `provider:model` or `openrouter:provider/model`
- **Sandbox**: isolated REPL environment per agent, supports sub-agent spawning
- **Monads**: composable interaction modes (prompter, REPL tool, multi-turn)

### Vertex AI / Gemini Support
- **No native Vertex AI support** — uses OpenRouter as the routing layer
- Gemini models accessed via `openrouter:google/gemini-...`
- Known models: OpenAI (gpt-4o, gpt-5) and Anthropic (claude-sonnet-4, claude-opus-4.5)
- To use Gemini: would go through OpenRouter, not direct Vertex API

### Key Differences from Our Architecture
| Aspect | Agentica | Our DSPy-based |
|--------|----------|---------------|
| LLM interaction | Direct API (OpenAI/Anthropic SDK) | DSPy Predict → litellm → API |
| REPL | Native sandbox with sub-agent spawning | SubprocessInterpreter with tool IPC |
| Sub-agents | `call_agent()` — full agent with own REPL | Nested RLM or tool-based |
| Prompt format | Raw messages (system + user + assistant) | DSPy signatures with field descriptions |
| Overhead | Minimal — direct API calls | DSPy adapter + litellm + field formatting |

### Does DSPy Hurt Performance?
Possible overhead:
1. **DSPy signature formatting** adds boilerplate to prompts (field descriptions, output format instructions)
2. **litellm** adds a layer of abstraction over the API
3. **JSONAdapter** may constrain the model's natural output format
4. **RLM action signature** includes `variables_info` and `repl_history` as formatted fields — verbose

Their approach: raw system prompt + direct API → model sees exactly what you intend, no formatting overhead.

### Recommendation
Consider building a thin RLM agent directly on litellm/API without DSPy for the flat batch solver. Keep DSPy for the sequential solver where optimizers (GEPA, MIPROv2) are useful.

## Files to Study
- `tmp/poetiq-arc-agi-solver/solve_parallel_coding.py` — multi-expert voting
- `tmp/poetiq-arc-agi-solver/solve_coding.py` — iterative refinement with feedback
- `tmp/arcgentica/arc_agent/agent.py` — sub-agent spawning via `call_agent()`
- `tmp/arcgentica/arc_agent/prompts.py` — analyze-hypothesize-implement-test pattern
- `tmp/ARC-AGI-3-Agents/agents/agent.py` — frame-based state accumulation
- `tmp/agentica-server/src/agentic/agent.py` — core agent class
- `tmp/agentica-server/src/agentic/monads/` — interaction modes (prompter, REPL)
- `tmp/agentica-server/src/inference/endpoint.py` — inference backends
