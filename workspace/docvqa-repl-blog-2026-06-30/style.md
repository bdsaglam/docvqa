# Style notes — DocVQA active-perception blog

Running constraints for the prose. Applied in every revision pass and the reader
test. (Add to the banned list as needed.)

## Title (locked)
**Perceive-Reason-Code: which part of a document agent actually does the work?**

Method/system name (official competition name): **Perceive-Reason-Code** — perceive
via a VLM call, reason in language, act by writing code. Introduced once in the
recipe section; usable as shorthand thereafter.

## Banned words / phrases
Do not use these in reader-facing prose (`sections/`, `draft.md`):

- **load-bearing** — replace with: carries it / does the work / what's essential /
  the part that matters / what the result rests on (pick by context).
- **delve into** — replace with: look at / dig into / get into / examine.

## Voice
- **Intro:** first-person ("We entered… it was a joint winner") — warmer, an
  invitation.
- **Body:** measured expository voice; use "we" only where natural ("we remove the
  REPL," "we found"). Do not force first-person, and not every sentence should be
  conversational. Rigor in the analytical sections > chattiness.

## Other governing rules (from earlier decisions)
- **No process/origin/change-history narration** (synthesis, not changelog). See
  memory `feedback_no_process_leak_in_artifacts`.
- **Terminology:** "active perception" / "depth-1 VLM-tool call"; avoid
  "recursive"/"delegation" as framing; "ReAct" = named baseline.
- **No legacy engineering solver names** (`flat_solo`, etc.) in prose.
- **Numbers:** current-code canonical. The competition submission's difference
  (DocVQA-specific prompts + OCR/search/voting on the same core) is admitted
  **once**, in the intro footnote `[^submission]` — do not re-litigate it in `win`
  or `ablations`. The OCR-on-top ablation stays a standalone finding.
