# Process Leak: Detecting and Clearing Editing-History Residue in Reader-Facing Artifacts

A specification for a skill/command that audits a finished artifact (blog post,
paper, README, spec, slide deck, code comments) and removes traces of *how it was
made* that have leaked into *what it says*.

This document is self-contained. An agent can build the skill/command from it
without other context.

---

## 1. The problem

Every reader-facing artifact should read as a **pure function of the current
decision set** — as if written by someone who only ever knew the final decisions,
addressing the *reader*, never carrying traces of the author's editing history or
the debates that shaped it.

A **process leak** is any sentence, phrase, comment, or structural choice that only
exists because of how the artifact came to be: a correction the author received, an
objection someone raised in review, a framing that was tried and abandoned, a value
that used to be different. The reader never saw that history, so the residue is at
best noise and at worst actively misleading — it makes them reconstruct a debate
they were never part of.

The canonical tell:

> "We entered the challenge with an approach, **not a model**."

The "not a model" rebuts an objection ("you just threw a big model at it") that the
reader never raised. It is there because *the author was once accused of it*. To the
reader it reads as defensiveness about a charge they never made — and it plants the
very suspicion it tries to dispel.

### Why clearing it matters

- **Defensive framing backfires.** Rebutting an unraised objection signals
  insecurity and *introduces* the objection. The reader thinks: "wait, should I have
  been worried about that?"
- **Hedges dilute authority.** "One honest note", "to be fair", "for completeness"
  advertise the author's carefulness instead of informing. They read as throat-clearing.
- **Chronology residue confuses.** "We switched from X to Y" forces the reader to
  hold a dead alternative (X) they never needed.
- **Seams show.** Collectively, leaks reveal that the artifact was argued into
  existence rather than known. That erodes trust in the parts that *are* solid.

The underlying principle: **artifacts are syntheses, not changelogs.** The
development process — pivots, dead ends, who asked what, what a number used to be —
appears only when the process is *deliberately the subject* (a postmortem, a "how we
got here" section chosen for the reader). Otherwise it must not leak.

---

## 2. The core test (leak vs. legitimate)

This is the whole skill. Everything else is pattern-matching to surface candidates;
this test decides each one. A naive auditor over-corrects by deleting every "not X"
and every caveat. Do not. Apply this test to each candidate:

> **Does this serve the reader's understanding of the current state, or does it
> betray our editing history / rebut an objection the reader was never shown?**

- **Serves the reader → keep.** A caveat the reader needs to interpret a result
  correctly. A contrast against an alternative the reader *was just shown* (e.g. the
  data on the preceding line). A single, neutral attribution of prior art.
- **Betrays history → cut or rewrite.** A rebuttal to an unseen objection. A phrase
  that narrates the author's own diligence. A reference to what something used to be.

Two quick sub-tests when the core test is ambiguous:

1. **The stranger test.** Would a first-time reader, with no knowledge of how this
   was made, understand why this sentence is here? If the sentence only makes sense
   to someone who watched the edit happen, it leaks.
2. **The earned-contrast test.** For any "X, not Y" or "not just Y": was Y *shown to
   the reader* in this artifact (data, a claim, a figure they just saw)? If yes, the
   contrast is earned — keep it. If Y exists only in the author's private history,
   cut it.

---

## 3. Leak taxonomy (what to look for)

Each category has a **signature**, **why it leaks**, and **surface cues** (words/
patterns that flag candidates for the core test). Cues surface candidates; they do
not decide. Always run the core test.

### L1. Defensive framing (rebutting an unseen objection)
- **Signature:** "X, not Y" · "not just X" · "rather than Y" · "isn't about Y" ·
  "that doesn't mean" · "this is not to say" — where Y is a position the reader was
  never shown.
- **Why it leaks:** The author is answering a charge from their private history. The
  reader inherits a suspicion they never had.
- **Cues:** ` not a`, ` not just`, `rather than`, `instead of`, `isn't `, `is not `,
  `doesn't make`, `that's not to say`.

### L2. Conversational hedge / self-referential meta-narration
- **Signature:** the author narrating their own carefulness or candor, or staging an
  imagined reader.
- **Why it leaks:** "honest note", "for completeness" describe the *author's*
  diligence, which is irrelevant to the reader. "A careful reader will ask" stages an
  objection to pre-empt it.
- **Cues:** `honest`, `to be honest`, `honestly`, `for completeness`, `hedge`,
  `in fairness`, `to be fair`, `it's worth noting`, `we should note`, `admittedly`,
  `full disclosure`, `a careful reader`, `you might ask`, `one might object`,
  `some will say`, `it should be said`.

### L3. Chronology residue
- **Signature:** any framing that references a prior state of the artifact or the
  work: before/after, old/new, tried-then-changed.
- **Why it leaks:** The reader only needs the current state. A dead alternative is
  cognitive load with no payoff.
- **Cues:** `previously`, `originally`, `at first`, `initially`, `used to`,
  `no longer`, `formerly`, `now ` (as in "now we do"), `updated`, `turns out`,
  `it might seem`, `we thought`, `earlier we`, `switched to`, `changed to`,
  `instead of` (temporal sense).

### L4. Over-emphasized disclaimer (protesting too much)
- **Signature:** insisting more than once that something is not novel / well known /
  obvious, or repeatedly pre-apologizing for a limitation.
- **Why it leaks:** Repetition of "this is nothing new" is residue of defending
  against a "you didn't invent this" reviewer. One neutral attribution of prior art
  is enough; three is defensiveness.
- **Cues:** `well established`, `well-known`, `nothing new`, `not novel`, `of course`,
  `as everyone knows`, `it's no secret`, `to be clear` (repeated).
- **Nuance:** A *single* honest attribution is legitimate and often required. The
  leak is the *repetition* and the defensive tone, not the fact.

### L5. Reversed-decision residue (local and structural)
- **Signature (local):** a sentence that argues hard against a framing the artifact
  no longer makes, or rests on a premise an earlier version established.
- **Signature (structural):** the *proportion, ordering, or emphasis* still reflects
  an abandoned decision, even when no single sentence names it. A section that argues
  at length against a position the artifact no longer states. A number left stale
  after the value it depended on changed.
- **Why it leaks:** When a decision changes, the old one lives diffusely — in what
  gets emphasized, what order things come in, what is treated as the default — not
  only in sentences that name it.
- **How to catch:** grep cannot. Re-derive the affected section from the *current*
  decisions and compare: does the emphasis/ordering still match a live decision, or a
  dead one? Then sweep the rest for premises the change invalidated.

### L6. Code-comment "used to" (code artifacts)
- **Signature:** a comment explaining what the code *used to* do, why it *changed*, or
  what was removed.
- **Why it leaks:** Comments describe the code as it is. History belongs in version
  control, not the source.
- **Cues:** `// was`, `# previously`, `# used to`, `// old`, `changed from`,
  `renamed from`, `no longer`, `removed the`, `deprecated (but`.

---

## 4. What is NOT a leak (guardrails against over-correction)

Deleting these is the most common failure of a naive auditor. All of these **stay**:

- **Earned contrasts.** "Perception is *not* optional; it is the thing the scaffold
  buys" — when the preceding data just showed perception is load-bearing. The reader
  was shown Y, so the "not Y" is earned. (Contrast with L1, where Y is unseen.)
- **Genuine caveats the reader needs.** "These ablations are validation-only" tells
  the reader how far to trust the result. Keep the *substance*; strip only the
  meta-narration wrapper ("two smaller hedges, for completeness: ...").
- **Forward references.** "How the state is represented turns out to matter later" —
  this orients the reader to the artifact's own structure, not the author's discovery
  order. (Watch the wording: "turns out" is a cue, but here it is reader-facing.)
- **Single, neutral prior-art attribution.** "The REPL-with-a-sub-call idea is not
  new" said once, as honest grounding, is fine. The leak is repetition (L4).
- **Scoping that narrows an over-reading of the reader's actual takeaway.** "The
  point here is only that a prompted agent is better off without X" — legitimate when
  it prevents the reader from over-generalizing a claim the artifact made.

The discriminator is always Section 2. When in doubt, ask: *was the thing being
contrasted or caveated shown to the reader in this artifact?*

---

## 5. How to clear (rewrite strategies)

Constraints that hold for every rewrite:
- **Never change substance:** no numbers, metric names, figure/citation references,
  or technical claims. Framing only.
- **Preserve voice:** match the artifact's register, sentence length, and
  punctuation conventions (e.g. if the author avoids em-dashes, the rewrite must too).
- **Prefer subtraction.** The best fix is usually deletion of the wrapper, keeping the
  positive claim.

| Category | Transformation | Before → After |
|---|---|---|
| L1 Defensive | State the positive claim; drop the rebuttal. | "an approach, **not a model**: let a model direct its perception" → "an approach: let a model direct its perception" |
| L2 Hedge | Delete the meta-wrapper; keep the substance as a plain statement. | "**One more honest note:** these test numbers sit below our validation numbers." → "These test numbers sit below our validation numbers." |
| L2 Staged reader | Remove the imagined objector; state the fact directly. | "One setup detail, **since a careful reader will ask**: these runs disable thinking." → "One setup detail: these runs disable thinking." |
| L2 Double-negative rebuttal | Replace "not-not" with the positive. | "That doesn't make the agent reason less ... is **not** answering-without-reasoning" → "The reasoning is still there; it just moves somewhere we can see it." |
| L3 Chronology | Delete the before-state; describe only the current state. | "We **switched from** JSON tool calls **to** code" → "The agent acts by writing code." (unless the comparison is the reader's point) |
| L4 Over-disclaimer | Keep one neutral attribution; delete the repeats and the defensive tone. | "That a code harness helps is **by now well established**" (3rd instance) → "A code harness helps a model" |
| L5 Reversed decision | Re-derive the section from current decisions; fix stale premises/numbers/emphasis. | (structural — no one-line fix; see Section 3 L5) |
| L6 Code comment | Delete the history; describe present behavior or nothing. | `# was 3, bumped to 5 after OOM` → `# batch size` (or delete) |

---

## 6. Detection procedure (the workflow the skill/command runs)

1. **Read the whole artifact top to bottom.** Leaks are contextual; a phrase is a
   leak or not depending on what the reader was shown before it. Do not audit by grep
   alone.
2. **Targeted surface pass.** Grep the cue library (Section 7) to collect candidates.
   This catches L1–L4 and L6.
3. **Structural pass (L5).** For any section whose framing you suspect changed:
   re-derive it from the current decision set and check whether its emphasis,
   ordering, and proportion match a live decision or a dead one. Separately, list the
   artifact's load-bearing numbers/premises and verify none is stale (a value that
   changed upstream after this text was written). This pass finds what grep cannot.
4. **Classify each candidate** with the Section 2 core test. Mark: leak (which
   category) or legitimate (why). Be precise, not trigger-happy — false positives that
   delete earned contrasts and real caveats are as bad as misses.
5. **Rewrite** the confirmed leaks per Section 5, honoring the constraints.
6. **Verify:** grep the cue library again to confirm the leak phrases are gone; spot
   the intentional keeps to confirm they survived; confirm no number/metric/citation
   moved and the voice constraints hold (e.g. em-dash count unchanged).

For a **command**, the target is a file path (or a diff); output is either a findings
report or applied edits. For a **skill**, it is invoked whenever the agent writes or
finalizes a reader-facing artifact, and it runs steps 1–6 as a self-check before
declaring the artifact done.

Recommended execution shape for large artifacts: run steps 1–4 in a **read-only
sub-agent** that returns a findings list (line, quote, category, why, proposed
rewrite, severity), then apply in the main agent so a human can review the diff. This
keeps the judgment (which is the hard part) auditable and prevents an over-eager
rewrite from silently changing substance.

---

## 7. Cue library (grep patterns for the surface pass)

Case-insensitive. These *surface candidates*; the core test decides. Expect false
positives — that is the design (better to over-surface and filter than to miss).

```
# L1 defensive
 not a | not an | not just | not merely | rather than | instead of
 isn't | is not | doesn't mean | that's not to say | this is not to

# L2 hedge / meta-narration
 honest | to be honest | honestly | for completeness | hedge | hedges
 in fairness | to be fair | it's worth noting | worth noting | we should note
 admittedly | full disclosure | a careful reader | you might ask
 one might object | some will say | it should be said | needless to say

# L3 chronology
 previously | originally | at first | initially | used to | no longer
 formerly | updated | turns out | it might seem | we thought | earlier we
 switched to | changed to | moved from | went from

# L4 over-disclaimer
 well established | well-known | nothing new | not novel | of course
 as everyone knows | it's no secret | to be clear

# L6 code comments
 // was | # was | # previously | # used to | // old | changed from
 renamed from | removed the | deprecated
```

---

## 8. Worked examples (real before/after)

From an audit of a technical blog post. Each was surfaced by a cue, then confirmed a
leak by the core test.

| # | Category | Before | After |
|---|---|---|---|
| 1 | L1 | "We entered the challenge with an approach, **not a model**: let a code-capable model direct its own perception." | "We entered the challenge with an approach: let a code-capable model direct its own perception." |
| 2 | L2 | "**One more honest note:** these test numbers sit below our validation numbers." | "These test numbers sit below our validation numbers." |
| 3 | L2 | "**Two smaller hedges, for completeness:** these ablations are validation-only ... **the honest shape of the evidence.**" | "Two limits on the evidence: these ablations are validation-only ... Neither moves the central picture." |
| 4 | L4 | "That a code harness helps a model is **by now well established**; what's less clear is which piece carries it." | "A code harness helps a model; what's less clear is which piece carries it." |
| 5 | L2 | "One setup detail, **since a careful reader will ask**: these runs disable thinking." | "One setup detail: these runs disable thinking." |
| 6 | L1 | "That doesn't make the agent reason less ... Thinking-off **is not** answering-without-reasoning." | "The reasoning is still there; thinking-off just moves it somewhere we can see it." |
| 7 | L2 | "**The honest read is** that these models are at least as perception-bound as reasoning-bound." | "On balance, these models are at least as perception-bound as reasoning-bound." |

Confirmed **not** leaks in the same audit (kept):
- "Perception is **not** optional; it is the thing the scaffold buys." — earned: the
  preceding data showed it.
- "The REPL-with-a-sub-call idea stands on a few **well-tested** ideas." — single
  neutral attribution.
- "How the state is represented **turns out** to matter later." — forward reference
  to the artifact's own structure, not discovery chronology.

---

## 9. Notes for the skill/command author

- **Skill (judgment-heavy, always-on self-check):** trigger when writing or
  finalizing any reader-facing artifact. Its value is the discrimination in Section 2,
  not the grep. Bake in the guardrails (Section 4) hard, because the dominant failure
  mode is over-correction. Emphasize: cues surface, the core test decides.
- **Command (run-on-target):** input a file path or diff; default to a **findings
  report** (non-destructive), with an opt-in apply mode. Report format: line, quoted
  text, category, one-line why, proposed rewrite, severity. Applied mode must not
  touch numbers/metrics/citations and must preserve voice constraints (detect and
  respect the artifact's em-dash / sentence-length conventions).
- **Both** should surface, per finding, *why it is a leak* (which unseen objection or
  history it betrays) — that reasoning is what lets a human trust or reject the call.
- Keep the two failure modes visible in the skill's own instructions: **missing a
  leak** (grep too narrow, no structural pass) and **over-correcting** (deleting
  earned contrasts and genuine caveats). A good run reports both what it cut and what
  it deliberately kept.
