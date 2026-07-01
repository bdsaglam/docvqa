## The cost of generality: it's slow

Everything good about this method — general model, no training, no domain
pipeline — is bought with one currency: **calls**. Perception happens a region at a
time, each region is a VLM call, and the calls are sequential because each one
depends on what the last one returned. The full method averages around **13 steps
per question**; the in-context-pixels variant, which never converges, runs more
than twice that and pins the cap.

| Configuration | Steps / question |
|---|---|
| Active-perception agent (ours) | ~13 |
| ReAct (no REPL) | ~5 |
| In-context pixels (no perception call) | ~30 (caps out on most questions) |
| Raw single pass (no scaffold) | 1 |

So the method trades latency and token cost for accuracy and generality. On the
heaviest documents it can run up against the model's context limit outright, and
the competition's self-consistency voting multiplies the cost several times over.
This isn't a small caveat — it's the reason you'd hesitate to put this exact
configuration in front of a latency-sensitive user.

We're not the first to hit this. MADQA makes the point sharply: an unconstrained
recursive agent can be flexible *and* ruinously expensive — in their setting one
burned on the order of 270M input tokens and several hundred dollars on a task it
then *lost* to a far cheaper retrieval agent. Flexibility has a bill attached.

It helps to be clear about what the extra steps buy, though. More steps mark a
*hard* document, not a path to a better answer — across questions, trajectory
length is mildly *negatively* correlated with correctness. The lever is the quality
of the perception loop, not its length; grinding longer is a symptom, not a fix.

The encouraging part is that we left the obvious efficiency levers untouched —
there's clear room, we just didn't need it to make the point:

- **Cut calls with cheap retrieval.** High-quality OCR run once as preprocessing,
  plus a searchable index, would let the agent jump to the right page instead of
  sweeping — fewer perception calls for the same evidence.
- **Make each call cheaper.** The reasoner and the VLM don't have to be the same
  model. A smaller, faster, or document-specialized VLM behind the perception call
  would cut per-call cost without touching the reasoning.

And this reframes the OCR result from earlier. We found OCR-on-top buys ~0
*accuracy* on these documents — but that was never its best use. Its real payoff is
likely **efficiency**: fewer and cheaper looks, not higher scores. The clean
extension isn't "OCR to answer better," it's "OCR to answer the same, faster."

Two smaller hedges, for completeness: these ablations are validation-only, and the
cross-benchmark length effect needs more trials before we'd lean on it. Neither
moves the central picture, but it's the honest shape of the evidence.

Step back from the bill for a moment, though, because there's a bigger idea hiding
in all this.
