## The task, and why it's hard

Document visual question answering is what it sounds like: you're handed a document
and a question in plain language, and you have to answer it. The catch is what
counts as a "document." In the ICDAR 2026 DocVQA challenge a single item might be a
one-page infographic or a 280-page annual report, and the answer might be a value
in a merged table cell, a label on an engineering drawing, a figure on a crowded
chart, a date in a form, or something you only get by reading two pages and doing
arithmetic.

So before any reasoning happens, there's a finding problem: locate the right page,
then the right region on it, *then* read — often compositing or computing over what
you found. That's the part general-purpose vision-language models struggle with.
Hand a VLM all the pages at once and it reads each at a fixed resolution with a
fixed slice of attention. On a sparse page that's fine. On a dense one it isn't.

Here's a concrete one. This is a chart from a financial report; the question asks
for a value on it.

![Figure 1: a dense financial chart; whole-page read misreads a label](figures/f1-nvidia-chart.png)
**Figure 1.** Left, the full page: two bar-and-line charts with around twenty tiny,
overlapping data labels. Right, a crop of the band that holds the answer. Read the
whole page in one pass and the model returns the wrong number — it grabs a
neighboring label. Crop to the region and zoom, and the right value is legible.
The information was always there; the model just couldn't afford to resolve it in
a single look.

That gap — between what's on the page and what a single read can resolve — is the
whole game. The question is what to do about it. The obvious moves are to use a
bigger model or to stuff more pages into the context window. The move that actually
worked was to let the model decide *where to look*.
