## Code as a substrate for thinking

Strip away the document-specific framing and here's what the REPL really is: a
**symbolic substrate** for a neural model. Code is the medium the model explores
in, composes in, and computes in — a place to hold and manipulate things its own
context can't. Active perception is just one instance of it: the model writes code
to *aim its own eyes*, and the code does the cropping and the arithmetic that the
network is bad at. The neural part proposes; the symbolic part executes and
remembers.

That a code substrate helps at **test time** is well established — it's the
through-line of the RLM-and-CodeAct literature, and these results add a clear
document-domain data point to it.

The question worth ending on points forward. Everything here exercises the substrate
at *inference*: the weights are frozen, and the code is scaffolding around them. What if the substrate were part of how the model *learns*
— not to make it better at deployment inside one particular harness, but to make
the **base model itself** better, in a way that transfers once the scaffolding is
gone? That's a sharper and more uncertain claim than "code helps agents," and it's
the one I keep coming back to.

Two things from this study make it feel concrete rather than idle. First, we
already know which form is trainable: the append-only trajectory ties the compacted
one on accuracy but keeps the clean, growing-prefix structure that learning methods
assume — and making folded trajectories trainable is itself an active problem.
Second, the oracle gap is just sitting there: the right answer is reachable about
24 points more often than the model reliably produces it. That's not noise — that's
an unspent signal, exactly the kind of thing a learning procedure exists to capture.

Whether training *through* a symbolic substrate yields a better base model, rather
than just a better-tuned agent, is genuinely open. We won a competition by letting a
model write code to look more carefully. The thread worth pulling next is whether
teaching it to do so leaves it smarter even when you take the code away.
