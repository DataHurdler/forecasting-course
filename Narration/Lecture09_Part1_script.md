# Lecture 9, Part 1 — Recording Script

**ECON 8310: Business Forecasting · RNNs and LSTMs**

Deck: `Slides/Lecture09_Part1_RNN_LSTM.pdf` (24 pages) · Measured runtime: see the timing guide

---

## How to use this document

- **`▶ SLIDE n — Title`** marks where to advance. The number is the PDF page.
- *Italic parentheticals* are stage directions. **[pause]** means stop for a beat.
- Slides 16 and 17 carry the lecture's real payload: the LSTM loses to the vanilla RNN, at all
  three window lengths. Deliver that plainly. Students expect the fancier model to win, and the
  fact that it doesn't is the most useful thing in the hour.

---

# ▶ SLIDE 1 — Title page

Lecture 9, Part 1: Recurrent Neural Networks and LSTMs.

For two weeks we've been building models that take a fixed block of numbers and produce a
forecast. Last week's CNN was the closest we've come to a model that understands time, and even
its filters only see about seven weeks back.

Today we get the architecture that was designed for sequences from the beginning — one that reads
data in order, one step at a time, and carries something forward as it goes.

**[pause]**

And then we're going to measure it against everything else in the course, and find that the more
sophisticated of the two models we build today comes second. That result is not a disappointment.
It's the lecture.

---

# ▶ SLIDE 2 — Lecture Outline

Five parts. First, precisely where Lectures 7 and 8 ran out of road — because the limitation is
specific and it's worth naming. Then the vanilla RNN, which is one equation applied repeatedly.
Then why that equation fails on long dependencies, which turns out to be a mathematical fact
rather than a tuning problem. Then the LSTM, built to fix exactly that. And finally PyTorch, and
an honest scoreboard.

---

# ▶ SLIDE 3 — Section divider: Where the Last Two Lectures Ran Out

Two architectures, two different ways of failing to see time.

---

# ▶ SLIDE 4 — Two Architectures That Cannot See Time

Let's be precise about what's missing, because "doesn't understand time" is too vague to act on.

Lecture 7's feedforward network treats each row independently. Everything it knows about *when*
something happened, you encoded by hand as a lag column. Shuffle the feature columns and the
model is unchanged. Time is not in the architecture; it's in your feature engineering.

Lecture 8's CNN does better — it genuinely sees local shape. But its **receptive field** is fixed
by the architecture. Stacked width-five and width-three filters reach about seven weeks back, and
no reasonable stack of small filters reaches an annual lag.

And worse, pooling deliberately discarded *where* in the window each pattern occurred. We measured
that costing about five hundred RMSE.

*(Point at the key box.)*

**[pause]**

Here's the assumption both of them share, and it's the one we're breaking today.

Both treat the input as a **fixed-size block**, to be processed all at once. Neither has any
notion of reading a sequence *in order* and carrying something forward from one step to the next.

That's the gap. Everything today closes it.

**[pause]**

It's worth noticing that we have been *compensating* for this gap all semester, quite
successfully. Lag columns are a workaround. Rolling means are a workaround. The Fourier terms in
Homework 4 are a workaround. Every one of them is you, the analyst, telling a memoryless model
something about time that it cannot work out for itself.

And they work well — LASSO on forty-six of those workarounds is still leading our scoreboard. So
this isn't a story where the old approach was broken. It's a story about whether the model can
*discover* the temporal structure instead of being handed it. That's the question today tests.

---

# ▶ SLIDE 5 — The Recurrent Idea: Carry State Forward

The recurrent idea is simple enough to state in one sentence, and I'd like you to have it before
any equations.

A recurrent network reads the sequence one step at a time, and maintains a **hidden state** — call
it h-t — which is a running summary of everything it has seen so far.

At each step it does two things. It updates that summary, using the new observation *and* the
previous summary. And optionally it emits an output.

Critically, the **same weights** are used at every step. That's weight sharing again — but along
*time* rather than along space, which is where the CNN shared them.

**[pause]**

Now the consequence, and it's a big claim.

There is no fixed window. In principle, h-t encodes the *entire* history — every observation from
step one to step t. The architecture imposes no limit whatsoever on how far back the model can
look. Compare that with the CNN's seven weeks, which was baked in by the filter sizes.

*(Beat. Then undercut it.)*

**In principle.**

Most of this lecture is about the gap between that promise and what a recurrent network actually
delivers. And then about the architecture that was invented specifically to close it.

---

# ▶ SLIDE 6 — Section divider: The Vanilla RNN

One equation, applied over and over — and the reason that is both the whole idea and the whole
problem.

---

# ▶ SLIDE 7 — Vanilla RNN: One Equation, Repeated

*(Point at the definition box.)*

Here's the recurrence. The new hidden state h-t is tanh of: W-h times the old hidden state, plus
W-x times the new input, plus a bias. And the output is a linear layer on top of the hidden state.

Read the first equation in words: **the new summary is a squashed mix of the old summary and the
new observation.** That is the entire model. Everything else today is consequences of that line.

The three weight matrices are the *same* at every time step. Not one per step — one, reused.

*(Now the picture under the box.)*

Across the top: twenty-six real weeks of Walmart food sales — the same window you saw last week,
the run-up to Christmas 2015. Each week drops into a cell. All twenty-six cells are drawn the same
because they **are** the same cell — one set of weights, reused at every step. The little arrows
along the row are the hidden state being handed forward, week to week. And only the last one,
h-twenty-six, goes into the linear layer that makes next week's forecast.

**[pause]**

Two things follow from that sharing, and both are genuinely nice.

First, a twenty-six-week window costs no more parameters than a five-week one. Compare the CNN,
where a longer window meant more filters or more layers. Here, window length is free.

Second, the same trained network handles sequences of *different* lengths without modification.
You can train on twenty-six weeks and run it on forty. Nothing in the architecture cares.

Our RNN below holds about **four and a half thousand parameters**. Lecture 7's small feedforward
network held twenty-two thousand six hundred fifty-seven. This is a much smaller model.

**[pause]**

One practical note. For forecasting we usually want a single number, not a sequence of outputs. So
we take the **final** hidden state — the summary after reading the whole window — and put one
linear layer on top of it. That's the pattern you'll write in Homework 5.

**[pause]**

A word on the hidden size, since it's the main thing you'll tune. It's how wide that running
summary is — how much the model is allowed to carry forward. Sixty-four is a sensible default for
our data. Too small and the summary can't hold enough to be useful; too large and you're back in
the overfitting regime from Lecture 7, with far more parameters than observations.

And notice the tanh in the recurrence isn't decorative. It squashes the state to between minus one
and one at every step, which keeps the running summary from growing without bound as the sequence
gets longer. Remove it and the state can drift off to enormous values. It's also, as we're about
to see, part of why the gradient dies.

---

# ▶ SLIDE 8 — Training It: Backpropagation Through Time

Training an RNN is ordinary backpropagation applied to an unusual graph, and the trick is to see
the graph correctly.

**Unroll** the recurrence across the window. Twenty-six time steps becomes a twenty-six-layer
feedforward network — one layer per step — in which every layer shares the same weights.

Once you see it that way, there's nothing new. Gradients flow backward from the loss at the final
step all the way to step one. Because W-h appears at *every* step, its gradient is a **sum** of
contributions from all twenty-six positions.

*(Point at the warning box. Slow down.)*

**[pause]**

And this is where it breaks.

Propagating a gradient back through k steps multiplies it by W-h, and by the slope of tanh, which
is never more than one — that's Lecture 7's right-hand panel — k times over. If the
relevant factor is below one, the gradient shrinks **exponentially** as it travels. If it's above
one, it explodes.

Either way, long-range signal does not survive the trip.

**[pause]**

The exploding case has an easy fix — `clip_grad_norm_` caps the size of the update, and we use it
in every recurrent model in this course. One line, and the problem is gone.

The vanishing case does not have an easy fix. That's what the next slide quantifies, and what the
whole second half of the lecture is about.

---

# ▶ SLIDE 9 — Why a Vanilla RNN Forgets

Let's put numbers on it, because "the gradient shrinks" is not vivid enough to change anyone's
behavior.

Suppose each backward step multiplies the gradient by a factor of zero-point-seven. That's an
unremarkable value for a tanh network — nothing pathological. That's the dashed gray line.

*(Walk the dashed line, slowly.)*

Ten steps back: zero-point-zero-two-eight. Barely present.

Twenty-six steps — our window length — about **one ten-thousandth.** Gone.

A year back: around ten to the minus eight. Invisible. There is no optimizer, no learning rate, no
amount of training that recovers a signal that small.

**[pause]**

Now notice the *shape*. The vertical axis is a log scale, and on a log scale the same factor every
step draws a **straight line**. How steep the line is *is* the factor. That's the whole mechanism
in one picture: repeated multiplication.

*(Point at the red line.)*

And the red line is not an example. That's our own RNN — the lab's model, trained on fifty-two-week
windows — and we measured how much gradient actually reaches each step back. Past the first few
steps it keeps more than the example, about zero-point-seven-nine per step; before training it was
zero-point-five-five, so training did stretch its memory. But it is still a straight line down.
Measured directly — not by multiplying out the slope, which the first few steps would throw off —
about four tenths of a percent survives to twenty-six weeks back. By the same week last year, one
thousandth of a percent.

**[pause]**

So the promise from slide five fails in practice. A vanilla RNN *can*, in principle, encode
arbitrarily long history. But the gradient that would *teach* it to do so has vanished long before
it reaches back that far.

In practice these networks learn dependencies of roughly ten steps. Not fifty-two.

*(Point at the citation.)*

And I want to stress one thing about the Bengio result. This is not an optimization bug that
better tuning fixes. It was **proved** to be inherent to the architecture. Which is why the fix
had to be a different architecture, and why the next section exists.

---

# ▶ SLIDE 10 — Section divider: LSTM — Long Short-Term Memory

Add a memory that information passes through, rather than being recomputed at every step.

---

# ▶ SLIDE 11 — LSTM: A Cell State and Three Gates

An LSTM carries **two** things forward instead of one.

A **cell state**, c-t — that's the long-term memory. And a **hidden state**, h-t — that's what the
rest of the network actually sees. Three gates control the traffic between them.

*(Walk the table.)*

The **forget gate** decides what fraction of the old memory to keep. The **input gate** decides
how much of the new candidate value to write in. The **output gate** decides how much of the
memory to expose to the outside world. And the last row is the **candidate**: the new content on
offer this week, a tanh of the same two inputs. The input gate decides how much of it gets written.

Here sigma means the sigmoid specifically — in Lecture 7 it stood for any activation. So each gate
outputs a number between zero and one. Think of it as a soft switch — not open or closed, but a
dial, and the network learns where to set it. The square brackets just mean last week's hidden
state and this week's input, stacked into one vector. The circled dot multiplies two vectors element
by element. And f-t here is the forget gate — not last week's filter.

**[pause]**

*(Point at the cell state update.)*

Now this line is the one that matters. Everything else is bookkeeping.

The new cell state equals the forget gate times the old cell state, **plus** the input gate times
the new candidate.

Note the operations. There's an element-wise multiply by a number between zero and one — that's
the forgetting. And then there's an **addition**.

Keep your eye on that plus sign. Two slides from now it will turn out to be the whole reason the
architecture works.

**[pause]**

One more thing before we move on: notice what the gates are functions *of*. Each one takes the
previous hidden state and the current input, runs them through a weight matrix, and squashes with
a sigmoid.

So the gates are **learned and input-dependent**. The network is not applying a fixed forgetting
rate. It decides, at every single time step, based on what it's currently looking at, how much to
forget and how much to write.

That's a genuinely different kind of model from anything we've built. A LASSO coefficient is the
same number for every observation. An LSTM gate is recomputed at every step of every sequence.
That flexibility is the appeal — and, as the scoreboard will show, it's also capacity you have to
pay for whether or not you need it.

---

# ▶ SLIDE 12 — What the Gates Are Doing, in Plain Terms

The equations are compact, but the behavior is genuinely intuitive. Let me put it in the language
of a store forecasting demand week by week.

*(Walk the three rows in a storytelling register.)*

The **forget** gate says: "the promotion that ran three months ago no longer explains anything —
drop it." The gate closes toward zero, and that part of memory decays away.

The **input** gate says: "this week is Thanksgiving and the spike is enormous — record it." The
gate opens toward one, and the event is written into the cell.

The **output** gate says: "I'm holding a memory of last December, but this week is April — don't
act on it yet." The information stays in the cell state without contaminating this week's
prediction.

**[pause]**

*(Now the part worth emphasizing.)*

The output gate is the one people skip when they explain this, and it's the subtle one.

It separates **what the model remembers** from **what the model uses right now**. Those are
different things, and a vanilla RNN cannot make the distinction — it has one hidden state, which
has to serve both jobs at once. Anything an RNN stores is immediately in play, affecting the very
next prediction.

The LSTM can hold something in reserve. That's a genuine capability, and it's why these models
work well on language, where a fact mentioned in sentence one might not matter again until
sentence forty.

---

# ▶ SLIDE 13 — Why the Cell State Fixes the Gradient

Back to that plus sign.

*(Point at the picture. Read it right to left — that's the direction a gradient travels.)*

Top lane: the vanilla RNN. Every backward hop multiplies by W-h and by the slope of the tanh. The
red arrow gets thinner at every hop, and by the left edge almost nothing arrives.

Bottom lane: the LSTM's cell state. Look at how the old memory enters — it's **additive**. The new
information comes in from below, the green plus. And notice what is *not* happening on the way
from c-t minus one to c-t: no weight matrix, no tanh. Just the forget gate.

When the forget gate sits near one, c-t is approximately c-t minus one, plus whatever's new. And a
gradient flowing backward through that addition passes essentially **undiminished** — the arrow
stays thick.

Repeated multiplication shrinks. Repeated addition does not.

**[pause]**

*(Point at the key box. This is the callback.)*

And you have seen this fix before — last week.

It is the ResNet skip connection. y equals F of x, **plus x**. A related problem: a signal that
has to pass through many layers. Same solution: give the gradient an additive path home.

*(Beat.)*

Here's the detail I enjoy. The LSTM is from 1997. ResNet is from 2015. And the 1997 LSTM had no
forget gate at all — its memory path was c-t equals c-t minus one plus the new part, which is
*exactly* ResNet's identity. The forget gate came in 2000, from Gers, Schmidhuber and Cummins, and
made that path learnable. So the idea was invented for **time**, forgotten by the computer vision
community, and then rediscovered eighteen years later for **depth**.

Same mathematics, two fields, two decades apart. Which is a decent argument for understanding
mechanisms rather than memorizing architectures — the mechanism transferred, the architecture
didn't.

*(Point at the gray line at the bottom.)*

One honest footnote. "Near one" is a condition, not a promise. On our panel, the trained forget
gates sit around one half — median point five two — so the cell keeps only about seven percent
of a memory after four weeks. The model *chose* to forget. Hold that thought; Check 2 explains
why it was right to.

---

# ▶ SLIDE 14 — Section divider: In PyTorch, and Does It Work

Four lines of model code, and an honest measurement against everything else this course has
fitted.

---

# ▶ SLIDE 15 — LSTM for Forecasting in PyTorch

The good news: after all that machinery, the code is short. `nn.LSTM` gives you the entire thing —
all three gates, the cell state, the whole update.

You define the LSTM layer, you define a linear layer on top, and in `forward` you run the sequence
through and take the last time step.

*(Point at the shape note.)*

Two things that will cost you time if you don't know them.

`batch_first=True` gives you shape `(batch, seq_len, features)`. That is the **opposite** of what
`Conv1d` wanted last week, which was `(batch, channels, time)`. If you're moving code between the
two — which Homework 5 asks you to do — this is where you'll get a shape error.

And `out[:, -1, :]` is the hidden state after the last step. That's the summary of the whole
window, which is what we put the linear layer on.

*(Point at the comment.)* For the vanilla RNN — the one that topped our scoreboard, and the one
Homework 5 asks for — swap in `nn.RNN`. It returns `out, h`, not `out, (h, c)`: there is no cell
state. Copy the LSTM line unchanged and Python will complain about unpacking.

Everything from Lecture 7 still applies: Adam, weight decay, `zero_grad()`, `shuffle=False`, split
by time. Plus one addition — `clip_grad_norm_` with a max norm of one, to handle the exploding
gradient case from slide 8.

**[pause]**

I want to flag how little of the theory shows up in the code, because it cuts both ways.

Everything from the last three slides — the cell state, the three gates, the additive update, the
whole reason this architecture exists — is inside `nn.LSTM`. One line. You could use this model
having understood none of it.

That's the appeal of the tooling, and it's also the risk. If you don't know that gating solves
long-range forgetting, you have no way to predict in advance that it will do nothing on a
twenty-six-week window. You'd just fit it, see a worse number, and have no account of why.

The measurement on the next slide is only *interpretable* because we spent twenty minutes on the
mechanism.

---

# ▶ SLIDE 16 — Does It Work? The Scoreboard So Far

*(This is the first payoff. Walk the table, then land both paragraphs.)*

Same weekly panel, same twenty-six-week windows, same test block as Homework 4. Everything here is
directly comparable.

LASSO with forty-six engineered features: seven forty-four. XGBoost: seven eighty-one. Then the
**vanilla RNN** at eight forty-two, on four thousand five hundred parameters. Random forest: eight
ninety-nine. The **LSTM** at nine eighty-seven, on eighteen thousand parameters. The best CNN
variant at one thousand thirty-two. Seasonal naive at the bottom, twenty-one fifty-two.

**[pause]**

Two readings, and the first is genuinely good news for recurrence.

**Recurrence beats convolution decisively** — eight forty-two against one thousand thirty-two. And
it did that on **four raw channels**, with no lag engineering at all, against forty-six hand-built
features for LASSO. That is a real argument for reading a sequence in order, and it's the
strongest result deep learning gets on our data.

*(Beat.)*

And the second reading. **The LSTM lost to the vanilla RNN** — at a fixed thirty-epoch budget.

That is not a typo. Hold onto the qualifier, though, because the next four slides are about what
that line does and does not mean.

---

# ▶ SLIDE 17 — Why the Gated Model Lost at 30 Epochs

*(Walk the table.)*

Thirteen weeks: RNN eight seventy-eight, LSTM nine ninety. Twenty-six weeks: eight forty-two
against nine eighty-seven. Fifty-two weeks: eight-oh-five against nine ninety-eight.

The RNN wins at all three window lengths.

**[pause]**

Now, be careful about how much that buys us. Three windows rules out a fluke of one *window*. It
does not rule out much else — all three rows are the same thirty series, the same split, the same
recency structure. It is one finding measured three ways, not three independent tests. Say that
out loud to the room; it is the kind of distinction that separates a result from a slogan.

*(Beat.)*

The natural reading is: an LSTM solves a problem **this data does not have**. Gating preserves
information across *long* dependency chains — that is what it was invented for — and weekly demand
is dominated by the recent past. And the gates are not free: eighteen thousand parameters against
four and a half thousand.

*(Point at the warning box.)*

**[pause]**

But before we sign off on that, notice what we have *not* checked. Both models trained for thirty
epochs, because thirty is what the training function defaults to. Nobody showed that thirty epochs
is enough for **both** of them — and one of them has four times the parameters.

A margin measured at one training budget is a claim about that budget.

So the next two slides do the two things this table owes us: **was the comparison fair**, and
**what in the data explains it**.

---

# ▶ SLIDE 18 — Check 1: Was It a Fair Fight?

*(Chart up. Take this one slowly — it reverses the previous slide.)*

Same two models, twenty-six-week window. This time we vary the training budget and watch the
margin. Red is the RNN, blue is the LSTM, and the dashed line is the thirty epochs we used.

At ten epochs, the LSTM is **ahead** by ninety-seven. At thirty — our setting — the RNN is ahead
by a hundred and forty-five; that's the bracket. At sixty, ninety-eight. At a hundred, the gap is
**minus nine**.

**[pause]** — *let that sit.*

Read the shape of it. The gap is **largest at exactly the budget we chose**. Not near the largest:
the largest anywhere in the range we looked at.

*(Trace the two lines.)*

And look at the curves separately, because they are different shapes. The RNN is finished by about
epoch forty-five — eight twenty-eight, eight thirty-three, eight thirty. Flat. The LSTM is still
coming down at a hundred: nine eighty-seven, nine twenty-six, eight twenty-one. It has not turned
over. **We ran out of patience, not the model.**

**[pause]**

So the honest verdict is not "the LSTM loses." It is three things.

At a fixed thirty-epoch budget, the RNN wins by a wide margin. Given enough epochs, the two
**tie**. And the RNN gets to its best answer roughly **three times faster on a quarter of the
parameters** — which is a real advantage, and worth having, and is an *optimization* advantage
rather than a verdict about gating.

*(Beat.)*

I want to be direct about what happened here. The first version of this slide deck did not have
this check, and the previous slide's table was presented as an architecture result. It is not one.
It took varying a parameter nobody thinks of as part of the experiment to find that out — which is
the same lesson as the seeds in Lecture 7, one level up.

---

# ▶ SLIDE 19 — Check 2: What in the Data Explains It

*(Three panels. This is the transferable slide; take the panels left to right.)*

Second check. Gating pays off when a value has to survive many steps that would otherwise
overwrite it. Does this panel have anything like that? Three measurements say no, and the first
two are on this slide.

*(Point at the left panel.)*

**One. Most of the variance is not sequential at all.** That's all thirty series of the lab's
panel on one axis — ten stores, three categories. Look at what separates them. Foods sit up
between ten and thirty thousand units a week, household around five, hobbies around two.
The dashed line is the grand mean, about eight thousand. The big differences are *which series*,
not *which week*.

*(Point at the triangle.)*

The middle panel puts a number on it, and it is a right triangle — Pythagoras, for variances.
Every week's distance from the grand mean splits into two pieces: the series' own average minus
the grand mean, and that week minus the series' average. Square them and add over all eight
thousand three hundred rows, and the cross term drops out exactly, because deviations from a mean
sum to zero. So the squares add. The between-series side is about six thousand eight hundred
units, the within-series side about two thousand one hundred, and the total about seven thousand
one hundred. Between squared over total squared: **ninety-one percent.** Nine percent is within a
series over time.

So the main thing a model has to get right is the level of the series in front of it, and the
last few weeks tell you that. No memory mechanism required.

**[pause]**

*(Point at the right panel.)*

**Two. The series are smooth** — and this one is counter-intuitive, so take it slowly. That's the
autocorrelation of each series around its own mean, averaged over the thirty. Point eight three at
lag one, and still point four three at lag fifty-two. That *looks* like long memory — and history
does help: on the scoreboard the RNN improved from eight seventy-eight to eight oh five as the
window grew from thirteen weeks to fifty-two. But it arrives smoothly. Because neighboring weeks
are nearly the same number, no single value has to survive many steps untouched; the gap is filled
in continuously along the way.

Gating earns its keep when something must survive steps that would overwrite it. Here nothing
overwrites anything.

---

# ▶ SLIDE 20 — Check 2, Continued: Nothing Arrives Early

**Three. Nothing in the inputs arrives early.** SNAP days correlate about point one four with
units in the same week. Two and four weeks ahead you do see something — minus point one three,
plus point one two — but that's the calendar, not a warning. SNAP days fall on the same dates every
month, so this week's count tells you the count two and four weeks out, and those weeks have their
own same-week effect. It's the same plus point one four, echoed. Event days are flat everywhere,
including at zero. There is no **announcement** structure in this data — nothing shows up at time
*t* that has to be stored until *t* plus *k*. That is the textbook use for a forget gate and an
input gate, and this panel does not contain one instance of it.

*(Point at the key box.)*

**[pause]**

And this is the part to carry out of the lecture — not the answer, the three questions. Before you
reach for a gated model, or an attention model, on your own series: how much of my variance is
cross-sectional rather than sequential? Is my series smooth enough that recent values already
summarize the old ones? Does anything in my inputs *lead* the target?

If the answers look like this panel's, the extra machinery has nothing to do.

---

# ▶ SLIDE 21 — Section divider: Key Takeaways

What recurrence buys, what it costs, and what is still missing.

---

# ▶ SLIDE 22 — RNN and LSTM in Context

*(Table first, then the two paragraphs.)*

Four architectures, three questions. Does it see order? How much can it actually remember? What
does it cost?

The FFN on lags: no order, and it remembers whatever you engineered. The CNN: local order only,
about seven steps. The vanilla RNN: yes to order, roughly ten to thirty steps in practice — note
that's *practice*, not the theoretical unlimited. The LSTM: yes, and hundreds of steps, at four
times the cost.

An LSTM holds roughly four times the parameters of a vanilla RNN at the same hidden size, because
each of the three gates plus the candidate needs its own weight matrix. That's the price of the
memory. On a short window you pay it for nothing.

**[pause]**

If you go looking, you'll also meet the **GRU** — the gated recurrent unit. It merges the forget
and input gates into one and drops the separate cell state, so it sits between the two models on
this table: more gating than a vanilla RNN, fewer parameters than an LSTM.

On data like ours, where the gates aren't earning their keep anyway, a GRU often performs about
like an LSTM at lower cost. It's a reasonable thing to try in your final project if you go down
this road — and a reasonable thing to skip, given what the scoreboard says about this whole family
on this dataset.

**[pause]**

*(Now set up Part 2.)*

And here's what's still missing, because it motivates next week.

An LSTM reads strictly left to right, one step at a time. Two consequences. It **cannot be
parallelized** across the sequence — step t needs step t minus one finished, so a thousand-step
sequence takes a thousand sequential operations no matter how much hardware you own.

And everything it knows about step three has to survive being squeezed through **every** hidden
state between three and the end. The gates make that survival more likely, but the information is
still passing through a bottleneck at every step.

Part 2 removes both constraints, with a mechanism that lets every position look directly at every
other position in a single operation.

---

# ▶ SLIDE 23 — Lecture 9, Part 1: Key Takeaways

One. A recurrent network reads a sequence in order and carries a hidden state. Weights are shared
across time, so window length costs no parameters.

Two. Training is backpropagation through time — unroll the recurrence and it's a deep network
whose layers share weights. Repeated multiplication shrinks the gradient exponentially: our own
RNN keeps about a thousandth of a percent of it a year back.

Three. An LSTM adds a cell state updated **additively**. Addition preserves gradients where
multiplication destroys them — the ResNet fix, eighteen years earlier.

Four. Three gates: forget what's stale, input what's new, output what's relevant now.

Five. On our panel, both recurrent models beat every CNN variant, on four raw channels. And both
still lose to a forty-six-coefficient LASSO.

And six. The LSTM lost to the vanilla RNN at a thirty-epoch budget, at all three window lengths —
and tied by a hundred epochs, with the LSTM still improving. Gating solves long-range
forgetting; a twenty-six-week window doesn't suffer from it. So the gates bought four times the
parameters and nothing else. **Fit the architecture to the problem.**

**[pause]**

Next week: attention and Transformers. Reading the whole sequence at once.

---

# ▶ SLIDE 24 — References

*(Advance and close. No narration needed.)*
