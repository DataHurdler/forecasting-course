# Lecture 9 Part 2 — Recording Script

**ECON 8310: Business Forecasting · Transformers**

Deck: `Slides/Lecture09_Part2_Transformers.pdf` (20 pages) · Measured runtime: see the timing guide

---

## How to use this document

- **`▶ SLIDE n — Title`** marks where to advance. The number is the PDF page.
- *Italic parentheticals* are stage directions. **[pause]** means stop for a beat.
- Slide 12 lists three advantages; slide 16 shows each one being irrelevant here. That pairing is
  the argument — deliver 12 with genuine enthusiasm so 16 has something to knock down.
- The positional-encoding ablation on slide 16 is the one number students remember. Land it.

---

# ▶ SLIDE 1 — Title page

Lecture 9, Part 2: Transformers.

Last week we built a recurrent network and found something surprising — the plain RNN beat the
LSTM at every window length we tried. Today we build the architecture that displaced both of them
everywhere else in machine learning.

**[pause]**

I want to be honest about the shape of this hour, because it mirrors last week. We're going to
spend most of the lecture on a mechanism that is genuinely one of the most important ideas of the
last decade — attention — and then measure it on our data and find it ties the LSTM.

That's not a reason to skip it. Attention is why the tools you use every day exist. But the gap
between "reshaped an entire field" and "helps with weekly demand forecasting" is exactly the gap
this course exists to teach you to see in advance.

---

# ▶ SLIDE 2 — Lecture Outline

Five parts. What recurrence could not do — two specific constraints. Then attention itself, which
is a weighted lookup and much less mysterious than the formula makes it look. Then the encoder
block, which is attention plus three things you already know. Then the implementation and an
honest measurement. And finally where all six sequence architectures sit relative to each other.

---

# ▶ SLIDE 3 — Section divider: What Recurrence Could Not Do

Two constraints that are built into reading a sequence one step at a time.

---

# ▶ SLIDE 4 — Two Problems Built Into Recurrence

Part 1 left us with a working recurrent forecaster. Both of its remaining problems come from the
same source: it reads strictly left to right. So let's follow one week through it — the same week
last year, week t minus fifty-two — and watch how it reaches the forecast.

*(Point at the top row. Trace the crimson path from left to right.)*

This is the recurrence. Week t minus fifty-two enters the very first hidden state, h-one. Only the
last state, h-fifty-two, feeds the forecast. So everything the model learned about that week has to
ride through all fifty-two states, one update at a time.

That's the first problem: **the information bottleneck.** Each of those states is one vector of
sixty-four numbers, and at every update the new week gets squeezed in with everything that came
before. Fifty-two squeezes. An LSTM's gates make survival more likely — that's what they're for —
but the whole history still lives in one fixed-size vector.

And look at the label above the row: each update waits for the one before. That's the second
problem, **the sequential bottleneck.** Update thirty cannot start until update twenty-nine is
done. It's a hard constraint, not an implementation detail: however many GPUs you own, those
fifty-two updates run one after another.

**[pause]**

*(Point at the bottom row.)*

Now attention. Same fifty-two weeks. The forecast is read from the last position, week t minus
one, and that position scores every week in the window directly. Every gray arc is one of those
lookups. The crimson one is week t minus fifty-two: one step, not fifty-two. And all fifty-two
lookups are computed at the same time — nothing waits for anything.

*(Point at the key box.)*

So attention removes both problems at once.

One honest caveat, and it's in the box. "One step" is a statement about the wiring — how many
operations sit between two weeks. It says nothing about whether a trained model actually uses that
path. We'll look at what ours used on slide seven.

---

# ▶ SLIDE 5 — Section divider: The Attention Mechanism

A learned, weighted lookup — which is a more useful way to read it than the formula suggests.

---

# ▶ SLIDE 6 — Scaled Dot-Product Attention

*(Point at the formula, then immediately translate it with the picture.)*

Here's the formula. Attention of Q, K, V equals softmax of Q K-transpose over root d-k, times V.

That's the whole mechanism. Every Transformer in the world is that line, repeated.

Now let me translate it, because the formula makes it look more exotic than it is. Read it as a
**lookup with soft matching.** Each week emits a **query** — "what am I looking for?" Every week
also offers a **key** — "here's what I have" — and a **value** — "here's what I'd contribute if
you pick me."

*(Point at the left panel.)*

The picture is a two-dimensional cartoon of what one head does; ours works in sixteen dimensions,
but the geometry is the same. The red arrow is the query. The gray arrows are keys from five
different weeks. A dot product has a very simple picture: drop each key straight down onto the
query's line, and the score is how far along the line it lands, times the query's length. Keys
pointing the same way as the query land far out — big scores. A key at right angles lands at
zero. A key pointing backwards lands on the negative side.

*(Middle panel.)*

The softmax turns those five scores into weights. Two things are guaranteed: every weight is
positive, and they sum to one. Last week gets more than half; the same week last year, pointing
the wrong way, gets almost nothing.

*(Right panel.)*

And the output is the weighted average of the values. Here's the consequence that's worth
remembering: because the weights are positive and sum to one, the output always lands **inside**
the shape the values make. Attention blends what's in the window. It can't extrapolate beyond it.

Score, normalize, average. That's it.

**[pause]**

And it's worth noticing you have built something very close to this already, by hand. In Homework
4, a rolling mean is a weighted average over recent positions — with weights you fixed in advance,
equal across the window. A Fourier term is a weighted combination too, with weights set by a
formula.

Attention is the same *shape* of operation with the weights **learned**, and recomputed for every
input. That's the whole upgrade. Not a new kind of mathematics — a new source for the numbers.

*(Point at the root d-k.)*

Why divide by root d-k? Because dot products grow with dimension. In high dimensions the raw
scores get large, and large inputs push the softmax toward a one-hot distribution — all the weight
on one position — where its gradient is nearly zero and nothing trains. Dividing by root d-k keeps
the scores in a range where the softmax stays soft.

It's a numerical fix, not a conceptual one, but leave it out and the model doesn't learn.
---

# ▶ SLIDE 7 — What That Buys a Forecaster

Let me make that concrete for our problem.

Attention can put weight on **any** week in the window, in one step. A recurrent model needs
fifty-two successive state updates to reach the same week last year — that's the top row on slide
four — and we saw last week what survives fifty-two multiplications. A CNN needs a receptive field
that wide, which it doesn't have. Distance costs attention nothing. That's the headline.

But "can" is a claim about the architecture. So we looked at what ours actually *did*.

*(Point at the heatmap. Explain the layout before the content.)*

This is our trained model — the one from the lab. Each row is one attention head; there are four
heads in each of two layers. Each column is a week in the window, from fifty-two weeks back on the
left to last week on the right. The color is how much weight the forecasting position put on that
week, averaged over the whole test year. White is less than an even share, dark red is a lot.

And we trained it twice, with two different seeds — that's the left and right panels.

**[pause]**

Read it. Almost all of the dark red is in the last two to seven weeks. That's where the model
looks. A couple of heads spread their weight almost evenly — that flat, pale row.

*(Point at the blue box on the left edge, then the callout.)*

And the same week last year? In the first run, nothing — no head gives it more than about an even share.
In the second run, one head in the first layer put eleven percent of its weight there: six times an
even share. Same data, same code, different seed — and one run found the annual lag and the other
didn't.

**[pause]**

Two lessons in that. First, **the weights are learned, not specified.** You never told the model
that lag fifty-two matters. Contrast Lecture 6, where you built the `lag52` column by hand and the
LASSO penalty merely decided whether to keep it. Here the model decides which weeks to consult —
and, it turns out, it mostly decides on the recent past.

Second, a caution about reading pictures like this one. Attention weights are **inspectable**, which
is genuinely useful — compared with an LSTM's hidden state, sixty-four numbers with no
interpretation at all, you can see where the model looked. But where it looked is not the same as
what mattered. A high weight tells you where the model consulted, not why it decided what it
decided, and there's a real literature arguing about how much to read into them.

---

# ▶ SLIDE 8 — Multi-Head Attention

One attention operation produces one set of weights — one notion of what is relevant. That's
limiting, because different kinds of relevance coexist in the same sequence.

**Multi-head attention** runs h attention operations in parallel, on different learned projections
of the same input, then concatenates the results.

Each head can specialize — and the heatmap on the last slide shows ours did. Most of the heads
watch the last few weeks, each at slightly different lags. One or two spread their weight almost
evenly across the window. And in one of the two training runs, one head found the annual lag.
Nobody assigned those jobs; the heads divided the work among themselves.

**[pause]**

*(Point at the cost paragraph — this is the counterintuitive part.)*

And the cost is not what you'd guess.

You might expect eight heads to cost eight times one head. They don't. Each head works in a
**lower-dimensional subspace** — d-model divided by h. So eight heads of sixteen dimensions each
cost about the same as one head of one hundred twenty-eight.

You're not adding capacity. You're partitioning the capacity you already had, and letting
different partitions specialize. The diversity is close to free, which is why nobody uses a single
head.

Our model below uses four heads with d-model of sixty-four — four heads of sixteen dimensions.

---

# ▶ SLIDE 9 — Section divider: The Transformer Encoder

Attention, plus the three pieces that make it trainable.

---

# ▶ SLIDE 10 — The Encoder Block

A Transformer encoder layer is four components in a fixed arrangement, and the good news is that
you already know three of them.

*(Walk the diagram left to right.)*

Each week comes in with its position added — more on that on the next slide. Then **multi-head
attention**, in red, because it's the only new part: every week attends to every week.

Then a plus sign, and a gray line arching over the top. That's the **skip connection** from Lecture
8 — the input is added back to attention's output, giving the gradient an additive path home. Then
**LayerNorm**, which rescales each week's vector so the numbers stay in a sensible range.

Then a **feed-forward sublayer** — a small MLP applied to each week independently. That is
literally the network from Lecture 7, run at every week of the window.

And around it, **another skip and another LayerNorm**.

The bracket underneath says stack N of those; our model stacks two. Nothing in the block is new to
you except attention itself.

That is genuinely worth pausing on. The architecture that reorganized machine learning is one new
mechanism, wrapped in a residual connection from 2015 and a two-layer network from the 1980s.

Progress in this field often looks like that from the inside — not a wholesale reinvention, but
one new component slotted into scaffolding that already worked. It's a reason to learn components
rather than model names.

**[pause]**

*(Now set up the next slide. This is a deliberate cliff.)*

And one thing is conspicuously missing. It is not an oversight, and I want you to spot it before
I say it.

Look at every operation in that block. Attention: a weighted average over weeks.
Feed-forward: applied to each week independently. LayerNorm: per week. Skip: per week.

**Nothing in there depends on the *order* of the input.**
---

# ▶ SLIDE 11 — Positional Encoding: Attention Cannot Tell Time

Let's make that precise, because it's the most important slide in the lecture.

Attention computes a weighted average over positions. Averaging does not care about order.
Shuffle the input window, and every attention output is **unchanged**. The technical term is that
attention is **permutation-invariant**.

**[pause]**

For language that's fatal — "dog bites man" and "man bites dog" would be identical inputs. For
forecasting it's arguably worse. A model that cannot distinguish last week from thirty weeks ago
is not forecasting at all; it's computing a weighted average of an unordered bag of numbers.

So the fix is to add position information **to the input itself**, before attention ever sees it.

*(Point at the formula, then straight to the dials.)*

The original paper adds thirty-two pairs of numbers to each week: a sine and a cosine, each pair at
its own speed. Here's the way to picture it. A sine and a cosine of the same angle are just the two
coordinates of a point on a circle. So each pair is a **clock hand**, and as you move through the
window, week by week, every hand turns — each at its own speed.

*(Walk the four dials left to right.)*

The first hand is fast: one full turn every six weeks. It tells neighboring weeks apart easily —
but it comes back around, so on its own it can't tell this week from six weeks ago. The next turns
once every twenty weeks. The third, once every sixty-three — slow enough that every week of the
year sits at its own angle. And the slowest barely moves across a whole year.

Read together, like the hands of a real clock, they pin down every week uniquely. Fast hands
separate nearby weeks; slow hands separate distant ones.

*(Point at the square on the right.)*

And here's the property that makes it work for learning. Move any week forward by, say, three
weeks, and **every hand turns by its own fixed angle** — the same angle no matter which week you
started from. So how similar two weeks look depends only on how far apart they are. That's what
this square shows: week against week, colored by similarity, and it's the same color along every
diagonal. The model can learn "three weeks ago" as a single idea, wherever it falls in the window.

You can also learn the position vectors instead of fixing them; both are used in practice.

**[pause]**

We measure exactly what removing this costs at the end of the lecture, on slide sixteen. Hold the
question.
---

# ▶ SLIDE 12 — Why Transformers Took Over

*(Deliver this slide with real enthusiasm. Slide 16 needs something to argue with.)*

Three properties, in the order they actually mattered commercially.

**Parallelism.** *(Point at the figure.)* The gray strip on top is the recurrence: fifty-two steps,
shaded in the order they have to run. The red square is attention: every week scored against every
other week, fifty-two by fifty-two — two thousand seven hundred and four scores. And no score waits
for another; they can all be computed at once.

I want to be clear that this — more than accuracy — is why the architecture won. It could absorb
far more data and far more compute per unit of wall-clock time than an RNN. When your competitive
advantage is training on more data, the architecture that trains faster wins by default.

One thing the picture does *not* say. A cell counts an operation, not its cost — one recurrent step
is a sixty-four by sixty-four matrix product, one score is a sixteen-number dot product. So don't
read the square as "fifty times more work". Read it as "no waiting".

**Direct long-range paths.** Any position reaches any other in one step — the arcs from slide four.
Nothing has to survive fifty-two multiplications to arrive. Compare the ten-to-the-minus-eight we
computed last week.

**Scale.** The architecture kept getting better as parameters and data grew, well past the point
where earlier architectures plateaued. That property is the entire foundation of the last five
years of AI. BERT is an encoder stack used for understanding; GPT is a decoder stack used for
generation. Both are this block, repeated.

**[pause]**

*(Point back at the square, then the line under it.)*

And here's the price, which is real. The square's area is the cost: order T-squared in both time
and memory, for every head in every layer. Look at the dashed corner — a twenty-six-week window is
a quarter of the square, six hundred seventy-six scores. Double the window and the recurrence takes
twice the steps, but attention computes four times the scores. At T equals fifty-two, that's
irrelevant. At T equals ten thousand, it's a hundred million scores per head, per layer, and it is the central
engineering constraint of the entire field — there's a whole literature on approximating it.

*(Now plant the doubt.)*

Read that list again with a forecaster's eye, though.

Every single one of those advantages is about **large data and long sequences**.

Hold that against what we actually have: a fifty-two-week window, and five thousand one hundred
sixty training examples.

---

# ▶ SLIDE 13 — Section divider: Transformers for Time Series

The implementation, the measurement, and an honest reading of it.

---

# ▶ SLIDE 14 — Transformer Encoder in PyTorch

`nn.TransformerEncoderLayer` gives you the entire block — multi-head attention, both residuals,
both LayerNorms, the feed-forward sublayer. `nn.TransformerEncoder` stacks N of them. You add a
linear layer to project your features up to d-model, and another to project down to one output.

Note `batch_first=True` again, giving `(batch, seq_len, features)` — same as the LSTM, opposite of
`Conv1d`.

**[pause]**

*(This is the practical warning. Emphasize it.)*

And here is the one that will cost someone in this room a weekend.

`nn.TransformerEncoderLayer` gives you the whole block — **but not the positional encoding.**

You add that yourself. It's the `+ self.pe` on the forward pass. And if you forget it, PyTorch
does not complain. There's no error, no warning, no shape mismatch. The model trains perfectly
happily on an unordered bag of weeks and returns a forecast.

It's a **silent failure**, which by now you should recognize as the most dangerous kind. We'll
quantify it in two slides.

**[pause]**

Add it to the catalogue we've been building all semester. `KFold` on a time series. Scaling
outside the fold. A rolling feature you forgot to shift. `shuffle=True` on a DataLoader whose rows
are ordered in time. And now a Transformer with no positional encoding.

Not one of those raises an exception. Every one of them returns a plausible number. The only
defense is procedural — you check them because they're on a list, not because something broke.

*(Point at the muted note.)*

One aside: for multi-step forecasting you also need a causal mask, so position t can't attend to
positions after it. We're doing one-step-ahead, so we don't need it here — but if you extend this
in your project, that's the thing to look up.

---

# ▶ SLIDE 15 — Does It Work?

*(Walk the table. Same panel, same test block as everything since Homework 3, and every sequence
model reading fifty-two-week windows.)*

LASSO with forty-six engineered features: seven forty-four. XGBoost: seven eighty-one. Vanilla
RNN: eight-oh-five, on four and a half thousand parameters. Random forest: eight ninety-nine.
**Transformer encoder, two layers: nine ninety** — on sixty-seven thousand parameters. LSTM: nine
ninety-eight. Best CNN: one thousand thirty-two — that one from Lecture 8, on twenty-six-week
windows. Seasonal naive: twenty-one fifty-two.

**[pause]**

So: the Transformer beats the CNN, and it **ties the LSTM** — nine ninety against nine
ninety-eight, a difference far smaller than the run-to-run spread. Calling that a win in either
direction would be reading noise.

And it loses to a vanilla RNN holding **one-fifteenth** as many parameters. And to a
forty-six-coefficient linear model.

*(Beat.)*

That is the state-of-the-art architecture of the last decade, finishing fifth of eight on our
data. Next slide is why, and the reason is not that we implemented it badly.

**[pause]**

Look at the parameter column once before we move on, because the trend across the whole table is
the real story of these three lectures.

Four and a half thousand parameters bought eight-oh-five. Eighteen thousand bought nine
ninety-eight. Sixty-seven thousand bought nine ninety — a tie with the eighteen thousand. And
forty-six coefficients bought seven forty-four.

Model size and accuracy are running in **opposite directions** on this dataset, or at best
sideways. That is not a universal law — on a large enough corpus the ordering flips entirely, which
is precisely why these architectures exist. But on the data in front of you, more capacity has
never once bought a meaningfully better forecast.

---

# ▶ SLIDE 16 — Reading That Result Honestly

Go back to slide twelve — the three advantages — and take them one at a time against this problem.

*(Walk the table.)*

**Parallel computation.** Completely real. And completely irrelevant here: the RNN trains in
seconds either way. Parallelism matters when training time is your binding constraint. Ours is
about ninety seconds.

**Direct long-range paths.** Real, and worth very little when the signal is dominated by the
recent past. Attention's superpower is reaching week t minus fifty-two cheaply. Our data mostly
wants week t minus one.

**Scales with data.** True, and five thousand one hundred sixty training windows is simply not
the regime where that pays. Transformers overtake other architectures somewhere, but that
somewhere is orders of magnitude from here.

**[pause]**

*(Now the ablation. This is the number they'll remember.)*

But I don't want you to leave thinking the mechanism did nothing, because one part of it is doing
very visible work — and we can measure it.

Take the same model and remove the positional encoding. Now it has to forecast from an unordered
bag of fifty-two weeks.

*(Point at the two-row table.)*

With positional encoding: nine ninety. Without: **one thousand two hundred eighty-two.**

That's a penalty of **two hundred ninety-two RMSE** — which drops it below even the 1D CNN.

**[pause]**

That gap is the price of permutation invariance, and it's the reason the sinusoids exist. It also
tells you the model genuinely is using temporal position when you give it to it.

And remember: forgetting that one line produces no error message. It produces a model that is
twenty-nine percent worse and looks completely normal.

---

# ▶ SLIDE 17 — Section divider: Key Takeaways

Six architectures, one series, and what the ranking actually taught us.

---

# ▶ SLIDE 18 — Sequence Architecture Comparison

*(Table, then the two paragraphs.)*

Four architectures on three axes. Long range: the CNN sees its receptive field only, the RNN about
ten to thirty steps, the LSTM hundreds, the Transformer unlimited — at quadratic cost. Parallel:
CNN yes, RNN no, LSTM no, Transformer yes. Data needed: modest, modest, more, and much more.

**[pause]**

Choose by the **structure of your problem**, not by the recency of the architecture.

Short window, signal in recent lags? A plain RNN — or honestly, a linear model with good features,
which is what won here. Long sequences, abundant data, dependencies at arbitrary distance? A
Transformer, and nothing else comes close.

*(Point at the last paragraph.)*

That second case is language. Which is exactly why this architecture reshaped that field, and why
it has been much slower to displace gradient boosting in tabular forecasting — where, notably, the
M5 competition itself was won by ensembles of the methods you learned in Lectures 5 and 6.

Not by a Transformer. By gradient boosting.

**[pause]**

*(A closing thought worth offering.)*

Let me anticipate a reasonable objection: if this architecture loses here, why spend an hour on it?

Two answers. The first is that you now have a *reason* — you can say, before fitting anything,
that a fifty-two-week window with recent-lag signal and five thousand training examples is not the
regime where attention pays. That prediction-in-advance is worth far more than the fitted number.

The second is that most of you will encounter this architecture outside forecasting, in tools you
use daily. Understanding that it is a soft lookup with learned weights, that it is
permutation-invariant unless told otherwise, and that its advantages are advantages at scale —
that understanding transfers well beyond this course.

---

# ▶ SLIDE 19 — Lecture 9 Part 2: Key Takeaways

One. Attention is a learned soft lookup: score every query against every key, softmax the scores,
average the values. Every week in the window is one step away.

Two. The price is quadratic in sequence length — irrelevant at fifty-two steps, the central
constraint at ten thousand.

Three. Multi-head attention runs several notions of relevance in parallel, at roughly the cost of
one, by splitting the dimension rather than adding to it.

Four. An encoder block is attention plus residual plus LayerNorm plus a feed-forward sublayer.
Only attention is new — the residuals are ResNet, the sublayer is Lecture 7.

Five. Attention is **permutation-invariant**, so position has to be added explicitly. PyTorch does
not do it for you, and omitting it fails silently — two hundred ninety-two RMSE, no error message.

And six. On our panel the Transformer tied the LSTM, beat the CNN, and lost to a vanilla RNN with
one-fifteenth the parameters. Its advantages are advantages **at scale**, and this problem has
none of the scale.

**[pause]**

Next time we change the question entirely. For ten weeks every model has handed you one number per
period. Bayesian methods hand you a distribution — and that turns out to answer questions a point
forecast simply cannot.

---

# ▶ SLIDE 20 — References

*(Advance and close. No narration needed.)*
