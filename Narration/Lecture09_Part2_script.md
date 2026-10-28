# Lecture 9, Part 2 — Recording Script

**ECON 8310: Business Forecasting · Attention and Transformers**

Deck: `Slides/Lecture09_Part2_Transformers.pdf` (20 pages) · Measured runtime: see the timing guide

---

## How to use this document

- **`▶ SLIDE n — Title`** marks where to advance. The number is the PDF page.
- *Italic parentheticals* are stage directions. **[pause]** means stop for a beat.
- Slide 12 lists three advantages; slide 16 shows each one being irrelevant here. That pairing is
  the argument — deliver 12 with genuine enthusiasm so 16 has something to knock down.
- The positional-encoding ablation on slide 16 is the one number students remember. Land it.
- Written to [`STYLE_GUIDE.md`](STYLE_GUIDE.md): whole sentences, read exactly as written.

---

# ▶ SLIDE 1 — Title page

This is Lecture 9, Part 2, on attention and Transformers.

Last week we built a recurrent network and found something surprising. At the thirty-epoch
training budget we used, the plain RNN beat the LSTM at every window length we tried, although the
two tied once we trained them for a hundred epochs. Today we build the architecture that displaced
both of them almost everywhere else in machine learning.

**[pause]**

I want to be honest about the shape of this hour, because it mirrors last week. We'll spend most
of the lecture on attention, which is genuinely one of the most important ideas of the last
decade. Then we'll measure it on our data, and we'll find that it ties the LSTM.

That isn't a reason to skip it. Attention is why the tools you use every day exist. But there's a
gap between an idea that reshaped an entire field and an idea that helps with weekly demand
forecasting, and this course exists to teach you to see that gap before you fit anything.

---

# ▶ SLIDE 2 — Lecture Outline

We'll take this in five parts. First, the two things recurrence could not do, both of them built
into reading a sequence one step at a time. Then attention itself, which turns out to be a weighted
lookup and a lot less mysterious than its formula makes it look. Third, the encoder block, which is
attention plus three things you already know. Fourth, the implementation in PyTorch and an honest
measurement on our data. And last, where the four sequence architectures we've studied sit
relative to each other.

---

# ▶ SLIDE 3 — Section divider: What Recurrence Could Not Do

This first section is about two limits that come with reading a sequence one step at a time.

---

# ▶ SLIDE 4 — Two Problems Built Into Recurrence

Part 1, which was last week, left us with a recurrent forecaster that works. Both of the problems it still has come from
the same source: it reads strictly from left to right. So let's follow one week through it, the
same week last year, which is week t minus fifty-two, and watch how it reaches the forecast.

*(Point at the top row and trace the crimson path from left to right.)*

This top row is the recurrence. Week t minus fifty-two enters the very first hidden state, h-one.
Only the last state, h-fifty-two, feeds the forecast. So everything the model learned about that
week has to ride through all fifty-two states, one update at a time.

That's the first problem, and it's called the **information bottleneck**. Each of those states is a
single vector of sixty-four numbers, and at every update the new week gets squeezed in with
everything that came before it. That's fifty-two squeezes. An LSTM's gates make it more likely that
the information survives, which is what the gates are for, but the whole history still has to live
in one vector of fixed size.

Now look at the label above the row: each update waits for the one before it. That's the second
problem, the **sequential bottleneck**. Update thirty can't start until update twenty-nine is
finished. This is a hard constraint, not a detail of the implementation. However many GPUs you own,
those fifty-two updates run one after another.

**[pause]**

*(Point at the bottom row.)*

Now here's attention, on the same fifty-two weeks. The forecast is read from the last position,
week t minus one, and that position scores every week in the window directly. Each gray arc is one
of those lookups, and the crimson arc is week t minus fifty-two. It takes one step to get there,
not fifty-two. And all fifty-two lookups are computed at the same time, so nothing waits for
anything.

*(Point at the key box.)*

So attention removes both problems at once.

There's one honest caveat, and it's in the box. When I say "one step," I'm describing the wiring,
meaning how many operations sit between two weeks. It says nothing about whether a trained model
actually uses that path. We'll look at what our model used on slide seven.

---

# ▶ SLIDE 5 — Section divider: The Attention Mechanism

Attention is a learned, weighted lookup, and that turns out to be a more useful way to think about
it than the formula suggests.

---

# ▶ SLIDE 6 — Scaled Dot-Product Attention

*(Point at the formula, then translate it right away with the picture.)*

Here's the formula. Attention of Q, K and V equals the softmax of Q times K-transpose, divided by
the square root of d-k, all multiplied by V.

That one line is the whole mechanism. Every Transformer in the world is that line, repeated.

Let me translate it, because the formula makes it look more exotic than it is. The best way to read
it is as a **lookup with soft matching**. Each week sends out a query, which asks, "What am I looking
for?" Every week also offers a key, which says, "Here's what I have," and a value, which says,
"Here's what I'd contribute if you pick me."

Those aren't three different inputs. Q, K and V are three learned projections of the same weeks,
meaning three weight matrices applied to one window. And d-k is just the length of each key. In our
model it's sixteen per head.

*(Point at the left panel.)*

This picture is a two-dimensional cartoon of what one head does. Ours works in sixteen dimensions,
but the geometry is the same. The red arrow is the query, and the gray arrows are keys from five
different weeks. A dot product has a very simple picture. Drop each key straight down onto the
query's line, and the score is how far along the line it lands, times the length of the query. Keys
that point the same way as the query land far out, so they get big scores. A key at a right angle
lands at zero, and a key pointing backwards lands on the negative side.

*(Point at the middle panel.)*

The softmax turns those five scores into weights, and it guarantees two things: every weight is
positive, and the weights add up to one. Here last week gets more than half the weight, and the
same week last year, which points the wrong way, gets almost nothing.

*(Point at the right panel.)*

The output is the weighted average of the values. Here's the consequence worth remembering. Because
the weights are positive and add up to one, the output always lands **inside** the shape the values
make. Attention blends what's already in the window. It can't extrapolate beyond it.

So attention does three things: it scores, it normalizes, and it averages.

**[pause]**

It's worth noticing that you've already built something very close to this by hand. In Homework 4,
a rolling mean is a weighted average over recent weeks, with weights you fixed in advance, all equal
across the window. A Fourier term is a weighted combination too, with its weights set by a formula.

Attention is the same kind of operation, except that the weights are **learned**, and they're
recomputed for every input. That's the whole upgrade. It isn't a new kind of mathematics, just a new
source for the numbers.

*(Point at the square root of d-k.)*

So why divide by the square root of d-k? Because dot products grow with the number of dimensions.
In high dimensions the raw scores get large, and large inputs push the softmax toward putting all of
its weight on a single position. When that happens its gradient is nearly zero, and nothing trains.
Dividing by the square root of d-k keeps the scores in a range where the softmax stays soft.

It's a numerical fix rather than a conceptual one, but if you leave it out, the model doesn't learn.

---

# ▶ SLIDE 7 — What That Buys a Forecaster

Let me make that concrete for our problem.

Attention can put weight on any week in the window, in a single step. A recurrent model needs
fifty-two successive state updates to reach the same week last year. That's the top row on slide
four, and last week we saw how little survives fifty-two multiplications. A CNN would need a
receptive field that wide, and ours doesn't have one. For attention, distance costs nothing, and
that's the headline.

But "can" is a claim about the architecture, so we looked at what our model actually did.

*(Point at the heatmap. Explain the layout before the content.)*

This is our trained model, the one from the lab. Each row is one attention head, and there are four
heads in each of two layers. Each column is a week in the window, running from fifty-two weeks back
on the left to last week on the right. The color shows how much weight the forecasting position put
on that week, averaged over the whole test year. White means less than an even share, and dark red
means a lot.

We trained it twice, with two different seeds, and those are the left and right panels.

**[pause]**

Now, read it. Almost all of the dark red sits in the last two to seven weeks, so that's where the
model looks. In each run, one or two heads spread their weight almost evenly across the window, and
those are the flat, pale rows.

*(Point at the blue box on the left edge, then at the callout.)*

What about the same week last year? In the first run, no head gives it more than about an even
share. In the second run, one head in the first layer put eleven percent of its weight there, which
is about six times an even share. It was the same data and the same code with a different seed, and
one run found the annual lag while the other didn't.

**[pause]**

There are two lessons in that. The first is that the weights are **learned, not specified**. You
never told the model that lag fifty-two matters. Compare that with Lecture 6, where you built the
lag fifty-two column by hand and the LASSO penalty only decided whether to keep it. Here the model
decides which weeks to consult, and it turns out that it mostly picks the recent past.

The second lesson is a caution about reading pictures like this one. Attention weights are
inspectable, and that's genuinely useful. An LSTM's hidden state is sixty-four numbers with no
interpretation at all, while here you can see where the model looked. But where it looked isn't the
same as what mattered. A high weight tells you which weeks the model consulted, not why it decided
what it decided, and researchers still argue about how much to read into these weights.

---

# ▶ SLIDE 8 — Multi-Head Attention

One attention operation produces one set of weights, which means one notion of what's relevant.
That's limiting, because different kinds of relevance exist side by side in the same sequence.

**Multi-head attention** runs h attention operations in parallel, each on its own learned projection
of the same input, and then concatenates the results.

Each head can specialize, and the heatmap on the last slide shows that ours did. Most of the heads
watch the last few weeks, each at slightly different lags. One or two spread their weight almost
evenly across the window. And in one of the two training runs, one head found the annual lag.
Nobody assigned those jobs. The heads divided the work among themselves.

**[pause]**

*(Point at the paragraph about cost. This is the counterintuitive part.)*

The cost isn't what you'd guess. You might expect eight heads to cost eight times as much as one,
but they don't. Each head works in a smaller space, with d-model divided by h dimensions. So eight
heads of sixteen dimensions each cost about the same as one head of a hundred and twenty-eight.

In other words, you're not adding capacity. You're dividing up the capacity you already had and
letting each part specialize. That variety comes almost for free, which is why nobody uses a single
head.

The note at the bottom gives our model's setting: four heads with a d-model of sixty-four, so four
heads of sixteen dimensions each.

---

# ▶ SLIDE 9 — Section divider: The Transformer Encoder

Next comes the encoder, which is attention plus the three pieces that make it trainable.

---

# ▶ SLIDE 10 — The Encoder Block

A Transformer encoder layer is four components in a fixed arrangement, and the good news is that
you already know three of them.

*(Walk the diagram from left to right.)*

Each week comes in with its position added, and I'll say more about that on the next slide. Next
comes **multi-head attention**, drawn in red because it's the only new part. Every week attends to
every week.

Then there's a plus sign, with a gray line arching over the top. That's the skip connection from
Lecture 8. The input is added back to attention's output, which gives the gradient an additive path
home. After that comes LayerNorm, which rescales each week's vector so the numbers stay in a
sensible range.

Then there's a feed-forward sublayer, a small network applied to each week on its own. That's
literally the network from Lecture 7, run at every week of the window. And around it there's another
skip connection and another LayerNorm.

The bracket underneath says to stack N of these blocks, and our model stacks two. Nothing in the
block is new to you except attention itself.

That's worth pausing on. The architecture that reorganized machine learning is one new mechanism,
wrapped in a residual connection from twenty fifteen and a two-layer network from the nineteen
eighties.

From the inside, progress in this field often looks like that. It's rarely a complete reinvention.
More often, one new component gets slotted into scaffolding that already worked. That's a good
reason to learn components rather than model names.

**[pause]**

*(Set up the next slide. This is a deliberate cliffhanger.)*

And one thing is conspicuously missing. It isn't an oversight, and I'd like you to spot it before I
say it.

Look at every operation in that block. Attention takes a weighted average over the weeks. The
feed-forward sublayer works on each week separately. So does LayerNorm, and so does the skip
connection.

**Nothing in there depends on the order of the input.**

---

# ▶ SLIDE 11 — Positional Encoding: Attention Cannot Tell Time

Let's make that precise, because this is the most important slide in the lecture.

Attention computes a weighted average over positions, and an average doesn't care about order. If
you shuffle the input window, each week's output simply moves with its week, and nothing else about
it changes. The technical term is that attention is **permutation-equivariant**: rearrange the input,
and the outputs are rearranged the same way. Each week sees the others only as an unordered bag.

**[pause]**

For language, that's fatal. "Dog bites man" and "man bites dog" contain exactly the same bag of
words. For forecasting it's arguably worse. Our model reads its forecast from the last position, so
it does know which week is most recent. But it can't tell two weeks ago from thirty weeks ago, and a
model that can't do that isn't really forecasting. It's taking a weighted average of an unordered
bag of numbers.

So the fix is to add information about position to the input itself, before attention ever sees it.

*(Point at the formula, then go straight to the dials.)*

Our model adds thirty-two pairs of numbers to each week, a sine and a cosine, with each pair turning
at its own speed. Here's a way to picture it. The sine and cosine of the same angle are just the two
coordinates of a point on a circle. So each pair is a **clock hand**, and as you move through the
window, week by week, every hand turns at its own speed.

*(Walk the four dials from left to right.)*

The first hand is fast. It makes one full turn every six weeks, so it tells neighboring weeks apart
easily. But it comes back around, so on its own it can't tell this week from six weeks ago. The next
hand turns once every twenty weeks. The third turns once every sixty-three weeks, which is slow
enough that every week of the year sits at its own angle. And the slowest hand barely moves across a
whole year.

Read together, like the hands of a real clock, they pin down every week uniquely. The fast hands
separate nearby weeks, and the slow hands separate distant ones.

*(Point at the square on the right.)*

Here's the property that makes this work for learning. Move any week forward by, say, three weeks,
and every hand turns by its own fixed angle. That angle is the same no matter which week you started
from, so how similar two weeks look depends only on how far apart they are. That's what this square
shows. It plots week against week, colored by similarity, and the color is the same all along every
diagonal. So the model can learn "three weeks ago" as a single idea, wherever it falls in the window.

You can also learn the position vectors instead of fixing them in advance. Both approaches are used
in practice.

**[pause]**

We'll measure exactly what removing this costs at the end of the lecture, on slide sixteen, so hold
that question until then.

---

# ▶ SLIDE 12 — Why Transformers Took Over

*(Deliver this slide with real enthusiasm. Slide 16 needs something to argue with.)*

There are three reasons the Transformer took over, and I'll take them in the order they actually
mattered commercially.

The first is **parallelism**.

*(Point at the figure.)*

The gray strip on top is the recurrence: fifty-two steps, shaded in the order they have to run. The
red square is attention. Every week is scored against every other week, fifty-two by fifty-two,
which comes to two thousand seven hundred and four scores. None of those scores waits for another,
so they can all be computed at once.

I want to be clear that this, more than accuracy, is why the architecture won. It could absorb far
more data and far more computing power per hour of training than an RNN could. When your
competitive advantage is training on more data, the architecture that trains faster wins by default.

There's one thing the picture doesn't say. A cell counts an operation, not what it costs. One
recurrent step is a sixty-four by sixty-four matrix product, while one score is a dot product of
sixteen numbers. So don't read the square as fifty times more work. Read it as no waiting.

The second reason is **direct long-range paths**. Any position reaches any other in one step, which
is what the arcs on slide four showed. Nothing has to survive fifty-two multiplications to arrive.
Compare that with the ten to the minus eight we computed last week.

The third reason is **scale**. The architecture kept getting better as parameters and data grew, well
past the point where earlier architectures leveled off. That property is the foundation of the last
five years of AI. BERT is a stack of encoders used for understanding text, and GPT is a stack of
decoders used for generating it. Both are this block, repeated.

**[pause]**

*(Point back at the square, then at the line under it.)*

Here's the price, and it's real. The area of the square is the cost: it grows with T squared, in both
time and memory, for every head in every layer. Look at the dashed corner. A twenty-six-week window
is a quarter of the square, or six hundred seventy-six scores. Double the window, and the recurrence
takes twice as many steps, but attention computes four times as many scores. At T equals fifty-two,
that doesn't matter. At T equals ten thousand, it's a hundred million scores per head, per layer, and
it's the central engineering constraint of the whole field. There's a large literature on
approximating it.

*(Now plant the doubt.)*

Now read that list again with a forecaster's eye. Every one of those advantages is about **large
data and long sequences**. Hold that against what we actually have: a fifty-two-week window and five
thousand one hundred sixty training windows.

---

# ▶ SLIDE 13 — Section divider: Transformers for Time Series

This section covers the implementation, the measurement, and an honest reading of the result.

---

# ▶ SLIDE 14 — Transformer Encoder in PyTorch

PyTorch's `nn.TransformerEncoderLayer` gives you the entire block: multi-head attention, both skip
connections, both LayerNorms, and the feed-forward sublayer. Then `nn.TransformerEncoder` stacks N
of them. You add one linear layer to project your features up to d-model, and another to project
down to a single output.

Notice `batch_first=True` again. It puts the batch first, then the sequence length, then the
features, which is the same order as the LSTM and the opposite of `Conv1d`.

**[pause]**

*(This is the practical warning. Emphasize it.)*

Here's the mistake that will cost someone in this room a weekend. `nn.TransformerEncoderLayer` gives
you the whole block, **but not the positional encoding**.

You add that yourself. It's the term that adds `self.pe` in the forward pass. And if you forget it,
PyTorch doesn't complain. There's no error, no warning, and no shape mismatch. The model trains quite
happily, still knowing which week is last and seeing the rest as an unordered bag, and it returns a
forecast.

That's a **silent failure**, and by now you should recognize it as the most dangerous kind. We'll put
a number on it in two slides.

**[pause]**

Add it to the catalogue we've been building all semester: using `KFold` on a time series, scaling
outside the fold, forgetting to shift a rolling feature, shuffling rows before the time split, and
now a Transformer with no positional encoding.

Not one of those raises an exception, and every one of them returns a plausible number. The only
defense is a procedure. You check for them because they're on a list, not because something broke.

*(Point at the note at the bottom.)*

One aside. For multi-step forecasting you also need a causal mask, so that position t can't attend
to the positions after it. We're forecasting one step ahead, so we don't need it here. But if you
extend this model in your project, that's the thing to look up.

---

# ▶ SLIDE 15 — Does It Work?

*(Walk the table. It's the same panel and the same test block as everything since Homework 3, and
the recurrent and attention models all read fifty-two-week windows.)*

Let's go down the table. LASSO with forty-six engineered features scores seven forty-four. XGBoost
scores seven eighty-one. The vanilla RNN scores eight-oh-five, with about four and a half thousand
parameters. The random forest scores eight ninety-nine. The **two-layer Transformer encoder scores
nine ninety**, with about sixty-seven thousand parameters. The LSTM scores nine ninety-eight. The best
CNN, which comes from Lecture 8 and reads twenty-six-week windows, scores one thousand thirty-two.
And seasonal naive scores twenty-one fifty-two.

Notice that the recurrent rows aren't the ones you saw last week. Last week's headline numbers,
eight forty-two and nine eighty-seven, were for twenty-six-week windows. These are the
fifty-two-week runs, so the RNN, the LSTM and the Transformer all read the same window.

**[pause]**

So the Transformer beats the CNN, and it **ties the LSTM**: nine ninety against nine ninety-eight.
That difference is far smaller than the spread from one training run to the next, so calling it a
win in either direction would be reading noise.

It also loses to a vanilla RNN with **one-fifteenth** as many parameters, and to a linear model with
forty-six coefficients.

That's the state-of-the-art architecture of the last decade, finishing fifth out of eight on our
data. The next slide explains why, and the reason isn't that we implemented it badly.

**[pause]**

Look at the parameter column once before we move on, because the trend across the whole table is the
real story of these three lectures.

About four and a half thousand parameters bought eight-oh-five. About eighteen thousand bought nine
ninety-eight. Sixty-seven thousand bought nine ninety, which ties the eighteen thousand. And
forty-six coefficients bought seven forty-four.

On this dataset, model size and accuracy run in **opposite directions**, or at best sideways. That
isn't a universal law. On a large enough body of data the ordering flips entirely, which is exactly
why these architectures exist. But in this table, adding capacity never bought a meaningfully better
forecast.

---

# ▶ SLIDE 16 — Reading That Result Honestly

Go back to slide twelve, with its three advantages, and let's take them one at a time against this
problem.

*(Walk the table.)*

Parallel computation is completely real, and here it's completely irrelevant. The RNN trains in
seconds either way. Parallelism matters when training time is what limits you, and here training
takes about ninety seconds.

Direct long-range paths are real too, but they're worth very little when the signal is dominated by
the recent past. Attention's superpower is reaching week t minus fifty-two cheaply, and our data
mostly wants week t minus one.

Scaling with data is also true, but five thousand one hundred sixty training windows is simply not
the regime where it pays. Transformers do overtake other architectures at some point, but that point
is orders of magnitude away from here.

**[pause]**

*(Now the ablation. This is the number they'll remember.)*

But I don't want you to leave thinking the mechanism did nothing, because one part of it is doing
very visible work, and we can measure it.

Take the same model and remove the positional encoding. It still knows last week, because that's
where it reads out its forecast, but the other fifty-one weeks become an unordered bag.

*(Point at the two-row table.)*

With positional encoding, the error is nine ninety. Without it, the error is **one thousand two
hundred eighty-two**.

That's a penalty of **two hundred ninety-two** in RMSE, which leaves it worse than even the 1D CNN.

**[pause]**

That gap is what the order of those fifty-one weeks is worth, and it's the reason the sine and cosine
terms exist. It also tells you that the model genuinely uses position in time when you give it that
information.

And remember that forgetting that one line produces no error message. It produces a model that's
twenty-nine percent worse and looks completely normal.

---

# ▶ SLIDE 17 — Section divider: Key Takeaways

To close, we'll put the four sequence architectures side by side and ask what the ranking actually
taught us.

---

# ▶ SLIDE 18 — Sequence Architecture Comparison

*(Walk the table, then the two paragraphs.)*

This table compares four architectures on three questions. The first is how far back each one can
reach. The CNN sees only its receptive field, the RNN about ten to thirty steps, and the LSTM
hundreds of steps. The Transformer has no limit, but its cost grows with the square of the window.
The second question is whether it can compute in parallel. The CNN can, the RNN and the LSTM can't,
and the Transformer can. The third is how much data it needs: modest amounts for the CNN and the
RNN, more for the LSTM, and much more for the Transformer.

**[pause]**

So choose by the **structure of your problem**, not by how recent the architecture is.

If the window is short and the signal sits in recent lags, use a plain RNN, or honestly, a linear
model with good features, which is what won here. If you have long sequences, abundant data and
dependencies at any distance, use a Transformer, and nothing else comes close.

*(Point at the last paragraph.)*

That second case is language. That's exactly why this architecture reshaped that field, and why it
has been much slower to displace gradient boosting in tabular forecasting. In fact, the M5
competition itself was won by gradient-boosted trees, the method you learned in Lecture 5. It wasn't
won by a Transformer.

**[pause]**

*(A closing thought worth offering.)*

Let me anticipate a reasonable objection. If this architecture loses here, why spend an hour on it?

I have two answers. The first is that you now have a reason. Before fitting anything, you can say
that a fifty-two-week window, a signal in the recent lags and about five thousand training windows
are not the conditions where attention pays off. Being able to predict that in advance is worth far
more than the fitted number.

The second answer is that most of you will meet this architecture outside forecasting, in tools you
use every day. Knowing that it's a soft lookup with learned weights, that it's blind to order unless
you tell it otherwise, and that its advantages are advantages at scale will serve you well beyond
this course.

---

# ▶ SLIDE 19 — Lecture 9, Part 2: Key Takeaways

Let me pull together the main points.

First, attention is a learned soft lookup. It scores every query against every key, turns the scores
into weights with a softmax, and averages the values. Every week in the window is one step away.

Second, its price grows with the square of the sequence length. That doesn't matter at fifty-two
steps, and it's the central constraint at ten thousand.

Third, multi-head attention runs several notions of relevance in parallel for roughly the cost of
one, because it splits the dimensions rather than adding to them.

Fourth, an encoder block is attention plus skip connections, LayerNorm and a feed-forward sublayer.
Only attention is new. The skip connections come from ResNet, and the sublayer is the network from
Lecture 7.

Fifth, attention **ignores order**, so position has to be added explicitly. PyTorch doesn't do it for
you, and leaving it out fails silently. It costs two hundred ninety-two in RMSE without a single
error message.

And last, on our panel the Transformer tied the LSTM, beat the CNN, and lost to a vanilla RNN with
one-fifteenth as many parameters. Its advantages are advantages **at scale**, and this problem has
none of the scale.

**[pause]**

Next time we change the question entirely. For ten weeks, every model has handed you one number per
period. Bayesian methods hand you a whole distribution, and that turns out to answer questions a
single-number forecast simply can't.

---

# ▶ SLIDE 20 — References

*(Advance and close. No narration is needed.)*
