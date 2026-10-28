# Lecture 9, Part 1 — Recording Script

**ECON 8310: Business Forecasting · RNNs and LSTMs**

Deck: `Slides/Lecture09_Part1_RNN_LSTM.pdf` (24 pages) · Measured runtime: see the timing guide

---

## How to use this document

- **`▶ SLIDE n — Title`** marks where to advance. The number is the PDF page.
- *Italic parentheticals* are stage directions. **[pause]** means stop for a beat.
- Slides 16 and 17 carry the lecture's real payload: the LSTM loses to the vanilla RNN, at all
  three window lengths, at thirty epochs (they tie by a hundred, which Checks 1 and 2 teach). Deliver that plainly. Students expect the fancier model to win, and the
  fact that it doesn't is the most useful thing in the hour.

---

# ▶ SLIDE 1 — Title page

This is Lecture 9, Part 1, on recurrent neural networks and LSTMs.

For the last two weeks, in Lecture 7 on feedforward networks and Lecture 8 on convolutional
networks, we've been building models that take a fixed block of numbers and produce a forecast.
Last week's CNN was the closest we've come to a model that understands time, and even its filters
only see about seven weeks back.

Today we get the architecture that was designed for sequences from the beginning. It reads the data
in order, one step at a time, and it carries something forward as it goes.

**[pause]**

Then we're going to measure it against everything else in the course, and we'll find that the more
sophisticated of the two models we build today comes second. That result isn't a disappointment.
It's the lecture.

---

# ▶ SLIDE 2 — Lecture Outline

We'll take this in five parts. First, exactly where Lectures 7 and 8 ran out of road, because the
limitation is specific and it's worth naming. Second, the vanilla RNN, which is one equation applied
over and over, and why that equation fails on long dependencies. That failure turns out to be a
mathematical fact rather than a tuning problem. Third, the LSTM, which was built to fix exactly that.
Fourth, the code in PyTorch, and an honest scoreboard. And last, the key takeaways: what recurrence
buys, what it costs, and what's still missing.

---

# ▶ SLIDE 3 — Section divider: Where the Last Two Lectures Ran Out

We start with the last two architectures we built, which fail to see time in two different ways.

---

# ▶ SLIDE 4 — Two Architectures That Cannot See Time

Let's be precise about what's missing, because saying a model doesn't understand time is too vague
to act on.

Lecture 7's feedforward network, from two weeks ago, treats each row independently. Everything it
knows about *when* something happened, you encoded by hand as a lag column. Shuffle the feature
columns, and the model is unchanged. Time isn't in the architecture. It's in your feature
engineering.

Lecture 8's CNN, from last week, does better, because it genuinely sees local shape. But each
filter's **receptive field** is fixed by the architecture we chose. Our two small kernels, one five
weeks wide and one three weeks wide, reach about seven weeks back. That was our choice, not a law of
convolution. More layers, or dilated filters that skip over inputs, would reach further. But no small
stack of plain filters reaches an annual lag.

Worse, the pooling layer deliberately threw away *where* in the window each pattern occurred, and we
measured that costing about five hundred in RMSE.

*(Point at the key box.)*

**[pause]**

Here's the assumption both of them share, and it's the one we're breaking today.

Both treat the input as a **fixed-size block**, to be processed all at once. Neither has any notion
of reading a sequence in order and carrying something forward from one step to the next.

That's the gap, and everything today is about closing it.

**[pause]**

It's worth noticing that we've been compensating for this gap all semester, and quite successfully.
Lag columns are a workaround. So are rolling means, and so are the Fourier terms in Homework 4. Every
one of them is you, the analyst, telling a memoryless model something about time that it can't work
out for itself.

And they work well. LASSO on forty-six of those workarounds is still leading our scoreboard. So this
isn't a story where the old approach was broken. It's a story about whether the model can *discover*
the temporal structure instead of being handed it, and that's the question today puts to the test.

---

# ▶ SLIDE 5 — The Recurrent Idea: Carry State Forward

The recurrent idea is simple enough to state in one sentence, and I'd like you to have it before we
see any equations.

A recurrent network reads the sequence one step at a time, and it keeps a **hidden state**, which
we'll call h-t. The hidden state is a running summary of everything the network has seen so far.

At each step it does two things. It updates that summary, using both the new observation and the
previous summary. And it can also emit an output, if we want one.

The crucial point is that the **same weights** are used at every step. That's weight sharing again,
but this time it runs along time rather than along space, which is where the CNN shared its weights.

**[pause]**

Now here's the consequence, and it's a big claim. The slide calls it the promise.

There's no fixed window. In principle, h-t encodes the *entire* history, every observation from step
one to step t. The architecture puts no limit at all on how far back the model can look. Compare
that with the CNN's seven weeks, which were baked in by the filter sizes.

*(Then undercut it.)*

**[pause]**

But all of that is true only **in principle**.

Most of this lecture is about the gap between that promise and what a recurrent network actually
delivers, and then about the architecture that was invented specifically to close it.

---

# ▶ SLIDE 6 — Section divider: The Vanilla RNN

The next section is the vanilla RNN. It's one equation, applied over and over, and that turns out to
be both the whole idea and the whole problem.

---

# ▶ SLIDE 7 — Vanilla RNN: One Equation, Repeated

*(Point at the definition box.)*

Here's the recurrence. The new hidden state, h-t, is the tanh of three terms added together: W-h
times the old hidden state, plus W-x times the new input, plus a bias. And the output is a linear
layer on top of the hidden state.

In words, the new summary is a squashed mix of the old summary and the new observation. That's the
entire model, and everything else today follows from that one line.

The three weight matrices are the *same* at every time step. There isn't one set per step. There's
one set, reused.

*(Point at the picture under the box.)*

Across the top are twenty-six real weeks of Walmart food sales. It's the same window you saw last
week in Lecture 8, the run-up to Christmas twenty fifteen. Each week drops into a cell. All
twenty-six cells are drawn the same because they **are** the same cell, with one set of weights
reused at every step. The little arrows along the row are the hidden state being handed forward from
week to week. And only the last one, h-twenty-six, goes into the linear layer that makes next week's
forecast.

**[pause]**

Two things follow from that sharing, and both are genuinely nice.

First, a twenty-six-week window costs no more parameters than a five-week one. Compare that with the
CNN, where a longer window meant more filters or more layers. Here, the length of the window is free.

Second, the same trained network handles sequences of *different* lengths without any changes. You
can train it on twenty-six weeks and run it on forty, and nothing in the architecture cares.

As the line under the picture says, our RNN holds about **four and a half thousand parameters**.
Lecture 7's small feedforward network held twenty-two thousand six hundred fifty-seven. So this is a
much smaller model.

There's one practical note. For forecasting we usually want a single number, not a sequence of
outputs. So we take the **final** hidden state, which is the summary after reading the whole window,
and we put one linear layer on top of it. That's the pattern you'll write in Homework 5.

**[pause]**

A word on the hidden size, since it's the main thing you'll tune. It's how wide that running summary
is, which means how much the model is allowed to carry forward. Sixty-four is a sensible default for
our data. If it's too small, the summary can't hold enough to be useful. If it's too large, you're
back in the overfitting regime from Lecture 7, with far more parameters than observations.

And notice that the tanh in the recurrence isn't decorative. It squashes the state to between minus
one and one at every step, which keeps the running summary from growing without bound as the
sequence gets longer. Take it out, and the state can drift off to enormous values. It's also, as
we're about to see, part of why the gradient dies.

---

# ▶ SLIDE 8 — Training It: Backpropagation Through Time

Training an RNN is ordinary backpropagation applied to an unusual graph, and the trick is to see the
graph correctly.

**Unroll** the recurrence across the window, and twenty-six time steps become a twenty-six-layer
feedforward network, with one layer per step and every layer sharing the same weights.

Once you see it that way, there's nothing new. Gradients flow backward from the loss at the final
step all the way to step one. Because W-h appears at *every* step, its gradient is a **sum** of
contributions from all twenty-six positions.

*(Point at the warning box, and slow down.)*

**[pause]**

And this is where it breaks.

To propagate a gradient back through k steps, you multiply it by W-h and by the slope of the tanh,
k times over. That slope is never more than one, which is what the right-hand panel of the
activation figure in Lecture 7 showed. If the factor you multiply by is below one, the gradient
shrinks **exponentially** as it travels. If it's above one, it can explode.

Either way, the long-range signal doesn't survive the trip.

**[pause]**

The exploding case has an easy fix. The function `clip_grad_norm_` caps the size of the update, and
we use it in every recurrent model in this course. It's one line, and the problem is gone.

The vanishing case doesn't have an easy fix. That's what the next slide puts numbers on, and it's
what the whole second half of the lecture is about.

---

# ▶ SLIDE 9 — Why a Vanilla RNN Forgets

Let's put numbers on it, because saying that the gradient shrinks isn't vivid enough to change
anyone's behavior.

Suppose each backward step multiplies the gradient by a factor of zero point seven. That's an
unremarkable value for a tanh network, nothing pathological, and it's the dashed gray line.

*(Walk the dashed line slowly.)*

Ten steps back, the gradient is zero point zero two eight, which is barely there.

At twenty-six steps, which is our window length, it's about **one ten-thousandth**. It's effectively
gone.

A year back, it's around ten to the minus eight, which is invisible. No optimizer, no learning rate
and no amount of training can recover a signal that small.

**[pause]**

Now notice the *shape*. The vertical axis is on a log scale, and on a log scale, the same factor at
every step draws a **straight line**. The steepness of the line *is* the factor. That's the whole
mechanism in one picture: repeated multiplication.

*(Point at the red line.)*

The red line isn't an example. It's our own RNN, the lab's model, trained on fifty-two-week
windows, and we measured how much gradient actually reaches each step back. Past the first few
steps, it keeps more than the example does, about zero point seven nine per step. Before training it
kept zero point five five, so training did stretch its memory. But it's still a straight line down.
We measured the survival directly, rather than multiplying out the slope, because the first few
steps would throw that calculation off. About four tenths of a percent of the gradient survives to
twenty-six weeks back. By the same week last year, it's one thousandth of a percent.

So the promise from slide five fails in practice. A vanilla RNN *can*, in principle, encode an
arbitrarily long history. But the gradient that would *teach* it to do that has vanished long before
it reaches that far back.

In practice these networks learn dependencies of roughly ten steps, not fifty-two.

*(Point at the citation.)*

**[pause]**

I want to stress one thing about the result from Bengio and colleagues. This isn't an optimization
bug that better tuning fixes. They **proved** that it's built into the architecture. That's why the
fix had to be a different architecture, and it's why the next section exists.

---

# ▶ SLIDE 10 — Section divider: LSTM — Long Short-Term Memory

The next section is the LSTM, which stands for long short-term memory. The idea is to add a memory
that information passes through, rather than one that's recomputed at every step.

---

# ▶ SLIDE 11 — LSTM: A Cell State and Three Gates

An LSTM carries **two** things forward instead of one.

The first is a cell state, c-t, which is the long-term memory. The second is a hidden state, h-t,
which is what the rest of the network actually sees. Three gates control the traffic between them.

*(Walk the table.)*

The forget gate decides what fraction of the old memory to keep. The input gate decides how much of
the new candidate value to write in. The output gate decides how much of the memory to expose to the
outside world. And the last row is the candidate, which is the new content on offer this week. It's
a tanh of the same two inputs, and the input gate decides how much of it gets written.

Here sigma means the sigmoid specifically, although in Lecture 7 it stood for any activation. So
each gate outputs a number between zero and one. Think of it as a soft switch. It isn't simply open
or closed. It's a dial, and the network learns where to set it. The square brackets just mean last
week's hidden state and this week's input, stacked into one vector. The circled dot multiplies two
vectors element by element. And f-t here is the forget gate, not last week's filter.

**[pause]**

*(Point at the cell state update.)*

Now this line is the one that matters. Everything else is bookkeeping.

The new cell state equals the forget gate times the old cell state, **plus** the input gate times the
new candidate.

Look at the operations. There's an element-by-element multiplication by a number between zero and
one, and that's the forgetting. Then there's an **addition**.

Keep your eye on that plus sign. Two slides from now, it'll turn out to be the whole reason the
architecture works. The equation beside it is how the output gate does its job: the hidden state is
the output gate times the tanh of the cell state.

**[pause]**

There's one more thing before we move on. Notice what the gates are functions *of*. Each one takes
the previous hidden state and the current input, runs them through a weight matrix, and squashes the
result with a sigmoid.

So the gates are **learned and input-dependent**. The network isn't applying a fixed rate of
forgetting. At every single time step, it decides how much to forget and how much to write, based on
what it's looking at right then.

That's a genuinely different kind of model from anything we've built. A LASSO coefficient is the
same number for every observation. An LSTM gate is recomputed at every step of every sequence. That
flexibility is the appeal, and as the scoreboard will show, it's also capacity you have to pay for
whether you need it or not.

---

# ▶ SLIDE 12 — What the Gates Are Doing, in Plain Terms

The equations are compact, but the behavior is genuinely intuitive. Let me put it in the language of
a store that forecasts demand week by week.

*(Walk the three rows as a story.)*

The forget gate says, "The promotion that ran three months ago no longer explains anything, so drop
it." The gate closes toward zero, and that part of the memory decays away.

The input gate says, "This week is Thanksgiving, and the spike is enormous, so record it." The gate
opens toward one, and the event is written into the cell.

The output gate says, "I'm holding a memory of last December, but this week is April, so don't act
on it yet." The information stays in the cell state without contaminating this week's prediction.

**[pause]**

*(This next part is the one to emphasize.)*

The output gate is the one people skip when they explain this, and it's the subtle one.

It separates **what the model remembers** from what the model uses right now. Those are different
things, and a vanilla RNN can't tell them apart. It has one hidden state, which has to do both jobs
at once. Anything an RNN stores is immediately in play, and it affects the very next prediction.

An LSTM can hold something in reserve. That's a genuine capability, and it's why these models work
well on language, where a fact mentioned in sentence one might not matter again until sentence
forty.

---

# ▶ SLIDE 13 — Why the Cell State Fixes the Gradient

So let's go back to that plus sign.

*(Point at the picture, and read it from right to left, which is the direction a gradient travels.)*

The top lane is the vanilla RNN. Every backward hop multiplies by W-h and by the slope of the tanh.
The red arrow gets thinner at every hop, and by the left edge almost nothing arrives.

The bottom lane is the LSTM's cell state. Look at how the old memory enters. It's **additive**. The
new information comes in from below, at the green plus sign. And notice what *isn't* happening on
the way from c-t minus one to c-t. There's no weight matrix and no tanh, just the forget gate.

When the forget gate sits near one, c-t is approximately c-t minus one, plus whatever's new. A
gradient flowing backward through that addition passes through essentially **undiminished**, so the
arrow stays thick.

Repeated multiplication shrinks a signal, and repeated addition doesn't.

**[pause]**

*(Point at the key box. This is the callback.)*

And you've seen this fix before, in Lecture 8, last week.

It's the ResNet skip connection: y equals F of x, **plus x**. The problem there was a related one, a
signal that has to pass through many layers, and the solution was the same: give the gradient an
additive path home.

Here's the detail I enjoy. The LSTM is from nineteen ninety-seven, and ResNet is from twenty
fifteen. The nineteen ninety-seven LSTM had no forget gate at all. Its memory path was c-t equals
c-t minus one plus the new part, which is *exactly* ResNet's identity. The forget gate came in two
thousand, from Gers, Schmidhuber and Cummins, and it made that path learnable. So the idea was
invented for **time**, forgotten by the computer vision community, and then rediscovered eighteen
years later for **depth**.

That's the same mathematics in two fields, almost two decades apart. It's a decent argument for
understanding mechanisms rather than memorizing architectures, because the mechanism transferred and
the architecture didn't.

*(Point at the gray line at the bottom.)*

Here's one honest footnote. "Near one" is a condition, not a promise. On our panel, the trained
forget gates sit around one half, with a median of zero point five two, so the cell keeps only about
seven percent of a memory after four weeks. The model *chose* to forget. Hold on to that thought,
because Check 2 explains why it was right to.

---

# ▶ SLIDE 14 — Section divider: In PyTorch, and Does It Work

The next section puts all this into PyTorch, which takes four lines of model code, and then measures
it honestly against everything else this course has fitted.

---

# ▶ SLIDE 15 — LSTM for Forecasting in PyTorch

The good news is that after all that machinery, the code is short. `nn.LSTM` gives you the entire
thing: all three gates, the cell state, and the whole update.

You define the LSTM layer, you define a linear layer on top, and in `forward` you run the sequence
through and take the last time step.

*(Point at the shape note.)*

There are two things here that will cost you time if you don't know them.

The first is that `batch_first=True` puts the batch first, then the sequence length, then the
features. That's the **opposite** of what `Conv1d` wanted last week, which was the batch, then the
channels, then time. If you're moving code between the two, and Homework 5 asks you to, this is
where you'll get a shape error.

The second is the indexing `out[:, -1, :]`, which takes the last position along the time axis. It's
the hidden state after the last step, which is the summary of the whole window, and it's what we put
the linear layer on.

*(Point at the comment.)*

For the vanilla RNN, which comes out best of the neural models on the next slide's scoreboard, and
which Homework 5 asks for, swap in `nn.RNN`. It returns the output and the hidden state, not the
output and a pair of hidden and cell states, because there's no cell state. If you copy the LSTM line
unchanged, Python will complain about unpacking.

Everything from Lecture 7 still applies: Adam, weight decay, `zero_grad()`, `shuffle=False`, and
splitting by time. There's one addition, which is `clip_grad_norm_` with a maximum norm of one, to
handle the exploding gradient case from slide eight.

**[pause]**

I want to point out how little of the theory shows up in the code, because that cuts both ways.

Everything from the last three slides is inside `nn.LSTM`: the cell state, the three gates, the
additive update, and the whole reason this architecture exists. It's one line. You could use this
model having understood none of it.

That's the appeal of the tooling, and it's also the risk. If you don't know that gating solves
long-range forgetting, you have no way to predict in advance that it'll do nothing on a
twenty-six-week window. You'd just fit it, see a worse number, and have no account of why.

The measurement on the next slide only makes sense because we spent twenty minutes on the mechanism.

---

# ▶ SLIDE 16 — Does It Work? The Scoreboard So Far

*(This is the first payoff. Walk the table, then land both paragraphs.)*

It's the same weekly panel, the same twenty-six-week windows, and the same test block as Homework 4,
so everything here is directly comparable.

LASSO with forty-six engineered features scores seven forty-four. XGBoost scores seven eighty-one.
Then comes the **vanilla RNN** at eight forty-two, with four and a half thousand parameters. The
random forest scores eight ninety-nine. The **LSTM** scores nine eighty-seven, with eighteen thousand
parameters. The best CNN variant scores one thousand thirty-two. And seasonal naive is at the bottom,
at twenty-one fifty-two.

*(Point at the two dashes.)*

The tree models have no parameter count, and that's not an omission. A tree's size isn't fixed
before training. It grows from the data. If you count their splits and leaves anyway, XGBoost holds
about sixteen thousand numbers, which is roughly the size of the LSTM. The random forest holds about
**two million**, more than the five thousand training rows, and it still lands between the two
recurrent models. Raw counts don't compare across model families, and averaging two hundred trees
cancels out most of that size.

**[pause]**

There are two readings here, and the first is genuinely good news for recurrence.

**The vanilla RNN beats convolution decisively**, eight forty-two against one thousand thirty-two.
The LSTM beats convolution too, but only just. At nine eighty-seven, it's forty-five better. And the
RNN did that on **four raw channels**, with no lag engineering at all, against forty-six hand-built
features for LASSO. That's a real argument for reading a sequence in order, and it's the strongest
result deep learning gets on our data.

Now for the second reading. **The LSTM lost to the vanilla RNN**, at a fixed budget of thirty epochs.

That isn't a typo. But hold on to the qualifier, because the next four slides are about what that
line does and doesn't mean.

---

# ▶ SLIDE 17 — Why the Gated Model Lost at 30 Epochs

*(Walk the table.)*

At thirteen weeks, the RNN scores eight seventy-eight and the LSTM scores nine ninety. At twenty-six
weeks, it's eight forty-two against nine eighty-seven. At fifty-two weeks, it's eight-oh-five against
nine ninety-eight.

So the RNN wins at all three window lengths.

**[pause]**

Now, be careful about how much that buys us. Three windows rule out a fluke of one window, and not
much else. All three rows use the same thirty series, the same split, and the same pull from the
most recent weeks. So it's one finding measured three ways, not three independent tests, which is
what the line at the top of the slide says. That's the kind of distinction that separates a result
from a slogan.

The natural reading is that an LSTM solves a problem **this data doesn't have**. Gating preserves
information across *long* chains of dependence, which is what it was invented for, and weekly demand
is dominated by the recent past. And the gates aren't free. They cost eighteen thousand parameters,
against four and a half thousand.

*(Point at the warning box.)*

**[pause]**

But before we sign off on that, notice what we *haven't* checked. Both models trained for thirty
epochs, because thirty is what the training function uses by default. Nobody showed that thirty
epochs is enough for **both** of them, and one of them has four times the parameters.

A margin measured at one training budget is a claim about that budget.

So the slides that follow make the two checks this table owes us. The first asks whether the
comparison was fair, and the second asks what in the data explains it.

---

# ▶ SLIDE 18 — Check 1: Was It a Fair Fight?

*(Bring up the chart, and take this one slowly, because it reverses the previous slide.)*

These are the same two models, on the twenty-six-week window. This time we vary the training budget
and watch the margin. Red is the RNN, blue is the LSTM, and the dashed line marks the thirty epochs
we used.

At ten epochs, the LSTM is **ahead** by ninety-seven. At thirty, which is our setting, the RNN is
ahead by a hundred and forty-five, and that's the bracket on the chart. At sixty epochs, the RNN
leads by ninety-eight. And at a hundred, the gap is **minus nine**.

*(Let that sit.)*

**[pause]**

Read the shape of it. The gap is **largest at exactly the budget we chose**. It isn't merely close to
the largest. It's the largest anywhere in the range we looked at.

*(Trace the two lines.)*

Now look at the two curves separately, because they have different shapes. The RNN is finished by
about epoch forty-five. After that it reads eight twenty-eight, eight thirty-three and eight thirty,
which is flat. The LSTM is still coming down at a hundred. It goes from nine eighty-seven at thirty
epochs to nine twenty-six at sixty and eight twenty-one at a hundred, and it hasn't turned over.
**We ran out of patience, not the model.**

**[pause]**

So the honest verdict isn't that the LSTM loses. It's three things.

First, at a fixed budget of thirty epochs, the RNN wins by a wide margin. Second, given enough
epochs, the two **tie**. And third, the RNN gets to its best answer roughly **three times faster, on
a quarter of the parameters**. That's a real advantage and worth having, but it's an advantage in
*optimization*, not a verdict about gating.

I want to be direct about what happened here. The first version of this slide deck didn't have this
check, and the previous slide's table was presented as a result about the architecture. It isn't
one. It took varying a setting nobody thinks of as part of the experiment to find that out. That's
the same lesson as the seeds in Lecture 7, two weeks ago, one level up.

---

# ▶ SLIDE 19 — Check 2: What in the Data Explains It

*(There are three panels. This is the slide that transfers to your own work, so take the panels from
left to right.)*

Here's the second check. Gating pays off when a value has to survive many steps that would otherwise
overwrite it. Does this panel have anything like that? Three measurements say no, and the first two
are on this slide.

*(Point at the left panel.)*

The first measurement is that most of the variance isn't sequential at all. This panel shows all
thirty series from the lab on one axis, which is ten stores times three categories. Look at what
separates them. The food series sit up between ten and thirty thousand units a week, household goods
sit around five thousand, and hobbies around two thousand. The dashed line is the grand mean, at
about eight thousand. The big differences are about *which series*, not *which week*.

*(Point at the triangle.)*

The middle panel puts a number on it, and it's a right triangle. It's Pythagoras, applied to
variances. Every week's distance from the grand mean splits into two pieces: the series' own average
minus the grand mean, and that week minus the series' average. Square them and add them up over all
eight thousand three hundred rows, and the cross term drops out exactly, because deviations from a
mean add up to zero. So the squares add. The between-series side is about six thousand eight hundred
units, the within-series side is about two thousand one hundred, and the total is about seven
thousand one hundred. Square the between side, divide by the square of the total, and you get
**ninety-one percent**. Only nine percent is variation within a series over time.

So the main thing a model has to get right is the level of the series in front of it, and the last
few weeks tell you that. It doesn't take a memory mechanism.

**[pause]**

*(Point at the right panel.)*

The second measurement is that history helps, but it arrives smoothly. This one is counterintuitive,
so let's take it slowly. The right panel is the autocorrelation of each series around its own mean,
averaged over the thirty series. It's zero point eight three at lag one, and still zero point four
three at lag fifty-two. That *looks* like long memory, and history does help. In the table two
slides back, the RNN improved from eight seventy-eight to eight-oh-five as the window grew from
thirteen weeks to fifty-two. But the help arrives smoothly. Because neighboring weeks are nearly the
same number, no single value has to survive many steps untouched. The gap is filled in continuously
along the way.

Gating earns its keep when something has to survive steps that would overwrite it, and here nothing
overwrites anything.

---

# ▶ SLIDE 20 — Check 2, Continued: Nothing Arrives Early

The third measurement is that nothing in the inputs arrives early. SNAP days correlate about zero
point one four with units in the same week. Two and four weeks ahead you do see something, minus
zero point one three and plus zero point one two, but that's the calendar, not a warning. SNAP days
fall on the same dates every month, so this week's count tells you the count two and four weeks out,
and those weeks have their own same-week effect. It's the same plus zero point one four, echoed.
Event days are flat everywhere, including in the same week. There's no **announcement** structure in
this data, nothing that shows up at time t and has to be stored until time t plus k. That's the
textbook use for a forget gate and an input gate, and this panel doesn't contain a single instance
of it.

*(Point at the key box.)*

**[pause]**

And this is the part to carry out of the lecture. It isn't the answer. It's the three questions.
Before you reach for a gated model, or an attention model, on your own series, ask these. How much of
my variance is cross-sectional rather than sequential? Is my series smooth enough that recent values
already summarize the old ones? And does anything in my inputs *lead* the target?

If the answers look like this panel's, the extra machinery has nothing to do.

---

# ▶ SLIDE 21 — Section divider: Key Takeaways

That brings us to the key takeaways: what recurrence buys, what it costs, and what's still missing.

---

# ▶ SLIDE 22 — RNN and LSTM in Context

*(Walk the table first, then the two paragraphs.)*

The table puts four architectures against three questions. Does it see order? How much can it
actually remember? And what does it cost?

The feedforward network on lags doesn't see order, and it remembers whatever you engineered. The CNN
sees local order only, about seven steps. The vanilla RNN does see order, and it remembers roughly
ten to thirty steps in practice. Note that this is *practice*, not the unlimited memory of the
theory. Those first three are all cheap. The LSTM sees order too, and it can remember hundreds of
steps, at four times the cost.

An LSTM holds roughly four times the parameters of a vanilla RNN at the same hidden size, because
each of the three gates and the candidate needs its own weight matrix. That's the price of the
memory, and on a short window you pay it for nothing.

**[pause]**

If you go looking, you'll also meet the GRU, the gated recurrent unit. It merges the forget and input
gates into one and drops the separate cell state, so it sits between the two models on this table.
It has more gating than a vanilla RNN and fewer parameters than an LSTM.

On data like ours, where the gates aren't earning their keep anyway, a GRU often performs about as
well as an LSTM at lower cost. It's a reasonable thing to try in your final project if you go down
this road, and a reasonable thing to skip, given what the scoreboard says about this whole family on
this dataset.

*(Now set up Part 2.)*

**[pause]**

Here's what's still missing, and it motivates next week.

An LSTM reads strictly from left to right, one step at a time, and that has two consequences. The
first is that it **can't be parallelized** across the sequence. Step t needs step t minus one to be
finished, so a thousand-step sequence takes a thousand operations in a row, no matter how much
hardware you own.

The second is that everything it knows about step three has to survive being squeezed through
**every** hidden state between step three and the end. The gates make that survival more likely, but
the information still passes through a bottleneck at every step.

Part 2, next week, removes both constraints, with a mechanism that lets every position look directly
at every other position in a single operation.

---

# ▶ SLIDE 23 — Lecture 9, Part 1: Key Takeaways

Let me pull together the main points.

First, a recurrent network reads a sequence in order and carries a hidden state. Its weights are
shared across time, so the length of the window costs no parameters.

Second, training is backpropagation through time. Unroll the recurrence, and it's a deep network
whose layers share their weights. Repeated multiplication shrinks the gradient exponentially, and our
own RNN keeps only about a thousandth of a percent of it a year back.

Third, an LSTM adds a cell state that's updated **additively**. Addition preserves gradients where
multiplication destroys them. It's the ResNet fix, eighteen years earlier.

Fourth, there are three gates. The forget gate drops what's stale, the input gate writes what's new,
and the output gate exposes what matters now.

Fifth, on our panel both recurrent models beat every CNN variant, on four raw channels. And both
still lose to a LASSO with forty-six coefficients.

And last, the LSTM lost to the vanilla RNN at a budget of thirty epochs, at all three window lengths,
and the two tied by a hundred epochs, with the LSTM still improving. So check that a margin survives
the training budget before you call it a result. Gating solves long-range forgetting, and a
twenty-six-week window doesn't suffer from it, so the gates bought four times the parameters and
nothing else. **Fit the architecture to the problem.**

**[pause]**

Next week, in Lecture 9, Part 2, we turn to attention and Transformers, which read the whole sequence
at once.

---

# ▶ SLIDE 24 — References

*(Advance and close. No narration is needed.)*
