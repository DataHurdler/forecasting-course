# Lab 9 Part 1 — Recording Script

**ECON 8310: Business Forecasting · Recurrence — and When Gates Are Not Worth It**

Lab: `Labs/Lecture09_Part1_lab.qmd` (5 steps) · Measured runtime: **~24 minutes** of narration
(the in-room version is budgeted at 40)

---

## How to use this document

- **`▶ STEP n — Title`** matches the lab's own headings exactly.
- *Italic parentheticals* are stage directions. **[pause]** is a beat; **[STOP — learner works]**
  is where you tell the viewer to pause the video and do something.
- Deterministic (`torch.manual_seed` per fit, `shuffle=False`). Fits are quick here — a few
  seconds each — so unlike Lab 8 there is no dead air to work around.

### This lab was restructured, and the reason matters

Writing this script turned up a **methodological** defect rather than an arithmetic one, and the
lab now has five steps instead of four.

Step 3 compared the RNN and the LSTM at a single 30-epoch budget, found gaps of 112 to 193, and
concluded the RNN wins. Every number was right. But the LSTM carries four times the parameters,
nobody had checked that 30 epochs was enough for **both** models, and Lab 7 — one week earlier —
exists to teach that a comparison run at one setting is not a result.

Measured at seven checkpoints out to 100 epochs, the margin **peaks at exactly 30 epochs** and
closes to nothing: the two models tie at 100, and the LSTM's curve has still not flattened.
So the lab now:

- **bounds Step 3's claim** to the budget it was measured at, and plants the doubt on purpose;
- **adds Step 4**, which varies the budget and reverses the reading;
- **rewrites Step 5** to *measure* the three data properties that explain why gating has nothing
  to do here, instead of asserting them.

The same correction went into the deck (two new frames), the deck narration, Lecture 13, and
HW05 Parts 2 and 3.

*(One more, purely mechanical: an earlier draft of Step 4 refitted both seeds inside the
checkpoint loop — 28 networks instead of 4. Correct numbers, seven times the runtime. Fixed
before shipping; it is the same silent-failure species the lab is about.)*

### In-room language that needs replacing

Step 3's "your turn" and the four closing **Discuss** blocks. Converted below.

---

# ▶ OPENING

*(Show the rendered lab, at the top of the document.)*

This is Lab 9, Part 1. It's the third architecture in three weeks, and it's the first one that has
no structural excuse.

Recall where the last two fell short. The feedforward network in Lab 7, two weeks ago, flattened the
window and lost the ordering. It couldn't tell you which of its one hundred four inputs came first.
The CNN in Lab 8, last week, kept the ordering but could only see seven weeks at a time, and you
proved that by arithmetic before you trained anything.

A recurrent network has neither limit. It reads the sequence in order, one step at a time, carrying
a hidden state forward. In principle, nothing stops it from using every week you give it.

**[pause]**

So today you'll build two of them, a plain RNN and an LSTM, and at the training budget anyone would
have used, the more sophisticated one loses.

I want to be clear that this isn't a trick or a botched implementation. It's a genuine result on
this data, it repeats at three window lengths, and working out exactly which claim the evidence
supports, and why, is the most transferable thing in this lab. You'll meet the same shape of
argument every time someone offers you a more powerful model.

---

# ▶ SETUP

*(Run the setup cell.)*

It's the same `build()` function as Lab 8, with the same panel, the same cutoff date, and the same
standardization, computed from the training rows only. And there are three window lengths again:
thirteen, twenty-six and fifty-two weeks.

*(Point at the return line and the callout.)*

One line is different from last week, and it's the one the callout is about.

The layout flips back. `Conv1d` wanted the batch first, then the channels, then time. But `nn.RNN`
and `nn.LSTM`, with `batch_first=True`, want the batch, then **time**, then the channels. So the
transpose you carefully added in Lab 8 has to come out again this week.

**[pause]**

And as always, neither library complains. Feed an RNN the `Conv1d` layout, and it'll read your four
channels as a four-step sequence, treat the twenty-six weeks as features, train to convergence, and
hand you a number.

That's now the fourth instance of the same failure mode in this course. First came shuffling the
rows before the time split, then a missing `zero_grad()` in Lab 7, then the axes transposed for the
convolution in Lab 8, and now the axes transposed back for the recurrence. None of them raises an
error. I keep pointing at this because it's the single biggest practical difference between this
material and the regression you did in Lecture 6, on regularization, where a shape error really is
an error.

---

# ▶ STEP 1 — A plain RNN

*(Show the `Recurrent` class, then run it.)*

There's one recurrent layer, and then a linear layer on the **final** hidden state, which is the
summary after the model has read the whole window.

```
vanilla RNN — 4,545 parameters, test RMSE 860
```

That's **four thousand five hundred forty-five** parameters. Hold that against Lab 7's feedforward
network, which had twenty-one thousand seven hundred sixty-one, and Lab 8's CNN, at eight thousand
one hundred sixty-one. This is the smallest model in the whole neural section of the course, and on
this seed, with a test RMSE of eight sixty, it's already the best.

*(Point at `clip_grad_norm_`.)*

One line deserves a word, because `clip_grad_norm_` isn't decoration. Backpropagating through
twenty-six steps multiplies the gradients together twenty-six times, and that product can explode as
easily as it can vanish. Clipping caps the size of the update. It costs nothing, and recurrent models
are exactly where you need it.

---

# ▶ STEP 2 — An LSTM

*(Run the cell.)*

The code is identical, with one word changed: `nn.RNN` becomes `nn.LSTM`.

```
LSTM        — 17,985 parameters, test RMSE 1,004

parameter ratio: 4.0x the RNN
```

**[pause]**

Account for the ratio before you look at the RMSE, because it's the whole economics of the
comparison. An LSTM has a forget gate, an input gate, an output gate and a candidate, which makes
four sets of weights where the plain RNN has one. So at the same hidden size, it has almost exactly
four times the parameters. The recurrent layer is exactly four times as big, and the linear head the
two models share at the end pulls the total ratio down to three point nine six, which the printout
rounds to four point oh.

And on this seed, the LSTM scores one thousand four, against the RNN's eight sixty.

---

# ▶ STEP 3 — Which wins, and does it hold?

*(Run the window table.)*

One comparison at one window isn't evidence. So we use three windows, with two seeds each.

```
 window  RNN  RNN_sd  LSTM  LSTM_sd winner
     13  878      33   990       21    RNN
     26  842      19   987       17    RNN
     52  805       5   998        3    RNN
```

**[pause]**

The RNN wins at all three. And unlike the window experiment in Lab 8, the seed question is settled
cleanly here. The spreads run from three to thirty-three, against gaps of one hundred twelve to one
hundred ninety-three, so the gaps are roughly three to sixty times the noise. This isn't a seed
artifact.

*(Show the callout.)*

Now read the callout carefully, because it does something the earlier version of this lab didn't. It
tells you to read the columns in a particular order.

Start with the winner column, and notice how the callout bounds the claim: at this training budget,
on this panel, the plain RNN wins at every window length. Three windows rule out a fluke of one
window, but they don't rule out much else. All three rows use the same thirty series, the same split,
and the same pull from the most recent weeks. So it's **one finding measured three ways**, not three
independent tests.

Then look at the parameter count. The model that lost has four times as many parameters as the model
that won.

And then, and this is the new part, be suspicious. Both models trained for exactly thirty epochs,
because that's what `train()` uses by default. Nothing in this table establishes that thirty epochs
is enough for *both* of them, and one of them has four times the parameters to fit.

**[STOP — learner works]**

Before you go on, pause the video. Compute the gap at each window, and say whether it's closing or
widening. Then predict what Step 4 will find.

*(Resume.)*

The gaps are one hundred twelve at thirteen weeks, one hundred forty-five at twenty-six, and one
hundred ninety-three at fifty-two, so the gap widens as the window grows. The table shows why: the
RNN keeps improving as the window gets longer, while the LSTM barely moves. Hold on to your
prediction for Step 4.

---

# ▶ STEP 4 — Was that a fair fight?

*(Start the cell, which takes two or three minutes, then talk over it.)*

This is the step the lab is really about, and I want to be honest that it was added after the fact.
The first version of this lab stopped at Step 3.

Lab 7 taught you not to trust a comparison run with one seed. This is the same objection, one level
up: not one seed, but one **budget**. So we train both models to a hundred epochs, and we record the
test RMSE at seven checkpoints along the way, instead of only at the end.

```
 epochs  RNN  LSTM  gap
     10 1099  1002  -97
     20  922  1033  111
     30  842   987  145
     45  834   953  119
     60  828   926   98
     80  833   885   52
    100  830   821   -9
```

*(Let it sit. Don't talk over this one.)*

**[pause]**

Read the gap column from top to bottom. At ten epochs it's **negative**, minus ninety-seven, so the
LSTM is ahead. At thirty it's **plus one hundred forty-five**. Then it falls to one hundred nineteen,
ninety-eight and fifty-two, and at a hundred epochs it's **minus nine**.

The gap is **largest at exactly the budget** we used in Step 3. It isn't merely near the largest.
It's the largest anywhere in the range.

*(Point at the two curves.)*

Now look at the shapes, because they're different. From epoch forty-five on, the RNN goes eight
thirty-four, eight twenty-eight, eight thirty-three, eight thirty. It's flat, and it's finished. Over
the same checkpoints the LSTM goes nine fifty-three, nine twenty-six, eight eighty-five, eight
twenty-one. It's still falling at epoch one hundred, and its curve hasn't turned over. **We ran out
of patience, not the model.**

**[pause]**

So the honest verdict isn't that the LSTM loses. It's three separate statements, and you need all
three.

First, at a fixed budget of thirty epochs, the RNN wins by a wide margin. That's true, and it's what
Step 3 measured. Second, given enough epochs, the two **tie**. And third, the RNN reaches its best
score roughly three times faster, on a quarter of the parameters.

That third one is a real advantage, and it's worth having. Convergence speed matters when you're
tuning, when you're retraining every night, and when compute is billed. But it's an advantage in
**optimization**, not a verdict on gating, and the Step 3 table can't tell those two claims apart.

**[STOP — learner works]**

The run-to-run spread in Step 3 was about twenty. From which checkpoint on is the gap smaller than
that? Pause the video, fill in the blank, and find out.

*(Resume.)*

Only the last checkpoint makes it. At a hundred epochs the gap is minus nine, inside the noise, and
at every earlier checkpoint it's fifty-two or more, in one direction or the other.

One more thing before we move on, and it's the part that transfers. Nobody thinks of the epoch count
as part of an experiment. It's boilerplate. You set it once, you copy it between projects, and you
stop seeing it. That's exactly what made it dangerous here: a number that felt like configuration
turned out to be the thing that decided the answer.

It was seeds in Lab 7, windows in Lab 8, and the budget here. It's the same mistake in three
costumes.

---

# ▶ STEP 5 — Why the gates buy nothing here

*(Run the gradient-shrink table.)*

Step 4 told us that the RNN's *margin* was mostly an artifact of the budget. It didn't tell us why,
given enough epochs, the LSTM only ties instead of pulling **ahead**, and for that you have to look
at the data.

Start with the problem gating exists to solve. In a plain RNN, gradients shrink as they propagate
backward, so the signal from far in the past never reaches the weights.

```
gradient shrink factor 0.7 per step:      26 steps back -> 0.000094
gradient shrink factor 0.9 per step:      26 steps back -> 0.064611
                                          52 steps back -> 0.004175
```

Notice that this is a matter of degree, and of **length**. At a shrink factor of zero point seven,
the signal is gone by twenty-six steps, and that's the catastrophe the textbooks draw. At zero point
nine, it's still zero point zero six five at twenty-six steps, and zero point zero zero four at
fifty-two. That's small, but it isn't zero. And twenty-six weeks is a short sequence. Vanishing
gradients are a real phenomenon that bites hard at hundreds of steps and only bites softly here.

*(Run the diagnostics cell.)*

Now let's stop asserting, and measure whether this dataset has the problem at all. There are three
questions.

```
1. between-series share of total variance : 91.0%

2. within-series autocorrelation of units
      lag  1: +0.828      lag 13: +0.695
      lag  4: +0.824      lag 26: +0.575
                          lag 52: +0.432

3. does a flag at t predict units at t+k?
      snap_days   t+0: +0.143   t+1: +0.030   t+2: -0.128   t+4: +0.116
      event_days  t+0: +0.004   t+1: +0.023   t+2: -0.009   t+4: -0.009
```

**[pause]**

The first answer is that **ninety-one percent** of the variance isn't sequential at all. It's
*between* series, meaning which store and which category. Only nine percent is within a series over
time. So the single biggest thing any model here has to get right is the **level** of the series in
front of it, and you can read that off the most recent few weeks. No memory mechanism is involved.

The second answer is counterintuitive, so slow down for it. The autocorrelation is zero point eight
three at lag one, and still zero point four three at lag fifty-two. That *looks* like long memory,
which would be exactly the case for an LSTM.

It's actually the opposite. Because neighboring weeks are nearly the same number, **the recent past
is a sufficient statistic for the distant past**. Week fifty-one already tells you most of what week
forty knew. Gating earns its keep when a value has to survive many steps that would otherwise
**overwrite** it. Here nothing overwrites anything, because the series was filled in continuously
the whole way along.

The third answer is that nothing in the inputs arrives early. SNAP days correlate zero point one four
with units in the same week. The printout shows minus zero point one three two weeks ahead and plus
zero point one two four weeks ahead, and that's the calendar, not a warning. SNAP days fall on the
same dates every month, so this week's count predicts the count two and four weeks out, and those
weeks carry their own same-week effect. It's the same zero point one four, echoed. Events are flat
everywhere, including in the same week. There's no **announcement** in this data, nothing that shows
up at time t and has to be held until time t plus k.

That's the textbook use for a forget gate and an input gate, and this panel doesn't contain a single
instance of it.

*(Show the closing paragraph of the step.)*

**[pause]**

Now here's the part to carry out of the lab. It isn't the answer. It's the three questions. Before
you reach for a gated model, or an attention model, on a series of your own, ask these. How much of
my variance is cross-sectional rather than sequential? Is my series smooth enough that recent values
already summarize the old ones? And does anything in my inputs **lead** the target?

If the answers look like this panel's, the extra machinery has nothing to do. If they don't, it
might be exactly what you need, and you'll know which before you spend a week finding out.

---

# ▶ BEFORE YOU LEAVE

**[STOP — learner works]** — *(replaces the in-room "Discuss")*

Here are four questions to work through on your own before you go.

The first is about the result. Using the three diagnostics from Step 5, describe a forecasting
problem where you'd expect the LSTM to beat the RNN outright. Be specific. What would the
between-series share look like? What would the autocorrelation look like? And what would have to be
true of the exogenous inputs?

**[pause]**

Here's the heart of the answer. You want a series where something far in the past has to be carried
forward through many irrelevant steps. Think of a promotion announced twelve weeks before it takes
effect, or a contract signed in January that governs deliveries in November. In the language of the
third diagnostic, that's an input that leads the target, and carrying it forward is exactly what
gating is for.

The second question is about the budget. Step 3 and Step 4 support different claims, from the same
two models on the same data. State the strongest claim you'd be willing to publish, in one sentence.
Then say what a reader would need to be told alongside it for that sentence to be honest.

The third question is about practice. You've now been caught twice, once by seeds in Lab 7 and once
by the training budget here. Both times, a comparison looked settled and wasn't. What's the general
form of that mistake, and what would you check the next time you see two models compared in a table?

The fourth question looks ahead to Homework 5. Part 2 of that homework asks you to fit an RNN, an
LSTM and a Transformer, and to explain where the extra LSTM parameters go and whether they bought
anything. You now have the RNN and LSTM part of that answer. What's your prediction for the
Transformer, and, more usefully, what result would make you wrong?

**[pause]**

Next week, in Lab 9, Part 2, you'll build the Transformer and find out. I won't spoil it, except to
say that the argument you just made about the LSTM is the argument you should try on attention as
well, and that you should check whether it survives.

---

# Appendix — expected output

Deterministic; `torch.manual_seed` per fit, `shuffle=False`. Most fits are ~6 s; **Step 4 trains
four models to 100 epochs and takes 2–3 minutes.**

| Quantity | Value |
|---|---|
| Tensor shape at W=26 | (5940, **26**, 4) — batch, time, channels (*not* the Conv1d layout) |
| Vanilla RNN | **4,545** parameters · seed 1 → 860 |
| LSTM | **17,985** parameters · seed 1 → 1,004 · ratio **4.0×** as printed (3.96; the recurrent layer alone is exactly 4×) |
| Window table — RNN | 13w: **878** (sd 33) · 26w: **842** (sd 19) · 52w: **805** (sd 5) |
| Window table — LSTM | 13w: 990 (sd 21) · 26w: 987 (sd 17) · 52w: 998 (sd 3) |
| Gaps vs spreads | 112 / 145 / 193 against sds of 3–33 — **not** a seed artifact |
| Convergence, RNN | 1099 · 922 · **842** · 834 · 828 · 833 · 830 (epochs 10→100) |
| Convergence, LSTM | 1002 · 1033 · 987 · 953 · 926 · 885 · **821** |
| Gap by epoch | −97 · +111 · **+145** · +119 · +98 · +52 · **−9** |
| Where the gap peaks | **epoch 30** — the budget Step 3 used |
| Gradient shrink | 0.7²⁶ = 0.000094 · 0.9²⁶ = 0.0646 · 0.9⁵² = 0.0042 |
| Between-series variance | **91.0%** |
| Within-series autocorrelation | 0.828 (lag 1) · 0.824 (4) · 0.695 (13) · 0.575 (26) · 0.432 (52) |
| `snap_days` lead correlation | +0.143 at t+0, then +0.030 / −0.128 / +0.116 — no lead |
| `event_days` lead correlation | +0.004 at t+0 — flat at every horizon |

**Three things to know before recording.**

*Step 4 reverses Step 3, and the recording should feel like that.* Do not soften Step 3 in
anticipation — deliver it as the result it appears to be, plant the suspicion the callout plants,
and let Step 4 do the work. The reversal is the lesson; pre-hedging it wastes the effect.

*Say plainly that Step 4 was added after the fact.* The first version of this lab stopped at Step
3 and the deck presented that table as an architecture verdict. Students trust a course more, not
less, when it shows its own corrections — and this one was found by asking "is 30 epochs enough
for both?", which is a question any of them could have asked.

*Step 5's second diagnostic is the one that gets misread.* An autocorrelation of 0.43 at lag 52
sounds like evidence **for** long memory. It is the reason long memory is unnecessary: smoothness
means recent values already summarize old ones. Expect the question, and answer it with the
overwrite framing — gating protects a value from being overwritten, and nothing here overwrites
anything.
