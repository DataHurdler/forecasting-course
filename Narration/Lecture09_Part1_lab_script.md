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

Step 3's "your turn" and the three closing **Discuss** blocks. Converted below.

---

# ▶ OPENING

*(Screen: rendered lab, top of document.)*

Lab 9, Part 1. This is the third architecture in three weeks, and it is the first one that has no
structural excuse.

Recall where the last two failed. The feedforward network in Lab 7 flattened the window and lost
the ordering — it could not tell you which of its 104 inputs came first. The CNN in Lab 8 kept
ordering but could only see seven weeks at a time, and you proved that by arithmetic before you
trained anything.

A recurrent network has neither limit. It reads the sequence in order, one step at a time,
carrying a hidden state forward. In principle nothing stops it using every week you give it.

**[pause]**

So today you build two of them — a plain RNN and an LSTM — and the more sophisticated one loses.

I want to be clear that this is not a trick or a botched implementation. It is a genuine result
about this data, it is repeated at three window lengths, and understanding *why* is the most
transferable thing in this lab. You will meet the same shape of argument every time someone offers
you a more powerful model.

---

# ▶ SETUP

*(Screen: run setup.)*

Same `build()` function as Lab 8, same panel, same cutoff, same standardization from training rows
only. Three window lengths again — 13, 26, 52.

*(Point at the return line and the callout.)*

One line is different from last week, and it is the one the callout is about.

**The layout flips back.** `Conv1d` wanted (batch, channels, time). `nn.RNN` and `nn.LSTM` with
`batch_first=True` want (batch, **time**, channels) — so the transpose you carefully added last
week has to come out again this week.

**[pause]**

And as always: neither library complains. Feed an RNN the Conv1d layout and it will read your four
channels as a four-step sequence, treat the 26 weeks as features, train to convergence, and hand
you a number.

That is now the fourth instance of the same failure mode in this course — shuffling before the
time split, missing `zero_grad()`, transposed axes for the convolution, and now transposed back for the recurrence.
None of them raise. I keep pointing at it because it is the single biggest practical difference
between this material and the regression you did in Lecture 6, where a shape error is an error.

---

# ▶ STEP 1 — A plain RNN

*(Screen: the `Recurrent` class, then run.)*

One recurrent layer, then a linear layer on the **final** hidden state — the summary after the
model has read the whole window.

```
vanilla RNN — 4,545 parameters, test RMSE 860
```

**Four and a half thousand parameters.** Hold that against Lab 7's feedforward network, which had
21,761, and Lab 8's CNN at 8,161. This is the smallest model in the entire neural section, and on
this seed it is already the best.

*(Point at `clip_grad_norm_`.)*

One line deserves a word. **`clip_grad_norm_` is not decoration.** Backpropagating through 26
steps multiplies gradients together 26 times, and that product can explode as easily as it can
vanish. Clipping caps the size of the update. It costs nothing, and recurrent models are exactly
where you need it.

---

# ▶ STEP 2 — An LSTM

*(Screen: run.)*

Identical code. One word changed — `nn.RNN` becomes `nn.LSTM`.

```
LSTM        — 17,985 parameters, test RMSE 1,004

parameter ratio: 4.0x the RNN
```

**[pause]**

Account for the ratio before you look at the RMSE, because it is the whole economics of the
comparison. An LSTM has a forget gate, an input gate, an output gate, and a candidate — four sets
of weights where the plain RNN has one. Same hidden size, four times the parameters — almost
exactly: the recurrent layer is exactly four times as big, and the shared linear head at the end
pulls the total ratio to 3.96, which the printout rounds to 4.0.

And on this seed it scores 1,004 against the RNN's 860.

---

# ▶ STEP 3 — Which wins, and does it hold?

*(Screen: run the window table.)*

One comparison at one window is not evidence. Three windows, two seeds each.

```
 window  RNN  RNN_sd  LSTM  LSTM_sd winner
     13  878      33   990       21    RNN
     26  842      19   987       17    RNN
     52  805       5   998        3    RNN
```

**[pause]**

The RNN wins at all three. And unlike Lab 8's window experiment, the seed question is settled
cleanly here: the spreads are 3 to 33 against gaps of 112 to 193. Three to sixty times the noise.
This is not a seed artifact.

*(Screen: the callout.)*

Now read the callout carefully, because it does something the earlier version of this lab did not.

**First**, it bounds the claim: *at this training budget, on this panel.* Three windows rules out
a fluke of one window. It does not rule out much else — all three rows are the same thirty series,
the same split, the same recency structure. **One finding measured three ways, not three
independent tests.**

**Second**, the parameter count: the model that lost has four times as many as the model that won.

**Third — and this is the new part — be suspicious.** Both models trained for exactly 30 epochs,
because that is what `train()` defaults to. Nothing in this table establishes that 30 epochs is
enough for *both* of them, and one of them has four times the parameters to fit.

**[STOP — learner works]**

Before you go on: compute the gap at each window and say whether it is closing or widening. Then
predict what Step 4 will find.

---

# ▶ STEP 4 — Was that a fair fight?

*(Screen: start the cell — about three minutes — then talk over it.)*

This is the step the lab is really about, and I want to be honest that it was added after the
fact. The first version of this lab stopped at Step 3.

Lab 7 taught you not to trust a comparison run at one seed. This is the same objection one level
up: **not one seed, one budget.** So train both models to 100 epochs and record the test RMSE at
seven checkpoints along the way, instead of only at the end.

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

**[pause]** — *let it sit. Do not talk over this one.*

Read the gap column down. At ten epochs it is **negative** — the LSTM is ahead. At thirty it is
**plus one forty-five**. Then 119, 98, 52, and at a hundred epochs it is **minus nine**.

**The gap is largest at exactly the budget we used in Step 3.** Not near the largest — the largest
anywhere in the range.

*(Point at the two curves.)*

And look at the shapes, because they are different. The RNN goes 834, 828, 833, 830 from epoch 45
onward. Flat. Finished. The LSTM goes 953, 926, 885, 821 — still falling at epoch 100, and its
curve has not turned over. **We ran out of patience, not the model.**

**[pause]**

So the honest verdict is not "the LSTM loses." It is three separate statements, and you need all
three.

At a fixed 30-epoch budget, the RNN wins by a wide margin — that is true and it is what Step 3
measured. Given enough epochs, the two **tie**. And the RNN reaches its best score roughly three
times faster on a quarter of the parameters.

That third one is a real advantage and worth having. Convergence speed matters when you are
tuning, when you are retraining nightly, when compute is billed. But it is an **optimization**
advantage, not a verdict on gating — and those are different claims that the Step 3 table cannot
tell apart.

**[STOP — learner works]**

The run-to-run spread in Step 3 was about 20. From which checkpoint on is the gap smaller than
that? Fill the blank in and find out.

*(Resume.)*

**[pause]**

One more thing before we move on, and it is the transferable part. Nobody thinks of the epoch
count as part of an experiment. It is boilerplate — you set it once, you copy it between
projects, you stop seeing it. That is precisely what made it dangerous here: a number that felt
like configuration turned out to be the thing determining the answer.

Seeds in Lab 7. Windows in Lab 8. Budget here. Same mistake, three costumes.

---

# ▶ STEP 5 — Why the gates buy nothing here

*(Screen: run the gradient-shrink table.)*

Step 4 told us the RNN's *margin* was mostly a budget artifact. It did not tell us why the LSTM
never pulls **ahead** — and for that you have to look at the data.

Start with the problem gating exists to solve. In a plain RNN, gradients shrink as they propagate
backward, so signal from far in the past never reaches the weights.

```
gradient shrink factor 0.7 per step:      26 steps back -> 0.000094
gradient shrink factor 0.9 per step:      26 steps back -> 0.064611
                                          52 steps back -> 0.004175
```

**[pause]**

Notice this is a matter of degree and of **length**. At a shrink factor of 0.7 the signal is
gone by 26 steps — that is the catastrophe the textbooks draw. At 0.9 it is 0.065 at 26 steps.
Small, but not zero. And 26 weeks is a short sequence. Vanishing gradients are a real phenomenon
that bites hard at hundreds of steps and only bites softly here.

*(Run the diagnostics cell.)*

Now stop asserting and measure whether this dataset has the problem at all. Three questions.

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

**One. Ninety-one percent of the variance is not sequential at all.** It is *between* series —
which store, which category. Only nine percent is within a series over time. So the single
biggest thing any model here must get right is the **level** of the series in front of it, and
that is readable off the most recent few weeks. No memory mechanism is involved in it.

**Two, and this one is counter-intuitive, so slow down.** The autocorrelation is 0.83 at lag 1 and
still 0.43 at lag 52. That *looks* like long memory — like exactly the case for an LSTM.

It is the opposite. Because neighboring weeks are nearly the same number, **the recent past is a
sufficient statistic for the distant past.** Week 51 already tells you most of what week 40 knew.
Gating earns its keep when a value must survive many steps that would otherwise **overwrite** it
— and here nothing overwrites anything, because the series was filled in continuously the whole
way along.

**Three. Nothing in the inputs arrives early.** SNAP correlates 0.14 with units in the same week.
The printout shows minus 0.13 two weeks ahead and plus 0.12 four weeks ahead — and that is the
calendar, not a warning. SNAP days fall on the same dates every month, so this week's count
predicts the count two and four weeks out, and those weeks carry their own same-week effect. The
same 0.14, echoed. Events are flat everywhere, including at zero. There is no
**announcement** in this data — nothing shows up at time *t* that must be held until *t + k*.

That is the textbook use for a forget gate and an input gate, and this panel does not contain a
single instance of it.

**[pause]**

*(Screen: the closing paragraph of the step.)*

And now the part to carry out of the lab, which is not the answer — it is the three questions.
Before you reach for a gated model, or an attention model, on a series of your own: how much of my
variance is cross-sectional rather than sequential? Is my series smooth enough that recent values
already summarize the old ones? Does anything in my inputs **lead** the target?

If the answers look like this panel's, the extra machinery has nothing to do. If they don't, it
might be exactly what you need — and you will know which, before you spend a week finding out.

---

# ▶ BEFORE YOU LEAVE

**[STOP — learner works]** — *(replaces the in-room "Discuss")*

Three questions for the board.

**The result.** Your table says the simpler model wins. Describe a forecasting problem where you
would expect the LSTM to win instead, and be specific about the property of the data that changes
the answer — not "more data," but what *kind* of dependence.

**[pause]** — *what you want back: a series where something far in the past must be carried
forward through many irrelevant steps. A promotion announced twelve weeks before it takes effect.
A contract signed in January that governs deliveries in November. That is what gating is for.*

**The practice.** Suppose you had run only `W=26`, one seed, and the LSTM had come out ahead.
Given Lab 7, what would you have needed to do before reporting that an LSTM beats an RNN on this
data?

**For Homework 5.** Part 2 asks you to fit an RNN, an LSTM, and a Transformer, and to explain
where the extra LSTM parameters go and whether they bought anything. You now have two thirds of
that answer. What is your prediction for the Transformer — and, more usefully, what result would
make you wrong?

**[pause]**

Next session you build the Transformer and find out. I will not spoil it, except to say that the
argument you just made about the LSTM is the argument you should try on attention as well, and
that you should check whether it survives.

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
