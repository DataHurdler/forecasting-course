# Lab 8 — Recording Script

**ECON 8310: Business Forecasting · Convolutions — What the Filter Can Actually See**

Lab: `Labs/Lecture08_lab.qmd` (5 steps) · Measured runtime: **~22 minutes** of narration
(the in-room version is budgeted at 40)

---

## How to use this document

- **`▶ STEP n — Title`** matches the lab's own headings exactly.
- *Italic parentheticals* are stage directions. **[pause]** is a beat; **[STOP — learner works]**
  is where you tell the viewer to pause the video and do something.
- Every network is deterministic (`torch.manual_seed` per model, `shuffle=False`), so the tables
  reproduce exactly.
- **This is the slowest lab in the course.** Step 3 trains nine networks and Step 4 six more.
  Budget about **four minutes** for Step 3 and **three** for Step 4 on a laptop, and plan a cut —
  do not narrate over a silent wait.

### Two corrections that were applied to the lab

1. **Step 4 was under-seeded, and it contradicted Lab 7.** The window comparison ran **two** seeds
   per window and concluded that a longer window "hurts, **monotonically**." The gaps it was
   reading were 64 and 27 RMSE — and Lab 7, one week earlier, had just established that on this
   panel a difference under about 100 RMSE from a couple of runs is not evidence. The lab was
   committing the exact error the previous lab exists to teach.
   I re-ran the experiment at **five** seeds per window. The ranking survives at the ends and
   dissolves in the middle: 13 weeks and 26 weeks are **indistinguishable** (975 ± 48 against
   984 ± 87), while 52 weeks is **clearly worse** (1,094 ± 35). Step 4 now runs **three** seeds
   per window — reusing Step 3's three `flatten` runs for W=26, so it costs six new fits, not
   nine — and prints the individual runs beside the mean. The callout no longer claims
   monotonicity; it claims what the data supports, which is still the lesson.
2. **The slow-cell warning was optimistic.** It said Steps 3 and 4 take "roughly two minutes
   apiece." Measured, Step 3 is about four minutes and Step 4 about three.

### In-room language that needs replacing

Step 3's "say it in one sentence" block, Step 4's "your turn", and the three closing **Discuss**
blocks. All converted below.

---

# ▶ OPENING

*(Screen: rendered lab, top of document.)*

Lab 8, and it starts with the failure we ended on last week.

Last week you took a 26-week window, flattened it into 104 numbers, and handed it to a network.
That model had no idea the window was ordered. Week 1 and week 26 were just two positions in a
list, and nothing told it that positions 1 and 5 were the same quantity four weeks apart.

A convolution knows. It slides a small filter along the time axis and looks for local shapes, and
the same filter is applied everywhere — so a pattern learned at week 3 is recognized at week 20.

**[pause]**

But today's real subject is not that convolutions work. It is **how far back one can actually
see** — which is far less than the window you give it, is entirely calculable before you train
anything, and is the single most useful thing to understand about this architecture.

And everything today runs on multiple seeds, because of last week.

---

# ▶ SETUP

*(Screen: run setup.)*

```
shapes at W=26: torch.Size([5940, 4, 26]) (batch, channels, time)
```

Same windowing as Lab 7, wrapped in a `build()` function so we can rebuild it at three window
lengths. Same panel, same cutoff, same train/test split, statistics from training rows only.

One change matters, and it is in the last line of `build`.

*(Point at the transpose.)*

`Conv1d` wants **(batch, channels, time)**. Your natural windowing produces **(batch, time,
channels)**. So there is a transpose, and the comment calls it "the whole point."

*(Screen: the warning callout.)*

Read this callout properly, because it is the same species of bug as last week's missing
`zero_grad()`.

Get the transpose backwards and **nothing crashes.** The shapes are still valid — you have four
of one thing and twenty-six of another, and `Conv1d` will happily convolve across the four
channels as though *they* were the time axis. It will slide a filter over "units, SNAP days, event
days, price" in that order, learn something with no meaning at all, and return a number that looks
like an RMSE.

**[pause]**

That is the third time this course has shown you the same shape of failure: `shuffle=True` on a
time series, a missing `zero_grad()`, and now a transposed axis. None of them raise. All of them
return a plausible number. This is what makes deep learning code different from regression code —
the language cannot tell you that your tensor means the wrong thing.

---

# ▶ STEP 1 — How far back can the filter see?

*(Screen: the receptive-field explanation, then run the cell.)*

Before we build anything, we do the arithmetic on paper. This is the calculation that decides
whether the architecture can possibly work, and it takes ten seconds.

Our network is `Conv1d(4, 32, kernel_size=5)` and then `Conv1d(32, 64, kernel_size=3)`.

A neuron in the first layer sees **five** consecutive weeks — that is what kernel size 5 means.

A neuron in the second layer sees three first-layer neurons. But those three overlap: each sees
five weeks, shifted by one. So between them they span 3 + (5 − 1) = **seven** weeks.

```
kernels (5, 3)      -> receptive field 7 weeks
kernels (5, 3, 3)   -> receptive field 9 weeks
kernels (5, 3, 3, 3)-> receptive field 11 weeks
```

**[pause]** — *let this sit. It is the lab in one table.*

**Seven weeks.** Not twenty-six. Seven.

And look at what stacking buys: each extra kernel-3 layer adds **two** weeks. Two. You go from
seven to nine to eleven, one whole convolutional layer at a time.

Hand this network a 52-week window and every convolution unit still sees only seven consecutive
weeks of it. The flatten layer after them gives each position its own weight, so the network as a
whole does see all fifty-two weeks — but it has to learn the annual relationship one weight per
position, with nothing shared across positions.

Hold that. Step 4 measures what it costs.

---

# ▶ STEP 2 — Build the CNN

*(Screen: run the build-and-train cell.)*

```
parameters: 8,161
test RMSE:  980
```

Two convolutional layers, then a head that flattens and maps to one output.

**[pause]**

Compare that parameter count with last week: **8,161 against 21,761.** The CNN is about a third
the size of the feedforward network — because the filters are *shared* across all time positions
instead of every input getting its own weight. And on this single seed it scores 980, against the
FFN's 1,109.

Smaller model, better number. That is the convolutional bargain: weight sharing buys you
efficiency, as long as the thing you are looking for is local.

Which — per Step 1 — means seven weeks local. Keep that qualifier attached.

---

# ▶ STEP 3 — Does pooling belong here?

*(Screen: start the cell — it takes about four minutes — then talk over the setup.)*

Lecture 8 argued that pooling buys **translation invariance.** An image classifier should not care
*where* in the frame the cat is; pooling throws away position and keeps the presence of the
pattern. That is exactly right for photographs.

The claim was that a forecaster usually *should* care where the spike was. A demand spike three
weeks ago and a demand spike twenty weeks ago are different facts about next week.

Three heads, three seeds each.

```
   head  mean  sd               runs
flatten  1032  80   [980, 1145, 972]
maxpool  1418  47 [1433, 1354, 1467]
avgpool  1534  81 [1643, 1448, 1509]
```

**[pause]**

**Read the spread before you read the means** — that is the habit last week was for.

Run-to-run standard deviation is 47 to 81. The gap between `flatten` and `avgpool` is **502**.
That is six to ten times the noise. Unlike last week's weight decay comparison, this difference
is unambiguously real, and you can say so with three runs precisely because you measured the
spread.

**[pause]**

The mechanism is worth stating exactly. `Flatten` keeps all 64 channels at all 20 surviving time
positions and hands the final layer 1,280 numbers — so the model knows *where* each pattern was
found. `AdaptiveAvgPool1d(1)` collapses the time axis to a single average per channel: 64 numbers.
It keeps "this pattern occurred" and discards "it occurred in week 22."

**[STOP — learner works]** — *(replaces "say it in one sentence")*

Pause and answer: why is averaging over position right for classifying photographs and wrong for
forecasting demand?

*(Resume.)*

Because in a photograph, position is nuisance — the cat is a cat wherever it sits, and a model
that has to learn "cat in the top left" and "cat in the middle" separately is wasting capacity.
In a forecast, position *is* the signal. Recency is the single most informative thing about a lag,
and average pooling deletes it.

Note that max pooling is less bad than average pooling — 1,418 against 1,534 — which fits: max at
least preserves the strongest response rather than diluting it across the window. Both are far
behind keeping the positions.

---

# ▶ STEP 4 — A longer window, the same receptive field

*(Screen: start the cell — about three minutes — then talk through what it does.)*

Now the experiment Step 1 set up, and it is the reason this lab exists.

Keep the architecture completely fixed — same two layers, same kernels, same seven-week receptive
field — and vary only how much history you hand it. Thirteen weeks, twenty-six, fifty-two. Three
seeds each, and we reuse Step 3's three `flatten` runs for the 26-week row rather than paying for
them twice.

```
 window  receptive_field  parameters  mean_RMSE  sd               runs
     13                7        7329        980  42   [1039, 960, 943]
     26                7        8161       1032  80    [980, 1145, 972]
     52                7        9825       1103  26  [1111, 1068, 1131]
```

**[pause]** — *let it sit.*

Quadrupling the window made the forecast **worse**.

*(Screen: the callout.)*

Now let me show you how to read this table honestly, because there are two different claims in it
and only one of them is safe.

**The middle comparison is not evidence.** Thirteen weeks against twenty-six is 980 against 1,032
— a gap of 52, when the 26-week row's own standard deviation is 80. Last week you decided that
differences smaller than about 100 RMSE, from a handful of runs, are not findings. That rule
applies to results you like as much as results you don't. Those two windows are
indistinguishable.

**The ends are evidence.** Thirteen weeks against fifty-two is 980 against 1,103 — **123 RMSE**,
and the spreads do not touch: the *worst* 13-week run, 1,039, still beats the *best* 52-week run,
1,068. Three seeds each, no overlap. That one you can report.

**[pause]**

So the finding is: giving this architecture four times the history made it measurably worse, and
giving it twice the history did nothing at all.

Why? Look at the two columns either side of the RMSE. The receptive field is **7** on every row —
it does not move, because nothing about the architecture changed. The parameter count goes 7,329,
8,161, 9,825 — because the flatten layer's input grows with the window.

So a longer window adds reach and capacity together. The flatten head can now weigh last year's
same week, but every extra week is also more weights to fit, from fewer training windows. Here the
capacity won, and the 52-week column is what overfitting looks like.

**More data into the model is not more information out of it, unless the architecture can reach
it.**

**[STOP — learner works]**

Fill in the blank in the next cell and find out how many kernel-3 layers you would have to stack
after that first kernel-5 layer to span a full 52 weeks.

*(Resume.)*

The answer is **24** — twenty-five convolutional layers in total, for a receptive field of 53
weeks. Twenty-three is not enough; it reaches 51.

Sit with that for a second. To let this architecture see one year, you would need a
twenty-five-layer network — on 5,940 training examples that overlap almost completely. Nobody is
going to do that, and if they did it would overfit spectacularly.

That is the honest limit of a plain 1D CNN for this problem, and it is why the next lecture is
about a different mechanism rather than a deeper stack of the same one. Real practitioners reach
for dilated convolutions here — filters with gaps in them, so the receptive field grows
multiplicatively instead of by two per layer. That is a genuine third option and worth knowing the
name of.

---

# ▶ STEP 5 — Where does it land?

*(Screen: run the scoreboard.)*

```
LASSO, 46 features (HW4)          744
XGBoost (HW4)                     781
1D CNN, flatten head (today)     1032
FFN, flattened window (Lab 7)    1084
Seasonal naive                   2152
```

**[pause]**

Say the honest version of this table.

The CNN at 1,032 and the FFN at 1,084 differ by **52** — and you measured the run-to-run spread at
80. So the correct statement is *not* "the CNN beat the feedforward network." It is: **the two
are indistinguishable on this problem.** One-third the parameters, a genuinely better inductive
bias, ordering preserved — and it bought nothing measurable.

Both are well behind a 46-coefficient LASSO at 744.

**[pause]**

That is not a failure of the lab and it is not a bad implementation. It is a result about the
problem. The useful signal in weekly store sales sits at long lags — last year's same week, the annual
shape — and this architecture reaches them only through the flatten layer, one weight per
position. At twenty-six weeks the window does not reach a year at all; at fifty-two it does, and
the extra weights cost more than the extra context bought.

Which is the honest reason to move on to Lecture 9, and not a reason to conclude that
convolutions are bad.

---

# ▶ BEFORE YOU LEAVE

**[STOP — learner works]** — *(replaces the in-room "Discuss")*

Three questions for the board.

**The architecture.** You have now seen two distinct ways to waste a window. The FFN flattens it
and loses the ordering entirely. The CNN preserves ordering but can only look at seven weeks at a
time. Without naming an architecture, describe what a model would need in order to use a whole
52-week window properly.

**[pause]** — *they are describing recurrence or attention, and next week they get both.*

**The practice.** Step 4's result — longer window, no better or worse forecast — is the sort of
thing a practitioner discovers after a week of tuning. What one calculation, done in Step 1 in ten
seconds, would have predicted it?

**For Homework 5.** Part 1 Question 4 asks you to run this window experiment and explain the
limit. You now have the answer. What it does *not* tell you is what to change. Name two
architectural options and say which you would try first — and note that "stack more conv layers"
is one of them, so work out how many you would need before you recommend it.

Solutions for the coded parts go up on Canvas after the deadline. Next week: a model that carries
state forward through the whole window.

---

# Appendix — expected output

Deterministic; `torch.manual_seed` per model, `shuffle=False`. **Slowest lab in the course** —
about eight minutes end to end, with Step 3 ≈ 4 min (9 fits) and Step 4 ≈ 3 min (6 new fits).

| Quantity | Value |
|---|---|
| Tensor shape at W=26 | (5940, 4, 26) — batch, **channels**, time |
| Receptive field | (5,3): **7** weeks · (5,3,3): 9 · (5,3,3,3): 11 — **+2 per layer** |
| CNN, W=26, flatten head | **8,161** parameters (FFN in Lab 7: 21,761) · seed 1 → 980 |
| Head comparison, 3 seeds | flatten **1032** (sd 80) · maxpool 1418 (sd 47) · avgpool 1534 (sd 81) |
| flatten vs avgpool gap | **502** — six to ten times the run-to-run spread |
| Window, 3 seeds — 13 | mean **980** · sd 42 · runs [1039, 960, 943] · 7,329 params |
| Window, 3 seeds — 26 | mean **1032** · sd 80 · runs [980, 1145, 972] · 8,161 params |
| Window, 3 seeds — 52 | mean **1103** · sd 26 · runs [1111, 1068, 1131] · 9,825 params |
| 13 vs 52 | **123 RMSE**, no overlap (worst 13 = 1,039 < best 52 = 1,068) |
| 13 vs 26 | 52 RMSE — **inside the noise**, not a finding |
| Five-seed confirmation (not in the lab) | 13: 975 ± 48 · 26: 984 ± 87 · 52: 1,094 ± 35 |
| Layers to span 52 weeks | **24** kernel-3 layers after the kernel-5 (25 total → 53 weeks) |
| Scoreboard | LASSO 744 · XGBoost 781 · **CNN 1032** · FFN 1084 · seasonal naive 2152 |

**Three things to know before recording.**

*Step 4 is the lab and it now contains a deliberate non-finding.* The middle comparison (13 vs 26,
a 52-point gap against an sd of 80) is there to be *rejected*, one week after Lab 7 taught the
rule. Do not let it get delivered as a result — the whole point is that a careful person declines
to read it, and then reads the ends of the table instead, where the evidence is clean.

*The head comparison in Step 3 is the opposite case and should sound different.* A 502-point gap
against a spread of 47–81 is about as unambiguous as this course gets. Say it with the confidence
the numbers earn — Step 3 and Step 4 together are what "measure the spread first" is *for*, one
example of each verdict.

*The deck agrees with this lab exactly.* Lecture 8's table carries 1,032 / 1,418 / 1,534 and
8,161 parameters, and Lecture 13's course scoreboard carries the 1,032. If you re-run and get
different numbers, something in the environment has changed — check before publishing, because
three other files quote these.
