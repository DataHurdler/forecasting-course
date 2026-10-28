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

*(Show the rendered lab, at the top of the document.)*

This is Lab 8, and it starts with the failure we ended on last week.

Last week, in Lab 7, you took a twenty-six-week window, flattened it into one hundred four numbers,
and handed it to a network. That model had no idea the window was ordered. Week one and week
twenty-six were just two positions in a list, and nothing told it that positions one and five held
the same quantity, one week apart.

A convolution knows. It slides a small filter along the time axis and looks for local shapes, and
the same filter is applied everywhere, so a pattern learned at week three is recognized at week
twenty.

**[pause]**

But today's real subject isn't that convolutions work. It's **how far back one can actually see**.
That turns out to be far less than the window you give it, it's entirely calculable before you
train anything, and it's the single most useful thing to understand about this architecture.

And every comparison today runs on three seeds, because of what last week's lab taught you about
trusting a single run.

---

# ▶ SETUP

*(Run the setup cell.)*

```
shapes at W=26: torch.Size([5940, 4, 26]) (batch, channels, time)
```

This is the same windowing as Lab 7, wrapped in a `build()` function so we can rebuild it at three
window lengths. It's the same panel, the same cutoff and the same split between training and test,
and the standardizing statistics come from the training rows only.

One change matters, and it's in the last line of `build`.

*(Point at the transpose.)*

Conv one-D wants its input as batch, then channels, then time. Your natural windowing produces
batch, then time, then channels. So there's a transpose, and the comment calls it "the whole
point."

*(Show the warning callout.)*

Read this callout properly, because half of it is the same species of bug as last week's missing
`zero_grad()`.

The loud half first. If you skip the transpose and hand the network batch, then time, then
channels, PyTorch stops you. The first layer expects four channels, it gets twenty-six, and the
error message says exactly that. That mistake costs you a minute.

The quiet half is the dangerous one. Suppose you fix the shape with `reshape` instead of a
transpose. The shape comes out exactly right, four by twenty-six, so **nothing crashes**. But
reshape doesn't move the axes. It pours the numbers into the new shape in their old order, so the
first "channel" now holds the first six and a half weeks of all four variables, interleaved:
units, SNAP days, event days, price, then units again. The model trains on that and returns a
number that looks like an RMSE.

**[pause]**

That's the third time this course has shown you the same shape of failure. First came a random
split made before the time split, then a missing `zero_grad()` in Lab 7, and now a reshape where a
transpose belonged.
None of them raises an error, and all of them return a plausible number. That's what makes deep
learning code different from regression code: the language can't tell you that your tensor means
the wrong thing.

---

# ▶ STEP 1 — How far back can the filter see?

*(Show the receptive-field explanation, then run the cell.)*

Before we build anything, we'll do the arithmetic on paper. This is the calculation that decides
whether the architecture can possibly work, and it takes ten seconds.

Our network is a Conv one-D layer that takes the four channels to thirty-two filters with a kernel
size of five, followed by a second Conv one-D layer that takes those thirty-two to sixty-four
filters with a kernel size of three.

A neuron in the first layer sees **five** consecutive weeks, because that's what a kernel size of
five means.

A neuron in the second layer sees three first-layer neurons. But those three overlap, since each
one sees five weeks, shifted by one. So between them they span three, plus five minus one, which
comes to **seven** weeks.

```
kernels (5, 3)      -> receptive field 7 weeks
kernels (5, 3, 3)   -> receptive field 9 weeks
kernels (5, 3, 3, 3)-> receptive field 11 weeks
```

*(Let the table sit for a moment. It's the whole lab in one table.)*

**[pause]**

That's **seven weeks**, not twenty-six.

And look at what stacking buys. Each extra layer with a kernel of three adds just two weeks. You go
from seven to nine to eleven, one whole convolutional layer at a time.

Hand this network a fifty-two-week window, and every convolution unit still sees only seven
consecutive weeks of it. The flatten layer after them gives each position its own weight, so the
network as a whole does see all fifty-two weeks. But it has to learn the annual relationship one
weight per position, with nothing shared across positions.

Hold that, because Step 4 measures what it costs.

---

# ▶ STEP 2 — Build the CNN

*(Run the build-and-train cell.)*

```
parameters: 8,161
test RMSE:  980
```

This is two convolutional layers, then a head that flattens and maps to one output.

**[pause]**

Compare that parameter count with last week's network. It's **eight thousand one hundred
sixty-one, against twenty-one thousand seven hundred sixty-one**. The CNN is about a third the size
of the feedforward network, because its filters are *shared* across all the time positions,
instead of every input getting its own weight. And on this single seed it scores nine eighty,
against the feedforward network's eleven-oh-nine.

So it's a smaller model with a better number. That's the convolutional bargain: weight sharing buys
you efficiency, as long as the thing you're looking for is local.

And after Step 1, you know that for these filters, local means seven weeks. Keep that qualifier
attached.

---

# ▶ STEP 3 — Does pooling belong here?

*(Start the cell, which takes about four minutes, then talk over the setup while it runs.)*

Lecture 8 argued that pooling buys **translation invariance**. An image classifier shouldn't care
*where* in the frame the cat is, so pooling throws away position and keeps the presence of the
pattern. That's exactly right for photographs.

The claim was that a forecaster usually *should* care where the spike was. A demand spike three
weeks ago and a demand spike twenty weeks ago are different facts about next week.

So we'll test three heads, with three seeds each.

```
   head  mean  sd               runs
flatten  1032  80   [980, 1145, 972]
maxpool  1418  47 [1433, 1354, 1467]
avgpool  1534  81 [1643, 1448, 1509]
```

**[pause]**

Now, **read the spread** before you read the means. That's the habit last week's lab was for.

The run-to-run standard deviation is forty-seven to eighty-one. The gap between the flatten head
and average pooling is **five hundred two**, which is six to ten times the noise. Unlike last
week's weight decay comparison, this difference is unambiguously real, and you can say so with
three runs precisely because you measured the spread.

**[pause]**

The mechanism is worth stating exactly. The flatten head keeps all sixty-four channels at all twenty
surviving time positions, and it hands the final layer one thousand two hundred eighty numbers, so
the model knows *where* each pattern was found. The adaptive average pooling layer collapses the
time axis to a single average for each channel, which leaves sixty-four numbers. It keeps "this
pattern occurred" and discards "it occurred in week twenty-two."

**[STOP — learner works]** — *(replaces "say it in one sentence")*

Pause the video and answer this: why is averaging over position right for classifying photographs,
and wrong for forecasting demand?

*(Resume.)*

It's because in a photograph, position is a nuisance. The cat is a cat wherever it sits, and a
model that has to learn "cat in the top left" and "cat in the middle" separately is wasting
capacity. In a forecast, position *is* the signal. Recency is the single most informative thing
about a lag, and average pooling deletes it.

Notice, too, that max pooling is less bad than average pooling, at fourteen eighteen against
fifteen thirty-four. That fits, because max pooling at least preserves the strongest response
rather than diluting it across the window. Both are far behind keeping the positions.

---

# ▶ STEP 4 — A longer window, the same receptive field

*(Start the cell, which takes about three minutes, then talk through what it does.)*

Now comes the experiment Step 1 set up, and it's the reason this lab exists.

Keep the architecture completely fixed, with the same two layers, the same kernels and the same
seven-week receptive field, and vary only how much history you hand it: thirteen weeks, twenty-six,
or fifty-two. There are three seeds each, and we reuse Step 3's three flatten runs for the
twenty-six-week row, rather than paying for them twice.

```
 window  receptive_field  parameters  mean_RMSE  sd               runs
     13                7        7329        980  42   [1039, 960, 943]
     26                7        8161       1032  80   [980, 1145, 972]
     52                7        9825       1103  26 [1111, 1068, 1131]
```

*(Let the table sit for a moment.)*

**[pause]**

Quadrupling the window made the forecast **worse**.

*(Show the callout.)*

Now let me show you how to read this table honestly, because there are two different claims in it,
and only one of them is safe.

The middle comparison isn't evidence. Thirteen weeks against twenty-six is nine eighty against one
thousand thirty-two, a gap of fifty-two, when the twenty-six-week row's own standard deviation is
eighty. Last week you decided that differences smaller than about a hundred in RMSE, from a handful
of runs, aren't findings. That rule applies to results you like as much as to results you don't.
Those two windows are indistinguishable.

The ends are evidence. Thirteen weeks against fifty-two is nine eighty against eleven-oh-three, a
gap of **one hundred twenty-three**, and the spreads don't touch. The *worst* thirteen-week run,
one thousand thirty-nine, still beats the *best* fifty-two-week run, one thousand sixty-eight.
That's three seeds each with no overlap, and that's a result you can report.

**[pause]**

So here's the finding. Giving this architecture four times the history made it measurably worse,
and giving it twice the history did nothing at all.

Why? Look at the two columns on either side of the RMSE. The receptive field is **seven** on every
row. It doesn't move, because nothing about the architecture changed. The parameter count goes from
seven thousand three hundred twenty-nine, to eight thousand one hundred sixty-one, to nine thousand
eight hundred twenty-five, because the flatten layer's input grows with the window.

So a longer window adds reach and capacity together. The flatten head can now weigh last year's
same week, but every extra week also means more weights to fit, from fewer training windows. Here
the capacity won, and the fifty-two-week row is what overfitting looks like.

The lesson is that **more history into the model isn't automatically better**. The context has to
be worth the weights it costs.

**[STOP — learner works]**

Now it's your turn. Pause the video, fill in the blank in the next cell, and find out how many
layers with a kernel of three you'd have to stack after that first layer, the one with a kernel of
five, to span a full fifty-two weeks.

*(Resume.)*

The answer is **twenty-four**. That's twenty-five convolutional layers in total, for a receptive
field of fifty-three weeks. Twenty-three isn't enough, because it only reaches fifty-one.

Sit with that for a second. To let this architecture see one year, you'd need a twenty-five-layer
network, trained on five thousand nine hundred forty examples that overlap almost completely.
Nobody is going to do that, and if they did, it would overfit spectacularly.

That's the honest limit of a plain 1D CNN for this problem, and it's why the next lecture is about a
different mechanism rather than a deeper stack of the same one. Real practitioners reach for dilated
convolutions here. Those are filters with gaps in them, so the receptive field grows multiplicatively
instead of by two per layer. That's a genuine third option, and it's worth knowing the name of.

---

# ▶ STEP 5 — Where does it land?

*(Run the scoreboard.)*

```
LASSO, 46 features (HW4)          744
XGBoost (HW4)                     781
1D CNN, flatten head (today)     1032
FFN, flattened window (Lab 7)    1084
Seasonal naive                   2152
```

**[pause]**

Let's say the honest version of this table.

The CNN at one thousand thirty-two and the feedforward network at one thousand eighty-four differ
by **fifty-two**, and you measured the run-to-run spread at eighty. So the correct statement is
*not* that the CNN beat the feedforward network. It's that **the two are indistinguishable on this
problem**. The CNN has about a third of the parameters and a genuinely better inductive bias,
meaning a better built-in assumption about the data, and it preserves the ordering. All of that
bought nothing measurable.

Both are well behind a LASSO with forty-six coefficients, at seven forty-four.

**[pause]**

That isn't a failure of the lab, and it isn't a bad implementation. It's a result about the
problem. The useful signal in weekly store sales sits at long lags, like last year's same week and
the shape of the year, and this architecture reaches them only through the flatten layer, one
weight per position. At twenty-six weeks, the window doesn't reach a year at all. At fifty-two it
does, and the extra weights cost more than the extra context bought.

That's the honest reason to move on to Lecture 9, next week, and it isn't a reason to conclude that
convolutions are bad.

---

# ▶ BEFORE YOU LEAVE

**[STOP — learner works]** — *(replaces the in-room "Discuss")*

Here are three questions to work through on your own before you go.

The first is about the architecture. You've now seen two distinct ways to waste a window. The
feedforward network flattens it and loses the ordering entirely. The CNN preserves the ordering,
but its filters can only look at seven weeks at a time. Without naming an architecture, describe
what a model would need in order to use a whole fifty-two-week window properly.

**[pause]**

Whatever words you used, what you're describing is either recurrence or attention. Recurrence comes
next week, in Lecture 9 Part 1, and attention follows the week after, in Part 2.

The second question is about practice. Step 4's result, a longer window and a worse forecast, is
the sort of thing a practitioner discovers after a week of tuning. What one calculation, done in ten
seconds back in Step 1, would have predicted it?

**[pause]**

It's the receptive field: seven weeks. However long the window gets, no filter sees more than seven
weeks of it, so a longer window can't show any filter a longer pattern. The only part of the network
that reaches the extra weeks is the flatten head, and it does that by giving every new position its
own weights, from seven thousand three hundred twenty-nine parameters at thirteen weeks to nine
thousand eight hundred twenty-five at fifty-two, learned from fewer training windows. That one
number told you the extra history would have to pay for those weights, and here it didn't.

The third question looks ahead to Homework 5. Question 4 in Part 1 of that homework asks you to run
this window experiment and explain the limit, and you now have the answer. What it does *not* tell
you is what to change. Name two architectural options, and say which you'd try first. Note that
"stack more convolutional layers" is one of them, so work out how many you'd need before you
recommend it.

Solutions go up on Canvas after class. Next week, we'll build a model that carries state forward
through the whole window.

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
