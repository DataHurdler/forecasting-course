# Lab 7 — Recording Script

**ECON 8310: Business Forecasting · Your First Network — and Why One Run Proves Nothing**

Lab: `Labs/Lecture07_lab.qmd` (5 steps) · Measured runtime: **~22 minutes** of narration
(the in-room version is budgeted at 40)

---

## How to use this document

- **`▶ STEP n — Title`** matches the lab's own headings exactly.
- *Italic parentheticals* are stage directions. **[pause]** is a beat; **[STOP — learner works]**
  is where you tell the viewer to pause the video and do something.
- Numbers are the lab's real output. **Every network here is deterministic** — `torch.manual_seed`
  inside `make_model` and `shuffle=False` in the loader mean seed 1 gives 1,109 every time, on
  every machine. That is unusual for a neural-network lab and it is what makes the seed
  experiment legible rather than mush.
- **The lab takes about 95 seconds to run**, nearly all of it Step 4's fifteen trainings. Plan a
  cut or a cutaway there; do not narrate over a progress-free wait.

### One correction that was applied to the lab

Step 5 said the broken model "lands around **7,400**." Measured, it is **7,325**. The claim built
on it — more than three times worse than the seasonal-naive benchmark — is unaffected (7,325 /
2,152 = 3.4). Changed because this course transcribes measured values exactly.

### Numbers carried in from elsewhere — all verified

Step 2 and the closing quote three figures from outside this lab. I checked all three:
**seasonal naive = 2,152** recomputes exactly on these same 1,590 test weeks; **LASSO 744** and
**XGBoost 781** match Lecture 8's table, Lecture 13's course scoreboard, and Labs 8 and 9 Part 2.
The scoreboard is internally consistent across the whole course.

### In-room language that needs replacing

Step 3's "say out loud what just happened", Step 4's "your turn", and the three closing
**Discuss** blocks. Converted below.

---

# ▶ OPENING

*(Screen: rendered lab, top of document.)*

Lab 7. Two things today, and I want to be honest at the start about which one matters.

The first is the machinery: turning a panel of thirty series into windowed training examples and
training a feedforward network on them. That is the part of Homework 5 most likely to cost you an
evening, so we build it together and you keep the code.

The second is the actual lesson, and it is not about neural networks specifically. You are going
to train the same model several times, on the same data, with the same settings, and get a
different answer every time. Then you are going to measure how big that variation is — and
discover it is larger than the effect you were about to go looking for.

**[pause]**

The data changes today. Labs 4 through 6 were one daily series. Homework 5 is the **weekly panel**
— thirty series, all ten stores by category — so that is what we use, and every number today is
comparable with your Homework 4 results on the same panel.

You need `torch`. Nothing today needs a GPU.

---

# ▶ STEP 1 — From a panel to windows

*(This covers the setup block as well — they run together.)*

*(Screen: run setup, then the windowing loop.)*

```
30 series, 8,310 rows
```

```
X shape (7530, 26, 4)  (windows, weeks, channels)
train 5,940   test 1,590
```

Here is the reframing that every method from here to the end of the course depends on.

A network cannot read a dataframe. It needs a stack of identically-shaped examples. So for each
series we slide a 26-week window along the history: those 26 weeks are the input, the units in
week 27 are the target, and the *target's* date decides whether the example is training or test.

Four channels — units, SNAP days, event days, average price. Twenty-six weeks. So each example is
a 26-by-4 block.

**[pause]**

Now notice the count, because it is a trap dressed as good news. **8,310 rows became 7,530
examples.** It looks like you have seven and a half thousand independent observations. You do not.
Most weeks appear as input in twenty-six different windows, and once more as a target. The
examples overlap almost completely with their neighbors.

So the effective sample size is far smaller than 7,530, and that matters for how much you should
trust anything this network tells you. Half of Homework 5's difficulty is remembering this.

*(Run the standardization block.)*

```
flattened to 104 features per example
```

Two details, both of which are graded in Homework 5.

**The statistics come from training data only.** `mu` and `sd` are computed under `train_mask` and
then applied to everything. Standardizing with statistics computed over the full dataset leaks
test-period information into training, and it will not raise an error — it will just quietly make
your results better than they are.

**And then we flatten.** Twenty-six weeks by four channels becomes 104 numbers in a row. A
feedforward network has no notion of sequence; it sees 104 unordered features. Hold that thought
— it is the answer to one of the closing questions, and it is what Lecture 8 exists to fix.

**This block is Homework 5 Part 1 Question 1.** Keep it.

---

# ▶ STEP 2 — Train a network

*(Screen: run the training cell.)*

```
parameters: 21,761
test RMSE:  1,109
```

Three layers — 104 in, 128, 64, one out — with ReLU and dropout between them. **Twenty-one
thousand seven hundred and sixty-one parameters**, trained on 5,940 overlapping examples.

*(Point at `shuffle=False`.)*

One line deserves attention. `shuffle=False` in the DataLoader. Almost every PyTorch tutorial you
will find sets it to `True`. Here it keeps every run identical, so a seed gives the same number
each time. Shuffling these training windows would not leak anything — each window carries its own
past, and they all come from before the cutoff. What leaks is a random split *before* the time
split, and — this is the theme of the lab — **nothing warns you.** No error, no warning, just a
number that is wrong in a way you cannot see.

**[pause]**

Now the result. 1,109 on the test weeks, against a seasonal-naive benchmark of 2,152. So the
network is roughly twice as good as the benchmark. On a panel where a typical week is about 9,300
units, that is an error of about 12%.

*(Screen: the "compare against your records" paragraph.)*

And now the comparison you should not skip. On this same panel, in Homework 4, **XGBoost managed
781** and a **46-coefficient LASSO managed 744.**

Twenty-two thousand parameters lost to forty-six coefficients. Not narrowly — by about half.

**[pause]**

I am not going to soften that, and you should not expect the rest of the lab to rescue it. It is
the honest result on this dataset, and *why* is one of your discussion questions. But hold the
number 1,109 loosely for about ninety seconds, because Step 3 is going to tell you how much that
particular number is worth.

---

# ▶ STEP 3 — Run it again

*(Screen: run the three seeds.)*

Same code. Same data. Same settings. Same machine. The only thing that changes is the random
seed — which controls the initial weights and which units dropout switches off.

```
seed 1:  test RMSE 1,109
seed 2:  test RMSE 1,176
seed 3:  test RMSE 946
```

**[STOP — learner works]** — *(replaces "say out loud what just happened")*

Pause the video and say what just happened, in one sentence, before I do.

*(Resume.)*

Nothing about the problem changed and the answer moved by **230 RMSE** — from 946 to 1,176.

**[pause]**

Sit with the size of that. 230 is not a rounding error. It is about 20% of the result. It is
larger than most differences you will ever see quoted between two competing architectures in a
blog post.

And notice which seed won. Seed 3 gives **946** — the best number anywhere in this lab. If I had
run seed 3 first and stopped there, I would have reported 946 and believed it. If I had run seed 2
first, I would have reported 1,176 and believed *that*. The reported number would depend entirely
on an integer nobody thinks of as part of the experiment.

That is the whole lab. Everything from here is working out the consequences.

---

# ▶ STEP 4 — So how would you compare two settings?

*(Screen: run the single-seed weight-decay cell.)*

Here is the trap, and it is the single most common mistake in applied deep learning.

You want to know whether weight decay helps. Three values, one training run each, pick the winner.

```
weight_decay=0.0     test RMSE 1,109
weight_decay=0.0001  test RMSE 1,007
weight_decay=0.01    test RMSE 1,142
```

Clear result. `1e-4` wins by **102 RMSE** over no weight decay at all. Nine percent. You would
write that down, you would put it in a slide, and honestly most people would put it in a paper.

**[pause]**

Now do it honestly. Five seeds per setting, and report the spread as well as the mean.

*(Run the five-seed cell. It takes about a minute.)*

```
weight_decay  mean  sd  min  max
           0  1084  86  946 1176
      0.0001  1088 134  991 1343
        0.01  1241 278 1050 1792
```

**[pause]** — *let this sit next to the previous table. This is the moment of the lab.*

The two means differ by **four**.

Not "the effect was smaller than we thought." Not "the advantage was modest." The 102-point
advantage that the single-seed experiment found **did not exist.** Seed 1 happened to be a good
draw for `1e-4` and an average draw for zero, and that is the entire finding.

*(Point at the min and max columns.)*

Look closer and it is worse than that. On `min` and on `max`, no weight decay is *better* than
`1e-4` — 946 against 991, 1,176 against 1,343. The setting with the tighter spread also has the
marginally better mean, by four. There is no effect here in any direction.

**[pause]**

Now the column the single run could never have produced: **`sd`**.

`1e-2` is worse on average — 1,241 — but the important number is 278, more than three times the
spread of no weight decay at all. It returned 1,050 once and 1,792 another time. That is a
different *kind* of bad from a setting that reliably lands at 1,240. One is a model that is
mediocre; the other is a model you cannot make promises about.

You cannot see that in one run. Ever.

**[STOP — learner works]**

Fill in the blank and answer the question in your own words: what is the smallest RMSE difference
you would be willing to call real on this problem?

*(Resume.)*

There is no single right answer, but there is a defensible way to reason. The run-to-run standard
deviation at your best setting is about **86**. Two single runs differing by less than that are
telling you nothing at all. If you want to claim a difference of that size is real, you need
several seeds per setting and you need to compare the distributions, not the winners.

Which means: **a difference under roughly 100 RMSE, from single runs, is not evidence.** And you
will notice that this disqualifies almost every architecture comparison you have ever read.

---

# ▶ STEP 5 — Break it on purpose

*(Screen: the two training functions side by side.)*

One last thing, and it is the same theme in a different costume.

Look at `train_broken`. One line is commented out — `opt.zero_grad()`. Everything else is
identical.

```
with    zero_grad():  1,109
without zero_grad():  7,325
```

**[pause]**

PyTorch **accumulates** gradients by default. It does not replace them, and nothing resets them
at the end of an epoch. If you never zero them, every update is driven by the sum of every gradient
computed since training began — sixty epochs' worth by the end — so the direction of each step has
less and less to do with the batch in front of you.

The result is 7,325. The seasonal-naive benchmark is 2,152. **The broken model is three and a half
times worse than predicting last year's value.**

And the point is not the number, it is everything that did not happen. No exception. No warning.
No NaN. The loop ran to completion in the normal amount of time and returned a float. If you had
not had a benchmark to compare against, you would have had no way to know.

**[pause]**

That is why Lecture 1 spent a whole session on benchmarks, and it is why every scoreboard in this
course starts with seasonal naive. A benchmark is not there to flatter your model. It is the only
thing standing between you and a silent bug.

---

# ▶ BEFORE YOU LEAVE

**[STOP — learner works]** — *(replaces the in-room "Discuss")*

Three questions for the board.

**The seeds.** You now know a single run tells you very little. But papers, blog posts, and vendor
benchmarks routinely report exactly one. When you read "our model achieves RMSE 1,050", what do
you need to know before believing it beats a baseline at 1,100? Be specific about what you would
ask for — and notice that today, a 50-point gap is well inside one standard deviation.

**[pause]**

**The comparison.** Twenty-two thousand parameters lost to a forty-six-coefficient LASSO, 1,084
against 744. Give one reason that is *not* "neural networks need more data." Think about what your
network can actually see: 104 unordered numbers. It does not know that feature 1 and feature 5 are
the same channel a week apart, or that the channels are four different quantities. It has to learn
the layout of its own input from 5,940 heavily overlapping examples — and the LASSO was handed
that structure by whoever engineered the features.

**The practice.** In Homework 5 you compare five architectures. Given today, what will you
actually do when one comes out 80 RMSE ahead of another?

**[pause]**

I'll tell you what the honest answer is, because it is also the assignment's answer: you say the
difference is inside the noise, and you say so *even when the architecture you like is ahead*.
Homework 5 Part 3 asks you to rank five architectures. Some of those gaps will be real and some
will be seeds. Knowing which is which is the skill being tested.

Solutions for the coded parts go up on Canvas after the deadline. Next week: a network that knows
its 104 inputs came from a sequence.

---

# Appendix — expected output

Deterministic — `torch.manual_seed` per model, `shuffle=False` in the loader. Seed 1 returns 1,109
on every run. Whole lab ≈ 95 seconds, of which Step 4's five-seed table is ~60.

| Quantity | Value |
|---|---|
| Panel | 30 series, 8,310 rows → **7,530** windows (train 5,940, test 1,590) |
| Window | 26 weeks × 4 channels → flattened to **104** features |
| Model | 3 layers (128, 64, 1), dropout 0.2 — **21,761** parameters |
| Seed 1 / 2 / 3 | **1,109** · 1,176 · **946** (range 230) |
| Single-seed weight decay | 0.0: 1,109 · 1e-4: **1,007** · 1e-2: 1,142 → apparent 102-point win |
| Five seeds, wd = 0 | mean **1,084** · sd **86** · min 946 · max 1,176 |
| Five seeds, wd = 1e-4 | mean **1,088** · sd 134 · min 991 · max 1,343 |
| Five seeds, wd = 1e-2 | mean 1,241 · sd **278** · min 1,050 · max 1,792 |
| Difference in means, 0 vs 1e-4 | **4** |
| Without `zero_grad()` | **7,325** (vs 1,109) |
| Seasonal naive on these test weeks | **2,152** — recomputed and verified |
| Homework 4 on this panel | LASSO (46 features) **744** · XGBoost **781** |
| Mean weekly units, test period | 9,301 |

**Three things to know before recording.**

*Step 4 is the lab; protect its screen time.* The two tables need to be visible together, and the
"differ by four" beat needs a real pause. If the recording is running long, compress Step 1's
narration, not this.

*Do not let Step 2's 1,109 settle as "the network's score."* It is one draw from a distribution
whose standard deviation is 86 and whose range across five seeds is 230. The mean, 1,084, is the
honest figure, and Step 3 arrives ninety seconds later to make that point — so flag it as
provisional when you first say it.

*The 45% loss to the LASSO is a real result and should not be hedged.* 1,084 against 744, on the
same panel and the same test weeks. It is the setup for Lecture 8 and for the closing question
about what a flattened window throws away, and understating it would waste both.
