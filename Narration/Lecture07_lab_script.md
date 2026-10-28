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

*(Show the rendered lab, at the top of the document.)*

This is Lab 7. We're doing two things today, and I want to be honest from the start about which one
matters.

The first is the machinery. We'll turn a panel of thirty series into windowed training examples,
and then train a feedforward network on them. That's the part of Homework 5 most likely to cost you
an evening, so we'll build it together, and you keep the code.

The second is the actual lesson, and it isn't about neural networks specifically. You're going to
train the same model several times, on the same data, with the same settings, and get a different
answer every time. Then you're going to measure how big that variation is, and discover that it's
larger than the effect you were about to go looking for.

**[pause]**

The data changes today. Labs 4 through 6, over the last three weeks, all used one daily series.
Homework 5 uses the **weekly panel**, which is thirty series, covering all ten stores by category.
So that's what we use today, and every number you see is comparable with your Homework 4 results on
the same panel.

You'll need torch, but nothing today needs a GPU.

---

# ▶ STEP 1 — From a panel to windows

*(This covers the setup block as well, because they run together.)*

*(Run the setup cell, then the windowing loop.)*

```
30 series, 8,310 rows
```

```
X shape (7530, 26, 4)  (windows, weeks, channels)
train 5,940   test 1,590
```

Here's the reframing that every method from here to the end of the course depends on.

A network can't read a dataframe. It needs a stack of examples that all have the same shape. So for
each series, we slide a twenty-six-week window along the history. Those twenty-six weeks are the
input, the units in week twenty-seven are the target, and the *target's* date decides whether the
example goes into training or test.

There are four channels: units, SNAP days, event days and the average price. With twenty-six weeks,
each example is a block of twenty-six by four.

**[pause]**

Now notice the count, because it's a trap dressed up as good news. **Eight thousand three hundred
ten rows became seven thousand five hundred thirty examples.** It looks as if you have seven and a
half thousand independent observations. You don't. Most weeks appear as input in twenty-six
different windows, and once more as a target, so the examples overlap almost completely with their
neighbors.

So the effective sample size is far smaller than seven thousand five hundred thirty, and that
matters for how much you should trust anything this network tells you. Half of the difficulty in
Homework 5 is remembering this.

*(Run the standardization block.)*

```
flattened to 104 features per example
```

There are two details here, and both of them are graded in Homework 5.

The first is that the statistics come from the training data only. The mean and the standard
deviation, mu and s-d in the code, are computed under the training mask and then applied to
everything. If you standardize with statistics computed over the full dataset, you leak information
from the test period into training. It won't raise an error. It will just quietly make your results
look better than they are.

The second is that we then flatten. Twenty-six weeks by four channels becomes a hundred and four
numbers in a row. A feedforward network has no notion of sequence, so it sees a hundred and four
unordered features. Hold on to that thought, because it's the answer to one of the closing
questions, and it's what Lecture 8, next week, exists to fix.

This block is what Homework 5, Part 1, Question 1 asks for, so **keep it**.

---

# ▶ STEP 2 — Train a network

*(Run the training cell.)*

```
parameters: 21,761
test RMSE:  1,109
```

The network has three layers. A hundred and four inputs go into a hundred twenty-eight units, then
sixty-four, then one output, with ReLU and dropout between them. That's **twenty-one thousand seven
hundred sixty-one parameters**, trained on five thousand nine hundred forty overlapping examples.

*(Point at `shuffle=False`.)*

One line deserves attention, and that's shuffle equals False, in the DataLoader. Almost every
PyTorch tutorial you'll find sets it to True. Here it keeps every run identical, so a given seed
gives the same number each time. Shuffling these training windows wouldn't leak anything, because
each window carries its own past, and they all come from before the cutoff. What leaks is a random
split made *before* the time split. And this is the theme of the lab: **nothing warns you**. There's
no error and no warning, just a number that's wrong in a way you can't see.

**[pause]**

Now the result. The network scores eleven-oh-nine on the test weeks, against a seasonal-naive
benchmark of twenty-one fifty-two, so it's roughly twice as good as the benchmark. On a panel where
a typical week is about nine thousand three hundred units, that's an error of about twelve percent.

*(Show the paragraph that begins "Compare that RMSE against your records.")*

Now for the comparison you shouldn't skip. On this same panel, in Homework 4, **XGBoost managed
seven eighty-one**, and a LASSO with forty-six coefficients managed **seven forty-four**.

So twenty-two thousand parameters lost to forty-six coefficients, and not narrowly. They lost by
about half.

**[pause]**

I'm not going to soften that, and you shouldn't expect the rest of the lab to rescue it. It's the
honest result on this dataset, and *why* it happens is one of your discussion questions. But hold
the number eleven-oh-nine loosely for about ninety seconds, because Step 3 is going to tell you how
much that particular number is worth.

---

# ▶ STEP 3 — Run it again

*(Run the three seeds.)*

It's the same code, the same data, the same settings and the same machine. The only thing that
changes is the random seed, which controls the initial weights and which units dropout switches off.

```
seed 1:  test RMSE 1,109
seed 2:  test RMSE 1,176
seed 3:  test RMSE 946
```

**[STOP — learner works]** — *(replaces "say out loud what just happened")*

Pause the video, and say what just happened in one sentence, before I do.

*(Resume.)*

Nothing about the problem changed, and the answer moved by **two hundred thirty** in RMSE, from nine
forty-six to eleven seventy-six.

**[pause]**

Sit with the size of that. Two hundred thirty isn't a rounding error. It's about twenty percent of
the result, and it's larger than most of the differences you'll ever see quoted between two
competing architectures in a blog post.

And notice which seed won. Seed three gives **nine forty-six**, the best number anywhere in this
lab. If I'd run seed three first and stopped there, I would have reported nine forty-six and
believed it. If I'd run seed two first, I would have reported eleven seventy-six and believed
*that*. The number I reported would depend entirely on an integer that nobody thinks of as part of
the experiment.

That's the whole lab. Everything from here on works out the consequences.

---

# ▶ STEP 4 — So how would you compare two settings?

*(Run the single-seed weight-decay cell.)*

Here's the trap, and it's the single most common mistake in applied deep learning.

You want to know whether weight decay helps. So you try three values, train once on each, and pick
the winner.

```
weight_decay=0.0     test RMSE 1,109
weight_decay=0.0001  test RMSE 1,007
weight_decay=0.01    test RMSE 1,142
```

That looks like a clear result. A weight decay of one e minus four wins by **a hundred and two** in
RMSE over no weight decay at all, which is nine percent. You'd write that down, you'd put it on a
slide, and, honestly, most people would put it in a paper.

**[pause]**

Now do it honestly, with five seeds per setting, and report the spread as well as the mean.

*(Run the five-seed cell. It takes about a minute.)*

```
weight_decay  mean  sd  min  max
           0  1084  86  946 1176
      0.0001  1088 134  991 1343
        0.01  1241 278 1050 1792
```

*(Let this table sit next to the previous one. This is the moment of the lab.)*

**[pause]**

The two means differ by **four**.

That isn't "the effect was smaller than we thought," and it isn't "the advantage was modest." The
advantage of a hundred and two points that the single-seed experiment found **didn't exist**. Seed
one happened to be a good draw for one e minus four and an average draw for zero, and that's the
entire finding.

*(Point at the min and max columns.)*

Look closer, and it's worse than that. On both the minimum and the maximum, no weight decay is
*better* than one e minus four: nine forty-six against nine ninety-one, and eleven seventy-six
against thirteen forty-three. The setting with the tighter spread also has the slightly better
mean, by four. There's no effect here in either direction.

**[pause]**

Now look at the column a single run could never have produced, which is the standard deviation, s-d.

The largest setting, one e minus two, which is zero point zero one, is worse on average, at twelve
forty-one. But the important number is its spread of two hundred seventy-eight, which is more than
three times the spread with no weight decay at all. It returned ten fifty once, and seventeen
ninety-two another time. That's a different *kind* of bad from a setting that reliably lands at
twelve forty. One is a model that's mediocre, and the other is a model you can't make promises
about.

You can't see that in one run, ever.

**[STOP — learner works]**

Fill in the blank, and answer the question in your own words: what's the smallest difference in
RMSE that you'd be willing to call real on this problem?

*(Resume.)*

There's no single right answer, but there is a defensible way to reason about it. The run-to-run
standard deviation at your best setting is about **eighty-six**. Two single runs that differ by less
than that are telling you nothing at all. If you want to claim that a difference of that size is
real, you need several seeds per setting, and you need to compare the distributions, not the
winners.

So a difference under roughly a hundred in RMSE, from single runs, **isn't evidence**. And you'll
notice that this rules out almost every architecture comparison you've ever read.

---

# ▶ STEP 5 — Break it on purpose

*(Show the two training functions side by side.)*

There's one last thing, and it's the same theme in a different costume.

Look at the function called train broken. One line is commented out, the call to zero-grad on the
optimizer, and everything else is identical.

```
with    zero_grad():  1,109
without zero_grad():  7,325
```

**[pause]**

PyTorch **accumulates** gradients by default. It doesn't replace them, and nothing resets them at
the end of an epoch. If you never zero them, every update is driven by the sum of every gradient
computed since training began, which is sixty epochs' worth by the end. So the direction of each
step has less and less to do with the batch in front of you.

The result is seven thousand three hundred twenty-five, and the seasonal-naive benchmark is
twenty-one fifty-two. **The broken model is more than three times worse than predicting last
year's value.**

And the point isn't the number. It's everything that didn't happen. There was no exception, no
warning, and no NaN, meaning no "not a number." The loop ran to completion in the normal amount of
time and returned an ordinary number. If you hadn't had a benchmark to compare against, you'd have
had no way to know.

**[pause]**

That's why Lecture 1, back in the first week, told you to always beat a benchmark first, and it's
why every scoreboard in this course starts with seasonal naive. A benchmark isn't there to flatter
your model. It's the only thing standing between you and a silent bug.

---

# ▶ BEFORE YOU LEAVE

**[STOP — learner works]** — *(replaces the in-room "Discuss")*

Here are three questions to work through before you go.

The first is about the seeds. You now know that a single run tells you very little. But papers,
blog posts and vendor benchmarks routinely report exactly one. When you read "our model achieves an
RMSE of ten fifty," what do you need to know before you believe it beats a baseline at eleven
hundred? Be specific about what you'd ask for, and notice that today, a gap of fifty points is well
inside one standard deviation.

**[pause]**

The second is about the comparison. Twenty-two thousand parameters lost to a LASSO with forty-six
coefficients, ten eighty-four against seven forty-four. Give one reason that *isn't* "neural
networks need more data." Think about what your network can actually see, which is a hundred and
four unordered numbers. It doesn't know that feature one and feature five are the same channel a
week apart, or that the channels are four different quantities. It has to learn the layout of its
own input from five thousand nine hundred forty heavily overlapping examples, and the LASSO was
handed that structure by whoever engineered the features.

The third is about practice. In Homework 5 you compare five architectures. Given what you saw
today, what will you actually do when one of them comes out eighty points ahead of another in RMSE?

**[pause]**

I'll tell you the honest answer, because it's also the assignment's answer. You say the difference
is inside the noise, and you say so *even when the architecture you like is ahead*. Part 3 of
Homework 5 asks you to rank five architectures. Some of those gaps will be real, and some will be
seeds. Knowing which is which is the skill being tested.

Solutions go up on Canvas after class. Next week, in Lab 8, you'll meet a network that knows its
hundred and four inputs came from a sequence.

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
