# Lab 5 — Recording Script

**ECON 8310: Business Forecasting · Tree Ensembles — Averaging, Then Boosting**

Lab: `Labs/Lecture05_lab.qmd` (6 steps) · Measured runtime: **~25 minutes** of narration
(the in-room version is budgeted at 40)

---

## How to use this document

- **`▶ STEP n — Title`** matches the lab's own headings exactly.
- *Italic parentheticals* are stage directions. **[pause]** is a beat; **[STOP — learner works]**
  is where you tell the viewer to pause the video and do something.
- Numbers are the lab's real output, verified by running the chunks end to end. Deterministic —
  `random_state=42` throughout. The whole lab fits in about 10 seconds on a laptop, forests and
  boosting included, so nothing here needs pre-computing.

### Two corrections that were applied to the lab

Writing this script turned up a claim in Step 6 that the measured output **contradicted outright**.
Both fixes are in the `.qmd` now.

1. **Step 6 — the important one.** The old grid mixed learning rate and tree count together —
   `(0.3,100), (0.1,100), (0.1,300), (0.05,1000)` — and the prose said the `lr=0.3` row had "the
   lowest training error of the four and the worst test error." Measured, it had **neither**: train
   95 (the second-lowest) and test **238**, the *best* test error in the grid. The paragraph's whole
   point — that boosting can overfit where a forest cannot — was true, but that grid did not show
   it, because moving both knobs at once slid along a ridge instead of climbing off it.
   The grid now **holds `lr=0.1` fixed and sweeps rounds**: 100 → 300 → 1,000 → 3,000, with two
   rate-change rows kept at the end. Test error now visibly bottoms at 300 and climbs back, which
   is the claim, demonstrated instead of asserted.
2. **Step 5.** The `avg_price` bullet said MDI gives it "a non-trivial score." It gives it 0.017 —
   ninth of ten. The real contrast is sharper and is now what the lab says: MDI ranks `avg_price`
   **above** `is_event`, while permutation scores price at exactly zero and the event flag at 22.

### In-room language that needs replacing

Four places, all converted below: Step 2's "say what the curve does", Step 4's "compare the two
numbers", Step 5's "find the disagreements", and the three closing **Discuss** blocks.

---

# ▶ OPENING

*(Show the rendered lab, at the top of the document.)*

This is Lab 5. It's the same store, the same aisle and the same ten features as Lab 4, last week,
which means daily sales for CA\_1 FOODS. Nothing about the data changes today. Everything that
changes is what we do with trees.

At the end of last week's lab, you built a bagged ensemble by accident. You averaged twenty
bootstrap trees, and the RMSE fell from three twenty-four to two sixty-six. Today we'll do that
properly, find out what the real algorithm adds on top of it, and then throw out averaging
altogether for something that works differently.

**[pause]**

Because the series hasn't changed, every number today lands on a scale you already know, and
that's the point of running Labs 4 and 5 as a pair. You've now fitted a GAM to this series in
Homework 2 and a single tree last week, and you'll fit four more models today. Last week's tree and
today's models are directly comparable, because they all saw exactly the same table.

You'll need `xgboost` installed, and on a Mac that also means running `brew install libomp`.

---

# ▶ SETUP

*(Run the setup cell.)*

```
Benchmark (same day last year)          625
Single tree (Lab 4)                     324
20 trees averaged by hand (Lab 4)       266
```

The scoreboard carries forward. Six twenty-five is the benchmark, which predicts each day with the
same day last year, the kind of benchmark we've used since Lecture 1. Three twenty-four was a single
tree, and two sixty-six was your hand-built bagging. Those last two numbers are hard-coded here on
purpose. They're last week's measured results, not refitted ones.

Everything today gets added to this list, and at the end you'll read it as one table.

---

# ▶ STEP 1 — The real thing

*(Run the forest.)*

```
Random Forest: 248
```

That's three hundred trees, with no tuning, in two lines of code.

**[pause]**

It scores two forty-eight, against your hand-built two sixty-six, and three twenty-four for a single
tree. So the library version buys about seven percent over what you built yourself, in three lines,
last week.

Sit with how small that is for a moment. **Most of the gain was already yours.** The idea of
resampling, refitting and averaging is where the improvement came from, and the professional
implementation adds a useful but modest amount on top. Two things account for that remainder:
three hundred trees instead of twenty, and deeper trees, since last week's stopped at depth five.

Notice what does *not* account for it. Scikit-learn's regression forest considers every feature at
every split unless you tell it otherwise, so this "random forest" is, strictly speaking, still
bagging. The second source of randomness, the one that makes it a forest, gets switched on in
Step 3.

That's worth saying, because the reverse impression is common, that the value lives in the
library. It doesn't. It's in the idea, and you already had it.

---

# ▶ STEP 2 — How many trees is enough?

*(Run the curve.)*

```
  n_estimators=5    test RMSE 288
  n_estimators=20   test RMSE 257
  n_estimators=50   test RMSE 251
  n_estimators=100  test RMSE 249
  n_estimators=300  test RMSE 248
  n_estimators=600  test RMSE 248
```

**[STOP — learner works]** — *(replaces "say what the curve does after about fifty trees")*

Pause the video, and describe this curve in one sentence. Then say what it implies for a compute
budget.

*(Resume.)*

It drops hard up to about fifty trees, and then it's flat. Fifty trees get you two fifty-one, and
six hundred get you two forty-eight. So you paid twelve times the compute for three units of RMSE,
on a series where a typical day is around two thousand nine hundred units.

**[pause]**

Now here's the property that makes this curve unusual, and please notice it, because Step 6 breaks
it: the curve **never turns up**. More trees never made the forest worse. That isn't the normal
behavior of a model knob. Usually "more" eventually means overfitting, and here it simply doesn't,
because each tree is fitted independently, and averaging more independent things can't make the
average worse.

So `n_estimators` in a forest isn't really a tuning parameter. It's a budget question. Pick the
point where the curve goes flat, spend nothing beyond it, and never worry that you've chosen badly.
Hold on to that thought for about twenty minutes.

---

# ▶ STEP 3 — The "random" in Random Forest

*(Run the `max_features` loop.)*

Here's the second kind of randomness. Bagging gives every tree a different sample of **rows**. A
random forest also gives every *split* a different random subset of **columns**. That's the entire
difference between the two algorithms, and it's why what you built last week was bagging and not a
forest. Strictly speaking, so was Step 1's default, and the "all features" row below is the same two
forty-eight.

The argument from Lecture 5, this week's lecture, goes like this. If one feature is strongly
predictive, every bagged tree splits on it first, so the trees are near-copies, and their errors
don't cancel. Deny each split most of the columns, and the trees are forced to be different.

You've already seen that failure mode. Last week, all twenty of your bootstrap trees found the day
of the week at the root.

```
  max_features=all features  test RMSE 248
  max_features=0.6           test RMSE 247
  max_features=sqrt          test RMSE 254
  max_features=0.2           test RMSE 257
```

*(Let the table sit for a moment. It isn't what students expect.)*

**[pause]**

The square-root setting is the textbook default. Here it's the third best of four, six units worse
than simply using every column. And zero point two, which is aggressive decorrelation, is worse
still.

*(Show the callout.)*

This isn't a bug, and it isn't a bad data set. It's the rule meeting its own boundary condition, and
the callout states the condition. You have **ten** features, and one of them carries most of the
signal. The square root of ten is about three, so at a typical split, most trees aren't allowed to
*look at* the day of the week. And the day of the week, as you measured last week, accounts for
roughly two thirds of the variance in this series.

Decorrelation is a trade. You give up some accuracy in each tree to gain independence between the
trees. That trade is a bargain when you have many partly redundant features and no single dominant
one. It's a bad trade when you have ten features and one of them is the answer.

**[pause]**

Notice the honest middle of the table. Zero point six is the best row, at two forty-seven, so mild
subsampling helps, by a hair. The finding isn't that decorrelation is wrong here. It's that the
standard *dose* is too strong for this set of features.

And don't expect Homework 3 to flip this ranking. Its forest has twenty-seven features, but last
week's sales, `units_lag1`, dominate there the way the day of the week dominates here, and the
square-root setting loses there too. **What you're learning is the condition, not the default.**

---

# ▶ STEP 4 — Validation for free

*(Run the out-of-bag fit.)*

A bootstrap sample of n rows, drawn with replacement, leaves out about a third of them. Those rows
never trained that tree, so they can score it, for free, with no holdout.

```
OOB R²   : 0.809
OOB RMSE : 297
test RMSE: 248
```

**[STOP — learner works]** — *(replaces "compare the two numbers and account for the gap")*

Pause the video here. The O-O-B estimate is two ninety-seven, and the test RMSE is two forty-eight,
so O-O-B is *pessimistic*, by about twenty percent. Work out why before I tell you. Then decide
which of the two you'd quote to someone who asks how this model will do next month.

*(Resume.)*

The obvious suspect is the wrong one. An O-O-B prediction for a given row comes from only the third
of the trees that didn't see it, which is about a hundred trees rather than three hundred. But
Step 2 told you what a smaller forest costs: a hundred trees score two forty-nine, against two
forty-eight. One unit can't explain a gap of forty-nine.

The gap comes from the rows. O-O-B rows are scattered across twenty twelve to twenty fifteen, the
whole training period. Year by year, the O-O-B error runs three eleven, three nineteen, two
eighty-three and two thirty-nine, so the earlier years are noisier. And the training period holds
three Christmas days when the store sold nothing. Drop those three days, and O-O-B falls to two
seventy-five. The test set, by contrast, is one solid block of the final year.

**[pause]**

So which number do you quote? Neither one, uncritically. O-O-B is the cheap sanity check you can
compute without giving up any data, and it's genuinely valuable when data is scarce. But it's
scoring the wrong period for a forecasting question, and it isn't a walk-forward estimate. If
someone asks how this model will do next month, the only honest answer comes from evaluating on
time you held out, in order. That's what the test column is, and it's what Homework 3 makes you do
properly.

---

# ▶ STEP 5 — Two importances, two different answers

*(Run the comparison table.)*

```
                MDI  permutation_ΔRMSE  MDI_rank  perm_rank
dow           0.583              284.2         1          1
units_lag28   0.119               36.8         2          4
units_lag1    0.086              116.9         3          2
units_lag7    0.053               35.6         4          5
units_lag365  0.036               10.2         5          8
doy           0.036               18.1         5          7
units_lag14   0.031                7.7         7          9
snap          0.027               40.4         8          3
avg_price     0.017                0.0         9         10
is_event      0.012               22.3        10          6
```

Let's start with the units, because they change what the numbers mean. M-D-I is a share of the
impurity reduction. It sums to one, and it's computed on the **training** data, by the fitting
process. Permutation importance is measured in RMSE: you shuffle that column in the **test** set and
see how much worse the forecast gets. So the day of the week at two eighty-four means, "Destroy the
day of the week, and you lose two hundred eighty-four units of accuracy." You can say that sentence
to a manager. You can't say "zero point five eight three" to anyone.

**[STOP — learner works]** — *(replaces "find the disagreements")*

Pause the video, and find the three disagreements before I name them.

*(Resume.)*

The first is the twenty-eight-day lag, `units_lag28`. It's second on M-D-I but fourth on
permutation, at thirty-six point eight. Meanwhile yesterday's sales, `units_lag1`, which M-D-I ranks
*below* it, are worth one hundred sixteen point nine. That's about three times as much. M-D-I counts
how often a feature was *used* for splitting, while permutation measures what happens when you take
it away. A feature used constantly for small refinements can lose to one used rarely for large ones.

The second disagreement is the one that matters commercially. The SNAP flag is **eighth** on M-D-I
and third on permutation, worth forty units of RMSE. SNAP benefits land on fixed days of the month,
and this is a food aisle. SNAP days average three thousand two hundred one units, against two
thousand eight hundred thirty-nine on other days. That's about thirteen percent higher, on four
hundred of the one thousand two hundred eleven training days. M-D-I nearly buried a real, actionable
effect.

The third is price against the event flag. M-D-I ranks `avg_price` *above* `is_event`, zero point
zero one seven to zero point zero one two. Permutation says price is worth **exactly zero**, and the
event flag is worth twenty-two.

That inversion has a known mechanism, and the footnote in Lecture 4, last week, named it: M-D-I is
biased toward features with many split points. The price column takes one hundred seventy-three
distinct values in the training set, so it offers one hundred seventy-two candidate split points,
and with that many chances, some of them will reduce impurity by luck alone. The event flag is
binary. It has one possible split, and no opportunity to get lucky. So M-D-I rewards opportunity,
and permutation rewards contribution.

**[pause]**

The practical rule is that M-D-I is free and permutation is **honest**. M-D-I comes out of the fit at
no cost, and it's fine for a quick screen. Anything you're going to act on, or say out loud to a
category manager, gets checked with permutation, on held-out data.

*(Show the your-turn plotting cell.)*

**[STOP — learner works]**

Now it's your turn. Pause the video, fill in the one blank, and plot the two rankings side by side.
The blank is the permutation series. Come back when you can see the SNAP bar jump.

---

# ▶ STEP 6 — Stop averaging. Start correcting.

*(Show the framing paragraph.)*

Everything so far today has been the same idea: build many trees independently, and average them.
Boosting is a different idea. The trees are built **in sequence**, and each new one is fitted to
what the ensemble so far got wrong.

Nothing is averaged. Every tree is a correction.

*(Run the grid.)*

```
  lr=0.1   n=100   train   171   test   244
  lr=0.1   n=300   train    95   test   243
  lr=0.1   n=1000  train    19   test   250
  lr=0.1   n=3000  train     0   test   253
  lr=0.3   n=100   train    95   test   238
  lr=0.05  n=1000  train    54   test   239
```

*(Let the first four rows sit on screen. This is the payoff of the lab.)*

**[pause]**

Read the first four rows as a curve. The learning rate is fixed, and only the number of rounds
changes, so it's the identical experiment you ran on the forest in Step 2.

Training error goes from one seventy-one, to ninety-five, to nineteen, to **zero**. Test error goes
from two forty-four, to two forty-three, to two fifty, to two fifty-three. It bottoms out around
three hundred rounds, and then it **turns up**.

Now put that next to Step 2, where you took a forest from five trees to six hundred and it never got
worse once.

**[pause]**

It's the same data, the same features and the same family of base learner, with opposite behavior
on the same knob. The reason is structural, and it's worth stating carefully. A forest's trees are
fitted independently of one another, so an extra tree only adds to an average, and it can't corrupt
what's already there. A boosted tree is fitted to the *residuals* of everything before it. So once
the real signal is used up, the next tree fits noise, and it gets added to the ensemble anyway, with
nothing to average it away.

So `n_estimators` in a forest is a budget, while `n_estimators` in boosting is a tuning parameter,
and getting it wrong costs you accuracy in a way no amount of forest tuning ever will. That's also
why boosting *has* a learning rate. It's the knob that decides how much damage each additional tree
is allowed to do.

The last two rows are the complement. A big rate with few rounds scores two thirty-eight, and a small
rate with many rounds scores two thirty-nine, so they end up in much the same place. Roughly
speaking, it's the product of the two that sets the budget. What you can't do is fix a rate and let
the rounds run.

*(Run the final scoreboard.)*

```
Benchmark (same day last year)       625
Single tree (Lab 4)                  324
20 trees averaged by hand (Lab 4)    266
Random Forest (300 trees)            248
XGBoost (lr=0.1, 100 trees)          244
```

Now, look at which XGBoost row we put on that scoreboard. It's the learning rate of zero point one
with a hundred trees, at two forty-four. It isn't the zero point three row at two thirty-eight,
which was the best number in the grid.

That's deliberate, and it isn't modesty. **Every** test RMSE in the grid above was computed on the
year we held out. If I pick the winner out of that column and then report it as my result, I've used
the test set to choose the model, and I no longer have an honest estimate of anything. Lecture 1, in
the first week, said this about forecast evaluation, and it doesn't stop being true just because the
grid has only six rows. So the scoreboard gets the conventional default. The way to actually choose
here is walk-forward validation inside the training period, and that's what you'll do in
Homework 4.

---

# ▶ BEFORE YOU LEAVE

*(Show the closing block.)*

Read the scoreboard as one object. It runs from six twenty-five down to two forty-four, a sixty-one
percent reduction against the benchmark, on one series, with the same ten features throughout.
Almost all of it arrived in two jumps: a single tree over the benchmark, and then averaging over the
single tree. The last two steps, the forest and boosting, bought twenty-two units between them.

**[STOP — learner works]** — *(replaces the in-room "Discuss")*

Here are three questions to work through on the discussion board.

The first is about the ranking. Boosting won, two forty-four to two forty-eight, by about one and a
half percent. Name one property of *this* data set that favors it. Then name a situation where you'd
still reach for the forest first, and think about two things in particular: how much tuning each
model needed today, and what happens to each one when nobody retunes it for a year.

**[pause]**

The answer to that second half is in Steps 2 and 6, and it's worth finding there yourself. The
forest was fire-and-forget, while boosting's best setting is a live decision, and a live decision
can go stale.

The second question is about the importances. You have two tables that disagree about SNAP, eighth
on one and third on the other. A category manager asks whether SNAP timing matters for stocking.
Which table do you show them, what exactly do you say, and what would you check before committing to
an answer? Be concrete about that last part.

The third question is about the arc. Across Labs 4 and 5, the error fell from three twenty-four to
two forty-four, on the same series with the same features. What was purchased with that, and what
was given up? Be specific about the thing you could do in Step 1 of Lab 4 that you can't do now.

**[pause]**

I'll give you the shape of that last answer, because it's the thread that runs through the whole
course, and I don't want it to be missed. In Lab 4, Step 1, you printed a tree and read a path out
loud: *it's a weekday, four weeks ago was busy, yesterday was busy, so predict two thousand nine
hundred fifty.* You can't do that now. There's no path to read. With three hundred trees, or a
hundred sequential corrections, the best account you can give of the model is a permutation
importance table.

You bought about twenty-five percent in accuracy, and you paid for it with the entire explanation.
Sometimes that's obviously the right trade. Sometimes you're standing in front of a category
manager, and it's obviously wrong. Knowing which is which is the actual skill.

Solutions for the coded parts go up on Canvas after class. Next week, in Lecture 6, we turn to
regularization, and to what you do when you have far more features than this.

---

# Appendix — expected output

Deterministic; `random_state=42` throughout. Whole lab runs in about 10 seconds.

| Quantity | Value |
|---|---|
| Scoreboard carried in | benchmark 625 · single tree 324 · 20 trees averaged 266 |
| Random Forest, 300 trees | **248** |
| Tree-count curve | 5: 288 · 20: 257 · 50: 251 · 100: 249 · 300: 248 · 600: 248 (never rises) |
| `max_features` | all: 248 · **0.6: 247** · sqrt: 254 · 0.2: 257 |
| OOB | R² 0.809 · RMSE **297** vs test **248** |
| MDI top three | `dow` 0.583 · `units_lag28` 0.119 · `units_lag1` 0.086 |
| Permutation top three (ΔRMSE) | `dow` 284.2 · `units_lag1` 116.9 · `snap` 40.4 |
| `snap` | MDI rank 8 → permutation rank 3 (40.4 RMSE) |
| `avg_price` vs `is_event` | MDI 0.017 > 0.012; permutation 0.0 < 22.3 |
| Cardinality behind that inversion | `avg_price` 173 distinct values · `is_event` 2 |
| SNAP effect in the data | 3,201 units on SNAP days vs 2,839 otherwise (+13%), 400 of 1,211 days |
| Boosting, fixed `lr=0.1` | n=100: 244 · n=300: **243** · n=1000: 250 · n=3000: 253 (turns up) |
| Boosting, rate changes | lr=0.3 n=100: 238 · lr=0.05 n=1000: 239 |
| Scoreboard entry (chosen as the default, not the winner) | XGBoost lr=0.1 n=100: **244** |

**Three things to know before recording.**

*Step 6's first four rows are the lab.* Everything else is setup for the contrast between a knob
that cannot hurt you (forest trees) and one that can (boosting rounds). If the recording runs
long, cut elsewhere.

*The `max_features` result is a genuine finding and students will read it as an error.* `sqrt` is
in every textbook and it loses here, by 6 units. Say plainly that this is the rule's boundary
condition — ten features, one dominant. Do not promise that Homework 3 restores the textbook
answer: its twenty-seven features are dominated by `units_lag1`, and `sqrt` loses there too. Do not
apologize for the dataset.

*Do not quietly promote the `lr=0.3` row to the scoreboard.* It is the best test number in the
grid (238) and it is tempting. Choosing it would be selecting on the test set, one lecture after
telling students not to — and the closing beat of Step 6 depends on **not** doing it.
