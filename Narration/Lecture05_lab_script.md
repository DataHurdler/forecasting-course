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

*(Screen: rendered lab, top of document.)*

Lab 5. Same store, same aisle, same ten features as last week — daily CA\_1 FOODS. Nothing about
the data changes today. Everything that changes is what we do with trees.

Last week you ended by accident with a bagged ensemble: twenty bootstrap trees, averaged, 324 down
to 266. Today we do that properly, find out what the real algorithm adds on top of it, and then
throw out averaging altogether for something that works differently.

**[pause]**

Because the series is unchanged, every number today lands on a scale you already know. That is the
point of running Labs 4 and 5 as a pair. You have now fitted a GAM to this series in Homework 2, a
single tree last week, and four more models today — and they are all directly comparable, because
they all saw exactly the same table.

You need `xgboost` installed. On a Mac that also means `brew install libomp`.

---

# ▶ SETUP

*(Screen: run setup.)*

```
Benchmark (same day last year)          625
Single tree (Lab 4)                     324
20 trees averaged by hand (Lab 4)       266
```

The scoreboard carries forward. 625 is the benchmark we have used since Lecture 1. 324 was a
single tree. 266 was your hand-built bagging, and those two numbers are hard-coded here
deliberately — they are last week's measured results, not refitted.

Everything today gets added to this list, and at the end you will read it as one table.

---

# ▶ STEP 1 — The real thing

*(Screen: run the forest.)*

```
Random Forest: 248
```

Three hundred trees, no tuning, two lines of code.

**[pause]**

248 against your hand-built 266, and against 324 for a single tree. So the library version buys
about 7% over what you built yourself in three lines last week.

Sit with how small that is for a moment. **Most of the gain was already yours.** The idea —
resample, refit, average — is where the improvement came from; the professional implementation
adds a useful but modest amount on top. Two things account for that remainder: three hundred trees
instead of twenty, and deeper trees — last week's stopped at depth five. Notice what does *not*
account for it: scikit-learn's regression forest considers every feature at every split unless you
tell it otherwise, so this "random forest" is, strictly, still bagging. The second source of
randomness — the one that makes it a forest — we switch on in Step 3.

That is worth saying because the reverse impression is common — that the value is in the library.
It isn't. It's in the idea, and you already had it.

---

# ▶ STEP 2 — How many trees is enough?

*(Screen: run the curve.)*

```
  n_estimators=5    test RMSE 288
  n_estimators=20   test RMSE 257
  n_estimators=50   test RMSE 251
  n_estimators=100  test RMSE 249
  n_estimators=300  test RMSE 248
  n_estimators=600  test RMSE 248
```

**[STOP — learner works]** — *(replaces "say what the curve does after about fifty trees")*

Pause and describe this curve in one sentence, then say what it implies for a compute budget.

*(Resume.)*

It drops hard to about fifty trees, and then it is flat. Fifty trees gets you 251; six hundred
gets you 248. You paid twelve times the compute for three units of RMSE — on a series where a
typical day is around 2,900 units.

**[pause]**

Now the property that makes this curve unusual, and please notice it, because Step 6 breaks it:
**the curve never turns up.** More trees never made the forest worse. That is not the normal
behavior of a model knob. Usually "more" eventually means overfitting, and here it simply does
not, because each tree is fitted independently and averaging more independent things cannot make
the average worse.

So `n_estimators` in a forest is not really a tuning parameter — it is a budget question. Pick the
point where the curve goes flat, spend nothing beyond it, and never worry that you have chosen
badly. Hold that thought for twenty minutes.

---

# ▶ STEP 3 — The "random" in Random Forest

*(Screen: run the max_features loop.)*

Here is the second randomness. Bagging gives every tree a different sample of **rows**. A random
forest also gives every *split* a different random subset of **columns**. That is the entire
difference between the two algorithms — and it is why what you built last week was bagging and
not a forest. So, strictly, was Step 1's default: the "all features" row below is the same 248.

The argument from Lecture 5: if one feature is strongly predictive, every bagged tree splits on
it first, so the trees are near-copies and their errors do not cancel. Deny each split most of the
columns and the trees are forced to be different.

You have already seen that failure mode. Last week, twenty out of twenty bootstrap trees found
`dow` at the root.

```
  max_features=all features  test RMSE 248
  max_features=0.6           test RMSE 247
  max_features=sqrt          test RMSE 254
  max_features=0.2           test RMSE 257
```

**[pause]** — *let it sit; this table is not what students expect.*

`max_features="sqrt"` is the textbook default. It is the third-best of four here, six units worse
than simply using every column. And 0.2 — aggressive decorrelation — is worse still.

*(Screen: the callout.)*

This is not a bug and it is not a bad dataset. It is the rule meeting its own boundary condition,
and the condition is stated in the callout: you have **ten** features, one of which carries most
of the signal. The square root of ten is about three, so at a typical split most trees are not
allowed to *look at* day-of-week — and day-of-week, as you measured last week, is roughly two
thirds of the variance in this series.

Decorrelation is a trade. You give up some accuracy per tree to gain independence between trees.
That trade is a bargain when you have many partly-redundant features and no single dominant one.
It is a bad trade when you have ten features and one of them is the answer.

**[pause]**

Notice the honest middle of the table: `0.6` is the best row, at 247. Mild subsampling helps by a
hair. So the finding is not "decorrelation is wrong here," it is "the standard *dose* is too
strong for this feature set."

Do not expect Homework 3 to flip this ranking. Its forest has twenty-seven features, but
`units_lag1` dominates there the way `dow` dominates here, and `sqrt` loses there too.
**What you are learning is the condition, not the default.**

---

# ▶ STEP 4 — Validation for free

*(Screen: run the OOB fit.)*

A bootstrap sample of n rows drawn with replacement leaves out about a third of them. Those rows
never trained that tree, so they can score it — for free, with no holdout.

```
OOB R²   : 0.809
OOB RMSE : 297
test RMSE: 248
```

**[STOP — learner works]** — *(replaces "compare the two numbers and account for the gap")*

Pause. The OOB estimate is 297 and the test RMSE is 248, so OOB is *pessimistic* by about 20%.
Work out why before I tell you, and then decide which of the two you would quote to someone asking
how this model will do next month.

*(Resume.)*

The obvious suspect is the wrong one. An OOB prediction for a given row comes from only the third
of trees that did not see it — about a hundred trees rather than three hundred. But Step 2 told you
what a smaller forest costs: a hundred trees score 249 against 248. One unit cannot explain 49.

The gap is the rows. OOB rows are scattered across 2012 to 2015, the whole training period. By
year, the OOB error runs 311, 319, 283 and 239: the earlier years are noisier. And the training
period holds three Christmas days when the store sold nothing; drop those three days and OOB falls
to 275. The test set is one solid block of the final year.

**[pause]**

So which do you quote? Neither, uncritically. OOB is the cheap sanity check you can compute
without giving up any data — genuinely valuable when data is scarce. But it is scoring the wrong
period for a forecasting question, and it is not a walk-forward estimate. If
someone asks how this does next month, the only honest answer comes from evaluating on time you
held out, in order — which is what the test column is, and what Homework 3 makes you do properly.

---

# ▶ STEP 5 — Two importances, two different answers

*(Screen: run the comparison table.)*

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

First, the units, because they change what the numbers mean. MDI is a share of impurity reduction
— it sums to one and it is computed on the **training** data by the fitting process. Permutation
importance is measured in **RMSE**: shuffle that column in the **test** set and see how much worse
the forecast gets. So `dow` at 284 means "destroy day-of-week and you lose 284 units of accuracy."
You can say that sentence to a manager. You cannot say "0.583" to anyone.

**[STOP — learner works]** — *(replaces "find the disagreements")*

Pause and find the three disagreements before I name them.

*(Resume.)*

**One.** `units_lag28` is second on MDI and fourth on permutation, at 36.8 — while `units_lag1`,
which MDI ranks *below* it, is worth 116.9. Three times as much. MDI counts how often a feature
was *used* for splitting; permutation measures what happens when you take it away. A feature used
constantly for small refinements can lose to one used rarely for large ones.

**Two, and this is the one that matters commercially.** `snap` is **eighth** on MDI and **third**
on permutation, worth 40 units of RMSE. SNAP benefits land on fixed days of the month and this is
a food aisle: SNAP days average 3,201 units against 2,839 otherwise, about 13% higher, on 400 of
1,211 training days. MDI nearly buried a real, actionable effect.

**Three.** Look at `avg_price` against `is_event`. MDI ranks price *above* the event flag —
0.017 to 0.012. Permutation says price is worth **exactly zero** and the event flag is worth 22.

That inversion has a known mechanism, and Lecture 4's footnote named it: **MDI is biased toward
features with many split points.** `avg_price` takes 173 distinct values in the training set, so it
offers 172 candidate split points, and with that many chances some will reduce impurity by luck
alone. `is_event` is binary — one possible split, no opportunity to get lucky. MDI rewards
opportunity; permutation rewards contribution.

**[pause]**

The practical rule: **MDI is free and permutation is honest.** MDI comes out of the fit at no cost
and is fine for a quick screen. Anything you are going to act on — or say out loud to a category
manager — gets checked with permutation, on held-out data.

*(Screen: the your-turn plotting cell.)*

**[STOP — learner works]** — fill in the one blank and plot the two rankings side by side. The
blank is the permutation series. Come back when you can see the `snap` bar jump.

---

# ▶ STEP 6 — Stop averaging. Start correcting.

*(Screen: the framing paragraph.)*

Everything so far today has been the same idea: build many trees independently, average them.
Boosting is a different idea. Trees are built **in sequence**, and each new one is fitted to what
the ensemble so far got **wrong**.

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

**[pause]** — *let the first four rows sit on screen. This is the payoff of the lab.*

Read the first four rows as a curve — the learning rate is fixed, only the number of rounds
changes. That is the identical experiment you ran in Step 2 on the forest.

Training error goes 171, 95, 19, **zero**. Test error goes 244, 243, 250, 253. It bottoms out
around three hundred rounds and then **turns up**.

Now put that next to Step 2, where you took a forest from five trees to six hundred and it never
got worse once.

**[pause]**

Same data, same features, same family of base learner. Opposite behavior on the same knob. The
reason is structural, and it is worth stating carefully: a forest's trees are fitted independently
of one another, so an extra one only adds to an average and cannot corrupt what is already there.
A boosted tree is fitted to the *residuals* of everything before it — so once the real signal is
used up, the next tree fits noise, and it is added to the ensemble anyway, with nothing to average
it away.

So: `n_estimators` in a forest is a budget. `n_estimators` in boosting is a tuning parameter, and
getting it wrong costs you accuracy in a way no amount of forest-tuning ever will. That is also
why boosting *has* a learning rate — it is the knob that decides how much damage each additional
tree is allowed to do.

The last two rows are the complement. A big rate with few rounds (238) and a small rate with many
(239) end up in much the same place: roughly, the product sets the budget. What you cannot do is
fix a rate and let the rounds run.

*(Run the final scoreboard.)*

```
Benchmark (same day last year)       625
Single tree (Lab 4)                  324
20 trees averaged by hand (Lab 4)    266
Random Forest (300 trees)            248
XGBoost (lr=0.1, 100 trees)          244
```

**[pause]**

Now, look at which XGBoost row we put on that scoreboard. `lr=0.1, n=100` — 244. Not the
`lr=0.3` row at 238, which was the best number in the grid.

That is deliberate, and it is not modesty. **Every test-RMSE number in the grid above was computed
on the year we held out.** If I pick the winner out of that column and then report it as my
result, I have used the test set to choose the model and I no longer have an honest estimate of
anything. Lecture 1 said this about forecast evaluation and it does not stop being true because
the grid has only six rows. So the scoreboard gets the conventional default, and the way to
actually choose here is walk-forward validation inside the training period — Homework 4.

---

# ▶ BEFORE YOU LEAVE

*(Screen: the closing block.)*

Read the scoreboard as one object. 625 to 244 — a 61% reduction against the benchmark, on one
series, with the same ten features throughout. Almost all of it arrived in two jumps: a single
tree over the benchmark, and averaging over the single tree. The last two steps, the forest and
boosting, bought 22 units between them.

**[STOP — learner works]** — *(replaces the in-room "Discuss")*

Three questions on the discussion board.

**The ranking.** Boosting won, 244 to 248 — by about one and a half percent. Name one property of
*this* dataset that favors it. Then name a situation where you would still reach for the forest
first, thinking about two things specifically: how much tuning each one needed today, and what
happens to each when nobody retunes it for a year.

**[pause]** — *the honest answer is in Steps 2 and 6, and students should find it there: the forest
was fire-and-forget, and boosting's best setting is a live decision that can go stale.*

**The importances.** You have two tables that disagree about `snap` — eighth on one, third on the
other. A category manager asks whether SNAP timing matters for stocking. Which table do you show
them, what exactly do you say, and what would you check before committing to an answer? Be
concrete about that last part.

**The arc.** Across Labs 4 and 5 the error fell from 324 to 244 on the same series with the same
features. What was purchased with that, and what was given up? Be specific about the thing you
could do in Step 1 of Lab 4 that you cannot do now.

**[pause]**

I'll say the shape of that last answer, because it is the through-line of the whole course and I
do not want it to be missed. In Lab 4, Step 1, you printed a tree and read a path out loud —
*weekday, four weeks ago was busy, yesterday was busy, predict 2,950.* You cannot do that now.
There is no path to read. Three hundred trees, or a hundred sequential corrections, and the best
account you can give of the model is a permutation importance table.

You bought 25% accuracy with the entire explanation. Sometimes that is obviously right. Sometimes
you are standing in front of a category manager and it is obviously wrong. Knowing which is which
is the actual skill.

Solutions for the coded parts go up on Canvas after the deadline. Next week: regularization, and
what to do when you have far more features than this.

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
