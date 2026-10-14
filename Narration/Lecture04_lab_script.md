# Lab 4 — Recording Script

**ECON 8310: Business Forecasting · Decision Trees — Reading One, and Finding Where It Breaks**

Lab: `Labs/Lecture04_lab.qmd` (4 steps) · Measured runtime: **~20 minutes** of narration
(the in-room version is budgeted at 40)

---

## How to use this document

- **`▶ STEP n — Title`** matches the lab's own headings exactly.
- *Italic parentheticals* are stage directions. **[pause]** is a beat; **[STOP — learner works]**
  is where you tell the viewer to pause the video and do something.
- Numbers are the lab's real output, verified by running the chunks end to end. Everything here
  is deterministic — `random_state=42` on every fit, and the bootstrap draws come from a seeded
  `RandomState`. Re-running gives the same table.

### One correction that was applied to the lab

Step 3 closed with "You just built a small Random Forest by hand." That is the wrong name, and
Lecture 5 contradicts it four days later: bagging and random forests differ *precisely* by the
feature subsampling this lab does not do, and L05 spends a whole slide on the problem bagging
does **not** solve. The line now says you built a **bagged ensemble** — a forest with the
decorrelation step still missing. If you remember the earlier wording, that is why it changed.

*(Lab 5's opening sentence had the same slip and gets the same fix, so the two labs agree.)*

### In-room language that needs replacing

Three places, all converted below:

1. Step 1 — "Read the root split aloud to the person next to you."
2. Step 1 callout — "Say why before moving on."
3. The closing **Discuss** block, converted to a discussion-board prompt.

---

# ▶ OPENING

*(Screen: rendered lab, top of document.)*

Lab 4, and the course changes character today.

Everything up to now worked the same way: you specified a structure — a trend and a season, an
AR term and an MA term, a smooth function of day-of-year — and then estimated it. You chose the
shape and the data filled in the numbers.

A tree does not work that way. You hand it a pile of columns and it finds the structure itself.
Nobody tells it that weekends matter.

**[pause]**

The series is daily units for CA\_1 FOODS — the same store, same aisle you forecast in
**Homework 2**. Not Lab 3; Lab 3 was electricity. Homework 2 was this series with a GAM, and it
was due before today, which is deliberate. It means you can put a tree on identical data and say
exactly what each method bought you. That comparison is the closing question.

Four things by the end: read a fitted tree out loud, watch the bias–variance tradeoff happen in
a table, discover why one tree is not enough, and find the one place trees reliably fail.

You need `scikit-learn` and `data/processed/m5_daily.csv`. Run `prep_m5.py` once if it is missing.

---

# ▶ SETUP

*(Screen: run setup.)*

```
1576 usable days — train 1211, test 365
benchmark (same day last year): RMSE 625
```

Just over four years of daily data, holding out the final year — May 2015 through May 2016.

The benchmark is the same one you have used since Lecture 1: predict today with the same day
last year. **625 units.** Write it down. Every number today gets measured against it.

**[pause]** — *let the `FEATURES` list sit on screen.*

Now look at what we handed the model, because this is the important part of the setup.

Ten columns: five lags, a SNAP flag, an event flag, day of week, day of year, and price.
And **no date index.** There is no time in `X_train` at all. Every row is an independent example,
and everything the model knows about *when* it is, is sitting in those columns because we put it
there.

That reframing — a time series flattened into an ordinary table — is what lets everything from
here to Lecture 9 be applied to forecasting at all. Trees, forests, boosting, neural networks:
none of them know what a time series is. They all consume this table.

---

# ▶ STEP 1 — Fit a tree and read it out loud

*(Screen: run the depth-3 fit and `export_text`.)*

Depth 3, deliberately. Eight leaves is small enough to read in full, and you should read it in
full, because being able to do that is most of what a tree is for.

```
|--- dow <= 4
|   |--- units_lag28 <= 2596
...
|--- dow >  4
|   |--- units_lag1 <= 3351
...

test RMSE: 337
```

**[pause]**

The root split is `dow <= 4`. That is Monday-through-Friday on one side, Saturday and Sunday on
the other — and the tree found it without being told that a week has a weekend in it.

**[STOP — learner works]** — *(replaces "read it aloud to the person next to you")*

Pause here and say the whole path out loud, in business language rather than in inequalities.
Not "`dow` less than or equal to 4.5, then `units_lag28` above 2596." Something closer to:
*it's a weekday, and four weeks ago was already busy, and last week was busy too, so predict
about 2,950.*

If you can do that for one leaf, you can do it for a store manager, and that is a thing no
model in Lectures 1 through 3 could offer you.

*(Resume. Point at the bottom-right of the printed tree.)*

One branch worth reading before we move on. On the weekend side, on days when yesterday was
already very busy, the tree splits on `is_event` — and it predicts **4,041** when there is no
event and **3,583** when there is. The event lowers the forecast by about 450 units. Events in
M5 include days like Christmas, when the store is shut or nearly so. The tree has no idea what
Christmas is. It found the hole in the data anyway.

**[pause]**

And the accuracy: **337 against a benchmark of 625.** Roughly a 46% improvement, from eight
leaves you can print on a slide.

*(Run the importances.)*

```
dow            0.789
units_lag28    0.106
units_lag1     0.064
units_lag7     0.030
is_event       0.011
```

Day of week takes seventy-nine percent of the importance. On its own, day-of-week means explain
about **66%** of the variance in training units — weekday average 2,632, weekend average 3,773,
a gap of 43%. There is no mystery about why the tree grabbed it first.

*(Screen: the callout comparing this with the lecture slide.)*

**[STOP — learner works]** — *(replaces "say why before moving on")*

Lecture 4's importance table, on the **weekly** version of this same store and aisle, had
`units_lag4` at 0.62 and day-of-week nowhere — it isn't in the table at all. Here, on **daily**
data, `dow` wins outright with 0.79 and the four-week lag drops to 0.11.

Same store. Same product. Opposite answer. Pause and work out why before I say it.

*(Resume.)*

Because weekly data has no day-of-week variation left in it. Aggregating to a week averages the
weekend and the weekdays together and destroys the largest single signal in the series. It was
never that the four-week lag mattered more; it was that the thing that mattered most had already
been summed away.

That is worth more than it looks. **Feature importance is a statement about the table you built,
not about the business.** Change the frequency and the ranking inverts.

---

# ▶ STEP 2 — Watch the bias–variance tradeoff happen

*(Screen: run the depth sweep.)*

Lecture 4 summarized this as a three-row table on a slide — stump, middle depths, full tree. Here
it is measured on your own data, and the next cell draws it as a curve.

```
depth  leaves  train_RMSE  test_RMSE
    1       2         443        413
    2       4         388        369
    3       8         354        337
    5      32         284        303
    8     179         184        313
   12     698          84        385
 None    1200           0        406
```

**[pause]** — *let this sit. Do not talk over it.*

Read the two columns separately.

**Training error** falls the whole way down and ends at **zero**. Not near zero — zero. Look at
the leaf count on that row: **1,200 leaves for 1,211 training days.** The tree has given almost
every single day its own leaf and memorized the answer. A model that has memorized the training
set has a training error of zero by construction, and that number tells you nothing whatsoever.

**Test error** does something else. It falls to **303 at depth 5**, and then turns around and
climbs back to **406** — worse than the depth-2 tree with four leaves.

The gap between the two columns is overfitting, drawn to scale. At depth 5 it's 284 against 303,
almost nothing. At full depth it's 0 against 406.

**[pause]**

Two things to carry out of this table.

First, the model that fits the training data best is the *worst* model in the table on data it
has not seen. That is not a subtlety, that is the whole lesson, and it will repeat in every
lecture for the rest of the course.

Second — and this is the practical one — the depth-5 tree beats the fully grown tree, and the
only thing separating them is one keyword argument. Nothing about the data changed. `max_depth`
is the entire difference between 303 and 406.

---

# ▶ STEP 3 — Why one tree is not enough

*(Screen: run the bootstrap loop.)*

Lecture 4 claimed a tree is unstable — nudge the data and the structure changes. Let's test that
properly rather than take it on faith.

Twenty bootstrap resamples: draw 1,211 rows *with replacement*, refit a depth-5 tree, record what
it split on first, and keep its test predictions.

```
root splits across 20 bootstrap trees:
  20 x  |--- dow <= 4.50

per-day spread of predictions across the 20 trees:
  average sd 160 units   worst day 1,137 units
```

**[pause]**

Now, that first result looks like it refutes the lecture. Twenty out of twenty found the same
root. The root is perfectly stable.

It is stable because day-of-week is *overwhelming* — a 43% gap survives any resample you can
draw. **A stable root is not a stable tree.** Look at the second number: on a typical test day
the twenty trees disagree by 160 units, and on the worst day they are spread over **1,137 units**
— about a third of an average day's sales. Every bit of that disagreement lives below the root,
in the splits you cannot see from this printout.

That is the instability the lecture was talking about, and you can only see it by looking at
predictions, not at structure.

*(Run the averaging cell.)*

```
average RMSE of a single bootstrap tree : 324
RMSE of the 20 trees averaged together  : 266
improvement: 18%
```

**[pause]** — *let it land before saying anything.*

Eighteen percent. No tuning, no new features, no extra data — the twenty trees saw *less* data
than the original fit, since a bootstrap sample leaves out about a third of the rows each time.

And note where 266 sits against Step 2's table: the best single tree you found by sweeping depth
was **303**. Averaging twenty individually *worse* trees — 324 on average — produced something
better than the best one of them and better than the best tree in the whole sweep.

That is the result the next lecture is built on, and it is worth being precise about the
mechanism. Nothing here reduced any individual tree's bias. What fell was the *variance* you
measured two cells ago: the disagreement averages out, the signal does not.

What you have built is called **bagging** — bootstrap aggregating. It is not yet a random forest.
A forest adds one more idea, which is to stop the trees from all looking at day-of-week first,
and you can already see why that might matter: twenty trees that agree perfectly on their root
are twenty trees making correlated mistakes. That is Lecture 5, and it is one keyword argument.

---

# ▶ STEP 4 — Where trees reliably fail

*(Screen: run the full-tree range cell.)*

Every lecture in this course should end by telling you where the method breaks. Here is where a
tree breaks, and it follows from one sentence: **a tree predicts the mean of the training rows in
a leaf.** Every prediction it will ever make is an average of values it has already seen.

```
training target range : [0, 5,016]
prediction range      : [0, 4,527]
```

The prediction range sits inside the training range, and it *has* to — not "usually does," cannot.
A tree can never forecast a value above the largest one in its training data, because there is no
arithmetic in it that could produce one. No trend term, no extrapolation, nothing. If this store
grows 20% next year, the tree will predict last year's ceiling forever.

**[pause]**

*(Run the extremes cell.)*

```
busiest 10% of test days  (n=37): actual 4,115   predicted 3,929   under by 186
quietest 10% of test days (n=37): actual 2,030   predicted 2,237   over by  207
```

Both ends are wrong, and — this is the point — **they are wrong in opposite directions**.
The busy days are under-forecast by 186 units. The quiet days are over-forecast by 207. The model
is pulling everything toward the middle.

Be precise about what this is and is not. The hard ceiling did not bite this year: the busiest
test day was 4,717 units and the training set had a 5,016-unit day, so nothing was clipped.
What you are seeing is the softer, more common version — leaf averaging shrinks the extremes
inward. It is not a large effect in absolute terms next to a full tree's RMSE of 406. It is a
*systematic* one, which is worse, because it will not average out over the year and it is exactly
the days you care most about.

**[STOP — learner works]**

Uncomment the scatter block and fill in the two blanks — actual on x, predicted on y, with the
45-degree line. Pause the video, run it, and come back when you can describe the shape.

*(Resume.)*

You should be looking at a cloud that is flatter than the 45-degree line. Below the line on the
right, above it on the left, hinged in the middle. Once you have seen that shape you will
recognize it everywhere, in every method that predicts by averaging.

---

# ▶ BEFORE YOU LEAVE

*(Screen: the closing block.)*

**[STOP — learner works]** — *(replaces the in-room "Discuss")*

Two questions on the discussion board before the next session.

**First, the failure.** You have just seen a tree pull its predictions toward the middle at both
ends. Give me the *mechanical* reason — and I mean mechanical, in terms of what a leaf prediction
literally is, not "the model is conservative."

Then the operational half. This store is planning inventory. One of those two errors costs more
than the other: under-forecasting a busy day means an empty shelf and a lost sale; over-forecasting
a quiet day means stock sitting in the back. Which is worse — and does your answer change if the
product is milk rather than canned goods?

**[pause]**

**Second, the comparison.** You fitted a GAM to this exact series in Homework 2, and a tree to it
today. Which one would you actually hand to a store manager, and what specifically would each
let you say that the other would not?

Be concrete. The GAM gives you a smooth curve per feature and a partial-dependence plot you can
show someone. The tree gives you a path — *weekday, four weeks ago was busy, yesterday was busy,
predict 2,950* — that a manager can follow and argue with. Those are genuinely different kinds of
explanation, and one of them will suit an audience better than the other.

**[pause]**

One last thing before you close the file. Look back at Step 2's table and notice what is *not*
in this lab: we chose depth 5 by looking at the test set. Every number in that column was
computed on the year we held out, and then we picked the winner from it. Lecture 1 told you what
that is worth. Fixing it properly needs walk-forward validation on the training data alone, and
that is Homework 3's job.

Solutions for the coded parts go up on Canvas after the deadline. Next week: what happens when
you stop the twenty trees from all looking at day-of-week first.

---

# Appendix — expected output

Deterministic; verified across repeated runs. `random_state=42` on every fit; the bootstrap uses
a seeded `RandomState`.

| Quantity | Value |
|---|---|
| Data | 1,576 usable days (train 1,211, test 365), 2015-05-24 → 2016-05-22 |
| Benchmark — same day last year | RMSE **625** |
| Depth-3 tree | test RMSE **337** · root split `dow <= 4` |
| Depth-3 importances | `dow` 0.789 · `units_lag28` 0.106 · `units_lag1` 0.064 · `units_lag7` 0.030 · `is_event` 0.011 |
| Lecture 4 slide (weekly, same series) | `units_lag4` 0.62 dominant; `dow` absent |
| Weekday vs weekend mean (train) | 2,632 vs 3,773 — a 43% gap; `dow` alone explains 66% of training variance |
| Depth sweep — train / test RMSE | 1: 443/413 · 2: 388/369 · 3: 354/337 · **5: 284/303** · 8: 184/313 · 12: 84/385 · None: **0/406** |
| Leaves at full depth | 1,200 (for 1,211 training rows) |
| Bootstrap root splits | 20 / 20 identical (`dow <= 4.50`) |
| Prediction spread across 20 trees | average sd 160 units · worst day 1,137 |
| Single bootstrap tree (mean) | RMSE **324** |
| Twenty trees averaged | RMSE **266** — an 18% improvement |
| Full tree — training target range | [0, 5,016] |
| Full tree — prediction range | [0, 4,527] |
| Busiest 10% (n=37) | actual 4,115 · predicted 3,929 · **under by 186** |
| Quietest 10% (n=37) | actual 2,030 · predicted 2,237 · **over by 207** |

**Three things to know before recording.**

*The stable root is the interesting part of Step 3, not a disappointment.* Twenty out of twenty
bootstrap trees find the same root, which reads at first like the lecture's instability claim
failing. It is not — the instability is in the prediction spread (160 average, 1,137 worst),
all of it below the root. Do not skip past the apparent contradiction; it is the reason the step
measures predictions rather than structure.

*Averaging beats the best single tree, not just the average one.* The comparison the lab prints
is 324 → 266. The stronger comparison, which is worth saying out loud, is that 266 also beats
**303** — the best tree in Step 2's entire depth sweep, fitted on all the data.

*The extrapolation ceiling did not bind this year.* Training max 5,016, test max 4,717, so no
test day was clipped. The mechanism is real and worth stating as an absolute (a tree *cannot*
predict above its training range), but the failure actually visible in the numbers is
regression toward the middle — 186 under at the top, 207 over at the bottom. Say both; do not
let the second be mistaken for evidence of the first.
