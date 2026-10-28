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

### Two corrections that were applied to the lab

**Step 4 (2026-10-19).** The lab used to blame the tree for under-forecasting the busiest days and
over-forecasting the quietest. That pattern is regression to the mean: days chosen by their actual
sales produce it for any imperfect forecaster, and a linear regression on the same features shows
more of it. The prose, the closing question and the script below now say so. The code is unchanged.

**Step 3.** The step closed with "You just built a small Random Forest by hand." That is the
wrong name, and Lecture 5 contradicts it four days later: bagging and random forests differ *precisely* by the
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

*(Show the rendered lab, at the top of the document.)*

This is Lab 4, and today the course changes character.

Everything up to now has worked the same way. You specified a structure, like a trend and a season,
an AR term and an MA term, or a smooth function of the day of the year, and then you estimated it.
You chose the shape, and the data filled in the numbers.

A tree doesn't work that way. You hand it a pile of columns, and it finds the structure itself.
Nobody tells it that weekends matter.

**[pause]**

The series is daily units for CA\_1 FOODS, the same store and the same aisle you forecast in
**Homework 2**. It isn't the Lab 3 data, because Lab 3, last week, used electricity demand.
Homework 2 was this series with a GAM, and it was due before today, which is deliberate. It means
you can put a tree on identical data and say exactly what each method bought you. That comparison
is the closing question.

By the end you'll have done four things. You'll have read a fitted tree out loud, watched the
bias–variance tradeoff happen in a table, discovered why one tree is not enough, and found the one
place trees reliably fail.

You need `scikit-learn` and `data/processed/m5_daily.csv`. If the file is missing, run
`prep_m5.py` once.

---

# ▶ SETUP

*(Run the setup cell.)*

```
1576 usable days — train 1211, test 365
benchmark (same day last year): RMSE 625
```

That's just over four years of daily data, holding out the final year, which runs from May twenty
fifteen through May twenty sixteen.

The benchmark is the same one you've used since Lecture 1: predict today with the same day last
year. It scores **six hundred twenty-five** units. Write that down, because every number today gets
measured against it.

*(Leave the `FEATURES` list on screen for a moment.)*

**[pause]**

Now look at what we handed the model, because this is the important part of the setup.

There are ten columns: five lags, a SNAP flag, an event flag, the day of the week, the day of the
year, and the price. And there's **no date index**. There's no time in `X_train` at all. Every row
is an independent example, and everything the model knows about *when* it is, it knows because we
put it in those columns.

That reframing, a time series flattened into an ordinary table, is what lets everything from here
to Lecture 9 be applied to forecasting at all. Trees, forests, boosting and neural networks: none of
them know what a time series is. They all consume this table.

---

# ▶ STEP 1 — Fit a tree and read it out loud

*(Run the depth-3 fit and `export_text`.)*

We're using depth three, and that's deliberate. Eight leaves is small enough to read in full, and
you should read it in full, because being able to do that is most of what a tree is for.

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

The root split is `dow <= 4`. That puts Monday through Friday on one side and Saturday and Sunday on
the other, and the tree found it without being told that a week has a weekend in it.

**[STOP — learner works]** — *(replaces "read it aloud to the person next to you")*

Pause here and say the whole path out loud, in business language rather than in inequalities.
Don't say "`dow` less than or equal to four point five, then `units_lag28` above two thousand five
hundred ninety-six." Say something closer to this: *it's a weekday, four weeks ago was already busy,
and the same day last week was busy too, so predict about two thousand nine hundred fifty.* If
you're on your own, write that sentence as a comment, the way the lab suggests, and add a guess
about which side of the root sells more. Then check your guess against the two node values on the
tree plot.

If you can do that for one leaf, you can do it for a store manager, and that's something no model
in Lectures 1 through 3 could offer you.

*(Resume, and point at the bottom right of the printed tree.)*

If you guessed that the weekend side sells more, you were right, and we'll put numbers on that gap
in a moment.

There's one branch worth reading before we move on. On the weekend side, on days when yesterday was
already very busy, the tree splits on `is_event`. It predicts **four thousand forty-one** when
there's no event and **three thousand five hundred eighty-three** when there is, so the event lowers
the forecast by more than four hundred fifty units. Events in M5 include days like Christmas, when
the store is shut or nearly so. The tree has no idea what Christmas is, and it found the hole in the
data anyway.

**[pause]**

And the accuracy is **three hundred thirty-seven**, against a benchmark of six hundred twenty-five.
That's roughly a forty-six percent improvement, from eight leaves you could print on a slide.

*(Run the importances.)*

```
dow            0.789
units_lag28    0.106
units_lag1     0.064
units_lag7     0.030
is_event       0.011
```

Day of week takes seventy-nine percent of the importance. On its own, the day-of-week averages
explain about sixty-six percent of the variance in training units. The weekday average is two
thousand six hundred thirty-two and the weekend average is three thousand seven hundred
seventy-three, a gap of forty-three percent. There's no mystery about why the tree grabbed it first.

*(Show the callout that compares this with the lecture slide.)*

**[STOP — learner works]** — *(replaces "say why before moving on")*

This week's lecture, Lecture 4, had an importance table on the **weekly** version of this same store
and aisle. There, `units_lag4` was at zero point six two, and day of week was nowhere. It isn't in
the table at all. Here, on **daily** data, `dow` wins outright with zero point seven nine, and the
four-week lag drops to zero point one one.

It's the same store and the same product, with the opposite answer. Pause and work out why before I
say it.

*(Resume.)*

It's because weekly data has no day-of-week variation left in it. Aggregating to a week averages the
weekend and the weekdays together, and that destroys the largest single signal in the series. It
was never that the four-week lag mattered more. The thing that mattered most had already been summed
away.

That's worth more than it looks. **Feature importance is a statement about the table you built, not
about the business.** Change the frequency, and the ranking inverts.

---

# ▶ STEP 2 — Watch the bias–variance tradeoff happen

*(Run the depth sweep.)*

Lecture 4, this week's lecture, drew this curve for the weekly series. Here it is measured on your
own daily data, and the next cell draws the same curve.

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

*(Let the table sit, and don't talk over it.)*

**[pause]**

Read the two columns separately.

The training error falls the whole way down and ends at **zero**. Not near zero, but zero. Look at
the leaf count on that row: one thousand two hundred leaves for one thousand two hundred eleven
training days. The tree has given almost every single day its own leaf and memorized the answer. A
model that has memorized the training set has a training error of zero by construction, and that
number tells you nothing whatsoever.

The test error does something else. It falls to **three hundred three** at depth five, and then it
turns around and climbs back to **four hundred six**, which is worse than the depth-two tree with
four leaves.

The gap between the two columns is overfitting, drawn to scale. At depth five it's two hundred
eighty-four against three hundred three, which is almost nothing. At full depth it's zero against
four hundred six.

**[pause]**

There are two things to carry out of this table.

First, the model that fits the training data best is worse on data it hasn't seen than every tree
from depth two to depth twelve. That isn't a subtlety. That's the whole lesson, and it will repeat
in every lecture for the rest of the course.

Second, and this is the practical one, the depth-five tree beats the fully grown tree, and the only
thing separating them is one keyword argument. Nothing about the data changed. `max_depth` is the
entire difference between three hundred three and four hundred six.

---

# ▶ STEP 3 — Why one tree is not enough

*(Run the bootstrap loop.)*

This week's lecture claimed that a tree is unstable: nudge the data, and the structure changes.
Let's test that properly, rather than take it on faith.

We draw twenty bootstrap resamples. Each one draws one thousand two hundred eleven rows *with
replacement*, refits a depth-five tree, records what it split on first, and keeps its test
predictions.

```
root splits across 20 bootstrap trees:
  20 x  |--- dow <= 4.50

per-day spread of predictions across the 20 trees:
  average sd 160 units   worst day 1,137 units
```

**[pause]**

Now, that first result looks like it refutes the lecture. Twenty out of twenty found the same root,
so the root is perfectly stable.

It's stable because day of week is *overwhelming*, and a forty-three percent gap survives any
resample you can draw. But **a stable root is not a stable tree.** Look at the second number. On a
typical test day, the twenty trees disagree by a hundred sixty units, and on the worst day they're
spread over one thousand one hundred thirty-seven units, which is more than a third of an average
day's sales. Every bit of that disagreement lives below the root, in the splits you can't see from
this printout.

That's the instability the lecture was talking about, and you can only see it by looking at
predictions, not at structure.

*(Run the averaging cell.)*

```
average RMSE of a single bootstrap tree : 324
RMSE of the 20 trees averaged together  : 266
improvement: 18%
```

*(Let it land before you say anything.)*

**[pause]**

That's an eighteen percent improvement, with no tuning, no new features and no extra data. In fact,
the twenty trees saw *less* of the data than the original fit, since a bootstrap sample leaves out
about a third of the rows each time.

And notice where two hundred sixty-six sits against Step 2's table. The best single tree you found by
sweeping depth scored **three hundred three**. Averaging twenty trees that are individually
*worse*, at three hundred twenty-four on average, produced something better than the best one of
them, and better than the best tree in the whole sweep.

That's the result the next lecture is built on, and it's worth being precise about the mechanism.
Nothing here reduced any individual tree's bias. What fell was the *variance* you measured two cells
ago. The disagreement averages out, and the signal doesn't.

What you've built is called **bagging**, which is short for bootstrap aggregating. It isn't yet a
random forest. A forest adds one more idea, which is to stop the trees from all looking at day of
week first, and you can already see why that might matter. Twenty trees that agree perfectly on
their root are twenty trees making correlated mistakes. That's Lecture 5, next week, and it's one
keyword argument.

---

# ▶ STEP 4 — Where trees reliably fail

*(Run the full-tree range cell.)*

Every lecture in this course should end by telling you where the method breaks. Here's where a tree
breaks, and it follows from one sentence: **a tree predicts the mean of the training rows in a
leaf.** Every prediction it will ever make is an average of values it has already seen.

```
training target range : [0, 5,016]
prediction range      : [0, 4,527]
```

The prediction range sits inside the training range, and it *has* to. It isn't that it usually
does. It cannot do anything else. A tree can never forecast a value above the largest one in its
training data, because there's no arithmetic in it that could produce one. There's no trend term,
no extrapolation, nothing. If this store grows twenty percent next year, the tree will predict last
year's ceiling forever.

**[pause]**

*(Run the extremes cell.)*

```
busiest 10% of test days  (n=37): actual 4,115   predicted 3,929   under by 186
quietest 10% of test days (n=37): actual 2,030   predicted 2,237   over by  207
```

Both ends are wrong, and **they're wrong in opposite directions**. The busy days are under-forecast
by a hundred eighty-six units, and the quiet days are over-forecast by two hundred seven.

Now let's be precise about what this is, because it isn't what it looks like. First, the hard
ceiling didn't bite this year. The busiest test day was four thousand seven hundred seventeen units,
and the training set had a day of five thousand sixteen, so nothing was clipped.

Second, look at how we chose these days: by their *actual* sales. Part of what made a day one of the
busiest was luck, a bigger crowd than its features suggested, and no forecast sees luck coming. So
any imperfect forecaster under-predicts the days that turned out busiest and over-predicts the days
that turned out quietest. That's called **regression to the mean**, and it isn't a tree failure. A
linear regression on the same features, with day of week entered as dummies, *can* extrapolate, and
it pulls toward the middle even harder: two hundred fifty-five under at the top, and two hundred
sixty-nine over at the bottom. And if you choose the busiest days by the tree's *prediction*
instead, the sign flips. Those days actually sold three thousand eight hundred twenty, against a
forecast of four thousand one hundred thirty-seven.

**[STOP — learner works]**

Uncomment the scatter block and fill in the two blanks, with actual on the x-axis, predicted on the
y-axis, and the forty-five-degree line drawn in. Pause the video, run it, and come back when you can
describe the shape.

*(Resume.)*

You should be looking at a cloud that's flatter than the forty-five-degree line. It sits below the
line on the right and above it on the left, hinged in the middle. Once you've seen that shape, you'll
recognize it everywhere. With the actual values on the x-axis, every imperfect forecast looks like
this, however good it is, because none of them can predict the luck.

---

# ▶ BEFORE YOU LEAVE

*(Show the closing block.)*

By now you should have found three things. A tree can never predict outside its training range. The
pull toward the middle at both ends is regression to the mean, not a tree failure. And averaging
twenty unstable trees beat any one of them.

**[STOP — learner works]** — *(replaces the in-room "Discuss")*

There are two questions for the discussion board before the next session.

The first is about the pull to the middle. You've just seen the busiest days under-forecast and the
quietest over-forecast. Explain why *any* imperfect forecaster shows that pattern when the days are
picked by their actual sales, and what you'd expect if they were picked by the forecast instead.

Then there's the operational half. This store is planning inventory, and one of those two errors
costs more than the other. Under-forecasting a busy day means an empty shelf and a lost sale, while
over-forecasting a quiet day means stock sitting in the back. Which is worse? And does your answer
change if the product is milk rather than canned goods?

**[pause]**

The second question is about the comparison. You fitted a GAM to this exact series in Homework 2,
and a tree to it today. Which one would you actually hand to a store manager, and what specifically
would each one let you say that the other wouldn't?

Be concrete. The GAM gives you a smooth curve for each feature, and a partial-dependence plot you can
show someone. The tree gives you a path that a manager can follow and argue with: *it's a weekday,
four weeks ago was busy, a week ago was busy, so predict two thousand nine hundred fifty.* Those are
genuinely different kinds of explanation, and one of them will suit an audience better than the
other.

**[pause]**

There's one last thing before you close the file. Look back at Step 2's table, and notice what's
missing from this lab: we chose depth five by looking at the test set. Every number in that column
was computed on the year we held out, and then we picked the winner from it. Lecture 1, on forecast
evaluation, told you what that's worth. Fixing it properly needs walk-forward validation on the
training data alone, and that's Homework 3's job.

Solutions for the coded parts go up on Canvas after class. Next week, we'll see what happens
when you stop the twenty trees from all looking at day of week first.

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
predict above its training range), but what the numbers show is regression to the mean — 186
under at the top, 207 over at the bottom — and it comes from choosing days by their actual value,
not from the tree: a linear regression on the same features shows more (255 and 269, not printed
by the lab). Say both; do not let the second be mistaken for evidence of the first.
