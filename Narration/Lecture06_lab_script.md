# Lab 6 — Recording Script

**ECON 8310: Business Forecasting · Regularization — Breaking OLS, Then Rescuing It**

Lab: `Labs/Lecture06_lab.qmd` (5 steps) · Measured runtime: **~22 minutes** of narration
(the in-room version is budgeted at 40)

---

## How to use this document

- **`▶ STEP n — Title`** matches the lab's own headings exactly.
- *Italic parentheticals* are stage directions. **[pause]** is a beat; **[STOP — learner works]**
  is where you tell the viewer to pause the video and do something.
- Numbers are the lab's real output, verified by running the chunks end to end. Deterministic;
  the whole lab takes about 6 seconds, LASSO paths and cross-validation included.

### Corrections (2026-10-19)

An earlier pass reported that nothing needed correcting. The full-course review found five things
it missed, now fixed here and in the lab. (1) The 62-column design is *exactly* collinear, rank 58
at every window size, so OLS has no unique coefficients and the `OLS_max_coef` column is the
solver's minimum-norm pick (Step 1). (2) The model *did* have day-of-week dummies; the finding is
that it dropped them and kept the weekly lags (Step 3). (3) Tied features in Step 3 printed in an
order that changed from run to run; the lab now sorts them, so the block below is what it prints.
(4) 1,210 days is about three years, not four, and n = 120 is not the best training fit in Step 1
(n = 80 fits better, 84 against 145). (5) Random K-fold is valid for a pure autoregression with
uncorrelated errors (Step 5).

The multiples-of-7 survivors and the CV-is-pessimistic argument in Step 5 still hold, and Step 5's
is quantified below in a way the lab does not print. See the last section for the one row a sharp
student will ask about.

### In-room language that needs replacing

Three places: Step 3's "which survivor is not a multiple of 7," Step 4's "decide whether that trade
is worth it," and the closing **Discuss** block. All converted below.

---

# ▶ OPENING

*(Show the rendered lab, at the top of the document.)*

This is Lab 6, and today you're going to break a model on purpose.

Lecture 6 made three claims about ordinary least squares when predictors pile up. Coefficient
variance grows with the number of parameters. Correlated predictors produce wild estimates that
offset each other. And once you have more predictors than observations, there's no unique solution
at all.

Those claims are easy to nod along to and hard to feel. So today you'll watch OLS fail, not just
degrade but *fail*, and then you'll fix it by changing one line.

**[pause]**

We're using the same series as the last two labs, Lab 4 on decision trees and Lab 5 on tree
ensembles: daily units for CA\_1 FOODS. But the feature set changes. Up to now you've used ten
hand-picked columns. Today we build what an analyst would actually build after an afternoon of
feature engineering, and that's where the trouble comes from.

---

# ▶ SETUP

*(Run the setup cell, and scroll back through the feature-building block as you talk.)*

Look at what the setup does, because nothing in it is unreasonable.

There are twenty-eight consecutive lags, because you don't know which one matters. There are a few
weekly lags, at thirty-five, forty-two, forty-nine, and fifty-six days, and last year's lags, at
three sixty-four, three sixty-five, and three sixty-six. There are rolling means and standard
deviations at five windows, three Fourier pairs for annual seasonality, and day-of-week dummies.
Then come SNAP, events, day of year, and price.

```
k = 62 features, n_train = 1210, n/k = 19.5
```

**[pause]**

That's sixty-two predictors: thirty-five lags, ten rolling statistics, six Fourier terms, seven
day-of-week dummies, and four others.

Nobody sat down and decided to fit a sixty-two-parameter model. Every one of those columns was a
defensible small decision, and the total is what routine feature engineering produces. That's the
honest version of how this problem arrives: not from ambition, but from diligence.

Right now the ratio is comfortable, at about twenty observations per parameter. Hold on to that
number, because Step 1 is going to take it apart.

---

# ▶ STEP 1 — Break it

*(Run the shrinking-window table.)*

We keep the features and the test set fixed, and we shrink only the training window. It starts at
about three years, then about one year, then two hundred days, then a hundred and twenty, and then
eighty. The problem stays the same throughout, and only the ratio changes.

```
   n  n/k OLS_max_coef OLS_test_RMSE Ridge_test_RMSE
1210 19.5          153           280             278
 400  6.5          636         2,071             323
 200  3.2        1,364         5,874             363
 120  1.9    1,049,739     3,578,141             621
  80  1.3      355,629     2,072,167             770
```

*(Let the table sit on screen, and don't talk over it.)*

**[pause]**

Read across the OLS columns, and start with the test RMSE. At twenty observations per parameter,
it's **two hundred eighty**, which is a perfectly respectable model. At six observations per
parameter, the error has multiplied by seven. At three, it's five thousand eight hundred
seventy-four.

Then the bottom two rows stop being a degradation and become an explosion. The test RMSE is **three
and a half million**, on a series where a busy day is four thousand units. The largest coefficient
climbs past a million right alongside it, but hold that column loosely. I'll come back to why in a
moment.

You may also notice that those bottom two rows aren't in order: a hundred and twenty rows does
worse than eighty on both columns. There's no lesson in that ordering. Once a design is this badly
conditioned, the size of the blow-up is essentially arbitrary. What matters is the scale of both
numbers.

Now here's the part I want you to take away, and it isn't in the printed table.

At n equals a hundred and twenty, that model's **training** RMSE is about **a hundred forty-five**.
It looks fine. If you fitted it and looked only at how well it fits the data you gave it, you would
ship it. And on data it hasn't seen, its error is three and a half million.

**[pause]**

Here's one technical point, because a careful student will ask about it. With sixty-two features
and eighty rows, there are still more rows than parameters, so we never crossed the line the
lecture warned about, where k is greater than n. **You don't need k greater than n to get this.**
Twenty-eight consecutive lags of a smooth series are nearly collinear with one another, and that
alone is enough to wreck the forecasts.

And here's the honest footnote on the coefficient column. This design isn't just nearly collinear.
It's *exactly* collinear, at every window size. The seven-, fourteen- and twenty-eight-day rolling
means are averages of lags that are already in the model, and the seven day-of-week dummies always
add up to one. That's four exact dependencies, so the sixty-two columns have a rank of fifty-eight,
and OLS has no unique set of coefficients anywhere in this table. `LinearRegression` doesn't
complain. It quietly hands back one particular solution, the minimum-norm one, so read that column
as the solver's pick, not as "the" OLS coefficients. The RMSE column is safe, because the same
identities hold in the test rows, so every one of those solutions makes the same forecasts.

*(Point at the last column.)*

Now look at the right-hand column. It's the same data, the same features, and the same rows, with
`Ridge(alpha=100)` in place of `LinearRegression()`. One word changed.

The numbers run two seventy-eight, three twenty-three, three sixty-three, six twenty-one, and seven
seventy.

It gets worse as the data gets scarcer, which it should. But it degrades *gracefully*, and at the
row where OLS is out by three and a half million, Ridge is at six twenty-one, which is roughly the
benchmark. It isn't good, but the model is still a model.

That's the opening argument of Lecture 6, and you just measured it.

---

# ▶ STEP 2 — Two penalties, two behaviors

*(Run the three-model comparison.)*

Now go back to the full training set, where OLS forecast well, and compare the two penalties.

```
  OLS    test RMSE   280   non-zero coefficients  62/62
  Ridge  test RMSE   278   non-zero coefficients  62/62
  LASSO  test RMSE   286   non-zero coefficients  21/62
```

**[pause]**

I want you to notice the disappointing thing first, because if I skip it, you'll notice it anyway
and trust me less.

On the full training set, regularization buys **essentially nothing**. Ridge is two units better
than OLS, and LASSO is six units worse. With nineteen and a half observations per parameter, there
was no problem to solve, and the penalty solved it.

That's the correct lesson, not a failed demo. Regularization is **insurance**, and Step 1 is the
fire. You don't judge insurance by what it pays out in a year when the house doesn't burn down.

**[pause]**

But look at the last column, because that's where the two penalties differ in kind and not just in
degree.

Ridge keeps all sixty-two coefficients and shrinks them toward zero. LASSO sets forty-one of them
to **exactly** zero, and keeps twenty-one.

So for six units of RMSE, about two percent, LASSO handed you a model with a third as many moving
parts. That isn't an accuracy trade. It's a trade for *communicability*. Twenty-one coefficients is
a model you can print and discuss, while sixty-two is a model you can only run.

Neither penalty is better. They encode different beliefs. Ridge believes that many predictors each
matter a little, and LASSO believes that a few matter and the rest are noise. Which belief is right
is a question about your data, and in Step 3 you get to see which one this series votes for.

---

# ▶ STEP 3 — The path, and the order features die in

*(Run the path plots.)*

This step runs sixty LASSO fits along a grid of penalties, from zero point one up to one thousand,
and it plots every coefficient's trajectory.

*(Point at the left panel.)*

Each line is one feature. At the left, where the penalty is weak, everything is alive. Move right,
and the lines collapse onto zero one after another. By the right-hand edge, and I checked this,
**every single coefficient is zero.** The model at an alpha of one thousand predicts the mean and
nothing else.

*(Point at the right panel.)*

The right panel counts the same thing, as model size against penalty. It's a staircase down from
sixty-two to zero. At an alpha of one hundred, nine features are left.

**[pause]**

Lecture 6 argued that the **order** in which features leave is a more stable finding than any one
coefficient table. Coefficients wobble when you resample, but the order of elimination mostly
doesn't. So let's read the order.

*(Run the survivors cell.)*

```
the last 8 features standing, in reverse order of elimination:
   lag28
   lag7
   lag35
   lag364
   lag14
   lag21
   lag1
   lag56
```

Two pairs leave at the same penalty step: lag twenty-eight with lag seven, and lag twenty-one with
lag one. The lab sorts each tie, so this list comes out the same on every run.

**[STOP — learner works]** — *(replaces "which survivor is not a multiple of 7")*

Pause the video here. You saw this path in the lecture, with the weekly lags drawn in red. Find the
one survivor that isn't a multiple of seven, and think about why the LASSO might keep it anyway.

*(Resume.)*

Twenty-eight, seven, thirty-five, three sixty-four, fourteen, twenty-one, and fifty-six. Every one
of them is a **multiple of seven** except lag one. And lag one survives for a different reason.
Yesterday is the freshest reading of the level the series is at right now, and no weekly lag gives
you that.

**[pause]**

Now think about what was available to be chosen. The feature set had twenty-eight consecutive
lags, from one, two, and three all the way to twenty-eight. It also had seven day-of-week dummies,
which are a direct encoding of the weekly cycle. At the weakest penalty, some of those dummies
carry the biggest coefficients in the model. Under heavy penalty, the LASSO dropped every one of
them.

Forced to give things up, it kept exactly the lags that align with the day of the week, and it
discarded the ones that don't. It also kept lag three sixty-four, which is fifty-two weeks back. So
it chose the same-day-last-year lag that is *also* a multiple of seven, rather than lag three
sixty-five.

That's why the dropout ordering is worth reporting to someone. It didn't just rank your columns.
It recovered the structure of the series, a weekly cycle plus an annual echo, from a pile of
undifferentiated lags. And it agrees with what the tree found on its own in Lab 4, when day of
week took seventy-nine percent of the importance. Two completely different model families reached
the same finding.

---

# ▶ STEP 4 — Choosing alpha, and the one-standard-error rule

*(Run the cross-validation block.)*

So far I've been picking penalties by hand. Now let's do it properly, with five-fold
`TimeSeriesSplit` and thirty candidate alphas. And this part matters: the scaler goes **inside**
the pipeline, so it's refitted on each fold's training rows and never sees the validation fold.

```
  CV minimum   alpha   19.64   CV   323   test   286   non-zero  21
  one-SE rule  alpha   59.68   CV   346   test   309   non-zero  11
```

There are two ways to read the same cross-validation curve.

The **CV minimum** takes the alpha with the best cross-validated score. That's the standard
choice, and it lands on twenty-one features.

The **one-standard-error rule** takes the *largest* alpha, meaning the simplest model, whose score
is still within one standard error of the best. The argument is that the cross-validation curve is
itself an estimate with noise on it, so a model that's statistically indistinguishable from the
best, but much simpler, is the better bet.

Here that means going from **twenty-one features to eleven**, and paying **twenty-three units of
test RMSE** for it. The test error goes from two eighty-six up to three oh nine, which is about
eight percent.

**[STOP — learner works]**

Pause the video, because there are two things to do. First, uncomment the plot and fill in the
blank, which is the one-standard-error index, so that you can see the band and both vertical lines.
Then answer the question the lab asks: is that trade worth it here?

*(Resume.)*

I'll give you the shape of the answer rather than the answer itself, because the honest reply is
that **the lab hasn't told you enough to decide.**

If this model is going into a dashboard that a person reads and questions, eleven coefficients
against twenty-one is a real gain, and eight percent is cheap. If it's one component inside an
automated replenishment system that nobody reads, you just paid eight percent for a property
nobody uses.

It's worth knowing what the eleven survivors are. They're lags one, seven, fourteen, twenty-one,
twenty-seven, twenty-eight, thirty-five, fifty-six, and three sixty-four, plus the
three-hundred-sixty-four-day rolling mean and the SNAP flag. That's nine lags, a level, and SNAP, a
model you could describe out loud in one sentence. And notice that SNAP made the cut. It's the
same feature that permutation importance rescued in Lab 5.

---

# ▶ STEP 5 — Why your CV number looks worse than the truth

*(Run the fold breakdown.)*

There's one loose end from Step 4 that you shouldn't let pass. The cross-validation score at the
minimum was **three twenty-three**, and the test RMSE of that same model was **two eighty-six**.
Cross-validation was pessimistic by about thirteen percent.

If those had gone the other way, you'd be worried. This direction has a specific cause, and it's
sitting in Step 1.

```
  fold 1: train n= 205 (n/k= 3.3)   val n=201   RMSE   433
  fold 2: train n= 406 (n/k= 6.5)   val n=201   RMSE   294
  fold 3: train n= 607 (n/k= 9.8)   val n=201   RMSE   325
  fold 4: train n= 808 (n/k=13.0)   val n=201   RMSE   257
  fold 5: train n=1009 (n/k=16.3)   val n=201   RMSE   308
```

**[pause]**

Look at fold one. It trains on two hundred five rows, which is about three observations per
parameter. That's the row from Step 1's table where OLS was already at five thousand eight hundred
seventy-four, and this fold is scored as though it were the model you're going to ship. It comes in
at four thirty-three, far worse than any other fold.

The model you actually ship trains on all twelve hundred and ten rows, and no fold ever saw that
much data.

So expanding-window cross-validation is **structurally pessimistic**. It averages in early folds
that were fitted on a fraction of your data. And you can put a number on what that costs. The five
folds average to three twenty-three. Drop fold one alone, and the remaining four average to **two
ninety-six**, against a test RMSE of **two eighty-six**. Almost the entire gap came from that one
starved fold.

**[pause]**

There are two conclusions, and the second is the one that gets violated in practice.

The first is to read a walk-forward cross-validation number as a **conservative** estimate. It's a
lower bound on your model's quality, not an unbiased estimate of it, and being conservative is a
fine property in a forecast you're going to stake inventory on.

The second, and please don't do this, is that the pessimism is **not** a reason to switch to random
k-fold splits. Random folds would shrink the number, and they'd do it by training on Thursday to
predict the previous Tuesday. To be fair to them, on a lag regression like this one, that does no
harm. The paper Lecture 6 cites, by Bergmeir and his co-authors, shows that k-fold is valid for a
pure autoregression with uncorrelated errors. But you rarely know that you're in that case. When
you aren't, the number you get back measures your model's ability to interpolate inside a period
it has already seen, which isn't the job. A pessimistic estimate of the right quantity beats an
accurate estimate of the wrong one.

---

# ▶ BEFORE YOU LEAVE

*(Run the scoreboard.)*

```
Benchmark (same day last year)    625
Single tree (Lab 4)               324
LASSO, 62 features (today)        286
Random Forest (Lab 5)             248
XGBoost (Lab 5)                   244
```

Today's model, with six times as many features as the tree ensembles had, lands **third**.

Say that plainly rather than skating past it. Sixty-two engineered features, a properly
cross-validated penalty, and a careful pipeline produced two eighty-six. That's comfortably better
than one tree, and comfortably worse than the Random Forest you fitted in two lines in Lab 5, with
just ten hand-picked columns.

**[STOP — learner works]** — *(replaces the in-room "Discuss")*

Here are three questions to work through before you go. Pause the video after each one.

The first is why trees win here. Name one property of daily store sales that favors a tree
ensemble over an additive linear model, however many lags you feed it. It's in the tree you read
out loud in Lab 4. Think about what happens on a weekend when yesterday was already busy, and
whether a model that adds up separate effects can express that.

**[pause]**

The second is the counterpoint, and this one matters for Homework 4. In Homework 4 you run this
comparison on the weekly panel of thirty series, with forty-six features, and the ranking may not
come out the same. What's different about that problem that could change the answer? Think about
how many observations per series you get at weekly frequency, and what pooling thirty series does
to the noise.

The third is the one to argue about. You've now fitted five model families to this one series: a
GAM in Homework 2, a tree, a forest, boosting, and a regularized linear model. Hand exactly one of
them to a store manager who won't retune it for a year and won't be retrained. Which one do you
choose, and is it the one with the lowest RMSE?

**[pause]**

Notice two properties that this lab measured and that RMSE doesn't see. The eleven-feature LASSO
is the only one of the five you could write down as a single equation. And Step 1 showed you
which models fail *loudly* and which fail quietly. OLS at a hundred and twenty rows had a beautiful
training fit, and its test error was three and a half million. A year from now, if this store's data
collection changes and the model quietly gets less data than it used to, which of your five would
you rather be running unattended?

Solutions for the coded parts go up on Canvas after class. Next week, in Lecture 7, we
leave linear models behind for a while and start on neural networks.

---

# Appendix — expected output

Deterministic; whole lab runs in about 6 seconds.

| Quantity | Value |
|---|---|
| Feature set | **62** — 35 lags, 10 rolling stats, 6 Fourier, 7 dow dummies, 4 other · n/k = 19.5 |
| OLS as n shrinks (max\|coef\| / test RMSE) | 1210: 153 / 280 · 400: 636 / 2,071 · 200: 1,364 / 5,874 · 120: **1,049,739 / 3,578,141** · 80: 355,629 / 2,072,167 |
| Ridge over the same range | 278 · 323 · 363 · 621 · 770 |
| OLS at n=120 — *training* RMSE | **145** (not printed by the lab; worth saying out loud) |
| Full data, three models | OLS 280 (62/62) · Ridge 278 (62/62) · LASSO 286 (**21**/62) |
| Last 8 LASSO survivors | lag28, lag7, lag35, lag364, lag14, lag21, lag1, lag56 — all multiples of 7 except `lag1` |
| Coefficients alive at alpha = 1,000 | **0** (9 at alpha = 100) |
| CV minimum | alpha 19.64 · CV 323 · test **286** · 21 features |
| One-SE rule | alpha 59.68 · CV 346 · test **309** · 11 features |
| One-SE survivors | lag1, lag7, lag14, lag21, lag27, lag28, lag35, lag56, lag364, roll364_mean, **snap** |
| CV folds | 433 · 294 · 325 · 257 · 308 → mean **323**; without fold 1, **296** (test 286) |
| Scoreboard | benchmark 625 · tree 324 · **LASSO 286** · RF 248 · XGBoost 244 |

**Three things to know before recording.**

*The bottom two rows of Step 1 are not monotone, and a sharp student will point at it.* n=120 is
worse than n=80 on both columns — 3.6 million against 2.1 million. There is no lesson in the
ordering; once a design matrix is that ill-conditioned (and this one is exactly singular, rank 58
of 62) the magnitude is essentially arbitrary and depends on which near-collinear directions
happen to line up. Say that rather than explaining it,
and steer back to the point, which is the *scale* of both numbers against a series that runs to
about 4,000 units a day.

*Step 2 is a disappointing table and must be delivered as a finding.* Regularization gains nothing
on the full training set — Ridge by 2, LASSO worse by 6. The temptation is to hurry past it. Don't:
the insurance framing only works if you first admit the premium bought nothing this year, and Step
1 is the claim on the policy.

*The CV-pessimism arithmetic is worth adding aloud.* The lab prints five fold RMSEs but never
averages them. 323 with fold 1, **296** without it, against a test of 286 — that turns "CV is
structurally pessimistic" from an assertion into a measurement, and it takes ten seconds to say.
