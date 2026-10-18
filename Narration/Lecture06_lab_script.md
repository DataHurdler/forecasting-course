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

### Nothing needed correcting this time

Every claim in the `.qmd` survived the measurement pass — including the two that sound like they
were written from expectation rather than output (the multiples-of-7 survivors, and the
CV-is-pessimistic argument in Step 5). Both are true, and Step 5's is quantified below in a way
the lab does not print. See the last section for the one row a sharp student will ask about.

### In-room language that needs replacing

Three places: Step 3's "which survivor is not a multiple of 7," Step 4's "decide whether that trade
is worth it," and the closing **Discuss** block. All converted below.

---

# ▶ OPENING

*(Screen: rendered lab, top of document.)*

Lab 6, and today you are going to break a model on purpose.

Lecture 6 made three claims about ordinary least squares when predictors pile up: coefficient
variance grows with the number of parameters, correlated predictors produce wild offsetting
estimates, and past the point where you have more predictors than observations there is no unique
solution at all.

Those are easy to nod along to and hard to feel. So today you will watch OLS fail — not
degrade, *fail* — and then fix it by changing one line.

**[pause]**

Same series as the last two labs: daily units for CA\_1 FOODS. But the feature set changes. Up to
now you have used ten hand-picked columns. Today we build what an analyst would actually build
after an afternoon of feature engineering, and that is where the trouble comes from.

---

# ▶ SETUP

*(Screen: run setup. Scroll back through the feature-building block as you talk.)*

Look at what the setup does, because nothing in it is unreasonable.

Twenty-eight consecutive lags, because you don't know which one matters. A few weekly lags — 35,
42, 49, 56. Last year's lags, 364, 365, 366. Rolling means and standard deviations at five
windows. Three Fourier pairs for annual seasonality. Day-of-week dummies. Then SNAP, events,
day-of-year and price.

```
k = 62 features, n_train = 1210, n/k = 19.5
```

**[pause]**

Sixty-two predictors. Thirty-five lags, ten rolling statistics, six Fourier terms, seven
day-of-week dummies, four others.

Nobody sat down and decided to fit a sixty-two-parameter model. Every one of those columns was a
defensible small decision, and the total is what routine feature engineering produces. That is the
honest version of how this problem arrives: not from ambition, from diligence.

Right now the ratio is comfortable — about twenty observations per parameter. Hold onto that
number, because Step 1 is going to take it apart.

---

# ▶ STEP 1 — Break it

*(Screen: run the shrinking-window table.)*

We keep the features and the test set fixed and shrink only the training window. Four years, then
about one, then two hundred days, then a hundred and twenty, then eighty. Same problem throughout.
Only the ratio changes.

```
   n  n/k OLS_max_coef OLS_test_RMSE Ridge_test_RMSE
1210 19.5          153           280             278
 400  6.5          636         2,071             323
 200  3.2        1,364         5,874             363
 120  1.9    1,049,739     3,578,141             621
  80  1.3      355,629     2,072,167             770
```

**[pause]** — *let it sit on screen. Do not talk over it.*

Read across the OLS columns. At twenty observations per parameter, the largest coefficient is
**153** and the test RMSE is **280** — a perfectly respectable model. At six observations per
parameter, the coefficient is 636 and the error has multiplied by seven. At three, it is 1,364 and
5,874.

And then the bottom two rows stop being a degradation and become an explosion. The largest
coefficient is **over a million.** The test RMSE is **three and a half million** — on a series
where a busy day is four thousand units.

**[pause]**

Now here is the part I want you to take away, and it is not in the printed table.

At n = 120, that model's **training** RMSE is about **145.** It looks fine. If you fitted it and
looked only at how it fits the data you gave it, you would ship it. It is the best-fitting model
on this slide and it is wrong by a factor of a thousand on data it has not seen.

**[pause]**

One technical point, because a careful student will ask. We never crossed the line the lecture
warned about. Sixty-two features and eighty rows — there are still more observations than
parameters, so OLS has a unique solution the whole way down this table. **You do not need k
greater than n to get this.** Twenty-eight consecutive lags of a smooth series are nearly
collinear with one another, and near-collinearity is enough. The matrix does not have to be
singular, only close to it.

*(Point at the last column.)*

Now the right-hand column. Same data, same features, same rows — `Ridge(alpha=100)` instead of
`LinearRegression()`. One word changed.

278, 323, 363, 621, 770.

It gets worse as data gets scarcer, which it should. But it degrades *gracefully*, and at the row
where OLS is out by three and a half million, Ridge is at 621 — roughly the benchmark. Not good,
but the model is still a model.

That is Lecture 6's opening argument, and you just measured it.

---

# ▶ STEP 2 — Two penalties, two behaviors

*(Screen: run the three-model comparison.)*

Back to the full training set, where OLS was fine, and now compare the two penalties.

```
  OLS    test RMSE   280   non-zero coefficients  62/62
  Ridge  test RMSE   278   non-zero coefficients  62/62
  LASSO  test RMSE   286   non-zero coefficients  21/62
```

**[pause]**

I want you to notice the disappointing thing first, because if I skip it you will notice it
anyway and trust me less.

**On the full training set, regularization buys essentially nothing.** Ridge is two units better
than OLS. LASSO is six units worse. With nineteen observations per parameter there was no problem
to solve, and the penalty solved it.

That is the correct lesson, not a failed demo. Regularization is **insurance**, and Step 1 is the
fire. You do not judge insurance by what it pays out in a year when the house doesn't burn down.

**[pause]**

But look at the last column, because that is where the two penalties differ in kind and not just
in degree.

Ridge keeps all sixty-two coefficients and shrinks them toward zero. LASSO sets forty-one of them
to **exactly** zero and keeps twenty-one.

So for six units of RMSE — about two percent — LASSO handed you a model with a third as many
moving parts. That is not an accuracy trade, it is a *communicability* trade. Twenty-one
coefficients is a model you can print and discuss. Sixty-two is a model you can only run.

Neither penalty is better. They encode different beliefs: Ridge believes many predictors each
matter a little, LASSO believes a few matter and the rest are noise. Which belief is right is a
question about your data, and in Step 3 you get to see which one this series votes for.

---

# ▶ STEP 3 — The path, and the order features die in

*(Screen: run the path plots.)*

Sixty LASSO fits along a grid of penalties, from 0.1 up to 1,000, and we plot every coefficient's
trajectory.

*(Point at the left panel.)*

Each line is one feature. At the left, weak penalty, everything is alive. Move right and lines
collapse onto zero one after another. By the right-hand edge — and I checked this — **every single
coefficient is zero.** The model at alpha = 1,000 predicts the mean and nothing else.

*(Point at the right panel.)*

The right panel is the same thing counted: model size against penalty. It is a staircase down from
62 to 0. At alpha = 100, nine features are left.

**[pause]**

Lecture 6 argued that the **order** features leave is a more stable finding than any one
coefficient table — coefficients wobble under resampling, the elimination order mostly does not.
So let's read the order.

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

**[STOP — learner works]** — *(replaces "which survivor is not a multiple of 7")*

Pause here. You saw this path in the lecture, with the weekly lags in red. Find the one survivor
that is not a multiple of seven, and think about why the LASSO might keep it anyway.

*(Resume.)*

28. 7. 35. 364. 14. 21. 56. Every one of them is a **multiple of seven** except `lag1`. And `lag1`
survives for a different reason: yesterday is the freshest reading of the level the series is at
right now, which no weekly lag gives you.

**[pause]**

Now think about what was available to be chosen. The feature set had twenty-eight consecutive
lags — 1, 2, 3, all the way to 28. There is no "week" feature. Nothing in the data told the model
that seven days is a meaningful interval. The day-of-week dummies were there as an alternative
route to the same information, and the LASSO dropped those too.

Under increasing penalty, forced to give things up, it kept exactly the lags that align with
day-of-week and discarded the ones that do not. It also kept `lag364` — which is fifty-two weeks,
so the same-day-last-year lag that is *also* a multiple of seven, rather than `lag365`.

That is why the dropout ordering is worth reporting to someone. It did not just rank your columns.
It recovered the structure of the series — a weekly cycle plus an annual echo — from a pile of
undifferentiated lags. And it agrees with what the tree found on its own in Lab 4, when
day-of-week took 79% of the importance. Two completely different model families, same finding.

---

# ▶ STEP 4 — Choosing alpha, and the one-standard-error rule

*(Screen: run the CV block.)*

So far I have been picking penalties by hand. Now do it properly: five-fold `TimeSeriesSplit`,
thirty candidate alphas, and — this matters — the scaler goes **inside** the pipeline, so it is
refitted on each fold's training rows and never sees the validation fold.

```
  CV minimum   alpha   19.64   CV   323   test   286   non-zero  21
  one-SE rule  alpha   59.68   CV   346   test   309   non-zero  11
```

Two ways to read the same CV curve.

The **CV minimum** takes the alpha with the best cross-validated score. Standard, and it lands on
21 features.

The **one-standard-error rule** takes the *largest* alpha — the simplest model — whose CV score is
still within one standard error of the best. The argument is that the CV curve is itself an
estimate with noise on it, so a model that is statistically indistinguishable from the best but
much simpler is the better bet.

Here that means going from **21 features to 11**, and paying **23 units of test RMSE** for it —
286 up to 309, about eight percent.

**[STOP — learner works]**

Two things to do. First, uncomment the plot and fill in the blank — the one-SE index — so you can
see the band and both verticals. Then answer the question the lab asks: is that trade worth it
here?

*(Resume.)*

I'll give you the shape of the answer rather than the answer, because the honest reply is that
**the lab has not told you enough to decide.**

If this model is going into a dashboard that a person reads and questions, eleven coefficients
against twenty-one is a real gain, and eight percent is cheap. If it is one component inside an
automated replenishment system that nobody reads, you just paid eight percent for a property
nobody consumes.

It is worth knowing what the eleven survivors are: `lag1`, `lag7`, `lag14`, `lag21`, `lag27`,
`lag28`, `lag35`, `lag56`, `lag364`, the 364-day rolling mean, and `snap`. Nine lags, a level, and
the SNAP flag. That is a model you could describe out loud in one sentence — and notice `snap`
survived to the very end, which is the same feature permutation importance rescued in Lab 5.

---

# ▶ STEP 5 — Why your CV number looks worse than the truth

*(Screen: run the fold breakdown.)*

One loose end from Step 4 that you should not let pass. The CV score at the minimum was **323**.
The test RMSE of that same model was **286**. Cross-validation was pessimistic by about 13%.

If those had gone the other way you would be worried. This direction has a specific cause, and it
is sitting in Step 1.

```
  fold 1: train n= 205 (n/k= 3.3)   val n=201   RMSE   433
  fold 2: train n= 406 (n/k= 6.5)   val n=201   RMSE   294
  fold 3: train n= 607 (n/k= 9.8)   val n=201   RMSE   325
  fold 4: train n= 808 (n/k=13.0)   val n=201   RMSE   257
  fold 5: train n=1009 (n/k=16.3)   val n=201   RMSE   308
```

**[pause]**

Look at fold 1. It trains on 205 rows — three observations per parameter. That is the row from
Step 1's table where OLS was already at 5,874, and it is scored as though it were the model you
are going to ship. It comes in at 433, far worse than any other fold.

The model you actually ship trains on all 1,210 rows. No fold ever saw that much data.

So expanding-window cross-validation is **structurally pessimistic**: it averages in early folds
fitted on a fraction of your data. And you can put a number on how much that costs. The five folds
average to 323. Drop fold 1 alone and the remaining four average to **296** — against a test RMSE
of **286.** Almost the entire gap was that one starved fold.

**[pause]**

Two conclusions, and the second is the one that gets violated in practice.

First: read a walk-forward CV number as a **conservative** estimate. It is a lower bound on your
model's quality, not an unbiased estimate of it, and being conservative is a fine property in a
forecast you are going to stake inventory on.

Second, and please do not do this: the pessimism is **not** a reason to switch to random k-fold
splits. Random folds would fix the number, and they would fix it by training on Thursday to
predict the previous Tuesday. The optimistic number you get back would be measuring your model's
ability to interpolate inside a period it has already seen, which is not the job. A pessimistic
estimate of the right quantity beats an accurate estimate of the wrong one.

---

# ▶ BEFORE YOU LEAVE

*(Screen: run the scoreboard.)*

```
Benchmark (same day last year)    625
Single tree (Lab 4)               324
LASSO, 62 features (today)        286
Random Forest (Lab 5)             248
XGBoost (Lab 5)                   244
```

Today's model, with six times as many features as the tree ensembles had, lands **third**.

Say that plainly rather than skating past it. Sixty-two engineered features, a properly
cross-validated penalty, and a careful pipeline produced 286 — comfortably better than one tree,
and comfortably worse than a Random Forest you fitted in two lines with no feature engineering
at all in Lab 5.

**[STOP — learner works]** — *(replaces the in-room "Discuss")*

Three questions for the board.

**Why trees win here.** Name one property of daily store sales that favours a tree ensemble over
an additive linear model, however many lags you feed it. It is in the tree you read out loud in
Lab 4 — think about what happens on a weekend when yesterday was already busy, and whether a model
that adds up separate effects can express that.

**[pause]**

**The counterpoint, and this one matters for Homework 4.** In Homework 4 you run this comparison on
the weekly, thirty-series panel with forty-six features. The ranking may not come out the same.
What is different about that problem that could change the answer? Think about how many
observations per series you get at weekly frequency, and what pooling thirty series does to the
noise.

**The one to argue about.** You have now fitted five model families to this one series — a GAM in
Homework 2, a tree, a forest, boosting, and a regularized linear model. Hand exactly one of them
to a store manager who will never retune it and will not be retrained. Which, and is it the one
with the lowest RMSE?

**[pause]**

Notice the two properties this lab measured that RMSE does not see. The eleven-feature LASSO is
the only model of the five you could write on a whiteboard. And Step 1 showed you which models
fail *loudly* versus quietly — OLS at n=120 had a beautiful training fit and was wrong by a factor
of a thousand. In a year, if this store's data collection changes and the model quietly gets less
data than it used to, which of your five would you rather be running unattended?

Solutions for the coded parts go up on Canvas after the deadline. Next week we leave linear models
behind for good and start on neural networks.

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
ordering; once a design matrix is that ill-conditioned the magnitude is essentially arbitrary and
depends on which near-collinear directions happen to line up. Say that rather than explaining it,
and steer back to the point, which is the *scale* of both numbers against a series that runs to
about 4,000 units a day.

*Step 2 is a disappointing table and must be delivered as a finding.* Regularization gains nothing
on the full training set — Ridge by 2, LASSO worse by 6. The temptation is to hurry past it. Don't:
the insurance framing only works if you first admit the premium bought nothing this year, and Step
1 is the claim on the policy.

*The CV-pessimism arithmetic is worth adding aloud.* The lab prints five fold RMSEs but never
averages them. 323 with fold 1, **296** without it, against a test of 286 — that turns "CV is
structurally pessimistic" from an assertion into a measurement, and it takes ten seconds to say.
