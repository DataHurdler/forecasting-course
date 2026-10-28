# Lab 11 Part 1 — Recording Script

**ECON 8310: Business Forecasting · Structural Time Series — and Two Ways to Get the Interval Wrong**

Lab: `Labs/Lecture11_Part1_lab.qmd` (4 steps) · Measured runtime: **~22 minutes** of narration
(the in-room version is budgeted at 40)

---

## How to use this document

- **`▶ STEP n — Title`** matches the lab's own headings exactly.
- *Italic parentheticals* are stage directions. **[pause]** is a beat; **[STOP — learner works]**
  is where you tell the viewer to pause the video and do something.
- `random_seed=42` on both fits and `default_rng(42)` on the forecast simulation, so the table
  reproduces. Two fits at about 20 seconds each — no dead air worth planning around.

### A defect this script found, and the fix that went in

The lab's Step 3 table used to print **1,585** for the frozen forecast and **1,578** for the
propagated one — inviting the reader to see a 7-unit difference between the two methods.

**The true difference is exactly zero.** Propagating the level adds steps with mean zero, so
E[level] is unchanged and the two methods have the *same* point forecast by construction. The
7 units were Monte Carlo noise in the predictive mean, and the table dressed noise as signal in
the one step where the whole lesson is that only the *intervals* differ.

The lab now computes the point forecast from the deterministic part — no simulation — so both
rows report **1,589**, identically, and the callout says why. Simulation is used for the interval
only, which is the one thing it is needed for.

*(Note for the record: the HW06 Part 1 solution key reports **1,587**, from a noisier simulated
estimate of the same quantity, as the deck's narration once did. 1,589 is the cleaner number — it
carries no forecast-simulation error at all. The gap is 2 units, 0.13%, and the deck's headline
claim "beat seasonal naive by 4%" is *more* exact at 1,589 (4.28%) than at 1,587 (4.40%). The deck
and its narration now read 1,589; aligning the solution key is pending owner approval.)*

### In-room language that needs replacing

Step 3's two "your turn" blocks, Step 4's "your turn", and the three closing **Discuss** blocks.

---

# ▶ OPENING

*(Show the rendered lab, at the top of the document.)*

This is Lab 11, Part 1. Last week's model, in Lab 10, had one parameter and no time in it. This week,
we point the same machinery at an actual time series.

A structural model writes a series as **parts you can name**: a level that drifts, a seasonal shape,
and noise. Then it puts a prior on each one. That's the appeal, and it's a real one. The output isn't
a forecast line. It's a distribution for every future week, and you can pull out the level and the
seasonality and look at each one separately.

**[pause]**

Today has an unusual shape, so let me tell you where it goes. You'll build the model, and you'll hit
a failure that the diagnostics catch. You'll fix it with a change that leaves the model itself almost
exactly as it was. Then you'll forecast two ways, and you'll find that the **theoretically correct**
way produces a *worse-calibrated* interval than the shortcut.

That last result isn't a bug in the lab. It's the model telling you something about itself, and
working out what takes the rest of the session.

---

# ▶ SETUP

*(Run the setup cell.)*

```
train 225 weeks, test 52 weeks | 6 Fourier coefficients standing in for 52
```

It's CA\_1 FOODS again, the series you've been forecasting since Homework 1, and this time we hold
out a full final year.

*(Point at the Fourier line.)*

One line deserves your attention: **six coefficients standing in for fifty-two**. Three sine-cosine
pairs describe the annual shape, instead of fifty-two weekly dummies. That's a modeling choice with a
cost and a benefit. You can't represent a one-week spike, but you have six parameters to estimate
instead of fifty-two, on two hundred twenty-five observations. Remember it, because it comes back in
Step 3 as a suspect.

Also notice the standardization. `y_z` is the series in standard units, because the priors below, an
Exponential with rate ten and a Normal with standard deviation zero point five, assume unit scale.
Priors aren't scale-free, and this is the line that makes them defensible.

---

# ▶ STEP 1 — Fit it the obvious way

*(Run the centered model.)*

The model is a drifting level plus seasonality. PyMC has a distribution for a random walk, so the
obvious thing is to reach for it: `pm.GaussianRandomWalk`. It reads exactly like the math.

```
R-hat        1.022   (want < 1.01)
ESS (bulk)   274     (want > 400)
divergences  0
```

**[pause]**

It **did not converge**, and I want you to look carefully at *how* you know.

R-hat is one point zero two two, above the threshold of one point zero one, which means the four
chains don't agree with each other. The effective sample size is two hundred seventy-four, below
four hundred. The draws are so autocorrelated that six thousand samples carry less information than
a few hundred independent ones.

And the divergences are zero.

That last line matters, because divergences are the diagnostic everyone learns first, and here
they're silent. The sampler never had to give up on a step. It just wandered badly. **Different
pathologies trip different instruments**, which is precisely why you check all three, every time,
rather than the one you remember.

Next week, in Lab 11, Part 2, you'll see the mirror image: a model where R-hat and the effective
sample size both pass, and forty-three divergences are the only thing objecting.

*(Point at `target_accept`.)*

And notice that this is **not** fixed by asking the sampler to try harder. `target_accept` is already
at zero point nine five. The problem isn't effort.

---

# ▶ STEP 2 — Reparameterize

*(Show the two model blocks side by side, then run.)*

The problem is the **shape of the posterior**, not the model.

A centered random walk couples the level to `sigma_level`. When the scale is small, the steps have
to be small, so the posterior narrows into a funnel that the sampler can't traverse with a single
step size. The fix is to write the walk as a cumulative sum of **standard** Normal draws, scaled
afterwards. Now `z` and `sigma_level` are independent in the posterior geometry.

```
R-hat        1.002   (want < 1.01)
ESS (bulk)   2376    (want > 400)
divergences  0
```

**[pause]**

R-hat went from one point zero two two to one point zero zero two. The effective sample size went
from two hundred seventy-four to **two thousand three hundred seventy-six**, which is nearly nine
times more information out of the same number of draws.

And here's the sentence to say slowly: **same data, same random walk, new coordinates.** A `cumsum`
of z times sigma, and a `GaussianRandomWalk` with that same sigma, describe the same walk. What
changed is the coordinate system the sampler explores.

That's worth internalizing, because it's unlike everything else in this course. It isn't a modeling
improvement, and it isn't a bias-variance trade. It's a **computational** fix to a **computational**
problem.

*(Read the callout under the code.)*

There's one honest footnote. The two fits are *almost* the same model, not exactly.
`GaussianRandomWalk` lets week one's level start anywhere, because its default start is a Normal
with a standard deviation of one hundred. PyMC warns about that default, and our Setup cell silences
warnings, so you never saw it. The `cumsum` version starts the walk near zero. That one difference
nudges sigma-level up, from about four hundred fifty units a week to about five hundred twenty. A
default you didn't choose is still a prior.

*(Show the components figure.)*

And now you get the payoff of a structural model, which is that you can look at the parts. The top
panel shows the level, a slow drift under the noise. The bottom panel shows the seasonal shape that
the six Fourier coefficients recovered. Neither of those is available from XGBoost at any price.

---

# ▶ STEP 3 — Forecast, two ways

*(Show the two methods, then run the table.)*

To forecast, you need the level at each future week, and there are two candidates.

The first is to hold the level constant. You carry the last estimated level forward, unchanged.
That's simple, and it asserts that the level stops drifting the moment your data ends, which isn't
what the model says.

The second is to propagate it. You let the level keep random-walking forward, exactly as it did
in-sample. This is what the fitted model actually implies, and it's the defensible choice.

```
             method  RMSE  coverage_%  mean_width
level held constant  1589          96        6340
   level propagated  1589         100       11667
```

*(Let the table sit for a moment.)*

**[pause]**

The point forecasts are **identical**, one thousand five hundred eighty-nine on both rows, and that's
not a coincidence. Propagating the level adds steps whose mean is zero, so the expected level is
unchanged, and RMSE simply can't tell these two apart. Everything that separates them is in the
**interval**, which differs by a factor of nearly two, and not in the direction you'd predict.

The theoretically correct method over-covers. A hundred percent of the weeks fell inside a
ninety-four percent interval, while the shortcut lands at ninety-six percent, close to nominal. So
the **defensible choice calibrated worse**.

*(Show the callout.)*

**[STOP — learner works]**

Pause the video, and work out why before I tell you. Then do the first "your turn," which prints the
two scale parameters and the drift accumulated over fifty-two weeks.

*(Resume.)*

Over fifty-two weeks, a random walk accumulates about the square root of fifty-two times
`sigma_level` of drift, and the square root of fifty-two is about seven. So if `sigma_level` is even
somewhat too large, the interval inflates fast, and it inflates with the *square root of the
horizon*.

And there's good reason to think it **is** too large, which brings back the Fourier choice from the
setup. The random walk is the most flexible component in this model. Anything the six coefficients
can't represent, like a promotion, a holiday that moves, or a single bad week, has nowhere else to
go, so it gets absorbed into the level. That inflates `sigma_level` beyond genuine drift, and the
forecast then propagates that inflated drift for a year.

So, and this is the sentence that matters, the frozen version is **not right**. It's wrong in a way
that happens to cancel an overestimated drift, on this series, at this horizon. Rely on that, and it
will fail somewhere else, silently, and you won't know which of the two errors moved.

*(Run the by-horizon table.)*

```
coverage / mean width, by horizon (nominal 94%)
             method      wk 1-13      wk 14-39      wk 40-52
level held constant 100% / 6,255   92% / 6,537  100% / 6,027
   level propagated 100% / 8,081 100% / 11,974 100% / 14,637
```

*(This table is the real lesson of the step.)*

**[pause]**

Read the **widths** across each row, not the coverage.

The frozen version is six thousand two hundred fifty-five wide in weeks one to thirteen, and six
thousand twenty-seven wide in weeks forty to fifty-two. It's *narrower* a year out than a month out.
It has no mechanism to widen, because nothing in it knows that time has passed.

The propagated version goes from eight thousand eighty-one, to eleven thousand nine hundred
seventy-four, to fourteen thousand six hundred thirty-seven. It nearly doubles.

So one of these models **knows** that forecasting further ahead is harder, and the other one
doesn't. That's a structural difference, not a scoring difference, and it's why the aggregate table
on its own would have misled you. On this series, at this horizon, the ignorance happened to cost
nothing. On a series with a genuinely drifting level, it would cost a great deal.

*(One more, if you have time.)*

And notice the only cell in the table that falls below nominal. The frozen version drops to
**ninety-two percent** in weeks fourteen to thirty-nine. Its respectable ninety-six percent average
is hiding a stretch where it's overconfident. Aggregates hide things, and that's the third time this
course has said so.

---

# ▶ STEP 4 — Calibrated is not the same as useful

*(Run the summary block.)*

Take the propagated version, the one the model implies, and ask it two **separate** questions.

```
coverage      100%   against a nominal 94%
mean width    11,667 units
              = 59% of average weekly units
RMSE          1,589
seasonal naive  1,660
```

The first question is whether it's calibrated. It is, and then some. It covered a hundred percent of
the weeks against a nominal ninety-four, so it isn't overconfident. It's conservative.

The second question is whether it's useful.

**[pause]**

The interval is **eleven thousand six hundred sixty-seven units wide on average**, on a series that
averages about nineteen thousand seven hundred. Call it plus or minus thirty percent, *on average*:
about twenty percent in the first quarter, and thirty-seven in the last. Now say that out loud as a
business sentence: "I'm ninety-four percent confident that demand six months from now is somewhere
between roughly thirty percent below and thirty percent above average."

Nobody places an inventory order against that. It's honest, and it's useless, and those aren't the
same axis.

**[STOP — learner works]**

Pause the video, and write a one-sentence answer to each question. Then answer the third question
the lab actually asks: does passing the first excuse failing the second?

*(Resume.)*

The answer to that third question is no. A hundred percent coverage against a nominal ninety-four
isn't praise. It's a miss in the conservative direction, and an interval can be honest and still be
too wide to act on. Passing the first doesn't excuse failing the second.

Then there's the point forecast: one thousand five hundred eighty-nine, against seasonal naive's one
thousand six hundred sixty. That's **about four percent better**, for a model that took twenty
seconds to sample, a reparameterization to converge, and a page of code.

If accuracy were the argument for this model, that number would be an embarrassment. But accuracy
isn't the argument. What you bought is a level and a seasonal component you can plot and defend, a
full distribution for every future week, and, the thing this whole lab has been about, an interval
that **admits when it doesn't know**. Whether that trade is worth four percent is a business
question, and it depends entirely on whether anyone downstream will use the uncertainty you went to
the trouble of producing.

---

# ▶ BEFORE YOU LEAVE

**[STOP — learner works]** — *(replaces the in-room "Discuss")*

Here are three questions to work through on your own before you go.

The first is about the two problems. The convergence failure was caught by a **diagnostic**: R-hat
and the effective sample size, computed from the fit alone, before you looked at any data you were
predicting. The over-wide interval was caught by nothing of the sort. It needed a **held-out year and
a coverage calculation**. What's the structural difference between those two kinds of error, and
what habit catches the second kind?

**[pause]**

Here's the difference worth landing. A diagnostic asks whether the computation worked, and a
held-out check asks whether the model is any good. No amount of the first substitutes for the
second, so the habit that catches the second kind is the held-out check. A perfectly converged
sampler will happily give you a useless interval.

The second question is about the payoff. The point forecast beat seasonal naive by about four
percent. Given that, what did fitting this model actually buy? Name one business question it can
answer that XGBoost from Homework 4 can't.

**[pause]**

Step 4 already gave you the first half: a level and a seasonal component you can defend, a full
distribution for every future week, and an interval that's honest about what it doesn't know. For
the business question, start from the one in this week's lecture: how much stock keeps the chance of
a stockout under five percent? XGBoost hands you one number per week, and that question needs a
distribution.

The third question is about the fix. The interval is honest and too wide to act on. Propose one
change to the **model**, not the forecasting code, that would narrow it, and say what you're assuming
in exchange.

**[pause]**

There are two good answers, and both are in this lab. One is a tighter prior on `sigma_level`, which
assumes the level really does drift slowly. The other is more seasonal flexibility, so that the
Fourier terms stop dumping their residual into the level. The second is the more interesting answer,
because it addresses the cause rather than the symptom.

Solutions for the coded parts go up on Canvas after class. Next week, in Lab 11, Part 2, we'll have
ten series instead of one, and we'll see what happens when some of them barely have any data.

---

# Appendix — expected output

`random_seed=42` on both fits, `default_rng(42)` on the forecast draws. Two fits, ~20 s each.

| Quantity | Value |
|---|---|
| Split | train **225** weeks, test **52** · 6 Fourier coefficients (3 pairs) for a 52-week period |
| **Centered** model | R-hat **1.022** · ESS **274** · divergences **0** |
| **Non-centered** model | R-hat **1.002** · ESS **2,376** · divergences 0 |
| ESS improvement | 274 → 2,376, about **8.7×**, from a reparameterization that changes no belief |
| Level held constant | RMSE **1,589** · coverage **96%** · mean width 6,340 |
| Level propagated | RMSE **1,589** (identical by construction) · coverage **100%** · mean width **11,667** |
| By horizon — frozen | 100% / 6,255 · **92%** / 6,537 · 100% / 6,027 (*narrower* a year out) |
| By horizon — propagated | 100% / 8,081 · 100% / 11,974 · 100% / 14,637 (**1.8× widening**) |
| Interval width in context | 11,667 on a mean of ~19,700 → about **±30% on average** (±20% in weeks 1–13, ±37% in weeks 40–52) |
| Seasonal naive | **1,660** → the model is about **4%** better (4.28%) |

**Three things to know before recording.**

*Step 1's failure shows zero divergences, and that is the teaching point.* R-hat and ESS catch it;
the diagnostic students learn first is silent. Next week's lab is the mirror image — R-hat and ESS
pass and 43 divergences are the loudest objection. Say both, so "check all three" stops sounding like
ritual.

*The by-horizon table matters more than the aggregate one.* The headline table makes it look like
a scoring question. The horizon table shows it is structural: the frozen interval is narrower at
week 50 than at week 1, because nothing in it knows time has passed. If the recording runs long,
cut elsewhere.

*Do not let "calibrated" pass as praise.* 100% coverage against a nominal 94% is a *failure* of
calibration in the conservative direction, and the lab's own framing — honest and too wide to act
on — is the one to keep. This is the lecture where a student is most likely to conclude that
Bayesian methods "worked" because a number came out above 94%.
