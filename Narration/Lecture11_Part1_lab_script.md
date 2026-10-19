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

*(Screen: rendered lab, top of document.)*

Lab 11, Part 1. Last week's model had one parameter and no time in it. This week we point the same
machinery at an actual time series.

A structural model writes a series as **parts you can name** — a level that drifts, a seasonal
shape, and noise — and puts a prior on each. That is the appeal, and it is a real one: the output
is not a forecast line, it is a distribution for every future week, and you can pull the level and
the seasonality out and look at them separately.

**[pause]**

Today has an unusual shape, so let me tell you where it goes. You will build the model and hit a
failure that the diagnostics catch. You will fix it with a change that alters nothing about the
model. Then you will forecast two ways — and find that the **theoretically correct** way produces
a *worse-calibrated* interval than the shortcut.

That last result is not a bug in the lab. It is the model telling you something about itself, and
working out what takes the rest of the session.

---

# ▶ SETUP

*(Screen: run setup.)*

```
train 225 weeks, test 52 weeks | 6 Fourier coefficients standing in for 52
```

CA\_1 FOODS again — the series you have been forecasting since Homework 2 — held out for a full
final year.

*(Point at the Fourier line.)*

One line deserves attention. **Six coefficients standing in for fifty-two.** Three sine-cosine
pairs describe the annual shape, rather than fifty-two weekly dummies. That is a modeling choice
with a cost and a benefit: you cannot represent a one-week spike, but you have six parameters to
estimate instead of fifty-two, on 225 observations. Remember it — it comes back in Step 3 as a
suspect.

Also note the standardization. `y_z` is the series in standard units, because the priors below —
`Exponential(10)`, `Normal(0, 0.5)` — assume unit scale. Priors are not scale-free, and this is
the line that makes them defensible.

---

# ▶ STEP 1 — Fit it the obvious way

*(Screen: run the centered model.)*

The model is a drifting level plus seasonality. PyMC has a distribution for a random walk, so the
obvious thing is to reach for it: `pm.GaussianRandomWalk`. It reads exactly like the maths.

```
R-hat        1.022   (want < 1.01)
ESS (bulk)   274     (want > 400)
divergences  0
```

**[pause]**

**It did not converge**, and I want you to look carefully at *how* you know.

R-hat is 1.022, above the 1.01 threshold — the four chains do not agree with each other.
Effective sample size is 274, below 400 — the draws are so autocorrelated that six thousand
samples carry less information than a few hundred independent ones.

And **divergences are zero.**

*(Beat.)*

That last line matters, because divergences are the diagnostic everyone learns first, and here
they are silent. The sampler never had to give up on a step; it just wandered badly. **Different
pathologies trip different instruments** — which is precisely why you check all three, every time,
rather than the one you remember.

Next week you will see the mirror image: a model where R-hat and ESS both pass and forty-three
divergences are the only thing objecting.

*(Point at `target_accept`.)*

And note this is **not** fixed by asking the sampler to try harder. `target_accept` is already at
0.95. The problem is not effort.

---

# ▶ STEP 2 — Reparameterize

*(Screen: the two model blocks side by side, then run.)*

The problem is the **shape of the posterior**, not the model.

A centered random walk couples the level to `sigma_level`: when the scale is small, the steps must
be small, so the posterior narrows into a funnel that the sampler cannot traverse at a single step
size. The fix is to write the walk as a cumulative sum of **standard** Normal draws, scaled
afterwards. Now `z` and `sigma_level` are independent in the posterior geometry.

```
R-hat        1.002   (want < 1.01)
ESS (bulk)   2376    (want > 400)
divergences  0
```

**[pause]**

R-hat 1.022 to 1.002. Effective sample size 274 to **2,376** — nearly nine times more information
out of the same number of draws.

And here is the sentence to say slowly: **same data, same random walk, new coordinates.**
`cumsum(z * sigma)` and `GaussianRandomWalk(sigma=sigma)` describe the same walk. What changed is
the coordinate system the sampler explores.

That is worth internalizing, because it is unlike everything else in this course. This is not a
modeling improvement or a bias-variance trade. It is a **computational** fix to a **computational**
problem.

*(One honest footnote, and read the callout under the code.)* The two fits are *almost* the same
model, not exactly. `GaussianRandomWalk` lets week one's level start anywhere, because its default
start is a Normal with standard deviation one hundred. PyMC warns about that default, and our Setup
cell silences warnings, so you never saw it. The `cumsum` version starts the walk near zero. That
one difference nudges sigma-level up, from about four hundred fifty units a week to about five
hundred twenty. A default you didn't choose is still a prior.

*(Screen: the components figure.)*

And now the payoff of a structural model — you can look at the parts. Top panel, the level: a slow
drift under the noise. Bottom panel, the seasonal shape the six Fourier coefficients recovered.
Neither of those is available from XGBoost at any price.

---

# ▶ STEP 3 — Forecast, two ways

*(Screen: the two methods, then run the table.)*

To forecast you need the level at each future week, and there are two candidates.

**Held constant.** Carry the last estimated level forward unchanged. Simple — and it asserts that
the level stops drifting the moment your data ends, which is not what the model says.

**Propagated.** Let the level keep random-walking forward exactly as it did in-sample. This is
what the fitted model actually implies. It is the defensible choice.

```
             method  RMSE  coverage_%  mean_width
level held constant  1589          96        6340
   level propagated  1589         100       11667
```

**[pause]** — *let it sit.*

The point forecasts are **identical** — 1,589 on both rows, and not by coincidence. Propagating
the level adds steps whose mean is zero, so the expected level is unchanged and RMSE simply cannot
tell these two apart. Everything separating them is in the **interval**, which differs by a factor
of nearly two, and not in the direction you would predict.

The theoretically correct method **over-covers**: 100% of weeks fell inside a 94% interval. The
shortcut lands at 96%, close to nominal. **The defensible choice calibrated worse.**

*(Screen: the callout.)*

**[STOP — learner works]**

Pause and work out why before I tell you. Then do the first "your turn" — print the two scale
parameters and the drift accumulated over 52 weeks.

*(Resume.)*

Over 52 weeks a random walk accumulates about $\sqrt{52}$ — call it seven — times `sigma_level` of
drift. So if `sigma_level` is even somewhat too large, the interval inflates fast, and it inflates
with the *square root of the horizon*.

**And there is good reason to think it is too large**, which brings back the Fourier choice from
the setup. The random walk is the most flexible component in this model. Anything the six
coefficients cannot represent — a promotion, a holiday that moves, a single bad week — has nowhere
else to go, so it gets absorbed into the level. That inflates `sigma_level` beyond genuine drift,
and the forecast then propagates that inflated drift for a year.

**[pause]**

So — and this is the sentence that matters — the frozen version is **not right**. It is wrong in a
way that happens to **cancel** an overestimated drift, on this series, at this horizon. Rely on
that and it will fail somewhere else, silently, and you will not know which of the two errors
moved.

*(Run the by-horizon table.)*

```
coverage / mean width, by horizon (nominal 94%)
             method      wk 1-13      wk 14-39      wk 40-52
level held constant 100% / 6,255   92% / 6,537  100% / 6,027
   level propagated 100% / 8,081 100% / 11,974 100% / 14,637
```

**[pause]** — *this table is the real lesson of the step.*

Read the **widths** across each row, not the coverage.

The frozen version: 6,255 at weeks 1–13, and 6,027 at weeks 40–52. It is *narrower* a year out
than a month out. It has no mechanism to widen, because nothing in it knows that time has passed.

The propagated version: 8,081, then 11,974, then 14,637. It nearly doubles.

**One of these models knows that forecasting further ahead is harder, and the other does not.**
That is a structural difference, not a scoring difference, and it is why the aggregate table on
its own would have misled you. On this series, at this horizon, the ignorance happened to cost
nothing. On a series with a genuinely drifting level it would cost a great deal.

*(One more, if you have time.)*

And notice the only sub-nominal cell in the table: the frozen version drops to **92%** in weeks
14–39. Its respectable 96% average is hiding a stretch where it is overconfident. Aggregates hide
things; that is the third time this course has said so.

---

# ▶ STEP 4 — Calibrated is not the same as useful

*(Screen: run the summary block.)*

Take the propagated version — the one the model implies — and ask it two **separate** questions.

```
coverage      100%   against a nominal 94%
mean width    11,667 units
              = 59% of average weekly units
RMSE          1,589
seasonal naive  1,660
```

**Question one: is it calibrated?** Yes, and then some. 100% against a nominal 94% — not
overconfident. Conservative.

**Question two: is it useful?**

**[pause]**

The interval is **11,667 units wide on average, on a series averaging about 19,700** — call it plus
or minus thirty percent, *on average*: about twenty percent in the first quarter, thirty-seven in the
last. Now say that out loud as a business sentence: *"I am 94% confident that demand six months
from now is somewhere between roughly thirty percent below and thirty percent above average."*

Nobody places an inventory order against that. It is honest and it is useless, and those are not
the same axis.

**[STOP — learner works]**

Write your one-sentence answer to each question, and then the third question the lab actually
asks: does passing the first excuse failing the second?

*(Resume.)*

And the point forecast: 1,589 against seasonal naive's 1,660. **About 4% better**, for a model
that took twenty seconds to sample, a reparameterization to converge, and a page of code.

*(Beat.)*

If accuracy were the argument for this model, that number would be an embarrassment. Accuracy is
not the argument. What you bought is a level and a seasonal component you can plot and defend, a
full distribution for every future week, and — the thing the whole lab has been about — an
interval that **admits when it does not know**. Whether that trade is worth 4% is a business
question, and it depends entirely on whether anyone downstream will use the uncertainty you went
to the trouble of producing.

---

# ▶ BEFORE YOU LEAVE

**[STOP — learner works]** — *(replaces the in-room "Discuss")*

Three questions for the board.

**The two problems.** The convergence failure was caught by a **diagnostic** — R-hat and ESS,
computed from the fit alone, before you looked at any data you were predicting. The over-wide
interval was caught by nothing of the sort: it needed a **held-out year and a coverage
calculation**. What is the structural difference between those two kinds of error, and what habit
catches the second kind?

**[pause]** — *the difference worth landing: a diagnostic asks "did the computation work?", a
held-out check asks "is the model any good?" — and no amount of the first substitutes for the
second. A perfectly converged sampler will happily give you a useless interval.*

**The payoff.** The point forecast beat seasonal naive by about four percent. Given that, what did
fitting this model actually buy? Name one business question it answers that XGBoost from Homework
4 cannot.

**The fix.** The interval is honest and too wide to act on. Propose one change to the **model** —
not the forecasting code — that would narrow it, and say what you are assuming in exchange.

**[pause]** — *the two good answers are both in this lab: a tighter prior on `sigma_level`, which
assumes the level really does drift slowly; or more seasonal flexibility so the Fourier terms stop
dumping their residual into the level. The second is the more interesting answer because it
addresses the cause rather than the symptom.*

Solutions for the coded parts go up on Canvas after the deadline. Next week: ten series instead of
one, and what happens when some of them barely have any data.

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
