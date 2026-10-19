# Lecture 11 Part 1 — Recording Script

**ECON 8310: Business Forecasting · Bayesian Statistics II — Time Series**

Deck: `Slides/Lecture11_Part1_BayesianTS.pdf` (20 pages) · Measured runtime: see the timing guide

---

## How to use this document

- **`▶ SLIDE n — Title`** marks where to advance. The number is the PDF page.
- *Italic parentheticals* are stage directions. **[pause]** means stop for a beat.
- Slide 4 sets expectations low on purpose and slides 16–17 collect on it. The argument of this
  lecture is that a **calibrated but useless** interval is still a better deliverable than a point
  forecast, because you can *establish* that it's useless. Do not soften that.
- Slides 12–13 are the first of two sampler failures in this pair of lectures. Treat them as a
  rehearsal for Part 2.

---

# ▶ SLIDE 1 — Title page

Lecture 11, Part 1: Bayesian Structural Time Series.

Last week we built the machinery — Bayes' theorem, priors, MCMC, and the three diagnostics. All of
it on a conversion rate, which is not a time series.

Today we point that machinery at the thing this course is actually about.

**[pause]**

And I'll tell you the ending now, because it's the interesting part. The model we build today will
beat the seasonal-naive benchmark by about four percent. That is a small win, and after ten weeks
of this course you should be suspicious of anyone who presents four percent as a triumph.

The argument for today is not the point forecast. It's what else comes back with it.

---

# ▶ SLIDE 2 — Lecture Outline

Five parts. Why you'd do this at all, given that four percent. Then the structural decomposition —
writing a series as parts you can name. Then fitting it in PyMC, where our first attempt fails.
Then reading the forecast, and the two very different questions you can ask about an interval. And
takeaways.

---

# ▶ SLIDE 3 — Section divider: Why Bayesian Time Series

Not a more accurate forecast — a different deliverable.

---

# ▶ SLIDE 4 — What Every Model So Far Gave You

Take stock for a second. Ten weeks of forecasting, and almost every model has returned the same
object: **one number per period.**

ETS, ARIMA, random forests, XGBoost, the LSTM — a point forecast. And at best, an interval derived
from asymptotic theory that, let's be honest, nobody checked.

**[pause]**

For a planner, that is often not enough. Consider the question: *how much stock do I need so the
chance of a stockout is under five percent?*

That is a question about a **distribution**. A point forecast cannot answer it — not because the
point forecast is inaccurate, but because it isn't the right kind of object. You can't extract a
fifth percentile from a single number.

*(Point at the key box.)*

A Bayesian time series model returns a full predictive distribution for every future period, plus
a distribution over each *component* — the trend, the seasonality, the noise, each with its own
uncertainty.

**[pause]**

*(Now set expectations honestly. This paragraph is load-bearing.)*

And let me set expectations properly, right now.

This will not beat XGBoost on RMSE. On our data it barely beats seasonal naive. The argument for
it is what it hands you, not where it ranks.

I'm saying that at the start because at the end of this lecture I'm going to hold it to exactly
that standard — and it's going to partially fail. That's a more useful hour than one where
everything works.

---

# ▶ SLIDE 5 — Three Things That Become Possible

Three capabilities you didn't have before.

*(Walk the table.)*

**Priors carry knowledge.** "Weekly growth is very unlikely to exceed twenty percent" is a sentence
you can put *into the model*, rather than hoping the data implies it. When you have short history,
that's the difference between a usable forecast and a wild one.

**Components get error bars.** Not just a forecast, but "how confident am I that there's a trend at
all?" — separately from the seasonality. You can decompose your uncertainty, not just your signal.

**Uncertainty compounds honestly.** Forecast fifty-two weeks out and the interval widens, because
the model propagates uncertainty forward rather than assuming it away.

**[pause]**

*(Point at the third row again — it's the one to dwell on.)*

That third row is the one people underestimate, so let me sharpen it, and be fair to ARIMA while
I do.

An ARIMA prediction interval widens too, and it also follows from a model you wrote down. Lecture 2
showed you how to check that model's residuals. The real difference is narrower: ARIMA's formula
plugs in **point estimates** of its parameters, as if they were known exactly. This model carries
their uncertainty into the band as well.

And each source of widening is a *named part of the model*. If you said the level follows a random
walk, then fifty-two steps of random walk uncertainty is what you get. You can trace the interval
back to a modeling choice and argue about that choice.

**[pause]**

That traceability is worth more than it sounds. When a planner tells you the interval is too wide
to use — which is exactly what will happen at the end of this lecture — you can point at the
specific component producing it and propose changing *that*.

*(Point at the price.)*

And the price is real: it's slower, it needs priors you must defend, and — as we'll see in twenty
minutes — it can fail silently in ways a least-squares fit simply cannot.

---

# ▶ SLIDE 6 — Section divider: The Structural Decomposition

Write the series as a sum of parts you can name, and put a prior on each.

---

# ▶ SLIDE 7 — Structural Time Series: One Equation

The whole idea is additive, and you have seen this shape before.

*(Point at the equation.)*

y-t equals mu-t, the level, plus s-t, the seasonality, plus epsilon-t, the noise.

That is the GAM decomposition from Lecture 3, with priors attached. Same additive structure, same
named components — now each one is a random variable with a distribution.

**[pause]**

*(Point at the figure, top row first.)*

Here is that equation, fitted, on CA_1 FOODS: the training weeks, February 2011 to May 2015. The
top row is what we observed. Below it are the three pieces the model split it into: the level in
blue, the seasonal shape in green, and what is left over, the residual, in gray. Add the bottom
three rows week by week and you get the top row exactly.

And every row is drawn to the same scale. That little bar is five thousand units in every panel. So
the heights tell you how much each piece contributes.

**[pause]**

Read them. The level does most of the work. It sinks to about fourteen thousand in the summer of
2011, climbs to around twenty-one thousand by 2014, and roughly flattens. The seasonal piece is
small: about a thousand units below the level in February, fifteen hundred above it in August. And
the residual is bigger than the seasonal piece, with swings of up to three thousand units either
way.

*(Point at the gray row.)*

Look at the residual closely and it is not shapeless. It rises and falls about once a month: about
half its variance sits at a period of four and a third weeks, which is a calendar month. The model
has no term for a monthly rhythm, so it lands in the noise. A likely suspect in California is SNAP,
whose benefits arrive in the first ten days of each month: the flags you used in Homework 2. It is
a covariate this model was never given.

*(One honest note on the blue line.)* Its first few weeks, falling steeply from about nineteen
thousand, partly reflect where this version of the model makes the walk begin. We'll come back to
that on slide 13.

**[pause]**

And here's why the split matters practically. **Each piece is a thing a business person can
name**, and each gets its own posterior.

"Demand is growing" is a statement about mu-t. "August is our peak" is a statement about s-t. "How
noisy is this store?" is sigma. Those are three separate questions, and the model answers them
separately, with uncertainty on each.

That's why the approach is called **structural**. You are not fitting a flexible curve to data and
hoping. You are stating what you believe the series is *made of*, and asking the data how much of
each part there is.

**[pause]**

Contrast ARIMA, which we spent Lecture 2 on. There, trend and seasonality are handled by
*differencing* — you remove them rather than model them — and the parameters that come out are
autocorrelation coefficients. Nobody has ever walked into a planning meeting and explained the
business implications of an MA(2) coefficient.

Here, every parameter is a quantity somebody outside the room can interpret.

**[pause]**

There's a real trade being made, though, and I don't want to sell only one side.

ARIMA's differencing is *automatic*. You don't have to know whether the trend is linear or
wandering or damped — you difference, and the trend is gone, whatever shape it had.

Structural modeling makes you **commit**. You have to say what kind of level you believe in
before you fit. Commit correctly and you get interpretable components and honest uncertainty.
Commit wrongly and you have baked a false assumption into the model, and the posterior will look
perfectly confident about it.

That's the same bargain as the DAG in Lecture 12: stating your assumptions gets you more, and
makes you responsible for them.

---

# ▶ SLIDE 8 — The Level: A Random Walk with a Prior on It

The simplest useful trend is a **local level**: the series has a level, and the level itself
drifts over time.

mu-t equals mu-t-minus-one plus a shock. Read as a sentence: **this week's level is last week's
level, plus a random disturbance.**

And that disturbance has a standard deviation, sigma-level, which you put a prior on. That single
parameter controls **how fast the level is allowed to move**.

*(Walk the two rows.)*

Small sigma-level: a nearly constant level. Rigid, but stable far into the future.

Large sigma-level: a level that chases recent data. Fits the training period beautifully, and the
forecasts wander off.

**[pause]**

*(Point at the callback.)*

And that is a familiar trade in new clothing.

It is the smoothing parameter from Lecture 1's exponential smoothing — alpha, deciding how fast to
forget. It is the penalty strength from Lecture 6. Now it's expressed as a **prior you can argue
about**, and — importantly — the data gets a say in it too, because sigma-level is estimated, not
fixed.

That's the sixth appearance of this idea coming into view. We'll name it properly next week.

---

# ▶ SLIDE 9 — Seasonality as Fourier Terms

Now seasonality. Our data is weekly with an annual cycle, so a naive approach would estimate
fifty-two separate seasonal parameters. That's a lot to estimate, and most of it would be noise.

Instead, build the seasonal shape out of a few smooth waves.

*(Point at the equation.)*

s-t is a sum over k of sine and cosine terms at increasing frequencies. P is the period —
fifty-two for us. K is how many waves you allow. K equals one is a single smooth annual cycle;
larger K lets the shape get more detailed, with sharper peaks.

And each a-k and b-k is just a coefficient with a Normal prior. Nothing exotic.

*(Point at the left panel.)*

Here is what our fit found, on CA_1 FOODS. The gray dots are the training weeks with the level
taken out, stacked by where they fall in the year. The blue wave is k equals one, once a year. The
green and the orange are k equals two and three, twice and three times a year. The red curve is
their sum — that is the seasonal shape the model uses. It peaks in mid-August, about fifteen
hundred units above the level, and bottoms out in mid-February.

**[pause]**

*(Point at the three dials.)*

Now the geometry, because it turns two numbers into one picture. A sine plus a cosine at the same
frequency is just one wave, shifted: a sine x plus b cosine x equals r sine of x plus phi. So draw
each pair, a-k and b-k, as an arrow from the origin. **The arrow's length is the wave's size. Its
angle is the wave's timing** — turn the arrow and the peak slides along the year.

The yearly arrow is about eleven hundred sixty units long. The other two are about two hundred
twenty and one hundred ninety. Each dial is drawn to its own scale, so read the lengths, not the
circles.

And here's what the dials don't show: the posterior for those two short arrows is about as wide as
the arrows themselves. The data pins down the yearly wave. The faster two are barely there.

**[pause]**

*(Point at the muted line — this is the third time they've seen this device.)*

You have met this exact construction **twice already**.

It is how Prophet encodes seasonality, in Lecture 3. And it is the Fourier feature block you
engineered by hand for Homework 4 — those `sin_1`, `cos_1`, `sin_2` columns.

Same device, third appearance. The difference now is that a-k and b-k come back as
**distributions**, so you can ask how confident you are that the seasonal peak is where you think
it is.

*(Point at the last line.)*

K trades flexibility against overfitting, exactly like spline knots in Lecture 3. We use K equals
three — six coefficients standing in for fifty-two parameters.

**[pause]**

Notice what that construction *assumes*, because it's an assumption and not a free lunch: it
assumes the seasonal shape is **smooth**.

Three sine-cosine pairs can describe a broad annual hump with a couple of gentle bumps. What they
cannot describe is a single enormous one-week spike — Black Friday, say — sitting on an otherwise
flat year. Fourier terms will smear that spike across neighboring weeks, because smooth curves
are all they can make.

If your business has sharp calendar events, the honest answer is to model them as **separate
indicator variables** rather than raising K until the waves can approximate a spike. That's what
Prophet does with its holiday regressors, and it's what you did by hand with the SNAP and event
flags in Homework 2.

---

# ▶ SLIDE 10 — Section divider: Fitting It in PyMC

The model in code, and the check you run before believing any of it.

---

# ▶ SLIDE 11 — The Model in PyMC

Here is everything from the last three slides, in about eight lines.

*(Walk the code.)*

`sigma_level` with an Exponential prior — it must be positive, so Exponential, not Normal. Then
`z`, a vector of standard Normals. Then the level is the **cumulative sum** of z times
sigma_level.

Stop on that line for a second, because it's the clever bit: `cumsum` of scaled Normal draws **is**
a random walk. That's the definition. Each step adds a scaled Normal shock to the running total.

Then `beta`, the Fourier coefficients. Then sigma, the observation noise. Then the likelihood: the
observed data is Normal around level plus the Fourier terms.

**[pause]**

*(This warning is practical and easy to get wrong.)*

One thing you must do, and it's bolded for a reason. **Standardize y first.**

These priors assume the data is on roughly unit scale. Our weekly units are in the tens of
thousands. An Exponential prior with rate ten on `sigma_level` is a sensible prior for a
standardized series and an absurdly tight one for raw units of twenty thousand — you'd be telling
the model the level essentially cannot move.

Standardize, fit, then transform the forecasts back. Getting this wrong produces a model that
fits terribly for reasons that look mysterious.

**[pause]**

This is the same lesson as Lecture 6's warning about standardizing before Ridge, and it will come
back one more time in Lecture 12 with the opposite sign — where standardizing a variable you then
*interpret* silently destroys the interpretation.

So the rule isn't "always standardize" or "never." It's: **know what scale each variable is on,
and know what your priors assume about that scale.** A prior is a statement about magnitudes. If
you don't know the magnitudes, you don't know what you've assumed.

---

# ▶ SLIDE 12 — The First Fit Failed — Here Is What That Looks Like

*(This is the honest-failure slide. Deliver it as a story, not a footnote.)*

Now something worth your attention.

The obvious way to write this model uses `pm.GaussianRandomWalk`, which is exactly what it sounds
like and exactly what the equation says. We wrote it that way first. And it sampled **badly**,
even with `target_accept` raised to point-nine-five.

*(Point at the top row of the figure.)*

This is sigma-level, the one parameter that misbehaved: four chains, one color each, the first
three hundred draws of fifteen hundred. Look at the texture. Each chain drifts: up for a stretch,
down for a stretch. A draw is mostly a copy of the one before it.

The numbers say the same thing, in red. R-hat: one-point-oh-two-two, against a threshold of
one-point-oh-one. Failed. ESS: two hundred seventy-four, against four hundred. Failed. Divergences:
zero. That one passed.

*(Point at the bottom row.)*

Now the bottom row, after the fix. Same parameter, same number of draws. This is what healthy
chains look like: a fuzzy band, each draw nearly independent of the last. R-hat one-point-oh-oh-two,
ESS two thousand three hundred seventy-six.

**[pause]**

Put a number on the difference. Six thousand draws in the top row carry about as much information
as two hundred seventy-four independent ones. The bottom row's six thousand are worth about two
thousand four hundred.

*(Now the important part.)*

Here is what I want you to take from this.

**Nothing crashed.** The model returned a forecast. The forecast, plotted, looked entirely
reasonable: a sensible level, a sensible seasonal shape, plausible intervals. If we had not run
the diagnostics, we would have shipped it and never known.

That is precisely why Lecture 10 made those three numbers non-negotiable. This is the failure mode
they exist for.

*(Point at the muted line.)* And yes, the two rows sit at different heights. Hold that thought for
one slide.

---

# ▶ SLIDE 13 — The Fix: Reparameterize, Do Not Sample More

*(Walk the table, left column, then right.)*

The fix was **reparameterization**, not more samples. Not a longer run, not a smaller step size:
a different way of writing the walk.

On the left, the centered form. The sampler works directly on all two hundred twenty-five levels,
and their plausible step size is sigma-level. So every time the sampler wants to try a different
sigma-level, all two hundred twenty-five levels have to move with it. It can't, in one step, so it
crawls. That's the drift you saw in the top row.

On the right, the non-centered form. The sampler works on z, independent standard Normals, and the
level is built from them afterwards: the cumulative sum of z times sigma-level. Now changing
sigma-level doesn't require moving anything else. The geometry is round, and the sampler can
navigate it.

*(Point at the warning box.)*

Same data, same six thousand draws. ESS went from two hundred seventy-four to two thousand three
hundred seventy-six. It is the standard fix: Martin, Kumar and Lao work it in section
four-point-six-point-one of their book. Next week we'll see the same fix applied to a hierarchy,
where it will matter even more.

**[pause]**

*(Point at the muted line. This is the answer to the different heights. Don't skip it.)*

Now the honest footnote, because it's a lesson in its own right.

The two versions are *almost* the same model, but not quite. `GaussianRandomWalk` needs to know
where the walk starts, and if you don't tell it, it lets week one's level start anywhere: a
Normal with standard deviation one hundred, on a standardized series. PyMC prints a warning saying
so. The lab switches warnings off, so you never see it. The cumsum version starts the walk at z
times sigma-level, which is close to zero, the series average.

That one difference moves the answer. With the start pinned near the average, the walk has to fall
fast in the first weeks, which you saw on the decomposition slide, and sigma-level comes out higher:
about five hundred twenty units a week, against about four hundred fifty. That's the gap between
the two rows. It is a slightly different model, not a sampling failure.

So two lessons, not one. Change the coordinates when the sampler crawls. And read the warnings you
were about to silence: a default you didn't choose is still a prior.

---

# ▶ SLIDE 14 — Section divider: Reading the Forecast

The point estimate, the interval, and whether either is any use.

---

# ▶ SLIDE 15 — Does It Work? The Point Forecast

CA_1 FOODS, fifty-two weeks held out.

*(Point at the table.)*

Bayesian structural model: RMSE one thousand five hundred eighty-nine. Seasonal naive — same week
last year, one line of code: one thousand six hundred sixty.

So we beat the benchmark by about **four percent**.

*(Point at the parenthesis.)* And don't set this benchmark beside the two thousand one hundred
fifty-two from Lectures 8 and 9. That one was seasonal naive across the whole thirty-series panel.
This is one series, CA_1 FOODS, the same benchmark as Homework 1.

**[pause]**

That is a real improvement and a small one. And I'll repeat what I said at the start: after ten
weeks of this course, you should be suspicious of anyone who reports four percent as a triumph.

*(Now the fair-hearing paragraph. Give it properly.)*

But let me also be fair to the model, because the comparison isn't like-for-like.

Look at what this model was *given*: the series, and a calendar. That's it. No lag features. No
price. No SNAP flags. No store identity. No cross-series information.

The XGBoost model in Homework 4 had forty-six engineered predictors and thirty series to learn
from, and beat naive by a much wider margin. It also had a great deal more information.

**[pause]**

And there's nothing stopping you from combining them, which is what a serious production system
usually does. Add price and promotion regressors to this structural model and the level has less
work to do, so it drifts less, so the interval narrows. You keep the predictive distribution and
you buy back some accuracy.

We're not doing that today because I want the mechanism visible rather than buried under
covariates. But it is the obvious next step, and it's a very reasonable direction for a final
project.

**[pause]**

So: the point forecast is not the reason to do this. Which brings us to the slide that is.

---

# ▶ SLIDE 16 — The Interval Is the Product

Every forecast comes with a ninety-four percent predictive interval. And there are **two**
questions you can ask about it, which are different questions, and this is the heart of the
lecture.

*(Walk the figure, left to right.)*

On the left, the last sixty training weeks: the gray dots are actual sales, the thin blue line is
the fit, and the pale band around it is the in-sample interval. At the dashed line we stop
looking. To the right are the fifty-two weeks we held out: black dots for what actually happened,
the blue line for the forecast, and the shaded band for the ninety-four percent interval.

**[pause]**

**Is it calibrated?** Count the black dots outside the band. There are none. Fifty-two of fifty-two
fell inside. So it is not overconfident. But it promised ninety-four percent, and ninety-four
percent of fifty-two would leave about three weeks outside, so it's actually *conservative* — a
bit too wide.

**Is it useful?** Look at how wide the band is. On average it's about eleven thousand seven
hundred units wide, on an average week of nineteen thousand seven hundred. That's plus or minus
thirty percent — **on average**.

*(Point at the two red arrows.)*

And "on average" is doing work in that sentence. One week out, the band is plus or minus sixteen
percent. A year out, it's plus or minus thirty-nine. It fans out as the forecast gets further
from the data. Honest, at every horizon. And too wide to order against at almost all of them.

---

# ▶ SLIDE 17 — Honest, and Too Wide to Order Against

*(This is the sentence of the lecture.)*

**Passing the first does not excuse failing the second.**

A model that says "demand will be somewhere between fourteen thousand and twenty-six thousand" is
telling you the truth and helping nobody. You cannot write a purchase order against that.

**[pause]**

And the cause is structural, not a bug.

*(Point at the formula.)*

Take the parameters as given, and today's level as given. Then the forecast h weeks out has
variance sigma-squared plus h times sigma-level-squared. The noise, sigma-squared, is the same
every week. The level's random-walk steps **add up** — one step of drift for every week you look
ahead.

*(Walk the figure.)*

The dotted line is the noise alone. The dashed curve is that formula, at the fitted values of the
parameters. Early on the noise dominates. After about seven weeks the drift term overtakes it,
and from there the band grows like the square root of h. The blue line is the band we actually
got. It sits a little above the dashed curve, because the real band also carries our uncertainty
about the parameters and about today's level.

So a level that is free to random-walk for fifty-two steps *drifts*, and the model is correctly
reporting the consequence of what we told it to believe.

The fix is a model with a **stronger opinion**: a damped level that reverts rather than
wandering, or covariates that pin it down — price, promotions, the things we withheld.

**[pause]**

*(And now the payoff. Slow down.)*

But notice what just happened, because this is the argument for the entire lecture.

**We could ask whether the uncertainty was honest.** We computed coverage against held-out data,
and got an answer.

For the models before this one, we never asked. ETS had intervals, and Prophet drew a band in
Homework 2, but we scored every model on its point forecast and never checked a band against
held-out weeks. **Here we did.**

A calibrated-but-too-wide interval is a worse forecast and a better *deliverable*, because it tells
you something true about how much you actually know. That is worth paying four percent for.

---

# ▶ SLIDE 18 — Section divider: Key Takeaways

What structural models are for.

---

# ▶ SLIDE 19 — Lecture 11, Part 1: Key Takeaways

One. A structural model writes the series as named parts — level, seasonality, noise — and gives
each one a prior and a posterior.

Two. The local level is a random walk whose step size you put a prior on. Small means rigid, large
means it chases the data. Same trade as a smoothing parameter or a penalty.

Three. Fourier terms build seasonality from a few waves instead of fifty-two parameters, and each
pair of coefficients is an arrow: its length is the wave's size, its angle the wave's timing. Same
device as Prophet in Lecture 3 and the features you built in Homework 4 — third appearance.

Four. **Check the sampler before the forecast.** Our first fit had R-hat one-point-oh-two and ESS
two seventy-four, while returning a perfectly plausible-looking answer. Reparameterizing fixed it.

Five. The point forecast beat seasonal naive by four percent. That is not the argument.

And six. The interval was **calibrated but too wide to act on** — plus or minus thirty percent on
average, and widening with the horizon — and being able to *establish* that is the actual product.

**[pause]**

Next time: hierarchical models. What to do when you have ten series and some of them barely
have any data — and a question you have to answer *before* you fit anything.

---

# ▶ SLIDE 20 — References

*(Advance and close. No narration needed.)*
