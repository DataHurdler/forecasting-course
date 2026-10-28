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

This is Lecture 11, Part 1, and today is about Bayesian structural time series.

Last week, in Lecture 10, we built the machinery: Bayes' theorem, priors, MCMC, and the three
diagnostics. We did all of it on a conversion rate, which isn't a time series.

Today we point that machinery at the thing this course is actually about.

**[pause]**

And I'll tell you the ending now, because it's the interesting part. The model we build today will
beat the seasonal-naive benchmark by about four percent. That's a small win, and after eleven weeks
of this course, you should be suspicious of anyone who presents four percent as a triumph.

So the argument for today isn't the point forecast. It's what else comes back with it.

---

# ▶ SLIDE 2 — Lecture Outline

We'll take this in five parts. First, why you'd do this at all, given that four percent. Second, the
structural decomposition, which means writing a series as parts you can name. Third, fitting it in
PyMC, where our first attempt fails. Fourth, reading the forecast, and the two very different
questions you can ask about an interval. And we'll finish with the takeaways.

---

# ▶ SLIDE 3 — Section divider: Why Bayesian Time Series

This first section is about why you'd fit a Bayesian time series at all. The answer isn't a more
accurate forecast. It's a different deliverable.

---

# ▶ SLIDE 4 — What Every Model So Far Gave You

Let's take stock for a second. Over ten weeks of forecasting, almost every model has returned the
same kind of object: **one number per period**.

ETS, ARIMA, random forests, XGBoost and the LSTM all hand you a point forecast. At best, you also
got an interval from the model's own error formula that, let's be honest, nobody checked.

**[pause]**

For a planner, that's often not enough. Consider this question: how much stock do I need so that
the chance of a stockout is under five percent?

That's a question about a **distribution**. A point forecast can't answer it, and that isn't because
the point forecast is inaccurate. It's because it isn't the right kind of object. The stock level
you need is the ninety-fifth percentile of demand, and you can't pull a ninety-fifth percentile out
of a single number.

*(Point at the key box.)*

A Bayesian time series model returns a full predictive distribution for every future period. It
also returns a distribution over each *component*, so the trend, the seasonality and the noise each
come with their own uncertainty.

**[pause]**

*(Now set expectations honestly. This paragraph is load-bearing.)*

And let me set expectations properly, right now.

This model won't beat XGBoost on RMSE. On our data, it barely beats seasonal naive. The argument for
it is what it hands you, not where it ranks.

I'm saying that at the start because at the end of this lecture, I'm going to hold it to exactly
that standard, and it's going to partially fail. That makes for a more useful hour than one where
everything works.

---

# ▶ SLIDE 5 — Three Things That Become Possible

There are three things you can do now that you couldn't do before.

*(Walk the table.)*

The first is that priors carry knowledge. "Weekly growth is very unlikely to exceed twenty percent"
is a sentence you can put *into the model*, rather than hoping the data implies it. When you have a
short history, that's the difference between a usable forecast and a wild one.

The second is that components get error bars. You don't just get a forecast. You can ask how
confident you are that there's a trend at all, separately from the seasonality. So you can
decompose your uncertainty, not just your signal.

The third is that uncertainty compounds honestly. Forecast fifty-two weeks out, and the interval
widens, because the model carries its uncertainty forward rather than assuming it away.

**[pause]**

*(Point at the third row again. It's the one to dwell on.)*

That third row is the one people underestimate, so let me sharpen it, and be fair to ARIMA while I
do.

An ARIMA prediction interval widens too, and it also follows from a model you wrote down. Lecture 2,
on ARIMA, showed you how to check that model's residuals. The real difference is narrower than that.
ARIMA's formula plugs in **point estimates** of its parameters, as if they were known exactly. This
model carries their uncertainty into the band as well.

And each source of widening is a *named part of the model*. If you said the level follows a random
walk, then fifty-two steps of random-walk uncertainty is what you get. You can trace the interval
back to a modeling choice, and argue about that choice.

**[pause]**

That traceability is worth more than it sounds. When a planner tells you the interval is too wide to
use, which is exactly what will happen at the end of this lecture, you can point at the specific
component that's producing it and propose changing *that*.

*(Point at the price.)*

And the price is real. The model is slower, it needs priors you have to defend, and, as we'll see in
twenty minutes, it can fail silently in ways a least-squares fit simply can't.

---

# ▶ SLIDE 6 — Section divider: The Structural Decomposition

The next section writes the series as a sum of parts you can name, and puts a prior on each one.

---

# ▶ SLIDE 7 — Structural Time Series: One Equation

The whole idea is additive, and you've seen this shape before.

*(Point at the equation.)*

It says that y-t equals mu-t, the level, plus s-t, the seasonality, plus epsilon-t, the noise.

That's the GAM decomposition from Lecture 3, with priors attached. It has the same additive
structure and the same named components, but now each one is a random variable with a distribution.

*(Point at the figure, top row first.)*

Here's that equation, fitted, on CA_1 FOODS, over the training weeks from February twenty eleven to
May twenty fifteen. The top row is what we observed. Below it are the three pieces the model split
it into: the level in blue, the seasonal shape in green, and what's left over, the residual, in
gray. Add the bottom three rows together, week by week, and you get the top row exactly.

Every row is drawn to the same scale. That little bar is five thousand units in every panel, so the
heights tell you how much each piece contributes.

**[pause]**

Now, read them. The level does most of the work. It sinks to about fourteen thousand in the summer
of twenty eleven, climbs to around twenty-one thousand by twenty fourteen, and then roughly
flattens. The seasonal piece is small. It sits about a thousand units below the level in February,
and about fifteen hundred above it in August. And the residual is bigger than the seasonal piece,
with swings of up to three thousand units either way.

*(Point at the gray row.)*

Look at the residual closely, and you'll see it isn't shapeless. It rises and falls about once a
month. In fact, about half its variance sits at a period of four and a third weeks, which is a
calendar month. The model has no term for a monthly rhythm, so that rhythm lands in the noise. A
likely suspect in California is SNAP, whose benefits arrive in the first ten days of each month.
Those are the flags you used in Homework 2, the GAMs assignment, and SNAP is a covariate this model
was never given.

*(Point at the start of the blue line.)*

There's one honest note to make about the blue line. Its first few weeks, falling steeply from about
nineteen thousand, partly reflect where this version of the model makes the walk begin. We'll come
back to that on slide thirteen.

And here's why the split matters in practice. **Each piece is something a business person can
name**, and each one gets its own posterior.

"Demand is growing" is a statement about mu-t. "August is our peak" is a statement about s-t. "How
noisy is this store?" is a question about sigma. Those are three separate questions, and the model
answers them separately, with uncertainty on each.

That's why the approach is called **structural**. You aren't fitting a flexible curve to the data
and hoping. You're stating what you believe the series is *made of*, and asking the data how much of
each part there is.

**[pause]**

Contrast that with ARIMA, which we spent Lecture 2 on. There, trend and seasonality are handled by
*differencing*. You remove them rather than model them, and the parameters that come out are
autocorrelation coefficients. Nobody has ever walked into a planning meeting and explained the
business implications of an MA-two coefficient.

Here, every parameter is a quantity that somebody outside the room can interpret.

There's a real trade being made, though, and I don't want to sell you only one side of it.

ARIMA's differencing is *automatic*. You don't have to know whether the trend is linear or wandering
or damped. You difference, and the trend is gone, whatever shape it had.

Structural modeling makes you **commit**. You have to say what kind of level you believe in before
you fit. Commit correctly, and you get interpretable components and honest uncertainty. Commit
wrongly, and you've baked a false assumption into the model, and the posterior will look perfectly
confident about it.

It's the same bargain you'll meet with the DAG in Lecture 12, on Bayesian regression, two weeks from
now. Stating your assumptions gets you more, and it makes you responsible for them.

---

# ▶ SLIDE 8 — The Level: A Random Walk with a Prior on It

The simplest useful trend is a **local level**. The series has a level, and the level itself drifts
over time.

The equation says mu-t equals mu-t-minus-one, plus a shock. Read as a sentence, it says that this
week's level is last week's level, plus a random disturbance.

That disturbance has a standard deviation, sigma-level, and you put a prior on it. That single
parameter controls **how fast the level is allowed to move**.

*(Walk the two rows.)*

A small sigma-level gives you a nearly constant level. It's rigid, but it stays stable far into the
future.

A large sigma-level gives you a level that chases recent data. It fits the training period
beautifully, and then the forecasts wander off.

**[pause]**

*(Point at the callback.)*

That's a familiar trade in new clothing.

It's the smoothing parameter from the exponential smoothing in Lecture 1, alpha, which decided how
fast to forget. It's also the penalty strength from Lecture 6, on regularization. Now it's expressed
as a **prior you can argue about**, and, just as important, the data gets a say in it too, because
sigma-level is estimated, not fixed.

So you're watching the sixth appearance of this idea come into view, and next week, in Lecture 11,
Part 2, we'll name it properly.

---

# ▶ SLIDE 9 — Seasonality as Fourier Terms

Now for seasonality. Our data is weekly with an annual cycle, so a naive approach would estimate
fifty-two separate seasonal parameters. That's a lot to estimate, and most of it would be noise.

Instead, we build the seasonal shape out of a few smooth waves.

*(Point at the equation.)*

The equation says that s-t is a sum over k of sine and cosine terms at increasing frequencies. P is
the period, which is fifty-two for us, and K is how many waves you allow. K equals one gives a single
smooth annual cycle, and a larger K lets the shape get more detailed, with sharper peaks.

Each a-k and b-k is just a coefficient with a Normal prior. There's nothing exotic here.

*(Point at the left panel.)*

Here's what our fit found on CA_1 FOODS. The gray dots are the training weeks with the level taken
out, stacked by where they fall in the year. The blue wave is k equals one, which repeats once a
year. The green and the orange are k equals two and three, which repeat twice and three times a
year. The red curve is their sum, and that's the seasonal shape the model uses. It peaks in
mid-August, about fifteen hundred units above the level, and it bottoms out in mid-February.

**[pause]**

*(Point at the three dials.)*

Now for the geometry, because it turns two numbers into one picture. A sine plus a cosine at the
same frequency is just one wave, shifted. In symbols, a sine x plus b cosine x equals r sine of x
plus phi. So you can draw each pair, a-k and b-k, as an arrow from the origin. **The arrow's length
is the wave's size, and its angle is the wave's timing.** Turn the arrow, and the peak slides along
the year.

The yearly arrow is about eleven hundred sixty units long. The other two are about two hundred twenty
and one hundred ninety. Each dial is drawn to its own scale, so read the lengths, not the circles.

And here's what the dials don't show. The posterior for those two short arrows is about as wide as
the arrows themselves. The data pins down the yearly wave, and the faster two are barely there.

*(Point at the muted line. This is the third time they've seen this device.)*

You've met this exact construction **twice already**.

It's how Prophet encodes seasonality, back in Lecture 3, on GAMs. And it's the Fourier feature block
you engineered by hand for Homework 4, the XGBoost assignment: those sine-one, cosine-one and
sine-two columns.

So it's the same device, making its third appearance. The difference now is that a-k and b-k come
back as **distributions**, so you can ask how confident you are that the seasonal peak is where you
think it is.

*(Point at the last line.)*

K trades flexibility against overfitting, exactly the way spline knots did in Lecture 3. We use K
equals three, which means six coefficients standing in for fifty-two parameters.

**[pause]**

Notice what that construction *assumes*, because it's an assumption and not a free lunch. It assumes
the seasonal shape is **smooth**.

Three sine-cosine pairs can describe a broad annual hump with a couple of gentle bumps. What they
can't describe is a single enormous one-week spike, Black Friday, say, sitting on an otherwise flat
year. Fourier terms will smear that spike across the neighboring weeks, because smooth curves are
all they can make.

If your business has sharp calendar events, the honest answer is to model them as **separate
indicator variables**, rather than raising K until the waves can approximate a spike. That's what
Prophet does with its holiday regressors, and it's what you did by hand with the SNAP and event flags
in Homework 2.

---

# ▶ SLIDE 10 — Section divider: Fitting It in PyMC

The next section puts the model in code, along with the check you run before believing any of it.

---

# ▶ SLIDE 11 — The Model in PyMC

Here's everything from the last three slides, in about eight lines.

*(Walk the code.)*

First comes `sigma_level`, with an Exponential prior. It has to be positive, so it gets an
Exponential, not a Normal. Then there's `z`, a vector of standard Normals. Then the level is the
**cumulative sum** of z times sigma-level.

Stop on that line for a second, because it's the clever bit. The `cumsum` of scaled Normal draws
**is** a random walk. That's the definition: each step adds a scaled Normal shock to the running
total.

Then comes `beta`, the Fourier coefficients, and then sigma, the observation noise. Last is the
likelihood, which says the observed data is Normal around the level plus the Fourier terms.

**[pause]**

*(This warning is practical and easy to get wrong.)*

There's one thing you have to do, and it's in bold on the slide for a reason. **Standardize y
first.**

These priors assume the data is on roughly unit scale, and our weekly units are in the tens of
thousands. An Exponential prior with rate ten on `sigma_level` is sensible for a standardized series,
and absurdly tight for raw units of twenty thousand. You'd be telling the model that the level
essentially can't move.

So standardize, fit, and then transform the forecasts back. Getting this wrong produces a model that
fits terribly, for reasons that look mysterious.

**[pause]**

This is the same lesson as the warning in Lecture 6, on regularization, about standardizing before
Ridge. And it'll come back one more time, with the opposite sign, in Lecture 12, two weeks from now.
There, standardizing a variable you then *interpret* silently destroys the interpretation.

So the rule isn't "always standardize," and it isn't "never standardize." It's this: **know what
scale each variable is on, and know what your priors assume about that scale.** A prior is a
statement about magnitudes. If you don't know the magnitudes, you don't know what you've assumed.

---

# ▶ SLIDE 12 — The First Fit Failed — Here Is What That Looks Like

*(This is the honest-failure slide. Deliver it as a story, not a footnote.)*

Now here's something worth your attention.

The obvious way to write this model uses `pm.GaussianRandomWalk`, which is exactly what it sounds
like and exactly what the equation says. We wrote it that way first. And it sampled **badly**, even
with `target_accept` raised to zero point nine five.

*(Point at the top row of the figure.)*

This is sigma-level, the one parameter that misbehaved. There are four chains, one color each,
showing the first three hundred draws of fifteen hundred. Look at the texture. Each chain drifts, up
for a stretch and then down for a stretch. A draw is mostly a copy of the one before it.

The numbers, in red, say the same thing. R-hat is one point zero two two, against a threshold of one
point zero one, so it failed. The effective sample size is two hundred seventy-four, against four
hundred, so that failed too. Divergences are zero, so that one passed. This is the failure I
promised you last week, the kind that only R-hat and the effective sample size can catch.

*(Point at the bottom row.)*

Now look at the bottom row, after the fix. It's the same parameter and the same number of draws.
This is what healthy chains look like: a fuzzy band, with each draw nearly independent of the last.
R-hat is one point zero zero two, and the effective sample size is two thousand three hundred
seventy-six.

**[pause]**

Let's put a number on the difference. Six thousand draws in the top row carry about as much
information as two hundred seventy-four independent ones. The bottom row's six thousand are worth
about two thousand four hundred.

*(Now the important part.)*

Here's what I want you to take from this: **nothing crashed**. The model returned a forecast, and
the forecast, plotted, looked entirely reasonable, with a sensible level, a sensible seasonal shape,
and plausible intervals. If we hadn't run the diagnostics, we would have shipped it and never known.

That's precisely why Lecture 10, last week, made those three numbers non-negotiable. This is the
failure mode they exist for.

*(Point at the muted line.)*

And yes, the two rows sit at different heights. Hold that thought for one slide.

---

# ▶ SLIDE 13 — The Fix: Reparameterize, Do Not Sample More

*(Walk the table, left column, then right.)*

The fix was **reparameterization**, not more samples. It wasn't a longer run, and it wasn't a smaller
step size. It was a different way of writing the walk.

On the left is the centered form. The sampler works directly on all two hundred twenty-five levels,
and their plausible step size is sigma-level. So every time the sampler wants to try a different
sigma-level, all two hundred twenty-five levels have to move with it. It can't do that in one step,
so it crawls, and that's the drift you saw in the top row.

On the right is the non-centered form. The sampler works on z, which are independent standard
Normals, and the level is built from them afterwards, as the cumulative sum of z times sigma-level.
Now changing sigma-level doesn't require moving anything else. The geometry is round, and the
sampler can navigate it.

*(Point at the warning box.)*

It's the same data and the same six thousand draws, and the effective sample size went from two
hundred seventy-four to two thousand three hundred seventy-six. This is the standard fix. Martin,
Kumar and Lao work through it in section four point six point one of their book. Next week, we'll
see the same fix applied to a hierarchy, where it will matter even more.

**[pause]**

*(Point at the muted line. This is the answer to the different heights. Don't skip it.)*

Now for the honest footnote, because it's a lesson in its own right.

The two versions are *almost* the same model, but not quite. `GaussianRandomWalk` needs to know
where the walk starts, and if you don't tell it, it lets week one's level start anywhere. Its
default start is a Normal with a standard deviation of one hundred, on a standardized series. PyMC
prints a warning saying so, but the lab switches warnings off, so you never see it. The cumsum
version starts the walk at z times sigma-level, which is close to zero, the series average.

That one difference moves the answer. With the start pinned near the average, the walk has to fall
fast in the first weeks, which you saw on the decomposition slide, and sigma-level comes out higher:
about five hundred twenty units a week, against about four hundred fifty. That's the gap between the
two rows. It's a slightly different model, not a sampling failure.

So there are two lessons here, not one. First, change the coordinates when the sampler crawls.
Second, read the warnings you were about to silence, because a default you didn't choose is still a
prior.

---

# ▶ SLIDE 14 — Section divider: Reading the Forecast

The next section is about the point estimate, the interval, and whether either one is any use.

---

# ▶ SLIDE 15 — Does It Work? The Point Forecast

This is CA_1 FOODS, with fifty-two weeks held out.

*(Point at the table.)*

The Bayesian structural model has an RMSE of one thousand five hundred eighty-nine. Seasonal naive,
which is the same week last year and takes one line of code, gets one thousand six hundred sixty.

So we beat the benchmark by about **four percent**.

*(Point at the note in the first line.)*

And don't set this benchmark beside the two thousand one hundred fifty-two from the neural network
lectures, Lectures 8 and 9. That one was seasonal naive across the whole thirty-series panel. This is
one series, CA_1 FOODS, and it's the same benchmark you had in Homework 1.

**[pause]**

That's a real improvement, and a small one. And I'll repeat what I said at the start: after eleven
weeks of this course, you should be suspicious of anyone who reports four percent as a triumph.

*(Now the fair-hearing paragraph. Give it properly.)*

But let me also be fair to the model, because the comparison isn't like for like.

Look at what this model was *given*: the series and a calendar, and that's all. It had no lag
features, no price, no SNAP flags, no store identity, and no information from other series.

The XGBoost model in Homework 4 had forty-six engineered predictors and thirty series to learn from,
and it beat naive by a much wider margin. It also had a great deal more information.

**[pause]**

And there's nothing stopping you from combining the two, which is what a serious production system
usually does. Add price and promotion regressors to this structural model, and the level has less
work to do. So it drifts less, and the interval narrows. You keep the predictive distribution, and
you buy back some accuracy.

We're not doing that today, because I want the mechanism to stay visible rather than buried under
covariates. But it's the obvious next step, and it's a very reasonable direction for a final
project.

So the point forecast is not the reason to do this. That brings us to the slide that is.

---

# ▶ SLIDE 16 — The Interval Is the Product

Every forecast comes with a ninety-four percent predictive interval. And there are **two** questions
you can ask about it. They're different questions, and they're the heart of this lecture.

*(Walk the figure, left to right.)*

On the left are the last sixty training weeks. The gray dots are actual sales, the thin blue line is
the fit, and the pale band around it is the in-sample interval. At the dashed line, we stop looking.
To the right are the fifty-two weeks we held out: black dots for what actually happened, the blue
line for the forecast, and the shaded band for the ninety-four percent interval.

**[pause]**

The first question is whether it's calibrated. Count the black dots outside the band, and you'll
find there are none. All fifty-two of the fifty-two weeks fell inside, so the interval isn't
overconfident. But it promised ninety-four percent, and ninety-four percent of fifty-two would leave
about three weeks outside. So it's actually *conservative*, a bit too wide.

The second question is whether it's useful. Look at how wide the band is. On average, it's about
eleven thousand seven hundred units wide, on an average week of nineteen thousand seven hundred.
That's plus or minus thirty percent, **on average**.

*(Point at the two red arrows.)*

And "on average" is doing work in that sentence. One week out, the band is plus or minus sixteen
percent. A year out, it's plus or minus thirty-nine. It fans out as the forecast gets further from
the data. It's honest at every horizon, and it's too wide to order against at almost all of them.

---

# ▶ SLIDE 17 — Honest, and Too Wide to Order Against

*(This is the sentence of the lecture.)*

Here's the sentence I want you to leave with: **passing the first does not excuse failing the
second.**

A model that says demand will be somewhere between fourteen thousand and twenty-six thousand is
telling you the truth and helping nobody. You can't write a purchase order against that.

**[pause]**

And the cause is structural. It isn't a bug.

*(Point at the formula.)*

Take the parameters as given, and take today's level as given. Then the forecast h weeks out has a
variance of sigma-squared plus h times sigma-level-squared. The noise, sigma-squared, is the same
every week. But the level's random-walk steps **add up**, with one step of drift for every week you
look ahead.

*(Walk the figure.)*

The dotted line is the noise alone. The dashed curve is that formula, at the fitted values of the
parameters. Early on, the noise dominates. After about seven weeks, the drift term overtakes it, and
from there the band grows like the square root of h. The blue line is the band we actually got. It
sits a little above the dashed curve, because the real band also carries our uncertainty about the
parameters and about today's level.

So a level that's free to random-walk for fifty-two steps *drifts*, and the model is correctly
reporting the consequence of what we told it to believe.

The fix is a model with a **stronger opinion**. That could be a damped level, which reverts instead
of wandering, or covariates that pin the level down, like price, promotions, and the other things we
withheld.

**[pause]**

*(And now the payoff. Slow down.)*

But notice what just happened, because this is the argument for the entire lecture.

We could **ask** whether the uncertainty was honest. We computed coverage against held-out data, and
we got an answer.

For the models before this one, we never asked. ETS had intervals, and Prophet drew a band in
Homework 2, but we scored every model on its point forecast and never checked a band against
held-out weeks. Here, we did.

A calibrated but too-wide interval is a worse forecast and a better *deliverable*, because it tells
you something true about how much you actually know. That's worth far more than the four percent.

---

# ▶ SLIDE 18 — Section divider: Key Takeaways

This last section pulls together what structural models are for.

---

# ▶ SLIDE 19 — Lecture 11, Part 1: Key Takeaways

Here are the main points.

First, a structural model writes the series as named parts, the level, the seasonality and the
noise, and it gives each one a prior and a posterior.

Second, the local level is a random walk whose step size you put a prior on. A small step size means
a rigid level, and a large one means a level that chases the data. It's the same trade as a
smoothing parameter or a penalty.

Third, Fourier terms build seasonality from a few waves instead of fifty-two parameters, and each
pair of coefficients is an arrow. Its length is the wave's size, and its angle is the wave's timing.
It's the same device as Prophet in Lecture 3 and the features you built in Homework 4, now making
its third appearance.

Fourth, **check the sampler before the forecast**. Our first fit had an R-hat of one point zero two
two and an effective sample size of two hundred seventy-four, while returning a perfectly
plausible-looking answer. Reparameterizing fixed it.

Fifth, the point forecast beat seasonal naive by four percent, and that isn't the argument.

And last, the interval was **calibrated but too wide to act on**. It was plus or minus thirty
percent on average, and it widened with the horizon. Being able to *establish* that is the actual
product.

**[pause]**

Next time, in Lecture 11, Part 2, we'll turn to hierarchical models. That's what to do when you have
ten series and some of them barely have any data, and it starts with a question you have to answer
*before* you fit anything.

---

# ▶ SLIDE 20 — References

*(Advance and close. No narration is needed.)*
