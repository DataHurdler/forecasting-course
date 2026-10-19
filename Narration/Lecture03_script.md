# Lecture 3 — Recording Script

**ECON 8310: Business Forecasting · Generalized Additive Models**

Deck: `Slides/Lecture03_GAMs.pdf` (20 pages) · Measured runtime: **~24 minutes** (22–27 depending on pace)

---

## How to use this document

- **`▶ SLIDE n — Title`** marks where to advance. The number is the PDF page.
- Written to be **spoken**. Short sentences. Contractions are fine.
- *Italic parentheticals* are stage directions — do not read them aloud.
- **[pause]** means actually stop for a beat.
- Change anything that doesn't sound like you.

---

# ▶ SLIDE 1 — Title page

Lecture 3: Generalized Additive Models.

This lecture sits at a hinge in the course. Behind us are the classical time series methods —
exponential smoothing, ARIMA, VAR. Ahead of us, starting next week, are the machine learning
methods: trees, forests, boosting, neural networks.

GAMs are the bridge. They are the last method we'll see that is genuinely nonlinear *and*
genuinely explainable at the same time. After today, every increase in flexibility costs you
something in interpretability.

So this is worth understanding well — partly for its own sake, and partly because it sets up
the tradeoff that the second half of this course keeps making.

**[pause]**

---

# ▶ SLIDE 2 — Lecture Outline

Five parts. The motivation — why we want something between linear and black-box. Then the
structure of a GAM and how smoothing is controlled. Then Prophet, which is a GAM built
specifically for time series. Then pyGAM, for the general case. And the takeaways.

---

# ▶ SLIDE 3 — Section divider: Motivation: Beyond Linear, Without Black Boxes

*(Divider.)*

Part one. Why we want flexibility without giving up the ability to explain what the model is
doing.

---

# ▶ SLIDE 4 — The Flexibility–Interpretability Tradeoff

Two columns, and this is one of the few genuine side-by-side comparisons in the course.

On the left, **linear models** — regression, ARIMA. Each coefficient has a clear meaning. You
can say "a one-point rise in unemployment lowers sales by four hundred units," and everyone in
the room knows what you mean. The properties are stable and well understood.

But they assume a constant slope everywhere. The effect of unemployment going from 4% to 5% is
assumed identical to it going from 9% to 10%. That's an assumption, and it's usually false.
They miss curvature and threshold effects entirely.

**[pause]**

On the right, **fully nonlinear models** — trees, neural networks. They capture almost any
pattern, and on complex data they're accurate.

But ask why a specific prediction came out the way it did, and you'll struggle. And because
interactions entangle the variables, you can't cleanly separate what any single variable
contributes.

**[pause]**

**GAMs sit between the two.** Each variable gets a flexible, nonlinear relationship. But the
model stays **additive** — the effects are added together, not multiplied or entangled — so
each one can still be plotted and explained on its own.

That is the whole bargain. And the rest of this lecture is about what it costs, because it
does cost something.

---

# ▶ SLIDE 5 — Why Additivity Matters for Interpretation

Let me make the additivity point concrete, because it's easy to nod along without seeing what's
at stake.

In an additive model, each variable's effect is the same whatever the other variables are doing.
In a non-additive model it isn't, and then you cannot describe the variables separately.

**[pause]**

*(Point at the left panel.)*

This is the electricity data you'll use in the lab. Average daily load, month by month. The red
line is weekdays, the blue line is weekends. Weekends are lower every month — offices and
factories are closed.

But look at the size of the gap. In July, weekdays run about four thousand and sixty megawatts
above weekends. In December, about two thousand. The weekend effect is twice as big in summer
as in winter. The likely reason, though the data alone can't prove it: office and shop air
conditioning runs on weekdays and stops at the weekend, while much of the winter load is home
heating, and homes are occupied every day.

So "what does the weekend do to demand?" has no single answer. It begins with "it depends on
the month."

**[pause]**

*(Point at the right panel.)*

Now fit an additive model: a month effect plus a weekend effect. Here is everything it can draw.
Two curves, exactly parallel. One weekend effect — about two thousand nine hundred and sixty
megawatts — in every month of the year. Too big in December, too small in July.

That is not a bad fit to blame on the estimator. It's the structure. In an additive model the
effect of one variable cannot depend on another, so the curves have to be parallel.

**[pause]**

And that's the trade, both halves of it. The price: an additive model cannot represent a gap
that changes with the month. The gift: because the effects don't depend on each other, each one
is a single curve. One curve for the month, one number for the weekend. You can plot each one
on its own and hand it to a stakeholder, and every plot is true whatever the other variable is
doing.

If you want the calculus version: in a GAM the marginal effect of $x_1$ is $f_1$-prime of $x_1$,
a function of $x_1$ alone. In a model with $x_1$ times $x_2$ in it, the derivative contains
$x_2$. Same idea as the parallel curves.

We'll come back to the price at the end of the lecture, and next week trees pay it for us.

---

# ▶ SLIDE 6 — Section divider: GAM Structure and Smoothing

Part two. The structure — additive nonlinear functions, penalized for wiggliness.

---

# ▶ SLIDE 7 — What Is a GAM?

Here's the definition. $y$ equals a constant, plus $f_1$ of $x_1$, plus $f_2$ of $x_2$, and so
on — where each $f$ is a smooth nonlinear function estimated from the data.

Compare that to linear regression, where each term is beta-$j$ *times* $x_j$. We've replaced
a coefficient with a function. That's the entire generalization.

Each $f_j$ is free to be linear, curved, U-shaped, or threshold-like. You don't specify the
shape. The data decides.

**[pause]**

Because the terms are *added* rather than multiplied, the effects stay independent and each
can be visualized on its own. And every function is fitted to minimize prediction error
subject to a smoothness penalty — that's two slides from now, and it's what stops the curves
from chasing noise.

**[pause]**

A retail example, to make it concrete.

Model weekly sales as $f_1$ of time, capturing the long-run trend. Plus $f_2$ of week-of-year,
capturing seasonality. Plus $f_3$ of unemployment, capturing economic sensitivity.

Three curves. Each plotted and interpreted separately. And none of them requiring you to hold
the others fixed to make sense of it.

Notice this is doing the same decomposition Prophet will do in twenty minutes — trend,
seasonality, external effects. Prophet is a GAM with the time series structure built in.

---

# ▶ SLIDE 8 — What the "Generalized" Actually Means

Someone should have asked this two slides ago, so let me answer it before we go on.

The method is called a **Generalized** Additive Model. So far I've shown you an *additive*
model. Where is the "generalized"?

**[pause]**

Everything we've modeled has been continuous and roughly symmetric — sales in dollars,
demand in units. For that kind of outcome, adding smooth functions together works directly.

But suppose your outcome is a **count**. Units sold at a small store: zero, one, two, seven.
An additive model can happily predict *minus three units*, which is not a thing.

Or suppose it's **binary**. Did this customer churn? Did we stock out? The answer is a
probability, between zero and one, and again an additive model will cheerfully predict one
point four.

**[pause]**

The fix is on the slide. Instead of modeling the mean directly, we model a **transformation**
of the mean — $g$ of the expected value of $y$ equals the additive predictor.

That $g$ is called the **link function**, and it's the entire "generalized."

Look at the table. For continuous symmetric data, the link is the **identity** — do nothing,
which is the case we've been in all along. For counts, the link is the **log**, which means
you model the log of the mean and exponentiate to get back, so the forecast can never go
negative. For binary outcomes, the link is the **logit**, which maps the whole real line into
zero-to-one, so the probability is bounded by construction.

**[pause]**

In pyGAM this is one word: `LinearGAM`, `PoissonGAM`, `LogisticGAM`. Same additive structure,
same splines, same smoothing penalty. Only the link changes.

And if this sounds familiar from a regression course — it should. It's exactly the same idea
as a generalized *linear* model. A GAM is a GLM where the linear terms have been replaced by
smooth functions. That's the whole family tree in one sentence.

---

# ▶ SLIDE 9 — Controlling Smoothness: The Penalty Term

If each $f$ can be any shape, what stops it from being a squiggle that passes through every
data point? This slide.

A GAM minimizes a **penalized** sum of squared errors. Two terms. The first is the ordinary
fit term — squared errors, exactly as in OLS. The second is the penalty: the integral of the
*squared second derivative*.

**[pause]**

The second derivative measures bending — how fast the slope is changing. A straight line has
a second derivative of zero everywhere. So look at the thin strips under each panel. *(Point
at the strips.)* Each one shades the squared bending of the curve above it, along the year, and
the shaded area *is* the penalty. Area, not a formula. Strictly, pyGAM charges a discrete
version of this integral — squared differences between neighboring spline coefficients — but
both are zero for exactly the same curves: straight lines.

Now the three panels. Same data in every one: a year of daily electricity demand, 2014, against
day of year. Same curve family: a hundred splines. The only thing that changes is lambda, the
price of bending.

**[pause]**

*(Point at the left panel.)* Lambda near zero. Bending is free, so the curve chases every wiggle
— every cold snap, every heat wave, every weekend dip. It uses about a hundred effective degrees
of freedom — the "edof" above each panel, the number of parameters the curve really uses:
basically all of them. Its strip is off the scale — eighty times the area of the middle one.

*(Point at the right panel.)* Lambda huge. Any bending at all is infinitely expensive, so the
curve can't bend, and what's left is a straight line. And not just any line — it's exactly the
ordinary least squares line. Its strip is empty, because a straight line never bends. So OLS
isn't a different method from a GAM. It's a GAM with the smoothness dial turned all the way up.

One technical aside. This only works because I used an ordinary spline basis here. The lab uses
a *cyclic* spline, which forces December thirty-first to meet January first. A straight line
can't do that unless it's flat, so with a cyclic spline the limit is a flat line, not the OLS
line.

**[pause]**

*(Point at the middle panel.)* And in between: lambda chosen by GCV, generalized
cross-validation — not by eye. You'll hear that refrain all semester: the tuning parameter is
selected by an estimate of out-of-sample error, never by what looks nice. GCV computes that
estimate from the fit itself — a shortcut to leave-one-out error, with nothing actually held
out. Forty-three effective degrees of freedom. It keeps the two humps, heating and cooling, and
drops most of the noise.

The proof is in the titles: each curve scored on the *next* year, 2015. The wiggly fit misses
by thirty-nine forty-six. The straight line by forty-six twenty. The GCV fit by thirty-six
twenty-one — the best of the three. The noise the left panel memorized doesn't come back next
year.

To be fair to the picture: GCV still keeps some week-long cold snaps that won't repeat, because
it treats neighboring days as independent, and they aren't. I checked every year from 2010 to
2016 the same way. The GCV fit beat the lambda-near-zero fit in seven years out of seven; the
straight line lost to it in six out of seven.

**[pause]**

Let me say a little more about splines, because "piecewise polynomials joined at knots" is one
of those phrases that sounds like an answer without being one.

Here's the picture. Take the range of your predictor and put down a handful of points along
it — those are the **knots**. Between each pair of knots, fit a low-order polynomial, usually
a cubic. Then require that where two pieces meet, they join up: same value, same slope, same
curvature. That's what makes it look like one smooth curve rather than a chain of segments.

**[pause]**

The number of knots controls how much wiggle is *available*. Few knots, and the curve can only
bend gently. Many knots, and it can bend almost anywhere.

Now, you might expect that choosing the number of knots is the crucial modeling decision. It
mostly isn't — and this is the elegant part of the penalized approach. You give the model
plenty of knots, more than you think you need, and then let **lambda** control how much of
that available flexibility actually gets used.

So the knots set the ceiling, and lambda sets where under the ceiling you land. That's why you
tune lambda — by GCV or cross-validation — and generally leave the knot count alone. One tuning
decision instead of two — and the one that's easier to make honestly.

---

# ▶ SLIDE 10 — Section divider: Prophet for Time-Series GAMs

Part three. Prophet — a GAM built specifically for business time series.

---

# ▶ SLIDE 11 — Prophet: Forecasting at Scale

Taylor and Letham built Prophet at Facebook, now Meta, for large-scale forecasting. And it is a
GAM in time-series clothing — look at the decomposition.

$y$ at time $t$ equals trend, plus seasonality, plus holidays, plus noise. Three additive
components. That is the GAM structure from two slides ago, with the terms chosen for what
business time series actually contain.

**[pause]**

**Trend** is piecewise linear or logistic, with *changepoints* — specific dates where the
slope is allowed to shift. That's important, because real business series do change direction:
a new competitor arrives, a store is renovated, a pandemic starts.

**Seasonality** is a Fourier series — a sum of sine waves. That's how Prophet handles weekly and
yearly cycles simultaneously — something none of our Lecture 1 or 2 methods could do. More on
how many waves in a moment.

**Holidays** are irregular events you supply yourself. Black Friday. A product launch. A
lockdown. These aren't periodic, so no seasonal term captures them, and you tell Prophet the
dates directly.

**[pause]**

Now — the design goals were practical rather than statistical. One fit-and-forecast call.
Robust to missing data and outliers. Interpretable, because every component plots on its own.

That's why it spread so quickly among analysts who aren't statisticians.

**[pause]**

Now let me be honest about its track record, because you'll hear Prophet talked about as if
it were state of the art, and it isn't.

In head-to-head comparisons, Prophet frequently **loses** to a well-tuned ARIMA or ETS on
forecast error. Our own textbook says so directly — the authors note that Prophet rarely gives
better accuracy than the alternatives. It did not win the M-competitions. It doesn't top
benchmark leaderboards.

**[pause]**

So why is it in this course, and why is it everywhere in industry?

Because accuracy is not the only thing being optimized. Consider what Prophet actually gives
an analyst: a defensible forecast in twenty minutes, from a series with gaps and outliers,
with a component plot you can put in front of a manager who will ask what drove it.

Getting an ARIMA to that same point takes stationarity testing, order identification,
residual checking, and someone who knows what a PACF is. If you have that person, use ARIMA
and get the better number. If you're one analyst covering four hundred product lines, you do
not have that person four hundred times.

**[pause]**

That's the real lesson, and it goes wider than Prophet. **The best model in a paper and the
best model in an organization are frequently different models**, and the gap between them is
usually maintenance cost and explainability, not accuracy. Part of your job is knowing which
one you're being asked for.

One knob to know: `changepoint_prior_scale`. High means the trend adapts quickly and risks
overfitting. Low means a stable trend. It's the main thing you'll tune.

---

# ▶ SLIDE 12 — Prophet's Seasonality: How Many Waves?

What does "a sum of sine waves" actually buy you? Here it is on the lab's electricity data.

*(Point at the gray dots.)*

Each dot is the average load on one day of the year, over the lab's five and a half years. Look
at the shape. It is not a wave. There's a winter hump in January, from heating. A trough in
April, when nobody heats or cools. A bigger peak in July, from air conditioning. And a second
trough in October.

**[pause]**

*(Point at the blue line.)*

The blue line is the best single wave — one sine curve with a period of one year. One wave has
one peak a year, so it has to choose. It can't be high in January *and* high in July. It ends up
nearly flat, and its error is about three thousand four hundred megawatts.

*(Point at the red line.)*

The red line adds a second wave with half the period — a hundred and eighty-two and a half days,
so it repeats twice a year. Now there are two peaks and two troughs, and the fit follows the
dots. The error falls to about thirteen hundred and fifty. A factor of two and a half.

*(Point at the bottom strip.)*

The strip underneath shows the two waves separately. The solid black one is the slow yearly
wave. The dashed red one is the fast half-year wave, and it does most of the work. Add them to
the average and you get the red curve on top.

**[pause]**

That's all Prophet's yearly seasonality is: a sum of these sine-and-cosine pairs. The number of
pairs is called the Fourier order, and Prophet's default for the yearly term is ten. More pairs,
more wiggles allowed. The lab has you force it down to one and see what happens on the test year.

---

# ▶ SLIDE 13 — Prophet: Code and Workflow

The code is genuinely this short, and it's exactly what you'll run in the lab.

Prophet wants a dataframe with two columns — `ds` for dates and `y` for values. It insists on
those names. Instantiate it, say which cycles you want — yearly and weekly on, daily off, because
we have one row per day — and fit. Then `predict` on the test dates, and you have a forecast.

**[pause]**

But the call that matters for this course is the last one: **`plot_components`**.

*(Point at the three panels.)*

This is what it shows, for the lab's model. Three panels, one per additive term.

On the left, the **trend**: a piecewise straight line, bending at the changepoints. The shaded
strip at the right is the lab's test year.

In the middle, the **weekly** part of the seasonality: weekdays a little above zero, Saturday
about eighteen hundred megawatts below, Sunday about twenty-five hundred below.

On the right, the **yearly** part: the winter hump, the April trough, the big summer peak, the
October trough. The same double-peaked shape you just saw on the previous slide — nobody told
Prophet about heating and cooling; the ten waves had enough room to find it.

Add the three panels together and you have the forecast. It's the additive structure made
visible — you're literally looking at the terms of the equation, one at a time.

That plot is what you put in front of a manager. Not the forecast line — the components. It
lets them check your model against their own knowledge of the business. "Yes, the summer peak
looks right. No, that changepoint in March is wrong, that was the fire." That conversation is
worth more than a small RMSE improvement.

**[pause]**

**Reach for Prophet** on seasonal series with known holiday effects, especially when someone
will ask you to explain the forecast.

**Reach for pyGAM** instead when the drivers are a few non-calendar predictors whose shape you
want to see — which is where we're going next. Not many correlated ones: that's the case where
GAM curves stop being trustworthy, as you'll see in a minute.

---

# ▶ SLIDE 14 — Section divider: pyGAM for Generalized Additive Modeling

Part four. pyGAM, for when you have several predictors and want a flexible effect for each.

---

# ▶ SLIDE 15 — pyGAM: Flexible Additive Modeling

Prophet is specialized for time structure. pyGAM is the general case: you specify a smooth
function per predictor and it fits the whole additive model.

The code shows the pattern. Import, then build the model by *adding terms together* — which
is a rather nice piece of API design, because the code mirrors the equation.

**[pause]**

Three term types, and that's the whole vocabulary.

**`s(j)`** is a smooth spline of feature $j$. The flexible nonlinear effect. This is the
default and the reason you're using a GAM.

**`l(j)`** forces a straight line. Worth using deliberately when theory says the effect should
be linear — don't spend flexibility where you don't need it.

**`f(j)`** handles a binary or categorical feature — a factor.

Then `gridsearch` tunes lambda by GCV — the same in-sample shortcut to leave-one-out error you
saw on the penalty slide — so you don't pick it yourself.

**[pause]**

One habit I'd ask you to build: after `gridsearch` returns, **look at the curves**. The search
optimizes an estimate of held-out error, and it will happily give you a curve that's
statistically optimal and economically absurd — sales rising as unemployment rises, say. When
that happens the model isn't broken; it's telling you something is confounded. But you only find
out if you look.

**[pause]**

And there's a specific failure worth naming, because it has a name and it will bite you:
**concurvity**.

You know multicollinearity from regression — two predictors so correlated that the model can't
tell which one is doing the work, so the coefficients go unstable and flip sign between
samples.

Concurvity is the same disease in a GAM, but worse, because the terms are *smooth functions*
rather than straight lines. One smooth term can mimic another even when the raw variables
aren't strongly correlated at all. A smooth function of time and a smooth function of a
trending economic indicator can trace nearly the same shape — and the model has no principled
way to allocate credit between them.

**[pause]**

The symptom is a partial dependence plot that looks wrong or wildly uncertain, and that
changes a lot when you drop an apparently unrelated variable.

The reason it matters more here than in ordinary regression is that GAMs are sold on
interpretability. Under concurvity, the individual curves become unreliable *even when the
overall forecast is fine*. So you get a model that predicts well and explains badly — which
is precisely the thing you chose a GAM to avoid.

pyGAM won't warn you. Check by dropping terms and seeing whether the remaining curves hold
their shape.

---

# ▶ SLIDE 16 — Interpreting GAM Results: Partial Dependence Plots

This is the payoff slide — the thing you actually deliver.

A partial dependence plot shows one estimated function, $f$-hat-$j$, across the range of values
you observed, with a confidence band around it. Reading the shape is most of the
interpretation. And this is a real one: Lab 3's model of daily electricity demand, a smooth
term in day of year plus a factor for day of week. You'll fit it yourself.

**[pause]**

*(Point at the left panel.)* Day of year. Demand is high in January, falls to a trough in late
April, climbs to the annual peak in mid-July, and falls again into the autumn. Two peaks:
heating in winter, air conditioning in summer. Nobody told the model that. It bent its way
there.

Now think back to the right-hand panel on the penalty slide — the straight line. A linear term in
day of year would have reported a gentle slope and concluded the calendar hardly matters. That's
exactly backwards. A flat line, a rising curve, a U-shape, a double hump: those are the shapes
you'll meet, and a linear term can only ever show you a straight line — rising, falling or flat.

*(Point at the right panel.)* Day of week. Weekdays sit together; Saturday and Sunday drop. The
bars are each day's band.

**[pause]**

Watch the band as well as the curve. Here it's about the same width all year, and that's worth
noticing, because it isn't the usual picture. Every day of the year appears four or five times
in the training data, and a cyclic spline has no ends, so the model is roughly equally sure everywhere.
On a predictor whose extremes are rare — unemployment above ten percent, say — the band fans out
exactly where the data thin, which is where a reader is most tempted to extrapolate. If the band
is wide, the curve there is a guess. Say so.

**[pause]**

And this is what makes a GAM worth its accuracy cost. The output isn't a number. It's a
sentence someone can act on. *(Point at the key box.)* "Plan for mid-July demand about eleven
thousand four hundred megawatts above the late-April trough, a winter peak about eighty-five
hundred above it, and weekends about three thousand below weekdays."

Notice the sentence talks in *gaps* — peak above trough — and the brackets on the plot draw
them. That's deliberate: the zero line of a smooth term is a bookkeeping choice, so the
differences are what mean something.

And the last clause on the slide is the point: no straight line in day of year has two peaks.

---

# ▶ SLIDE 17 — Section divider: Key Takeaways and Roadmap

Let's pull it together — and set up where the course goes next.

---

# ▶ SLIDE 18 — GAMs vs. Other Methods: When to Choose Each

This table is the honest summary of the whole course so far, and it's worth reading down the
columns rather than across.

ARIMA and ETS: not nonlinear, no interactions, highly interpretable, univariate time series.
Linear regression: same, with manual interactions if you build them.

GAM: **nonlinear, yes. Interactions, only the ones you specify. Interpretable, high.**

Random Forest: nonlinear and interactions, medium interpretability. XGBoost: same, but low.

**[pause]**

The column that decides it is usually **Interactions**.

A GAM is additive by construction. That's the source of its interpretability, and it's also a
real limit — though not an absolute one. If the effect of price genuinely depends on whether
it's a holiday week, a GAM can represent that only if you add the interaction term yourself: a
tensor-product smooth, `te()` in pyGAM, or a smooth of price fitted separately by holiday. You
have to suspect the interaction first. A tree finds it on its own — that's exactly what a tree
does, and it's why the next three lectures exist.

So the additivity is a bargain, not a free lunch. You buy explainability with the ability to
find interactions automatically.

**[pause]**

**Choose a GAM when** you must explain each predictor's effect. When those effects are curved
rather than straight. When a time series has complex trend and seasonality — reach for
Prophet. And when interpretability matters more than the last few points of accuracy.

That last clause is the honest one. Whether a more flexible model actually buys those last few
points is an empirical question, and Lectures 4 through 6 test it on our data. The answer is
not always the one the reputation predicts.

---

# ▶ SLIDE 19 — Lecture 3: Key Takeaways

Five things.

**One.** GAMs give nonlinear flexibility while keeping interpretability, because the model is
additive. Each variable's effect can be visualized and explained independently.

**Two.** The smoothness penalty prevents overfitting. Lambda trades fit against smoothness,
and it's chosen by GCV or cross-validation — never by eye.

**Three.** Prophet decomposes a series into trend, seasonality, and holidays using a GAM.
Ideal for business series with seasonal patterns and known irregular events.

**Four.** pyGAM handles the general case. `s()` for smooth, `l()` for linear, `f()` for
categorical, and `gridsearch` for lambda.

**Five.** Partial dependence plots are what make results actionable. They're what you deliver.

**[pause]**

And the meta-point, which matters more than any of the five: this is the last method in the
course that is both genuinely nonlinear and genuinely explainable. From next week, every
increase in flexibility costs you something in interpretability, and part of your job as a
forecaster is deciding when that trade is worth making.

---

# ▶ SLIDE 20 — References

Readings are in the syllabus — section 7.7 for regression splines, and 12.1 and 12.2 for
complex seasonality and Prophet.

Next week: Decision Trees. We move to tree-based methods, and the first thing we gain is
exactly what we gave up today — interactions, captured automatically.

See you then.

---

## Timing guide

| Segment | Slides | Target |
|---|---|---:|
| Opening & motivation | 1–5 | ~9 min |
| GAM structure, links & smoothing | 6–9 | ~10 min |
| Prophet | 10–13 | ~10 min |
| pyGAM & interpretation | 14–16 | ~9 min |
| Close | 17–20 | ~5 min |

If you need to compress, slide 15's term-type list survives trimming. Do **not** compress
slides 5, 16, or 18 — the additivity argument, the partial dependence plot, and the
interactions tradeoff are the three ideas this lecture exists to deliver.
