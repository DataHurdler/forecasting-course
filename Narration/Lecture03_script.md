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

This is Lecture 3, on generalized additive models, which everyone calls GAMs.

This lecture sits at a hinge in the course. Behind us are the classical time series methods:
exponential smoothing in Lecture 1, and ARIMA and VAR in Lecture 2, last week. Ahead of us,
starting next week, are the machine learning methods, which means trees, forests, boosting and
neural networks.

GAMs are the bridge between the two. They're the last method we'll see that is genuinely
nonlinear *and* genuinely explainable at the same time. After today, every increase in flexibility
costs you something in interpretability.

So this one is worth understanding well, partly for its own sake, and partly because it sets up
the tradeoff that the second half of this course keeps making.

**[pause]**

---

# ▶ SLIDE 2 — Lecture Outline

We'll take this in five parts. First, the motivation, which is why we'd want something between a
linear model and a black box. Second, the structure of a GAM and how its smoothness is
controlled. Third, Prophet, which is a GAM built specifically for time series. Fourth, pyGAM, for
the general case. And we'll finish with the takeaways.

---

# ▶ SLIDE 3 — Section divider: Motivation: Beyond Linear, Without Black Boxes

This first section is about why we want flexibility without giving up the ability to explain what
the model is doing.

---

# ▶ SLIDE 4 — The Flexibility–Interpretability Tradeoff

This slide has two columns, and it's one of the few genuine side-by-side comparisons in the
course.

On the left are the **linear models**, like regression and ARIMA. Each coefficient has a clear
meaning. You can say, "A one-point rise in unemployment lowers sales by four hundred units," and
everyone in the room knows what you mean. Their properties are stable and well understood.

But they assume one constant slope everywhere. The effect of unemployment going from four percent
to five percent is assumed to be identical to the effect of it going from nine percent to ten
percent. That's an assumption, and it's usually false. So linear models miss curves and threshold
effects entirely.

**[pause]**

On the right are the **fully nonlinear models**, like trees and neural networks. They capture
almost any pattern, and on complex data they're accurate.

But ask why one specific prediction came out the way it did, and you'll struggle. And because
interactions entangle the variables, you can't cleanly separate what any single variable
contributes.

**[pause]**

So GAMs sit **between the two**. Each variable gets its own flexible, nonlinear curve. But the
model stays additive, meaning the effects are added together rather than multiplied or entangled,
so each one can still be plotted and explained on its own.

That's the whole bargain. The rest of this lecture is about what it costs, because it does cost
something.

---

# ▶ SLIDE 5 — Why Additivity Matters for Interpretation

Let me make the additivity point concrete, because it's easy to nod along without seeing what's
at stake.

In an additive model, each variable's effect is the same whatever the other variables are doing.
In a model that isn't additive, that's no longer true, and then you can't describe the variables
separately.

*(Point at the left panel.)*

This is Lab 3's daily electricity load for P-J-M East, the data you'll use in the lab, averaged
month by month. The red line is weekdays, and the blue line is weekends. Weekends are lower in
every month, because offices and factories are closed.

But look at the size of the gap. In July, weekdays run about four thousand and sixty megawatts
above weekends. In December, the gap is about two thousand. So the weekend effect is twice as big
in summer as in winter. Here's the likely reason, although the data alone can't prove it. Office
and shop air conditioning runs on weekdays and stops at the weekend, while much of the winter
load is home heating, and homes are occupied every day.

So the question "what does the weekend do to demand?" has no single answer. The answer begins
with "it depends on the month."

**[pause]**

*(Point at the right panel.)*

Now fit an additive model, a month effect plus a weekend effect, and here's everything it can
draw: two curves, exactly parallel. There's one weekend effect, about two thousand nine hundred
and sixty megawatts, in every month of the year. That's too big in December and too small in
July.

You can't blame that bad fit on the estimator. It comes from the structure. In an additive model,
the effect of one variable can't depend on another, so the curves have to be parallel.

And that's the trade, both halves of it. The price is that an additive model can't represent a
gap that changes with the month. The gift is that, because the effects don't depend on each other,
each one is a single curve: one curve for the month, and one number for the weekend. You can plot
each one on its own and hand it to a stakeholder, and every plot is true whatever the other
variable is doing.

**[pause]**

If you want the calculus version, here it is. In a GAM, the marginal effect of x-one is f-one
prime of x-one, which is a function of x-one alone. In a model with x-one times x-two in it, the
derivative contains x-two. It's the same idea as the parallel curves.

We'll come back to the price at the end of the lecture, and next week, in Lecture 4, trees pay it
for us.

---

# ▶ SLIDE 6 — Section divider: GAM Structure and Smoothing

The second section covers the structure of a GAM, which is a set of additive nonlinear functions,
penalized for wiggliness.

---

# ▶ SLIDE 7 — What Is a GAM?

Here's the definition: y equals a constant, alpha, plus f-one of x-one, plus f-two of x-two, and
so on, plus an error term. Each f is a smooth nonlinear function estimated from the data.

Compare that with linear regression, where each term is beta-j *times* x-j. We've replaced a
coefficient with a function, and that's the entire generalization.

Each f-j is free to be linear, curved, U-shaped, or threshold-like. You don't specify the shape.
The data decides.

**[pause]**

Because the terms are *added* rather than multiplied, the effects stay independent, and each one
can be plotted on its own. And every function is fitted to minimize prediction error, subject to
a smoothness penalty. That penalty is two slides from now, and it's what stops the curves from
chasing noise.

Here's a retail example, to make it concrete.

Model weekly sales as f-one of time, which captures the long-run trend, plus f-two of week of
year, which captures seasonality, plus f-three of unemployment, which captures economic
sensitivity.

That gives you three curves, and you plot and interpret each one separately. None of them needs
you to hold the others fixed to make sense of it.

**[pause]**

Notice that this is the same decomposition Prophet will do later in this lecture: trend,
seasonality and external effects. Prophet is a GAM with the time series structure built in.

---

# ▶ SLIDE 8 — What the "Generalized" Actually Means

Someone should have asked this as soon as the definition went up, so let me answer it before we
go on.

The method is called a **generalized** additive model, but so far I've only shown you an additive
model. So where does the "generalized" come in?

Everything we've modeled so far has been continuous and roughly symmetric, like sales in dollars
or demand in units. For that kind of outcome, adding smooth functions together works directly.

But suppose your outcome is a **count**, like units sold at a small store: zero, one, two, seven.
An additive model can happily predict minus three units, which isn't a thing.

Or suppose it's **binary**. Did this customer churn? Did we stock out? The answer is a
probability between zero and one, and again an additive model will cheerfully predict one point
four.

**[pause]**

The fix is on the slide. Instead of modeling the mean directly, we model a **transformation** of
the mean: g of the expected value of y equals the additive predictor.

That g is called the link function, and it's the entire "generalized."

Now look at the table. For continuous, symmetric data, the distribution is Gaussian and the link
is the identity, which means do nothing. That's the case we've been in all along. For counts, the
distribution is Poisson and the link is the log. You model the log of the mean and exponentiate
to get back, so the forecast can never go negative. For binary outcomes, the distribution is
binomial and the link is the logit, which maps the whole real line into the range from zero to
one, so the probability stays inside that range by construction.

**[pause]**

In pyGAM, this is one word: `LinearGAM`, `PoissonGAM` or `LogisticGAM`. The additive structure,
the splines and the smoothing penalty are all the same. Only the link changes, and everything
else in this lecture is unchanged.

And if this sounds familiar from a regression course, it should. It's exactly the same idea as a
generalized *linear* model. A GAM is a G-L-M where the linear terms have been replaced by smooth
functions. That's the whole family tree in one sentence.

---

# ▶ SLIDE 9 — Controlling Smoothness: The Penalty Term

If each f can be any shape, what stops it from being a squiggle that passes through every data
point? That's what this slide answers.

A GAM minimizes a **penalized** sum of squared errors, and it has two terms. The first is the
ordinary fit term, the squared errors, exactly as in O-L-S. The second is the penalty: lambda
times the integral of the *squared second derivative*.

The second derivative measures bending, meaning how fast the slope is changing. A straight line
has a second derivative of zero everywhere.

*(Point at the strips.)*

Now look at the thin strips under each panel. Each one shades the squared bending of the curve
above it, along the year, and the shaded area *is* the penalty. So you can read the penalty as an
area rather than a formula. Strictly speaking, pyGAM charges a discrete version of this integral,
the squared differences between neighboring spline coefficients, but both versions are zero for
exactly the same curves, which are straight lines.

Now for the three panels. Every one shows the same data, a year of daily electricity demand from
twenty fourteen, against day of year. Every one uses the same family of curves, a hundred
splines, which are cubic pieces joined at knots. The only thing that changes is lambda, the price
of bending.

**[pause]**

*(Point at the left panel.)*

On the left, lambda is near zero. Bending is free, so the curve chases every wiggle: every cold
snap, every heat wave and every weekend dip. It uses about a hundred effective degrees of freedom.
That's the e-dof number above each panel, and it counts the parameters the curve really uses.
Here, that's basically all of them. Its strip is off the scale, with eighty times the area of the
middle one.

*(Point at the right panel.)*

On the right, lambda is huge. Any bending at all is infinitely expensive, so the curve can't bend,
and what's left is a straight line. And it isn't just any line. It's exactly the ordinary least
squares line. Its strip is empty, because a straight line never bends. So O-L-S isn't a different
method from a GAM. It's a GAM with the smoothness dial turned all the way up.

There's one technical aside here. This only works because I used an ordinary spline basis for
this picture. The lab uses a *cyclic* spline, which forces December thirty-first to meet January
first. A straight line can't do that unless it's flat, so with a cyclic spline the limit is a flat
line, not the O-L-S line.

*(Point at the middle panel.)*

In the middle, lambda is chosen by G-C-V, which stands for generalized cross-validation, and not
by eye. You'll hear that refrain all semester: the tuning parameter is selected by an estimate of
out-of-sample error, never by what looks nice. G-C-V computes that estimate from the fit itself.
It's a shortcut to leave-one-out error, with nothing actually held out. This curve uses
forty-three effective degrees of freedom. It keeps the two humps, heating and cooling, and drops
most of the noise.

The proof is in the panel titles, where each curve is scored on the *next* year, twenty fifteen.
The wiggly fit misses by thirty-nine forty-six. The straight line misses by forty-six twenty. And
the G-C-V fit misses by thirty-six twenty-one, the best of the three. The noise the left panel
memorized doesn't come back the next year.

To be fair to the picture, G-C-V still keeps some week-long cold snaps that won't repeat. That's
because it treats neighboring days as independent, and they aren't. I checked every year from
twenty ten to twenty sixteen the same way. The G-C-V fit beat the lambda-near-zero fit in seven
years out of seven, and the straight line lost to it in six out of seven.

**[pause]**

Let me say a little more about splines, because "piecewise polynomials joined at knots" is one of
those phrases that sounds like an answer without being one.

Here's the picture. Take the range of your predictor, and put down a handful of points along it.
Those are the knots. Between each pair of knots, fit a low-order polynomial, usually a cubic. Then
require that, where two pieces meet, they join up, with the same value, the same slope and the
same curvature. That's what makes the result look like one smooth curve rather than a chain of
segments.

The number of knots controls how much wiggle is *available*. With few knots, the curve can only
bend gently. With many knots, it can bend almost anywhere.

Now, you might expect that choosing the number of knots is the crucial modeling decision. It
mostly isn't, and this is the elegant part of the penalized approach. You give the model plenty of
knots, more than you think you need, and then you let **lambda** control how much of that
available flexibility actually gets used.

So the knots set the ceiling, and lambda sets where you land under the ceiling. That's why you
tune lambda, by G-C-V or cross-validation, and generally leave the knot count alone. It's one
tuning decision instead of two, and it's the one that's easier to make honestly.

---

# ▶ SLIDE 10 — Section divider: Prophet for Time-Series GAMs

The third section is about Prophet, a GAM built specifically for business time series.

---

# ▶ SLIDE 11 — Prophet: Forecasting at Scale

Taylor and Letham built Prophet at Facebook, which is now Meta, for forecasting at large scale.
It's a GAM in time series clothing, and you can see that in the decomposition.

Written out, y at time t equals a trend, plus seasonality, plus holidays, plus noise. That's three
additive components. It's the GAM structure from a few slides ago, with the terms chosen for what
business time series actually contain.

The first component is the **trend**, which is piecewise linear or logistic, with changepoints.
Those are specific dates where the slope is allowed to shift. That matters, because real business
series do change direction: a new competitor arrives, a store is renovated, or a pandemic starts.

The second component is the seasonality, which is a Fourier series, meaning a sum of sine waves.
That's how Prophet handles weekly and yearly cycles at the same time, which none of our methods
from Lecture 1, on exponential smoothing, or Lecture 2, on ARIMA, could do. I'll say more about
how many waves in a moment.

The third component is the holidays, which are irregular events you supply yourself, like Black
Friday, a product launch, or a lockdown. These aren't periodic, so no seasonal term captures them,
and you tell Prophet the dates directly.

**[pause]**

The design goals were practical rather than statistical. Prophet takes one fit-and-forecast call,
it's robust to missing data and outliers, and it's interpretable, because every component plots on
its own. In other words, it was built for analysts rather than statisticians, which is why it
spread so quickly among analysts who aren't statisticians themselves.

Now let me be honest about its track record, because you'll hear Prophet talked about as if it
were state of the art, and it isn't.

In head-to-head comparisons, Prophet frequently **loses** to a well-tuned ARIMA or ETS on forecast
error. Our own textbook says so directly. The authors note that Prophet rarely gives better
accuracy than the alternatives. It didn't win the M-competitions, and it doesn't top benchmark
leaderboards.

So why is it in this course, and why is it everywhere in industry?

It's because accuracy isn't the only thing being optimized. Consider what Prophet actually gives
an analyst. It's a defensible forecast in twenty minutes, from a series with gaps and outliers,
with a component plot you can put in front of a manager who will ask what drove it.

Getting an ARIMA to that same point takes stationarity testing, order identification, residual
checking, and someone who can read a P-A-C-F, the partial autocorrelation plot from Lecture 2. If
you have that person, use ARIMA and get the better number. But if you're one analyst covering four
hundred product lines, you don't have that person four hundred times.

**[pause]**

That's the real lesson, and it goes wider than Prophet. The best model in a paper and the best
model in an organization are **frequently different models**, and the gap between them is usually
maintenance cost and explainability, not accuracy. Part of your job is knowing which one you're
being asked for.

There's one knob to know about, `changepoint_prior_scale`. A high value means the trend adapts
quickly and risks overfitting. A low value means a stable trend. It's the main thing you'll tune.

---

# ▶ SLIDE 12 — Prophet's Seasonality: How Many Waves?

So what does "a sum of sine waves" actually buy you? Here it is on the lab's electricity data.

*(Point at the gray dots.)*

Each dot is the average load on one day of the year, over the lab's five and a half years. Look at
the shape, because it isn't a wave. There's a winter hump in January, from heating. There's a
trough in April, when nobody heats or cools. There's a bigger peak in July, from air conditioning.
And there's a second trough in October.

**[pause]**

*(Point at the blue line.)*

The blue line is the best single wave, one sine curve with a period of one year. One wave has one
peak a year, so it has to choose. It can't be high in January *and* high in July. So it ends up
nearly flat, and its error is about three thousand four hundred megawatts.

*(Point at the red line.)*

The red line adds a second wave with half the period, a hundred and eighty-two and a half days, so
it repeats twice a year. Now there are two peaks and two troughs, and the fit follows the dots.
The error falls to about thirteen hundred and fifty, which is a factor of two and a half.

*(Point at the bottom strip.)*

The strip underneath shows the two waves separately. The solid black one is the slow yearly wave.
The dashed red one is the fast half-year wave, and it does most of the work. Add them to the
average, and you get the red curve on top.

**[pause]**

That's all Prophet's yearly seasonality is: a sum of these sine-and-cosine pairs. The number of
pairs is called the Fourier order, and Prophet's default for the yearly term is ten. More pairs
allow more wiggles. In the lab, you'll force the order down to one and see what happens on the
test year.

---

# ▶ SLIDE 13 — Prophet: Code and Workflow

The code really is this short, and it's exactly what you'll run in the lab.

Prophet wants a data frame with two columns, `ds` for the dates and `y` for the values, and it
insists on those names. You create the model, say which cycles you want, and fit it. Here yearly
and weekly are on, and daily is off, because we have one row per day. Then you call `predict` on
the test dates, and you have a forecast.

**[pause]**

But the call that matters for this course is the last one, `plot_components`.

*(Point at the three panels.)*

This is what it shows for the lab's model: three panels, one for each additive term.

On the left is the trend, a piecewise straight line that bends at the changepoints. The shaded
strip at the right is the lab's test year.

In the middle is the weekly part of the seasonality. Weekdays sit a little above zero, Saturday is
about eighteen hundred megawatts below, and Sunday is about twenty-five hundred below.

On the right is the yearly part: the winter hump, the April trough, the big summer peak, and the
October trough. It's the same double-peaked shape you saw on the previous slide. Nobody told
Prophet about heating and cooling. The ten waves had enough room to find it.

Add the three panels together, and you have the forecast. It's the additive structure made
visible, because you're literally looking at the terms of the equation, one at a time.

That plot is what you put in front of a manager, not the forecast line. The components let them
check your model against their own knowledge of the business. They might say, "Yes, the summer
peak looks right. No, that changepoint in March is wrong, because that was the fire." That
conversation is worth more than a small RMSE improvement.

**[pause]**

So when should you use which tool? Reach for Prophet on seasonal series with known holiday
effects, especially when someone will ask you to explain the forecast.

Reach for pyGAM instead when you want to see the shape of a few non-calendar drivers, which is
where we're going next. I mean a few drivers, not many correlated ones. That's the case where GAM
curves stop being trustworthy, as you'll see in a minute.

---

# ▶ SLIDE 14 — Section divider: pyGAM for Generalized Additive Modeling

The fourth section is about pyGAM, for when you have several predictors and want a flexible effect
for each one.

---

# ▶ SLIDE 15 — pyGAM: Flexible Additive Modeling

Prophet is specialized for time structure. pyGAM is the general case: you specify a smooth
function for each predictor, and it fits the whole additive model.

The code shows the pattern. You import, and then you build the model by *adding terms together*,
which is a rather nice piece of design, because the code mirrors the equation.

There are three term types, and they're the whole vocabulary.

The first, s of j, is a smooth spline of feature j. It's the flexible nonlinear effect, it's the
default, and it's the reason you're using a GAM.

The second, l of j, forces a straight line. It's worth using deliberately when theory says the
effect should be linear, so that you don't spend flexibility where you don't need it.

The third, f of j, handles a binary or categorical feature, which is called a factor.

Then `gridsearch` tunes lambda by G-C-V, the same in-sample shortcut to leave-one-out error you saw
on the penalty slide, so you don't pick it yourself. The last two lines pull out the partial
dependence for the first term, with a ninety-five percent band, and make predictions on new data.

**[pause]**

There's one habit I'd ask you to build. After `gridsearch` returns, **look at the curves**. The
search optimizes an estimate of held-out error, and it will happily give you a curve that's
statistically optimal and economically absurd, such as sales rising as unemployment rises. When
that happens, the model isn't broken. It's telling you that something is confounded. But you only
find out if you look.

There's also a specific failure worth naming, because it has a name and it will bite you. It's
called **concurvity**.

You know multicollinearity from regression. Two predictors are so correlated that the model can't
tell which one is doing the work, so the coefficients go unstable and flip sign between samples.

Concurvity is the same disease in a GAM, but worse, because the terms are *smooth functions*
rather than straight lines. One smooth term can mimic another, even when the raw variables aren't
strongly correlated at all. A smooth function of time and a smooth function of a trending economic
indicator can trace nearly the same shape, and the model has no principled way to split the credit
between them.

**[pause]**

The symptom is a partial dependence plot that looks wrong or wildly uncertain, and that changes a
lot when you drop a variable that seems unrelated.

The reason it matters more here than in ordinary regression is that GAMs are sold on
interpretability. Under concurvity, the individual curves become unreliable, *even when the
overall forecast is fine*. So you get a model that predicts well and explains badly, which is
precisely the thing you chose a GAM to avoid.

pyGAM won't warn you. To check, drop terms, and see whether the remaining curves hold their shape.

---

# ▶ SLIDE 16 — Interpreting GAM Results: Partial Dependence Plots

This is the payoff slide, because it's the thing you actually deliver.

A partial dependence plot shows one estimated function, f-hat-j, across the range of values you
observed, with a confidence band around it. Reading the shape is most of the interpretation. And
this is a real one. It's Lab 3's model of daily electricity demand, with a smooth term in day of
year plus a factor for day of week, and you'll fit it yourself.

*(Point at the left panel.)*

The left panel is day of year. Demand is high in January, falls to a trough in late April, climbs
to the annual peak in mid-July, and drops again in the fall. So there are two peaks, heating in
winter and air conditioning in summer. Nobody told the model that. It bent its way there.

Now think back to the right-hand panel on the penalty slide, the straight line. A linear term in
day of year would have reported a gentle slope and concluded that the calendar hardly matters,
which is exactly backwards. A flat line, a rising curve, a U-shape and a double hump are all
shapes you'll meet, and a linear term can only ever show you a straight line, whether it's rising,
falling or flat.

*(Point at the right panel.)*

The right panel is day of week. The weekdays sit together, and Saturday and Sunday drop. The bars
are each day's band.

**[pause]**

Watch the band as well as the curve. Here it's about the same width all year, and that's worth
noticing, because it isn't the usual picture. Every day of the year appears four or five times in
the training data, and a cyclic spline has no ends, so the model is roughly equally sure
everywhere. On a predictor whose extremes are rare, such as unemployment above ten percent, the
band fans out exactly where the data thin, which is where a reader is most tempted to extrapolate.
If the band is wide, the curve there is a guess, so say so.

**[pause]**

And this is what makes a GAM worth its cost in accuracy. The output isn't a number. It's a
sentence someone can act on.

*(Point at the key box.)*

"Plan for mid-July demand about eleven thousand four hundred megawatts above the late-April
trough, a winter peak about eighty-five hundred above it, and weekends about three thousand below
weekdays."

Notice that the sentence talks in *gaps*, peak above trough, and the brackets on the plot draw
those gaps. That's deliberate. The zero line of a smooth term is a bookkeeping choice, so the
differences are what mean something.

And the last clause on the slide is the point: no straight line in day of year has two peaks.

---

# ▶ SLIDE 17 — Section divider: Key Takeaways and Roadmap

Let's pull it all together, and set up where the course goes next. The line on this divider says
it in brief: GAMs are flexible, interpretable, and underutilized in business forecasting.

---

# ▶ SLIDE 18 — GAMs vs. Other Methods: When to Choose Each

This table is the honest summary of the course so far, and it's worth reading down the columns
rather than across.

ARIMA and ETS aren't nonlinear, don't handle interactions, are highly interpretable, and are built
for univariate time series. Linear regression is the same, except that it handles interactions
manually, if you build them yourself, and it's for a few predictors.

The GAM row reads like this. Nonlinear, **yes**. Interactions, only the ones you specify.
Interpretable, high. And it's for a few predictors, when you need to explain their effects.

Random forest is nonlinear and finds interactions, with medium interpretability, and it's for many
predictors in tabular data. XGBoost is the same, but with low interpretability, and it's for many
predictors when accuracy comes first.

**[pause]**

The column that usually decides it is **Interactions**.

A GAM is additive by construction. That's the source of its interpretability, and it's also a real
limit, though not an absolute one. If the effect of price genuinely depends on whether it's a
holiday week, a GAM can represent that only if you add the interaction term yourself. That could
be a tensor-product smooth, which pyGAM calls t-e, or a smooth of price fitted separately for
holiday and non-holiday weeks. Either way, you have to suspect the interaction first. A tree finds
it on its own. That's exactly what a tree does, and it's why the next two lectures, on decision
trees and tree ensembles, exist.

So the additivity is a bargain, not a free lunch. You buy explainability with the ability to find
interactions automatically.

**[pause]**

So choose a GAM when you must explain each predictor's effect, and when those effects are curved
rather than straight. Choose one when a time series has complex trend and seasonality, and in that
case, reach for Prophet. And choose one when interpretability matters more than the last few
points of accuracy.

That last condition is the honest one. Whether a more flexible model actually buys those last few
points is an empirical question, and Lectures 4 through 6 test it on our data. The answer isn't
always the one the reputation predicts.

---

# ▶ SLIDE 19 — Lecture 3: Key Takeaways

Here are the five takeaways.

First, GAMs give you nonlinear flexibility while keeping interpretability, because the model is
additive. The model says that y equals alpha plus a sum of smooth functions, so each variable's
effect can be plotted and explained on its own.

Second, the smoothness penalty prices bending, and that's what prevents overfitting. Lambda trades
fit against smoothness, and it's chosen by G-C-V or cross-validation, never by eye.

Third, Prophet is a GAM for time series. It decomposes a series into trend, seasonality and
holidays, which makes it ideal for business series with seasonal patterns and known irregular
events.

Fourth, pyGAM handles the general case, with s for smooth terms, l for linear terms, f for
categorical terms, and `gridsearch` to tune lambda.

The last one is that partial dependence plots are what make the results actionable. They turn the
model into sentences a stakeholder can act on, and they're what you deliver.

**[pause]**

And there's a bigger point, which matters more than any of the five. This is the last method in
the course that is both genuinely nonlinear and genuinely explainable. From next week, every
increase in flexibility costs you something in interpretability, and part of your job as a
forecaster is deciding when that trade is worth making.

---

# ▶ SLIDE 20 — References

The two works on this slide are Simon Wood's book on generalized additive models, and Taylor and
Letham's paper introducing Prophet, called Forecasting at Scale. Your readings are in the
syllabus. In the textbook, they're sections seven point four and seven point seven, for useful
predictors and regression splines, and sections twelve point one and twelve point two, for complex
seasonality and Prophet.

Next week is Lecture 4, on decision trees. We move to tree-based methods, and the first thing we
gain is exactly what we gave up today: interactions, captured automatically.

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
