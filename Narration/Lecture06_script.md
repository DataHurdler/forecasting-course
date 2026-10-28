# Lecture 6 — Recording Script

**ECON 8310: Business Forecasting · Regularization & Model Selection**

Deck: `Slides/Lecture06_Regularization.pdf` (21 pages) · Measured runtime: see the timing guide

---

## How to use this document

- **`▶ SLIDE n — Title`** marks where to advance. The number is the PDF page.
- *Italic parentheticals* are stage directions. **[pause]** means stop for a beat.
- This lecture closes the first half of the course. The callbacks to Lectures 3, 4, and 5 are
  load-bearing — they are what turn six weeks of separate methods into one idea.

---

# ▶ SLIDE 1 — Title page

This is Lecture 6, on regularization and model selection.

This lecture closes the first half of the course, and it has a job beyond its own content. For
five weeks we've been collecting methods: exponential smoothing, ARIMA, GAMs, trees, and tree
ensembles. Today I want to show you that several of them have been doing the same thing in
different clothing.

**[pause]**

Here's the setup. Everything so far has assumed that you know which predictors belong in the
model, and you don't. In practice, you sit down with a weekly sales series and you can generate
predictors almost without limit. There are lags one through fifty-two, week-of-year dummies,
holiday flags, promotions, prices, competitor prices, macro series, and every interaction you can
think of.

Feature engineering is cheap, but estimating what you engineered is not.

So today's question is this: when you have more predictors than you can safely estimate, what
decides which ones survive? The answer turns out to be the same one we already gave for GAMs, in
Lecture 3, and for trees, in Lecture 4.

---

# ▶ SLIDE 2 — Lecture Outline

We'll take this in five parts. First comes the problem, which is why too many predictors break
ordinary least squares. Second come the shrinkage methods: Ridge, LASSO, and Elastic Net. Third is
choosing the penalty, which is where most people go wrong. Fourth is how to do all of this in
practice, in Python. And last come the key takeaways and the road ahead, including a map of which
method to reach for when.

---

# ▶ SLIDE 3 — Section divider: The Problem — Too Many Predictors

Let's begin with part one, the problem of too many predictors. Feature engineering gives us more
predictors than we can safely estimate, so something has to decide which ones survive.

---

# ▶ SLIDE 4 — Why Regularize?

Start with what OLS actually does. It minimizes the sum of squared residuals, and nothing else. It
has no opinion about how many predictors you handed it, or about how large the coefficients get.

So let's hand it a lot. Lab 6 builds the feature set an analyst would actually build for daily
sales: twenty-eight lags, rolling means, Fourier terms, and day-of-week dummies. It ends up with
sixty-two predictors without anyone setting out to build that many. Then it fits OLS on shorter
and shorter training windows, and it scores every fit on the same test year.

**[pause]**

*(Point at the horizontal axis.)*

Across the bottom is the number of training rows per predictor, n over k. On the left it's
nineteen and a half, which is twelve hundred and ten rows for sixty-two predictors. As you move
right, the window shrinks to four hundred rows, then two hundred, then a hundred and twenty, and
then eighty. By the right edge, there's barely more than one row per predictor.

Up the side is the test RMSE, and notice that the scale is logarithmic. Each gridline is ten
times the one below it.

*(Trace the red line.)*

OLS is in red. With plenty of data it's fine, at two hundred eighty. At six and a half rows per
predictor, it's already above two thousand. At three, it's nearly six thousand. And at under two
rows per predictor, it's **three point six million**, on a series that sells nearly three thousand
units a day.

Nothing about the problem changed. It's the same features, the same target, and the same test
year. Only the ratio of rows to parameters changed.

*(Trace the blue line.)*

Now look at Ridge, in blue. It uses the same features, with one line of code changed. It starts
at two seventy-eight, and at the right edge it's at seven hundred seventy. That's a little worse
than the dashed benchmark, same day last year, at six twenty-five, but it's in the same world.
Ridge degrades gracefully across the whole range.

**[pause]**

That gap, between a red line that leaves the chart and a blue line that barely moves, is what this
lecture is about. You'll reproduce every point of it yourself in Lab 6.

---

# ▶ SLIDE 5 — Regularization: Fit Plus a Penalty

So why does OLS blow up like that? Three things go wrong, and they're worth separating, because
they have different fixes.

The situation isn't exotic, either. A weekly model with lags, calendar dummies, promotions, and a
couple of macro series easily carries sixty predictors, and three years of weekly history is only
a hundred and fifty-six observations. That's the right-hand side of the chart you just saw.

The first problem is that variance grows with k. Every predictor you add gives the model another
dimension in which to chase noise. All of the coefficients get estimated less precisely, including
the ones you cared about.

The second problem is that correlated predictors produce unstable coefficients, and this is the one
you'll actually see in your output. Put lag one and lag two in the same model. They move together,
almost by construction, so OLS can't tell which one deserves the credit. You get a coefficient of
plus eight hundred on one and minus seven hundred and forty on the other, and they nearly cancel.
The fit is fine, but the individual numbers are nonsense. If you report them to a stakeholder,
you're reporting noise with a decimal point on it.

The third problem is that once k exceeds n, there's no unique solution at all. It isn't that the
estimate is bad. There is no estimate. Infinitely many coefficient vectors fit the training data
perfectly.

**[pause]**

Regularization's answer is to stop minimizing fit alone, and to minimize fit *plus a penalty on the
size of the coefficients*.

Look at the shape of that expression on the slide, because you've seen it before. There's fit on
the left, a complexity penalty on the right, and lambda setting the exchange rate between them.

That's the GAM smoothing penalty from Lecture 3. It's also cost-complexity pruning from Lecture 4.
It's the same skeleton with a different penalty, and by the end of today I'll show you that it's
been hiding in one more place you didn't expect.

And the bargain is always the same: you accept a little bias to buy a large reduction in variance.
For a forecast, where all you care about is the error on data you haven't seen, that trade is
almost always worth making.

---

# ▶ SLIDE 6 — Subset Selection: The Classical Answer

Before shrinkage, the standard approach was to *count* predictors rather than shrink them. It's
worth understanding properly, both because you'll meet it and because the way it fails is
instructive.

Best subset selection fits every possible model, all two-to-the-k of them, and picks the winner.
It's exact. It's also computationally hopeless past about thirty predictors, since two to the
thirtieth power is over a billion models.

Stepwise selection is its practical cousin. You add the strongest predictor, or drop the weakest
one, one at a time. It's tractable, and you'll notice that it's the same greedy logic as CART, from
Lecture 4, with the same weakness. It can miss the best combination entirely, because it never
reconsiders an earlier decision.

Either way, you need a criterion to compare models of different sizes, such as adjusted R-squared,
AIC and its corrected version AICc, BIC, or cross-validation.

**[pause]**

Now look at the warning box, because it holds the real objection, and it's a subtle one.

Subset selection is **discrete**. A predictor is either in or out, and there's no middle ground.

That means small changes in the data flip predictors in and out of the model. Add three weeks of
data, rerun it, and you get a different set of variables, sometimes a substantially different set.

So the *selected model* is itself a high-variance object.

Stay with that for a moment, because it's easy to miss. You adopted subset selection to reduce
variance. And you did reduce the variance of the coefficients, given the model you picked. But you
introduced variance in *which model you chose*, and that variance is invisible. It doesn't show up
in any standard error your software prints.

You haven't removed the instability. You've moved it somewhere harder to see.

**[pause]**

This is the approach in section seven point five of FPP, Forecasting: Principles and Practice,
which is your reading this week along with chapter six of ISLP. It's still taught, and it's still
reasonable when you have a handful of candidate predictors and you need a story you can defend.

Shrinkage replaces the in-or-out switch with a continuous dial, which is more stable and easier to
tune.

---

# ▶ SLIDE 7 — Section divider: Shrinkage Methods

That brings us to part two, the shrinkage methods. The core idea fits in one line: keep every
predictor, but pull its coefficient toward zero.

---

# ▶ SLIDE 8 — Ridge Regression (ℓ₂)

Ridge adds lambda times the sum of the squared coefficients.

Think about what that does. Squaring means large coefficients are punished disproportionately. A
coefficient of ten contributes a hundred to the penalty, while ten coefficients of one contribute
ten in total. So Ridge has a strong preference for spreading an effect around rather than
concentrating it.

At lambda equals zero, you have OLS exactly. As lambda goes to infinity, every coefficient goes to
zero. Everything useful happens in between.

**[pause]**

Ridge shrinks every coefficient toward zero, but it sets **none** of them exactly to zero. Hold on
to that, because it's the whole contrast with the next slide.

Its particular strength is exactly the failure I described on slide five. Remember the plus eight
hundred and the minus seven hundred and forty? Ridge won't do that. Rather than letting two
collinear variables fight over the same coefficient, it splits the effect between them, so you get
something like plus thirty and plus twenty-eight instead. That's far more stable, and far more
honest about what the data can actually distinguish.

So use Ridge when you believe most of your predictors carry a little signal, and you want to keep
them all.

**[pause]**

Now the practical warning, and I want to be emphatic about it, because it's the single most common
implementation error with any of these methods. **Standardize first.**

The penalty isn't scale-invariant. Lambda times beta squared depends entirely on what units beta is
measured in. Measure a predictor in dollars, and its coefficient is small, so it's barely
penalized. Measure the same predictor in thousands of dollars, and its coefficient is a thousand
times larger, so it's penalized a million times harder.

It's the same variable, carrying the same information, and it gets completely different treatment.

Get this wrong, and your *units* decide which variables survive, not your data. We'll see the
right way to do it on the code slide, and it isn't what most people do first.

---

# ▶ SLIDE 9 — LASSO (ℓ₁)

Now swap the squared penalty for an absolute one, so you take the sum of absolute values instead
of the sum of squares.

That looks like a small change, but it isn't. The behavior changes in kind, not just in degree.

LASSO sets some coefficients **exactly** to zero. They aren't just small. They're zero. That means
it performs selection and estimation in a single step, and it leaves you with a sparse model you
can actually read aloud to a stakeholder.

That last part matters more than it sounds. A model with eight nonzero coefficients is something
you can put on a slide and defend in a meeting. A model with sixty small ones is not, even if its
RMSE is a hair better.

So use LASSO when you believe only a handful of predictors genuinely matter, and you want the model
to tell you which ones.

**[pause]**

But be very careful about what you conclude, because this is where LASSO gets misused.

Among a group of highly correlated predictors, LASSO keeps **one**, more or less arbitrarily, and
zeros out the rest.

So suppose lag one, lag two, and lag four are all correlated and all genuinely informative, and
LASSO keeps lag four. It's tempting, and I've seen this in professional work, to report that lag
four is "the important lag."

It isn't. It's the one that happened to win a coin toss between near-identical variables. Rerun it
on slightly different data, and lag one wins instead.

The variable that survived isn't the important one. It's the survivor. Those are different claims,
and only one of them is supported.

---

# ▶ SLIDE 10 — Why LASSO Zeros Out and Ridge Does Not

So why does one zero out while the other doesn't? The geometry answers it, and today we'll draw it
on our own data rather than on a cartoon.

Both methods can be written the same way: minimize squared error subject to a *budget* on
coefficient size. You can spend the budget however you like, but you can't exceed it. Every lambda
in the penalty form corresponds to some budget t, so this is the same problem stated the other way
around.

The **shape** of that budget is the entire difference.

*(Point at the figure, axes first.)*

This is Lab 6's data cut down to two predictors so that we can draw it: daily units regressed on
lag seven and lag fourteen, with everything standardized. The horizontal axis is the coefficient on
lag seven, and the vertical axis is the coefficient on lag fourteen. The black dot at the top right
is OLS, at about point five on lag seven and point three four on lag fourteen.

The gray rings around it are ellipses of equal squared error. Every point on one ring fits the data
equally well, and the fit gets worse as the rings grow outward. They're tilted because the two lags
are correlated, at about point seven five.

*(Point at the left panel.)*

On the left is Ridge. A budget on the sum of squared coefficients is a **disc** in two dimensions,
smooth and without corners. Grow the ellipses out from OLS until one first touches the disc. That's
the black curve, and the touch point is the blue dot, at about point oh seven on *each* lag. Both
lags are kept. The blue line shows where the solution goes as you loosen the budget, all the way
out to OLS, and it never touches an axis. So Ridge coefficients get small, but they don't get to
zero.

*(Point at the right panel.)*

On the right is LASSO. A budget on the sum of absolute values is a **diamond**, and a diamond has
corners sitting exactly on the axes. The ellipses are the same, but now the touch point is the
corner, where lag fourteen is exactly zero. Follow the red line. As the budget grows, the solution
slides out along the lag-seven axis with lag fourteen pinned at zero, until t reaches about point
one six. Only then does lag fourteen enter, and the solution heads straight for OLS.

That's the whole mechanism. A corner is a point where one coefficient is zero.

**[pause]**

Now look closely at the black curve on the right, because I want to be honest about it. It runs
almost exactly along the diamond's face. The corner wins here, but only just, and only while the
budget is small: below point one six, which is about a fifth of OLS's own total of point eight
four. The reason is that lag seven and lag fourteen are nearly interchangeable. Their correlations
with sales are point seven five and point seven two, so lag seven wins by a hair. That's the coin
toss from the previous slide, drawn.

So the textbook line, that the contours are likely to hit a corner first, needs a qualifier.
Corners win when the budget is tight. Loosen it, and LASSO keeps both.

That leads to the point about interpretation that I most want you to take from this slide.

The sparsity of LASSO is a **geometric accident** of the L-one ball having corners. It isn't a
statistical test of significance.

When LASSO zeros a coefficient, it's saying, "This predictor isn't worth its share of the budget at
this lambda." It is *not* saying, "This predictor has been proven irrelevant." There's no null
hypothesis anywhere in what we just did, and no p-value.

**[pause]**

I labor this because LASSO output gets read as if it were a significance table, in papers and in
business reports, and it isn't one. James and his co-authors work through the geometry carefully
in section six point two of ISLP, for any number of predictors.

---

# ▶ SLIDE 11 — Elastic Net: Both Penalties at Once

Elastic Net asks why you should have to choose, and it uses both penalties.

That gives you two knobs now, so keep them straight. **Lambda** sets the total amount of penalty.
**Alpha** sets the mix between the two types. Alpha equals one is pure LASSO, alpha equals zero is
pure Ridge, and anything in between is a blend.

What you get is LASSO's ability to zero predictors out, plus Ridge's **grouping effect**:
correlated predictors now enter or leave the model *together*, instead of one surviving
arbitrarily.

That's the direct fix for the coin-toss problem from two slides back.

**[pause]**

For forecasting, this is usually the **right default**, and the reason is structural rather than a
matter of taste.

Our predictors are lagged values of the same series. Lag one and lag two are correlated almost by
construction, because they're the same series shifted by one period.

That's precisely the situation where pure LASSO picks a winner at random and pure Ridge refuses to
simplify anything. Elastic Net is built for it.

So if you want a default for the homework, start with Elastic Net, tune the mix, and let the data
tell you how far toward LASSO or toward Ridge it wants to sit.

---

# ▶ SLIDE 12 — The Grouping Effect, Drawn

Here's the grouping effect, on the same picture as before.

*(Point at the left panel.)*

On the left is the diamond from slide ten, unchanged. It has the same lag seven and lag fourteen,
the same ellipses, and the same budget. The ellipse touches the corner, and lag fourteen is exactly
zero.

*(Point at the right panel.)*

On the right is the elastic-net ball, with the mix alpha at one half, so it's half LASSO and half
Ridge. I've drawn it with the same reach along each axis, point one, so that the two shapes are
directly comparable.

It's still pointed where it meets the axes. That's the LASSO half, and it's why Elastic Net can
still zero things out. Everywhere else the edge is curved, and that's the Ridge half.

**[pause]**

Now be honest about what you can see. At a budget this small, the curve is slight, and the ball
looks almost exactly like the diamond. When the coefficients are this small, the squared term is
tiny next to the absolute one.

And yet look at where the ellipse touches it. It isn't the corner but the face, at point oh seven
on lag seven and point oh four on lag fourteen. Both lags are kept.

Why does such a small bend matter so much? Because of what we saw on the diamond slide. These two
lags are nearly interchangeable, so the ellipses run almost parallel to the face, and the corner
only won by a hair. Blunt the corner even slightly, and the hair goes the other way.

Follow the green line. The corner still holds for the very smallest budgets, so the solution starts
out along the lag-seven axis, but only out to about point oh two, against point one six for LASSO.
After that, both lags travel together toward OLS.

**[pause]**

That's the grouping effect: correlated predictors enter together, instead of one of them winning a
coin toss. And it's why I'd start the homework with Elastic Net.

---

# ▶ SLIDE 13 — Section divider: Choosing the Penalty

That brings us to part three, choosing the penalty, and this is the part people rush. Lambda isn't
estimated from the model. It's *tuned*, and how you tune it decides whether the model generalizes.

---

# ▶ SLIDE 14 — The Regularization Path

The regularization path is the best diagnostic these methods give you, and it's underused.

The idea is to fit the model across a whole grid of lambda values, and trace each coefficient as it
moves. This is that picture for Lab 6's LASSO, with all sixty-two features and every coefficient on
a standardized scale.

*(Point at the horizontal axis.)*

Penalty strength runs from left to right, on a log scale. One warning about the label:
scikit-learn calls lambda `alpha`, but it's the same thing.

*(Sweep from left to right.)*

Read it from left to right. At **small lambda**, on the left, the coefficients sit near their OLS
values. Fifty-six of the sixty-two are nonzero, and the biggest ones are day-of-week dummies and
yesterday's sales. As **lambda grows**, they shrink, and under LASSO they drop to zero one at a
time. You can see the gray lines collapsing onto the zero line at around one to one and a half on
the axis. At **large lambda**, only a few are left. And past about two point seven, everything is
gone, and you're left with the intercept, which is just the mean forecast.

*(Point at the list on the right.)*

Now the colors. Every red line is a lag that's a multiple of seven days: one week back, two weeks
back, and so on out to eight weeks, plus fifty-two weeks. The list on the right is the order they
left in, with the last to leave at the top. Lag seven and lag twenty-eight leave together, then lag
thirty-five, lag three sixty-four, and lag fourteen. Then yesterday and lag twenty-one leave
together, and then lag fifty-six.

Seven of the last eight are multiples of seven. And the model had another way to describe the
week: seven day-of-week dummies, which carry some of the biggest coefficients at the weakest
penalty. Under increasing penalty, it dropped every one of those dummies, kept the lags that line
up with the day of the week, and discarded the lags that don't.

**[pause]**

The order in which predictors leave is the **useful output**.

Think about what that ordering means. A predictor that survives until very high lambda is one that
keeps earning its place as the penalty becomes punishing. The last survivors are the ones that
hold up under the most pressure.

You've already watched a tiny version of this. On the diamond, as you shrink the budget, lag
fourteen is the first to go, and lag seven is the last one standing.

That ordering is far more stable across resamples than any single model's coefficient table. Move
your training window a few weeks, and the exact coefficients at your chosen lambda will shift. The
order in which they drop out mostly won't.

So when you report, in the homework and in your final project, report the ordering. Saying that
under increasing penalty the last predictors standing were the weekly lags is a much stronger claim
than quoting a coefficient on lag seven of, say, one hundred twenty-four, a number that moves with
every choice of lambda. The ordering is more honest about what the data supports, and it's the
version that survives someone rerunning your analysis.

---

# ▶ SLIDE 15 — Choosing λ by Cross-Validation

So how do you pick lambda? You tune it, in three steps.

First, define a grid of values, usually spaced on a log scale. Lambda matters multiplicatively, so
you want to try point zero one, point one, one, and ten, not one through ten.

Second, for each lambda, run walk-forward cross-validation and record the mean validation RMSE.

Third, pick the lambda that minimizes the cross-validation error, and then refit on all the
training data.

**[pause]**

Next is the **one-standard-error rule**, which is worth knowing and takes ten seconds to apply.

Instead of taking the outright minimum, you take the **largest** lambda whose cross-validation
error is within one standard error of that minimum.

Here's the reasoning. The cross-validation curve is usually flat near its bottom, and the exact
location of the minimum is itself noisy. Rerun with a different split, and it moves. Anything
within one standard error is statistically indistinguishable from the best. So among all of those
equally good options, you take the one with the most regularization. You get a simpler model, and
you give up essentially no accuracy for it.

**[pause]**

And now the warning box. It's the same warning you've heard from me since Lecture 1, with one
honest qualification this time. The box says to **default** to `TimeSeriesSplit`.

Random folds put future observations in the training set and past observations in validation.
You're letting the model see the future in order to predict the past.

Here's the qualification, and it comes from the paper cited on the slide. Bergmeir, Hyndman and Koo
show that for a purely autoregressive model whose errors are uncorrelated, ordinary K-fold is
actually valid, because each row carries its own lags with it. On Lab 6's lag regression, shuffled
K-fold does no harm.

But that guarantee is gone once the series trends, or once the errors are correlated from one
period to the next, and with business data you rarely know in advance which case you're in. When
it fails, it fails silently. There's no crash and no warning. Your cross-validation error just
looks better than it should, and the forecast disappoints in production.

So default to `TimeSeriesSplit`, as in Lecture 1, and as in the walk-forward evaluation you ran in
Lab 2. A walk-forward split can't train on the future, whatever the errors do.

---

# ▶ SLIDE 16 — Section divider: In Practice

That brings us to part four, putting it into practice. It comes down to three things: standardize,
tune on a time-aware split, and read the path.

---

# ▶ SLIDE 17 — Regularization in Python

The code is short, and one line of it is doing something important that's easy to miss.

We import `ElasticNetCV`, `make_pipeline`, `StandardScaler`, and `TimeSeriesSplit`. Then we build a
pipeline, which is a scaler followed by Elastic Net, with a list of candidate `l1_ratio` values and
`TimeSeriesSplit` for the cross-validation. Then we fit it.

**[pause]**

The part that matters is `make_pipeline`, and I want to explain **why**, rather than just assert it.

The obvious thing to do, and what almost everyone does the first time, is to standardize your whole
dataset once, up front, and then run cross-validation on the scaled data.

That's leakage.

Here's why. `StandardScaler` computes a mean and a standard deviation. If you compute those on the
full sample, they include the validation folds. So every training fold has been scaled using
information from data it isn't supposed to have seen.

It's a small leak, and it won't wreck you the way random K-fold can. But it's real, it biases your
cross-validation error in the optimistic direction, and it's completely avoidable.

Wrapping the scaler in a pipeline fixes it. The pipeline refits the scaler *inside each fold*,
using only that fold's training data. It's one function call, and the leak is gone.

**[pause]**

There are two smaller notes.

First, `l1_ratio` plays the role of alpha, the mix between Ridge and LASSO. Give it a list, and the
cross-validation searches over the mix and the penalty strength together. There's one piece of
small print: `sklearn` puts a half in front of the squared term, so `l1_ratio` and the slide's
alpha agree only at zero and one. An `l1_ratio` of one half is our alpha of two thirds, which is a
little more LASSO than it sounds.

Second, there's a naming trap that will bite you: **`sklearn` calls the penalty strength `alpha`,
not `lambda`.** That's unfortunate, because in the textbook notation, and on the slide behind me,
alpha is the *mix*. So `sklearn`'s `alpha` is the textbook's lambda, and `sklearn`'s `l1_ratio` is,
near enough, the textbook's alpha.

Read the docstring, not your memory.

---

# ▶ SLIDE 18 — Section divider: Key Takeaways and Roadmap

That brings us to the last part, the key takeaways and the road ahead. Here's the line this whole
lecture has been building toward: shrinkage is the linear-model counterpart to what trees do by
pruning.

---

# ▶ SLIDE 19 — Linear Methods: Which One When

This table is your decision guide, and it's the thing to photograph.

Use OLS when k is much smaller than n and the predictors are uncorrelated. That's a real
situation, just not a common one in forecasting.

Use stepwise selection when you have few candidates and you need a story you can tell.

Use Ridge when you have many predictors that are all mildly useful, and use LASSO when you believe
only a few predictors truly matter.

Use Elastic Net when you have many *correlated* predictors, which, as we said, describes most
forecasting problems.

**[pause]**

Now for the connection to Lecture 5, last week, on tree ensembles, which is the point of the whole
lecture.

Trees and shrinkage look like completely different technologies, but they solve the same problem
in different geometries.

Look at XGBoost's hyperparameters, the ones on last week's hyperparameter slide. `reg_lambda` is
**literally a Ridge penalty on the leaf weights**. It has the same L-two form, applied to leaf
values instead of regression coefficients. If you want the LASSO version, XGBoost has that too:
`reg_alpha` is an L-one penalty on the same leaf weights. And `gamma` is a different kind of
charge, a fixed price for every leaf, however small its weight. That's the alpha from
cost-complexity pruning in Lecture 4, back again.

You were doing regularization last week, even though nobody called it that.

**[pause]**

So regularization isn't a linear-model topic. It's **how every flexible model is kept honest**.

You've now met it four times: the GAM smoothing penalty in Lecture 3, cost-complexity pruning in
Lecture 4, shrinkage on trees in Lecture 5, and Ridge and LASSO today.

Next week, in Lecture 7, it appears a fifth time, in neural networks, as weight decay. Weight decay
is Ridge: the same L-two penalty, applied to the network's weights under a different name.

If you take one thing from the first half of this course, take that. The methods multiply, but the
underlying ideas don't.

---

# ▶ SLIDE 20 — Lecture 6: Key Takeaways

Let me leave you with five takeaways.

The first is that with many predictors, OLS overfits. Coefficient variance grows with k, and once
k exceeds n, there's no unique solution at all.

The second is that subset selection counts predictors in or out. It's discrete, greedy, and
unstable: small changes in the data flip the selected model, and that instability is invisible in
your output.

The third is that shrinkage replaces the switch with a dial. Ridge shrinks everything smoothly.
LASSO shrinks and also zeros some coefficients out, through the corners of the diamond, and only
when the budget is tight enough. Elastic Net does both, and it keeps correlated predictors
together.

The fourth is to standardize first, inside each cross-validation fold, because none of these
penalties is scale-invariant. And lambda is tuned, not estimated, with `TimeSeriesSplit` and never
with random K-fold.

The last one is that the regularization path, meaning the order in which predictors drop out, is a
more stable finding than any single coefficient table. Report it.

**[pause]**

That closes the first half of the course. Everything so far has been either a model you specify
and estimate, or an algorithm that partitions your data. Both are things you can, with some effort,
look inside and explain.

Next week, in Lecture 7, we start neural networks, and that changes. We'll trade a good deal of
interpretability for flexibility, and the same fit-plus-penalty skeleton comes with us, this time
under the name weight decay.

Homework 4 is where today's material and last week's meet. It has you tune XGBoost, read a LASSO
regularization path, and compare Elastic Net against LASSO and Ridge. Build the path plot, because
it's the fastest way to develop intuition for what lambda is actually doing.

Homework 4 also has you sweep `reg_lambda` in its own small XGBoost search, so watch what it does.
It's the cheapest way to see, with your own hands, that the two halves of that assignment are the
same idea.

---

# ▶ SLIDE 21 — References

The references are on the slide, and there are two I'd single out.

Tibshirani, nineteen ninety-six, is the original LASSO paper, and it's unusually readable for a
foundational statistics paper. It's worth an hour if you're curious where this came from.

Zou and Hastie, two thousand five, introduces Elastic Net, and its motivating section is
essentially the correlated-predictor problem we spent this lecture on.

For the homework, chapter six of ISLP is the one to read. It covers this material at exactly the
level you need, with Python code alongside.

See you next week for neural networks.

---

## Timing guide

| Segment | Slides | Target |
|---|---|---:|
| Opening & the problem | 1–6 | ~8 min |
| Ridge, LASSO, geometry | 7–12 | ~11 min |
| Choosing lambda | 13–15 | ~6 min |
| Python & the synthesis | 16–21 | ~6 min |
| **Total** | | **~31 min** |

If you must compress, slide 19's method table survives trimming — it's a reference students will
photograph rather than absorb live. Do **not** compress slides 10 or 14: the ℓ₁-corner figure and
the regularization path are the two things that make the rest more than a list of estimators.

Slide 19's XGBoost connection is the payoff for the whole first half of the course. If you are
running long, cut elsewhere and protect it.
