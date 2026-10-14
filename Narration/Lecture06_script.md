# Lecture 6 — Recording Script

**ECON 8310: Business Forecasting · Regularization & Model Selection**

Deck: `Slides/Lecture06_Regularization.pdf` (19 pages) · Measured runtime: see the timing guide

---

## How to use this document

- **`▶ SLIDE n — Title`** marks where to advance. The number is the PDF page.
- *Italic parentheticals* are stage directions. **[pause]** means stop for a beat.
- This lecture closes the first half of the course. The callbacks to Lectures 3, 4, and 5 are
  load-bearing — they are what turn six weeks of separate methods into one idea.

---

# ▶ SLIDE 1 — Title page

Lecture 6: Regularization and Model Selection.

This one closes the first half of the course, and it has a job beyond its own content. For five
weeks we've been collecting methods — smoothing, ARIMA, GAMs, trees, ensembles. Today I want to
show you that several of them have been doing the same thing in different clothing.

**[pause]**

Here's the setup. Everything so far has assumed you know which predictors belong in the model.
You don't. In practice you sit down with a weekly sales series and you can generate predictors
almost without limit: lags one through fifty-two, week-of-year dummies, holiday flags,
promotions, prices, competitor prices, macro series, and every interaction you can think of.

Feature engineering is cheap. Estimating what you engineered is not.

So the question today is: when you have more predictors than you can safely estimate, what
decides which ones survive? And the answer turns out to be the same answer we already gave for
GAMs and for trees.

---

# ▶ SLIDE 2 — Lecture Outline

Four parts. The problem — why too many predictors breaks ordinary least squares. Shrinkage —
Ridge, LASSO, and Elastic Net. Choosing the penalty, which is where most people go wrong. And
then practice, plus a map of which method to reach for when.

---

# ▶ SLIDE 3 — Section divider: The Problem — Too Many Predictors

*(Divider.)*

Feature engineering gives us more predictors than we can safely estimate. Something has to
decide which ones survive.

---

# ▶ SLIDE 4 — Why Regularize?

Start with what OLS actually does. It minimizes the sum of squared residuals. Nothing else. It
has no opinion about how many predictors you handed it or how large the coefficients get.

Now put a realistic number on it. A weekly model with lags, calendar dummies, promotions, and a
couple of macro series carries sixty predictors easily. And three years of weekly history is a
hundred and fifty-six observations.

Sixty predictors, a hundred and fifty observations. OLS will fit that sample beautifully. It
will also forecast badly.

**[pause]**

Three things go wrong, and they're worth separating because they have different fixes.

**First, variance grows with k.** Every predictor you add gives the model another dimension in
which to chase noise. The coefficients get less precisely estimated, all of them, including the
ones you cared about.

**Second, correlated predictors produce unstable coefficients.** This is the one you'll actually
see in your output. Put lag one and lag two in the same model — and they move together, almost
by construction — and OLS can't tell which deserves the credit. So you get a coefficient of plus
eight hundred on one and minus seven hundred and forty on the other. They nearly cancel. The
fit is fine. The individual numbers are nonsense, and if you report them to a stakeholder you
are reporting noise with a decimal point on it.

**Third, once k exceeds n there is no unique solution at all.** Not a bad estimate — no estimate.
Infinitely many coefficient vectors fit the training data perfectly.

**[pause]**

Regularization's answer is to stop minimizing fit alone, and minimize fit *plus a penalty on
coefficient size*.

Look at the shape of that expression on the slide, because you have seen it before. Fit on the
left. A complexity penalty on the right. Lambda setting the exchange rate between them.

That is the GAM smoothing penalty from Lecture 3. That is cost-complexity pruning from Lecture
4. Same skeleton, different penalty — and by the end of today I'll show you it's been hiding in
one more place you didn't expect.

And the bargain is always the same: you accept a little bias to buy a large reduction in
variance. For a forecast — where all you care about is the error on data you haven't seen —
that trade is almost always worth making.

---

# ▶ SLIDE 5 — Subset Selection: The Classical Answer

Before shrinkage, the standard approach was to *count* predictors rather than shrink them. It's
worth understanding properly, both because you'll meet it and because its failure mode is
instructive.

**Best subset** fits every possible model — all two-to-the-k of them — and picks the winner. It
is exact. It is also computationally hopeless past about thirty predictors. Two to the thirty is
over a billion models.

**Stepwise** is the practical cousin. Add the strongest predictor, or drop the weakest, one at a
time. It's tractable, and you'll notice it's the same greedy logic as CART from Lecture 4 — and
it has the same weakness. It can miss the best combination entirely, because it never
reconsiders an earlier decision.

Either way you need a criterion to compare models of different sizes: adjusted R-squared, AIC,
AICc, BIC, or cross-validation.

**[pause]**

Now the warning box, because this is the real objection and it's subtle.

Subset selection is **discrete**. A predictor is in, or it is out. There is no middle.

And that means small changes in the data flip predictors in and out of the model. Add three
weeks of data, rerun, and you get a different set of variables — sometimes a substantially
different set.

So the *selected model* is itself a high-variance object.

Sit with that, because it's easy to miss. You adopted subset selection to reduce variance. And
you did reduce the variance of the coefficients, conditional on the model. But you introduced
variance in *which model you chose*, and that variance is invisible — it doesn't show up in any
standard error your software prints.

You haven't removed the instability. You've moved it somewhere harder to see.

**[pause]**

This is the approach in FPP section seven point five, which is your reading this week alongside
ISLP chapter six. It's still taught, and it's still reasonable when you have a handful of
candidate predictors and you need a story you can defend.

Shrinkage replaces the in-or-out switch with a continuous dial. More stable, and easier to tune.

---

# ▶ SLIDE 6 — Section divider: Shrinkage Methods

*(Divider.)*

The core idea in one line: keep every predictor, but pull its coefficient toward zero.

---

# ▶ SLIDE 7 — Ridge Regression (ℓ₂)

Ridge adds lambda times the sum of squared coefficients.

Read what that does. Squaring means large coefficients are punished disproportionately — a
coefficient of ten contributes a hundred to the penalty, while ten coefficients of one
contribute ten total. So Ridge has a strong preference for spreading effect around rather than
concentrating it.

At lambda equals zero you have OLS exactly. As lambda goes to infinity every coefficient goes to
zero. Everything useful happens in between.

**[pause]**

Ridge shrinks every coefficient toward zero but sets **none** of them exactly to zero. Hold onto
that, because it's the whole contrast with the next slide.

Its particular strength is exactly the failure I described two slides ago. Remember plus eight
hundred and minus seven hundred and forty? Ridge won't do that. Rather than letting two
collinear variables fight over the same coefficient, it splits the effect between them — you get
something like plus thirty and plus twenty-eight instead. Far more stable, and far more
honest about what the data can actually distinguish.

**Use Ridge when** you believe most of your predictors carry a little signal and you want to keep
them all.

**[pause]**

Now the practical warning, and I want to be emphatic because it is the single most common
implementation error with any of these methods.

**Standardize first.**

The penalty is not scale-invariant. Lambda times beta-squared depends entirely on what units
beta is in. Measure a predictor in dollars and its coefficient is small, so it's barely
penalized. Measure the same predictor in thousands of dollars and its coefficient is a thousand
times larger, so it's penalized a million times harder.

Same variable. Same information. Completely different treatment.

Get this wrong and your *units* decide which variables survive. Not your data. We'll see the
right way to do it in the code slide, and it is not what most people do first.

---

# ▶ SLIDE 8 — LASSO (ℓ₁)

Now swap the squared penalty for an absolute one. Sum of absolute values instead of sum of
squares.

That looks like a small change. It is not — the behavior changes qualitatively.

LASSO sets some coefficients **exactly** to zero. Not small. Zero. Which means it performs
selection and estimation in a single step, and it leaves you with a sparse model you can
actually read aloud to a stakeholder.

That last part matters more than it sounds. A model with eight nonzero coefficients is something
you can put on a slide and defend in a meeting. A model with sixty small ones is not, even if
its RMSE is a hair better.

**Use LASSO when** you believe only a handful of predictors genuinely matter and you want the
model to tell you which.

**[pause]**

But now be very careful about what you conclude, because this is where LASSO gets misused.

Among a group of highly correlated predictors, LASSO keeps **one** — more or less arbitrarily —
and zeros the rest.

So suppose lag one, lag two, and lag four are all correlated, all genuinely informative, and
LASSO keeps lag four. It is tempting — and I've seen this in professional work — to report that
lag four is "the important lag."

It isn't. It's the one that happened to win a coin toss between near-identical variables. Rerun
on slightly different data and lag one wins instead.

The variable that survived is not the important one. It's the survivor. Those are different
claims, and only one of them is supported.

---

# ▶ SLIDE 9 — Why LASSO Zeros Out and Ridge Does Not

So why does one zero out and the other doesn't? The geometry answers it, and today we draw it on
our own data rather than on a cartoon.

Both methods can be written the same way: minimize squared error subject to a *budget* on
coefficient size. Spend the budget however you like, but don't exceed it. Every lambda in the
penalty form corresponds to some budget t, so this is the same problem stated the other way round.

The **shape** of that budget is the entire difference.

**[pause]**

*(Point at the figure, axes first.)*

This is Lab 6's data cut down to two predictors, so that we can draw it: daily units on lag seven
and lag fourteen, everything standardized. The horizontal axis is the coefficient on lag seven, the
vertical axis the coefficient on lag fourteen. The black dot top right is OLS — about point five on
lag seven and point three four on lag fourteen.

The gray rings around it are ellipses of equal squared error. Every point on one ring fits the data
equally well, and the fit gets worse as the rings grow outward. They're tilted because the two lags
are correlated, about point seven five.

**[pause]**

*(Left panel.)*

**Ridge.** A budget on the sum of squared coefficients is a **disc** in two dimensions. Smooth, no
corners. Grow the ellipses out from OLS until one first touches the disc — that's the black curve —
and the touch point is the blue dot: about point oh seven on *each* lag. Both kept. The blue line
shows where the solution goes as you loosen the budget, all the way out to OLS, and it never touches
an axis. So ridge coefficients get small. They don't get to zero.

*(Right panel.)*

**LASSO.** A budget on the sum of absolute values is a **diamond**, and a diamond has corners
sitting exactly on the axes. Same ellipses. Now the touch point is the corner, where lag fourteen is
exactly zero. Follow the red line: as the budget grows, the solution slides out along the lag-seven
axis with lag fourteen pinned at zero, until t reaches about point one six. Only then does lag
fourteen enter, and the solution heads straight for OLS.

That's the whole mechanism. A corner is a point where one coefficient is zero.

**[pause]**

Now look closely at the black curve on the right, because I want to be honest about it. It runs
almost exactly along the diamond's face. The corner wins here, but only just, and only while the
budget is small — below point one six, about a fifth of OLS's own total of point eight four. The
reason is that lag seven and lag fourteen are nearly interchangeable: their correlations with sales
are point seven five and point seven two. Lag seven wins by a hair. That is the coin toss from the
previous slide, drawn.

So the textbook line — the contours are likely to hit a corner first — needs a qualifier. Corners
win when the budget is tight. Loosen it and LASSO keeps both.

**[pause]**

Which leads to the interpretive point I most want you to take from this slide.

The sparsity of LASSO is a **geometric accident** of the ℓ₁ ball having corners. It is not a
statistical test of significance.

When LASSO zeros a coefficient it is saying "this predictor isn't worth its share of the budget
at this lambda." It is *not* saying "this predictor has been proven irrelevant." There's no null
hypothesis anywhere in what we just did, and no p-value.

I labor this because LASSO output gets read as if it were a significance table, in papers and in
business reports, and it isn't one. James and co-authors work through the geometry carefully in
section six point two, for any number of predictors.

---

# ▶ SLIDE 10 — Elastic Net: Both Penalties at Once

Elastic Net says: why choose? Use both penalties.

Two knobs now, and keep them straight. **Lambda** sets the total amount of penalty. **Alpha**
sets the mix between the two types — alpha equals one is pure LASSO, alpha equals zero is pure
Ridge, and anything between is a blend.

What you get is LASSO's ability to zero predictors out, plus Ridge's **grouping effect**:
correlated predictors now enter or leave the model *together*, instead of one surviving
arbitrarily.

Which is the direct fix for the coin-toss problem two slides back.

**[pause]**

**For forecasting, this is usually the right default**, and the reason is structural rather than
a matter of taste.

Our predictors are lagged values of the same series. Lag one and lag two are correlated almost by
construction — they're the same series shifted by one week.

That is precisely the situation where pure LASSO picks a winner at random, and pure Ridge refuses
to simplify anything. Elastic Net is built for it.

So if you want a default for the homework: start with Elastic Net, tune the mix, and let the data
tell you how far toward LASSO or Ridge it wants to sit.

---

# ▶ SLIDE 11 — Section divider: Choosing the Penalty

*(Divider.)*

And here's the part people rush. Lambda is not estimated from the model. It is *tuned* — and how
you tune it decides whether the model generalizes.

---

# ▶ SLIDE 12 — The Regularization Path

The regularization path is the best diagnostic these methods give you, and it's underused.

The idea: fit the model across a whole grid of lambda values, and trace each coefficient as it
moves.

Read it left to right. At **small lambda** the coefficients sit near their OLS values — the model
is dense, and unstable in the way we discussed. As **lambda grows** they shrink, and under LASSO
they drop to zero one at a time. At **large lambda** only the most robust predictors are left.
And in the limit, everything is gone and you're left with the intercept, which is just the mean
forecast — the naive benchmark from Lecture 1.

**[pause]**

Now here's the part I actually want you to use.

**The order in which predictors leave is the useful output.**

Think about what that ordering means. A predictor that survives until very high lambda is one
that keeps earning its place as the penalty gets punishing. The last survivors are the ones that
hold up under the most pressure.

You've already watched a tiny version of this. On the diamond, shrink the budget and lag fourteen
is the first to go; lag seven is the last one standing.

That ordering is far more stable across resamples than any single model's coefficient table. Move
your training window a few weeks and the exact coefficients at your chosen lambda will shift.
The dropout *order* mostly won't.

So when you report — in the homework and in your final project — report the ordering. "Under
increasing penalty, the last three predictors standing were the four-week lag, the fifty-two-week
lag, and SNAP days" is a much stronger claim than "the coefficient on lag four was two thousand
one hundred." It's more honest about what the data supports, and it's the version that survives
someone rerunning your analysis.

---

# ▶ SLIDE 13 — Choosing λ by Cross-Validation

So how do you pick lambda? You tune it. Three steps.

**One.** Define a grid, usually log-spaced — lambda matters multiplicatively, so you want to try
point zero one, point one, one, ten, not one through ten.

**Two.** For each lambda, run walk-forward cross-validation and record the mean validation RMSE.

**Three.** Pick the lambda that minimizes CV error, then refit on all the training data.

**[pause]**

Now, the **one-standard-error rule**, which is worth knowing and takes ten seconds to apply.

Instead of taking the outright minimum, take the **largest** lambda whose CV error is within one
standard error of that minimum.

The reasoning: the CV curve is usually flat near its bottom, and the exact location of the
minimum is itself noisy — rerun with a different split and it moves. Anything within one standard
error is statistically indistinguishable from the best. So among all those equally-good options,
take the one with the most regularization: simpler model, more stable, and essentially no
accuracy given up.

**[pause]**

And now the warning box, which is the same warning I have given you in every lecture, and I am
going to keep giving it.

**Never use KFold here.**

Random folds put future observations in the training set and past observations in validation. You
are letting the model see the future to predict the past.

The consequence is nasty specifically because it's silent. There's no crash. No warning. Your CV
error just looks excellent — often dramatically better than what you'd get honestly — and then
the forecast fails in production and you have no idea why.

Bergmeir and co-authors have the careful treatment of this if you want it.

Use `TimeSeriesSplit`, exactly as in Lecture 1's walk-forward evaluation. Every time.

---

# ▶ SLIDE 14 — Section divider: In Practice

*(Divider.)*

Three things: standardize, tune on a time-aware split, and read the path.

---

# ▶ SLIDE 15 — Regularization in Python

The code is short, and one line of it is doing something important that's easy to miss.

We import `ElasticNetCV`, `make_pipeline`, `StandardScaler`, and `TimeSeriesSplit`. Then we build
a pipeline: scaler, then Elastic Net with a list of candidate `l1_ratio` values and
`TimeSeriesSplit` for the CV. Fit it.

**[pause]**

**`make_pipeline` is the part that matters**, and I want to explain why rather than just assert
it.

The obvious thing to do — and what almost everyone does the first time — is standardize your
whole dataset once, up front, then run cross-validation on the scaled data.

That is leakage.

Because `StandardScaler` computes a mean and a standard deviation. If you compute those on the
full sample, they incorporate the validation folds. So every training fold has been scaled using
information from data it isn't supposed to have seen.

It's a small leak — it won't wreck you the way KFold will — but it's real, it biases your CV
error optimistically, and it is completely avoidable.

Wrapping the scaler in a pipeline fixes it. The pipeline refits the scaler *inside each fold*,
using only that fold's training data. One function call, and the leak is gone.

**[pause]**

Two smaller notes.

`l1_ratio` is alpha — the Ridge-LASSO mix. Give it a list and the CV searches over the mix and
the penalty strength together.

And a naming trap that will bite you: **`sklearn` calls the penalty strength `alpha`, not
`lambda`.** Which is unfortunate, because in the textbook notation — and on the slide behind me —
alpha is the *mix*. So `sklearn`'s `alpha` is the textbook's lambda, and `sklearn`'s `l1_ratio`
is the textbook's alpha.

Read the docstring, not your memory.

---

# ▶ SLIDE 16 — Section divider: Key Takeaways and Roadmap

*(Divider.)*

And here's the line that this whole lecture has been building toward: shrinkage is the
linear-model counterpart to what trees do by pruning.

---

# ▶ SLIDE 17 — Linear Methods: Which One When

The table is your decision guide, and it's the thing to photograph.

**OLS** when k is much smaller than n and predictors are uncorrelated. That's a real situation,
just not a common one in forecasting.

**Stepwise** when you have few candidates and need a story you can tell.

**Ridge** when you have many predictors that are all mildly useful.

**LASSO** when you believe few predictors truly matter.

**Elastic Net** when you have many *correlated* predictors — which, as we said, is most
forecasting problems.

**[pause]**

Now the connection to Lecture 5, which is the point of the whole lecture.

Trees and shrinkage look like completely different technologies. They solve the same problem in
different geometries.

Look at XGBoost's hyperparameters — the ones on the regularization slide last week.
`reg_lambda` is **literally a Ridge penalty on the leaf weights**. Same ℓ₂ form,
applied to leaf values instead of regression coefficients. If you want the LASSO version, XGBoost
has that too: `reg_alpha`, an ℓ₁ penalty on the same leaf weights. And `gamma` is a different kind
of charge: a fixed price for every leaf, however small its weight. That's cost-complexity pruning's
alpha from Lecture 4, back again.

You were doing regularization last week. Nobody called it that.

**[pause]**

So: **regularization is not a linear-model topic. It is how every flexible model is kept honest.**

And you've now met it four times. The GAM smoothing penalty in Lecture 3. Cost-complexity pruning
in Lecture 4. Shrinkage on trees in Lecture 5. Ridge and LASSO today.

Next week it appears a fifth time, in neural networks, as weight decay and dropout. Weight decay
is Ridge — the same ℓ₂ penalty, applied to network weights, under a different name.

If you take one thing from the first half of this course, take that. The methods multiply. The
underlying ideas don't.

---

# ▶ SLIDE 18 — Lecture 6: Key Takeaways

Five things.

**One.** With many predictors OLS overfits. Coefficient variance grows with k, and once k exceeds
n there's no unique solution at all.

**Two.** Subset selection counts predictors in or out. It's discrete, greedy, and unstable — small
data changes flip the selected model, and that instability is invisible in your output.

**Three.** Shrinkage replaces the switch with a dial. Ridge shrinks everything smoothly. LASSO
shrinks and zeros out — through the corners of the diamond, and only when the budget is tight
enough. Elastic Net does both and keeps correlated predictors together.

**Four.** Standardize first, inside each CV fold — none of these penalties are scale-invariant.
And lambda is tuned, not estimated: `TimeSeriesSplit`, never random K-fold.

**Five.** The regularization path — the order in which predictors drop out — is a more stable
finding than any single coefficient table. Report it.

**[pause]**

That closes the first half of the course. Everything so far has been a model you specify and
estimate, or an algorithm that partitions your data. Both are things you can, with effort, look
inside and explain.

Next week we start neural networks, and that changes. We'll trade a good deal of interpretability
for flexibility — and the same penalties come with us, wearing the names weight decay and dropout.

Assignment 4 is where today's material and last week's meet: XGBoost tuning alongside a LASSO
regularization path, and Elastic Net against Ridge. Build the path plot — it's the fastest way to
develop intuition for what lambda is actually doing.

And if you want to see the connection I just drew with your own hands, add `reg_lambda` to your
XGBoost grid and watch what it does. It isn't required, but it's the cheapest way to convince
yourself the two halves of that assignment are the same idea.

---

# ▶ SLIDE 19 — References

References are on the slide. Two I'd single out.

**Tibshirani nineteen ninety-six** is the original LASSO paper, and it's unusually readable for a
foundational statistics paper — worth an hour if you're curious where this came from.

**Zou and Hastie two thousand five** introduces Elastic Net, and the motivating section is
essentially the correlated-predictor problem we spent this lecture on.

For the homework, **ISLP chapter six** is the one to read. It covers this material at exactly the
level you need, with Python code alongside.

See you next week for neural networks.

---

## Timing guide

| Segment | Slides | Target |
|---|---|---:|
| Opening & the problem | 1–5 | ~7 min |
| Ridge, LASSO, geometry | 6–10 | ~9 min |
| Choosing lambda | 11–13 | ~5 min |
| Python & the synthesis | 14–19 | ~6 min |
| **Total** | | **~27 min** |

If you must compress, slide 17's method table survives trimming — it's a reference students will
photograph rather than absorb live. Do **not** compress slides 9 or 12: the ℓ₁-corner figure and
the regularization path are the two things that make the rest more than a list of estimators.

Slide 17's XGBoost connection is the payoff for the whole first half of the course. If you are
running long, cut elsewhere and protect it.
