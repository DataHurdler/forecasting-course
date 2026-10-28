# Lecture 12 — Recording Script

**ECON 8310: Business Forecasting · Bayesian Statistics III — Bayesian Linear Regression**

Deck: `Slides/Lecture12_BayesianRegression.pdf` (17 pages) · Measured runtime: see the timing guide

---

## How to use this document

- **`▶ SLIDE n — Title`** marks where to advance. The number is the PDF page.
- *Italic parentheticals* are stage directions. **[pause]** means stop for a beat.
- Slide 5's "do not standardize log price" is not a style note. It is the lecture's second-biggest
  point, and the lab is built around what happens when you ignore it. Give it real weight.
- Slide 12 is the one students carry into their projects: a *tighter* interval on the wrong model.
  Say "confidently, precisely wrong" out loud.

---

# ▶ SLIDE 1 — Title page

This is Lecture 12, Bayesian Statistics Part Three, and today is about Bayesian linear regression.

This is the last technical lecture of the course, and it closes a loop. We started fitting linear
regressions in Lecture 2, the week of ARIMA and VAR models, and we spent Lecture 6, on
regularization, learning to shrink them. Today we fit the same model one more time, with priors on
the coefficients, and what comes back is something quite different from a table of estimates and
standard errors.

**[pause]**

And along the way, we're going to do something this course hasn't done yet. We're going to ask what
a coefficient *means*, causally. The question isn't whether it predicts. It's what would happen if
we intervened.

That's a different question, and it needs a different kind of thinking. Getting it wrong is how
confident analysts produce expensive mistakes.

---

# ▶ SLIDE 2 — Lecture Outline

We'll take this in four parts. First, Bayesian linear regression itself, which will mostly feel
familiar. Second, reading a posterior coefficient, with three questions you can ask a distribution
that you can't ask a point estimate. Third, causal structure and DAGs, which is where the
coefficient starts to mean something. And last, a map of everything we've built this semester.

---

# ▶ SLIDE 3 — Section divider: Bayesian Linear Regression

This first section is the same regression you already know, except that it returns a distribution
for every coefficient.

---

# ▶ SLIDE 4 — OLS vs. Bayesian Regression: What Changes?

You've been fitting linear models since Lecture 2, and regularizing them since Lecture 6.

OLS solves for a single best coefficient vector, using the closed-form expression at the top of the
slide. That gives you one number per predictor, plus a standard error that rests on large-sample
theory.

The Bayesian version replaces that with the update rule from Lecture 10, on Bayesian foundations:
the posterior over beta is proportional to the likelihood times the prior.

*(Point at the key box.)*

**[pause]**

The consequence is the thing to hold on to: **every coefficient comes back as a distribution.**

It isn't an estimate with an error bar computed from an asymptotic formula. It's a full posterior, a
cloud of plausible values, and you can query it however you like.

*(Point at the last paragraph. This is a satisfying callback.)*

Here's something that should make Lecture 6 click in hindsight. **Regularization is a prior.** Put a
Normal prior on beta and find the single most probable value, which is the peak of the posterior,
called its *mode*, and you get exactly the Ridge estimate. Do the same with a Laplace prior, the
double exponential with a sharp point at zero, and the mode is exactly the LASSO estimate.

The penalty you tuned by cross-validation in Lecture 6 was a statement of belief all along. You were
saying that coefficients are probably small. You just weren't saying it in the notation of
probability.

So these are the same estimates in two vocabularies, with one precision worth keeping. The match is
at the posterior *mode*. That's where LASSO's zeros come from: the Laplace prior's sharp point puts
the peak of the posterior exactly at zero for a weak coefficient. The posterior *mean* under the
same prior is never exactly zero. So a Bayesian who uses a Laplace prior and reports posterior means
gets shrinkage, not sparsity.

**[pause]**

And the equivalence is genuinely useful, not just cute, because it tells you what a penalty is
*assuming*.

Ridge's Normal prior says that coefficients are probably small and symmetric around zero, with no
particular preference for exactly zero. LASSO's Laplace prior is more peaked at zero and has fatter
tails. It says that most coefficients are tiny and a few are large.

Once you see the penalty as a belief, its behavior stops being a rule to memorize and becomes
something you can reason about.

---

# ▶ SLIDE 5 — Bayesian Linear Regression in PyMC

Here's the model. We regress log units on log price, a SNAP indicator and an event indicator, plus
one intercept per series and a trend, and slides eleven and twelve will explain why those last two
are there. Every coefficient gets a weakly informative Normal prior, and sigma gets an Exponential.
Nothing here is new after Lecture 10.

*(Point at the last line under the code.)*

A few names in the code need defining. `J` is the number of series, which is thirty. `idx` maps each
row to its series, so that each row picks up its own series' intercept. And `yr` counts the years
since the first week, which is what the trend multiplies.

**[pause]**

*(Now the warning. This is the lecture's second-biggest point, so don't rush it.)*

But look at the bold line under the code, because it runs against a habit you've built all
semester.

**Do not standardize log price.**

You've standardized predictors in nearly every model since Lecture 6. It makes penalties comparable,
it helps samplers, and it's a good default. Here, it will quietly destroy your answer.

Here's why. With units and price both in logs, `b_price` is an **elasticity**, the percent change in
demand per percent change in price. That's a quantity a category manager already uses and can act
on directly.

If you standardize log price, the coefficient becomes the change in log units per **standard
deviation** of log price. That's still a valid number, but it's no longer an elasticity. And every
percentage claim you build on it downstream will be off by exactly the factor you divided by.

**[pause]**

The lab does this on purpose, and the business answer moves by a factor of about twenty-two. The
diagnostics stay clean the whole time, and nothing gives you any kind of warning.

*(Point at the centering note.)*

There's one thing you *should* do, and that's subtract each series' own mean log price. That's what
`lpc` is: log price, centered within each series. Centering isn't scaling. It shifts the variable
without changing its units, so the elasticity survives intact. And it fixed a real sampling
problem: the effective sample size went from one hundred eighty-four to three thousand ninety-seven.

So center, but don't scale. That's the rule.

There's a general principle underneath it, and it outlives this example. For every transformation
you apply, ask whether your coefficient **still answers the question you asked**.

Logging both sides gives you an elasticity, which you can interpret. Centering preserves it. Scaling
destroys it. And taking a difference turns it into something else again.

None of those operations is wrong. What's wrong is applying one out of habit and then reading the
coefficient as though you hadn't. The transformation and the interpretation have to be decided
together. If you can't say what one unit of your variable *is*, you can't say what your coefficient
means.

---

# ▶ SLIDE 6 — Section divider: Reading the Posterior

The next section is about three questions you can ask a distribution that you can't ask a point
estimate.

---

# ▶ SLIDE 7 — Reading a Posterior Coefficient

Here's the fitted model, controlling for series and trend.

*(Walk the table.)*

Start with `b_price`. It's minus zero point four nine three, with an HDI from minus zero point five
nine six to minus zero point three eight seven. That's **inelastic**: the whole interval sits well
above minus one, which is the line between elastic and inelastic demand. We'll come back to what
that buys you.

Next, `b_trend` is zero point one one zero, with a tight interval. That's about eleven percent
growth per year.

Then `b_snap` is zero point zero three six, which is about a three point seven percent lift in a
SNAP week. But that's one average across all thirty series. Last week's hierarchical model, in
Lecture 11, Part 2, put the lift for FOODS near eight percent.

And `b_event` is minus zero point zero zero three, with an interval from minus zero point zero one
one to plus zero point zero zero four. That interval **spans zero**.

**[pause]**

Here's a quick note on the **HDI**, the highest density interval. It's the shortest interval that
contains ninety-four percent of the posterior. And unlike a confidence interval, it means what
everyone wants it to mean: there's a ninety-four percent probability that the coefficient is inside
it.

*(Point at the last paragraph.)*

**[pause]**

Now, the last row is the useful one, and I want to be precise about why.

The interval for `b_event` spans zero, so the data don't establish a direction. Events might raise
demand slightly, or they might lower it slightly.

In most courses, that's where the story ends: it's "not statistically significant," so you move on.
Hold on to it instead. Two slides from now, we'll ask this coefficient a second question. We won't
ask which way it points. We'll ask whether it's big enough to matter at all.

---

# ▶ SLIDE 8 — Asking the Posterior a Direct Question

Because the posterior is a set of samples, any question about beta is answered by **counting**.

*(Point at the code box.)*

You pull the samples into an array. Then you ask what fraction of them are below zero, what fraction
are below minus one, and what fraction are below minus zero point five.

That's three one-line questions and three direct answers, with no test statistic, no null
hypothesis and no table.

**[pause]**

*(Point at the figure.)*

Here's what that counting looks like. The blue curve is the posterior for the price elasticity, and
the area under it is one, which stands for all four thousand draws. Every question in the box is the
share of that area on one side of a line.

*(Point at the dotted line at zero, on the right.)*

Start with the question of whether it's below zero. The whole curve is, so the probability is **one
point zero zero**. Demand falls when price rises, and there's no doubt about it. That's fine, but
it's the boring question, and a p-value would have told you something similar.

*(Point at the dashed line at minus one.)*

Now for the one a pricing manager cares about. Minus one is the line between **elastic** and
**inelastic** demand, and it's labeled revenue break-even for a reason. To its left, a one percent
price rise loses more than one percent of volume, so revenue falls. To its right, revenue rises.

Look how far the curve is from that line. The probability that the elasticity is below minus one is
**zero**. Not one draw out of four thousand is down there. The black bar under the curve is the
ninety-four percent HDI, and all of it sits between minus one and zero. So demand here is inelastic,
in every draw.

*(Point at the red area.)*

Then there's a finer question, and the data do *not* settle it. Does a one percent price rise cost
more than half a percent of volume? That's the red area, everything to the left of minus zero point
five, and it's **zero point four six** of the total. The line cuts the curve almost in half, so it's
essentially a coin flip.

So the posterior gives you three answers of different kinds. The direction is settled. The side of
minus one is settled. And the finer question about size is honestly open. "Statistically
significant" reports only the first of those. The question the business actually asks is *how big*,
and the posterior tells you which questions about size the data can answer and which they can't.

The usual p-value, tested against zero, gives you only the direction. It never draws that map.

**[pause]**

*(Worth making concrete, since students will ask what to do with 0.46.)*

So a reasonable question is what a manager does with forty-six percent.

They don't get a yes or a no on that question, and that's the point, because the evidence is
balanced. What they get is the ability to reason about the *consequences* of being wrong in each
direction. Suppose that planning for the larger volume loss and being wrong is cheap, while planning
for the smaller one and being wrong is expensive. Then forty-six percent is more than enough reason
to plan cautiously.

That's a decision-theoretic argument, and it needs a probability as an input. A significance test
can't supply one.

---

# ▶ SLIDE 9 — Too Small to Matter? A ROPE

Let's go back to the event coefficient from two slides ago. Its interval spans zero, so the data give
no direction.

A **ROPE**, which stands for region of practical equivalence, asks a different question. It doesn't
ask whether the effect is zero. It asks whether the effect is too small to act on.

*(Point at the figure.)*

The blue curve is the posterior for the effect of an event week, in percent. The black bar is its
ninety-four percent HDI, which runs from minus one point one percent to plus zero point four
percent.

*(Point at the light gray band and its dashed edges.)*

The wide gray band is the ROPE. It covers every effect between about minus five and plus five
percent a week, which we're declaring too small to matter. **One hundred percent** of the posterior
sits inside it, every single draw.

*(Point at the darker band in the middle.)*

The darker band is a much stricter ROPE, about plus or minus one percent. Even there, ninety-five
percent of the posterior is inside.

**[pause]**

*(Point at the sentence under the figure. Say it plainly, because this is a choice, not a result.)*

So where does the width come from? Not from the data. It's a business choice. Neither the lab nor
this deck has a natural one, so we use the width Homework 7 uses, which is an absolute coefficient
below zero point zero five, or about five percent a week. A manager with thin margins might set it
narrower, and we've just seen that at one percent the answer barely moves. Homework 7 asks you to
compute one yourself, for every predictor.

*(Point at the key box.)*

**[pause]**

Here's why it's worth doing. "Not significant" lumps together two completely different situations.
One is an effect you can't detect because it's tiny. The other is an effect you can't detect because
your data is thin.

Those call for opposite actions. If the effect is genuinely tiny, you stop investigating. You have
your answer, and it's that events don't matter much for these series. If you can't detect it because
the data is thin, you collect more data or find a better design, because the question is still open.

A p-value above zero point zero five is compatible with both, and it gives you no way to tell which
one you're in. The ROPE tells you right away. If the posterior is narrow and inside the band, you're
in the first situation. If it's wide and spilling out of the band, you're in the second. Here, it's
the first.

---

# ▶ SLIDE 10 — Section divider: Causal Structure and DAGs

The third section asks which coefficients mean what you think they mean.

---

# ▶ SLIDE 11 — Directed Acyclic Graphs

Everything so far has been about *estimating* a coefficient well. Now the question is what it means.

A **DAG**, which stands for directed acyclic graph, is a picture of what you believe causes what. The
nodes are variables, and the arrows run from cause to effect. Acyclic means that nothing causes
itself: no path of arrows ever leads back to where it started.

Crucially, you draw it **before** the model, from what you know about the domain. And its purpose is
specific. It tells you **which variables belong in the regression.**

*(Point at the figure. Walk it from left to right.)*

Here's ours. Price is on the left and units sold is on the right, and the thick blue arrow between
them is `b_price`, the one effect we want.

*(Point at the two red boxes.)*

Now look at the two red boxes. The first is the **series**, meaning which store and which category,
and it sets both the price level and the volume. Food is cheap and sells in volume, while hobby items
cost more and sell less. The second is **time**: over these five years, prices drifted up and volumes
grew. Each red box has an arrow into price *and* an arrow into units.

*(Point at the confounder definition on the right.)*

That's the definition of a **confounder**: it causes both your predictor and your outcome. You
**must** control for it. Otherwise its influence leaks into your coefficient, and you credit your
variable with someone else's effect. That's why the model on slide five has one intercept per series
and a trend. Each red box is a term in the model.

*(Point at the gray box.)*

SNAP and events are in the model too, but look at their arrows. They point into units only. They
explain demand, which tightens the estimate, but they aren't confounders. If you left them out,
`b_price` would be noisier, but it wouldn't be wrong.

**[pause]**

*(Point at the collider definition.)*

A **collider** is the opposite shape, because it's caused by both. And controlling for a collider
**creates** a spurious association that wasn't there to begin with. The slide cites Richard
McElreath's book, *Statistical Rethinking*, for that point.

There isn't a collider in our picture, but it's easy to build one. Revenue is price times units, so
both arrows point into it. If you hold revenue fixed in the regression, price and units are forced
to move in opposite directions. You'd get an elasticity of exactly minus one, manufactured entirely
by the control.

*(Point at the bold line under the columns.)*

That's why "add every control you have" is bad advice, and it surprises people.

More controls aren't safer. Conditioning on a collider actively manufactures a relationship out of
nothing. So which variables to include is a question about **causal structure**, and no amount of
model fit will answer it, not a cross-validation score and not an information criterion. Two models
can fit identically and mean opposite things.

**[pause]**

*(Point at the muted caveat, and deliver it plainly.)*

Here's one honest limit. Whether a given DAG is *right* is a question about your business, not a
statistical one. This course uses DAGs to reason about **what a coefficient means**. Estimating
causal effects credibly, with instruments, natural experiments and all of that, is a separate
subject and a separate course.

---

# ▶ SLIDE 12 — Confounding, Measured

Let's make that concrete on our own data, because the size of the effect is startling.

Two things move price and units together in our panel, and neither of them is price causing demand.
They're the two red boxes from the DAG.

*(Point at the left panel.)*

The first is the **category**. Each color is a category, and each one is its own cloud. Food is cheap
and sells in volume, and hobby and household items cost more and sell less. Fit one line through all
of it, and you get a slope of about **minus two point eight**. But look at what that line is doing.
It's joining the clouds. It measures the difference between food and hobbies, not what happens when
a price changes.

*(Point at the middle panel.)*

So subtract each series' own mean, which removes the category. Now the slope **flips sign**, to plus
one point seven: higher prices, more units. That's the second confounder, **time**. The shading is
the year, with light for twenty eleven and dark for twenty sixteen. Over those five years, prices
drifted up by a percent or two a year, and volumes grew about ten percent a year. So within any one
series, the later weeks are both dearer and busier. Time has an arrow into price *and* into units,
which makes it a confounder, exactly like the category.

*(Point at the right panel.)*

Remove the trend as well, and what's left is the price response, with a slope of about **minus zero
point four nine**. That's what the model with series effects and a trend estimates: minus zero point
four nine three, with an HDI from minus zero point five nine six to minus zero point three eight
seven.

**[pause]**

*(Now the punchline. Slow down.)*

The uncontrolled estimate, minus two point eight zero five, is nearly **six times too large**. It
claims that demand collapses when price rises, with a one percent price increase costing almost
three percent of volume.

And here's the part I want you to remember for the rest of your career.

It is **not noisier**. Its interval runs from minus two point eight five six to minus two point seven
five six, a width of about zero point one. The controlled one is roughly zero point two wide.

**The wrong answer has an interval half the width of the right one.**

*(Point at the warning box.)*

**[pause]**

It's confidently, precisely wrong. And nothing in the output flags it.

A tight credible interval says the model is **sure**. It says nothing whatsoever about whether the
model is answering **your question**.

Bayesian machinery doesn't protect you from leaving out a confounder. Neither does more data, because
more data would make that wrong interval even tighter. Only the DAG protects you, and the DAG is
drawn before you fit anything. And notice that it took both arrows. If you control for the category
but forget time, you get the wrong *sign*.

---

# ▶ SLIDE 13 — Scenario Analysis with Posterior Samples

Here's the payoff for having a distribution instead of a number.

Because the posterior is samples, you can push a business scenario through it and get back a
**distribution of outcomes**.

*(Point at the code.)*

It takes three lines. You pull the elasticity samples, you apply the price change to each one, and
you report the mean and the interval.

**[pause]**

With an elasticity near minus zero point four nine three, a five percent price rise implies roughly a
**two point four percent fall in units**, with ninety-four percent of the posterior between minus two
point nine and minus one point nine percent.

So you can report the pessimistic end, which is what a planner actually wants. You don't just say
that demand will fall about two percent. You say that demand will fall about two percent, and that
it's unlikely to be worse than three.

*(Point at the decision paragraph.)*

**[pause]**

Then the decision follows directly. Revenue is price times units, so a five percent price rise
multiplies revenue by one point zero five, raised to the power of one plus the elasticity. That's
above one exactly when the elasticity is above minus one, which is to say when demand is inelastic.

And slide eight already told us that every posterior draw is above minus one. So revenue rises in
every draw, by about **two and a half percent** on average, with a probability of **one point zero
zero**.

Notice how the two slides fit together. The exact size of the elasticity is uncertain. We couldn't
even say whether a one percent rise costs more than half a percent of volume. But the revenue
question only cares which side of minus one we're on, and on that the posterior is unanimous. A
decision can be settled even when the parameter behind it isn't pinned down, and the posterior shows
you which decisions those are.

*(Point at the muted note.)*

There's one caveat, and it's the one from slide eleven. This is valid only to the extent that the
**DAG** is. It's a causal claim about intervening on price, and it inherits every assumption in that
diagram.

---

# ▶ SLIDE 14 — Section divider: Course Synthesis and Next Steps

The last section steps back to look at the whole course, from classical time series to Bayesian
hierarchical models, which together make a complete forecasting toolkit.

---

# ▶ SLIDE 15 — Course Method Map

*(Walk the table at a steady pace. Students will photograph this slide.)*

Here's everything we've built, on four axes: what data it takes, what uncertainty it gives you, how
interpretable it is, and what it's good at.

ETS and ARIMA, from Lectures 1 and 2, take a univariate series and give you confidence intervals.
They're highly interpretable, fast and automatic. And remember that the benchmark beat every one of
them in Homework 1, which is the first thing this table should remind you of.

GAMs and Prophet, from Lecture 3, take a series plus predictors. They give you limited uncertainty,
they're interpretable, and they're good at being nonlinear and interpretable at the same time.

Decision trees, from Lecture 4, take tabular data and give you no uncertainty, but they give you a
rule-based explanation anyone can read.

Random forests, from Lecture 5, are robust and need minimal tuning, and the out-of-bag error comes
free. That's an error estimate, though, not an interval.

XGBoost is also from Lecture 5. It captures interactions, and it scored seven eighty-one on our
board. That number came from an earlier version of xgboost. The version pinned for this course
gives seven ninety-two, which still leaves it in second place.

Ridge and LASSO, from Lecture 6, are linear and interpretable, and they hold the best number on our
scoreboard, seven forty-four. The simplest model in the table beat the boosted trees.

Feedforward nets and 1D CNNs, from Lectures 7 and 8, give you flexible nonlinear fits, but their
interpretability is low.

LSTMs and Transformers, from Lecture 9, handle sequences and long-range dependencies, and their
interpretability is very low.

And the Bayesian models, from Lectures 10 through 12, take any data and give you a **full
posterior**. They're highly interpretable, and they're strongest exactly where samples are small.

**[pause]**

*(Point at the key box.)*

The box says it plainly: **no single method dominates.** That's the honest summary of fourteen
weeks, and it's what the scoreboard has been telling you all semester.

So choose on four things: your data type and size, whether you need uncertainty quantified, whether
someone needs to understand the model, and how fast you need it deployed.

And in practice, **combine** them. Use whichever model wins your scoreboard for the point forecast,
which on ours was LASSO, ahead of XGBoost. Then add a Bayesian model for the intervals, because
they're honest. You aren't obliged to pick one row of that table.

---

# ▶ SLIDE 16 — Lecture 12: Key Takeaways

Here are the main points.

First, Bayesian linear regression places priors on the coefficients and returns a full posterior for
each one. Its mode is the Ridge estimate under a Normal prior, and the LASSO estimate under a Laplace
prior.

Second, ask the posterior directly. It gives you the mean, the HDI, the probability that the
coefficient is past any threshold you care about, and how much of it sits inside a ROPE. That's
richer than a p-value, and it answers the question people actually ask.

Third, draw the **DAG** before you specify the regression. Control for the confounders, which here
meant both the series and time, and never control for a collider.

Fourth, use posterior samples for scenario analysis. Push any counterfactual through the posterior
draws, and you get a full distribution of outcomes for that scenario, not a point projection.

And last, no single method is best for every forecasting problem. Build an intuition for when to
reach for each one, and pair an accurate point forecast with Bayesian intervals.

**[pause]**

*(Close the course.)*

Here's the one to carry out of this room, and it's really the lesson of the whole semester.

The uncontrolled elasticity was nearly six times too large, with an interval **half the width** of
the correct one. Precision is not accuracy, and a confident model is not a correct model. What
caught it wasn't a diagnostic, a score or a cross-validation fold. It was a picture you draw before
fitting anything, from what you know about the business.

That habit is the most transferable thing in this course. Next week, in Lecture 13, we'll pull the
whole semester together in the synthesis.

---

# ▶ SLIDE 17 — References

*(Advance and close. No narration is needed.)*
