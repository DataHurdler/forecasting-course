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

Lecture 12: Bayesian Statistics III — Bayesian Linear Regression.

This is the last technical lecture of the course, and it closes a loop. We started in Lecture 2
with linear regression. We spent Lecture 6 regularizing it. Today we fit the same model one more
time, with priors on the coefficients — and get back something quite different from a table of
estimates and standard errors.

**[pause]**

And along the way we're going to do something this course hasn't done yet: ask what a coefficient
*means*, causally. Not "does it predict" but "if we intervened, what would happen."

That's a different question, it needs a different kind of thinking, and getting it wrong is how
confident analysts produce expensive mistakes.

---

# ▶ SLIDE 2 — Lecture Outline

Four parts. Bayesian linear regression, which is mostly familiar. Then reading a posterior
coefficient — three questions you can ask a distribution that you cannot ask a point estimate.
Then causal structure and DAGs, which is where the coefficient starts meaning something. And then
a map of everything we've built this semester.

---

# ▶ SLIDE 3 — Section divider: Bayesian Linear Regression

The same regression you already know — returning a distribution for every coefficient.

---

# ▶ SLIDE 4 — OLS vs. Bayesian Regression: What Changes?

You have been fitting linear models since Lecture 2, and regularizing them since Lecture 6.

OLS solves for a single best coefficient vector — the closed-form expression on the slide. One
number per predictor, plus a standard error that rests on large-sample theory.

The Bayesian version replaces that with the update rule from Lecture 10: the posterior over beta
is proportional to the likelihood times the prior.

*(Point at the key box.)*

**[pause]**

And the consequence is the thing to hold onto: **every coefficient comes back as a distribution.**

Not an estimate with an error bar computed from an asymptotic formula. A full posterior — a cloud
of plausible values, which you can query however you like.

*(Point at the last paragraph. This is a satisfying callback.)*

**[pause]**

And here's something that should make Lecture 6 click retroactively.

**Regularization is a prior.** Put a Normal prior on beta, find the single most probable value —
the peak of the posterior, its *mode* — and you get exactly the Ridge estimate. Do the same with a
Laplace prior, the double-exponential with a sharp point at zero, and the mode is exactly the LASSO
estimate.

The penalty you tuned by cross-validation in Lecture 6 was a statement of belief all along. You
were saying "coefficients are probably small," you just weren't saying it in probability
notation.

Same estimates, two vocabularies — with one precision worth keeping. The match is at the posterior
*mode*. That's where LASSO's zeros come from: the Laplace prior's sharp point puts the peak of the
posterior exactly at zero for a weak coefficient. The posterior *mean* under the same prior is never
exactly zero. A Bayesian who uses a Laplace prior and reports posterior means gets shrinkage, not
sparsity.

**[pause]**

And the equivalence is genuinely useful, not just cute. It tells you what a penalty is *assuming*.

Ridge's Normal prior says coefficients are probably small and symmetric around zero, with no
particular preference for exactly zero. LASSO's Laplace prior is more peaked at zero and has fatter
tails — it says most coefficients are tiny and a few are large.

Once you see the penalty as a belief, the behavior stops being a rule to memorize and becomes
something you can reason about.

---

# ▶ SLIDE 5 — Bayesian Linear Regression in PyMC

Here's the model. Log units regressed on log price, a SNAP indicator and an event indicator —
plus one intercept per series and a trend, which slide 12 will explain. Weakly informative Normal
priors on every coefficient, Exponential on sigma. Nothing here is new after Lecture 10.

**[pause]**

*(Now the warning. This is the lecture's second-biggest point — do not rush it.)*

But look at the bolded line, because it runs against a habit you have built all semester.

**Do not standardize log price.**

You have standardized predictors in nearly every model since Lecture 6. It makes penalties
comparable, it helps samplers, it's a good default. And here it will quietly destroy your answer.

Here's why. With units and price both in logs, `b_price` is an **elasticity** — the percent change
in demand per percent change in price. That is a quantity a category manager already uses and can
act on directly.

Standardize log price, and the coefficient becomes "change in log units per **standard deviation**
of log price." Still a valid number. No longer an elasticity. And every percentage claim you build
on it downstream will be wrong by exactly the factor you divided by.

**[pause]**

*(Beat.)*

The lab does this deliberately, and the business answer moves by a factor of twenty-two. With
clean diagnostics throughout. No warning of any kind.

*(Point at the centering note.)*

One thing you *should* do: subtract each series' own mean log price. That's **centering**, not
scaling — it shifts the variable without changing its units, so the elasticity survives intact. And
it fixed our sampling badly: effective sample size went from one hundred eighty-four to three
thousand ninety-seven.

Center, don't scale. That's the rule.

**[pause]**

And there's a general principle underneath it that outlives this example.

Ask, for every transformation you apply: **does my coefficient still answer the question I asked?**

Logging both sides gives you an elasticity, which is interpretable. Centering preserves it.
Scaling destroys it. Taking a difference changes it into something else again.

None of those are wrong operations. What's wrong is applying one out of habit and then reading the
coefficient as though you hadn't. The transformation and the interpretation have to be decided
together — and if you can't say what one unit of your variable *is*, you cannot say what your
coefficient means.

---

# ▶ SLIDE 6 — Section divider: Reading the Posterior

Three questions you can ask a distribution that you cannot ask a point estimate.

---

# ▶ SLIDE 7 — Reading a Posterior Coefficient

Here's the fitted model, controlling for series and trend.

*(Walk the table.)*

`b_price`: minus point-four-nine-three, HDI from minus point-five-nine-six to minus
point-three-eight-seven. **Inelastic**: the whole interval sits well above minus one, the line
between elastic and inelastic demand. We'll come back to what that buys you.

`b_trend`: point-one-one-zero, a tight interval. About eleven percent growth per year.

`b_snap`: point-oh-three-six. About a three-point-seven percent SNAP-week lift — but that is one
average across all thirty series. Last week's FOODS hierarchy put the FOODS lift near eight percent.

`b_event`: minus point-oh-oh-three, interval from minus point-oh-one-one to plus point-oh-oh-four.
**Spans zero.**

**[pause]**

Quick note on the **HDI** — highest density interval. It's the shortest interval containing
ninety-four percent of the posterior. And unlike a confidence interval, it means what everyone
wants it to mean: there is a ninety-four percent probability the coefficient is in there.

*(Point at the last paragraph.)*

**[pause]**

Now, the last row is the useful one, and I want to be precise about why.

`b_event`'s interval spans zero, so the data do not establish a direction. Events might raise
demand slightly; they might lower it slightly.

In most courses that's where the story ends: "not statistically significant," move on. Hold onto
it instead. Two slides from now we'll ask this coefficient a second question — not which way it
points, but whether it's big enough to matter at all.

---

# ▶ SLIDE 8 — Asking the Posterior a Direct Question

Because the posterior is a set of samples, any question about beta is answered by **counting**.

*(Point at the code box.)*

Pull the samples into an array. Then: what fraction are below zero? What fraction are below minus
one? What fraction are below minus point-five?

Three one-line questions, three direct answers. No test statistic, no null hypothesis, no table.

**[pause]**

*(Point at the figure.)*

And here's what that counting looks like. The blue curve is the posterior for the price
elasticity, and the area under it is one — all four thousand draws. Every question in the box is
the share of that area on one side of a line.

*(Point at the dotted line at zero, on the right.)*

Below zero: the whole curve. The probability is **one point zero zero**. Demand falls when price
rises, and there is no doubt about it. Fine. That's the boring question, and a p-value would have
told you something similar.

*(Point at the dashed line at minus one.)*

Now the one a pricing manager cares about. Minus one is the line between **elastic** and
**inelastic** demand, and it's labeled revenue break-even for a reason. To its left, a one percent
price rise loses more than one percent of volume, so revenue falls. To its right, revenue rises.

Look how far the curve is from that line. The probability the elasticity is below minus one is
**zero** — not one draw out of four thousand. The black bar under the curve is the ninety-four
percent HDI, and all of it sits between minus one and zero. Demand here is inelastic, in every
draw.

**[pause]**

*(Point at the red area.)*

And then a finer question, which the data do *not* settle. Does a one percent price rise cost more
than half a percent of volume? That's the red area, everything left of minus point-five, and it is
**point four six** of the total. The line cuts the curve almost in half. Essentially a coin flip.

So the posterior gives you three answers of different kinds: the direction, settled; the side of
minus one, settled; the finer size question, honestly open. "Statistically significant" reports
only the first. The question the business actually asks is *how big*, and the posterior tells you
which size questions the data can answer and which they can't.

The usual p-value, tested against zero, gives you only the direction. It never draws that map.

**[pause]**

*(Worth making concrete, since students will ask what to do with 0.46.)*

And a reasonable question is what a manager does with forty-six percent.

They do not get a yes or a no on that question, and that's the point — the evidence is balanced.
What they get is the ability to reason about the *consequences* of being wrong in each direction.
If planning for the larger volume loss and being wrong is cheap, and planning for the smaller one
and being wrong is expensive, then forty-six percent is more than enough to plan cautiously.

That's a decision-theoretic argument, and it needs a probability as an input. A significance test
cannot supply one.

---

# ▶ SLIDE 9 — Too Small to Matter? A ROPE

Back to the event coefficient from two slides ago. Its interval spans zero, so the data give no
direction.

A **ROPE** — a region of practical equivalence — asks a different question. Not "is it zero?" but
"is it too small to act on?"

*(Point at the figure.)*

The blue curve is the posterior for the event-week effect, in percent. The black bar is its
ninety-four percent HDI: minus one point one percent to plus point four.

*(Point at the light gray band and its dashed edges.)*

The wide gray band is the ROPE: every effect between about minus five and plus five percent a
week, which we are declaring too small to matter. **One hundred percent** of the posterior sits
inside it. Every draw.

*(Point at the darker band in the middle.)*

The darker band is a much stricter ROPE, about plus or minus one percent. Even there, ninety-five
percent of the posterior is inside.

**[pause]**

*(Point at the sentence under the figure. Say it plainly — this is a choice, not a result.)*

Where does the width come from? Not from the data. It's a business choice. Neither the lab nor
this deck has a natural one, so we use the width Homework 7 uses: an absolute coefficient below
point-oh-five, which is about five percent a week. A manager with thin margins might set it
narrower — and we've just seen that at one percent the answer barely moves. Homework 7 asks you to
compute one yourself, for every predictor.

*(Point at the key box.)*

**[pause]**

And this is why it's worth doing. "Not significant" conflates two completely different situations:
an effect you can't detect because it's tiny, and an effect you can't detect because your data is
thin.

Those call for opposite actions. An effect that is genuinely tiny means stop investigating — you
have your answer, and it's that events don't matter much for these series. An effect you cannot
detect because the data is thin means collect more data, or find a better design, because the
question is still open.

A p-value above point-oh-five is compatible with both, and gives you no way to tell which you're
in. The ROPE tells you immediately: narrow and inside the band is the first; wide and spilling out
of it is the second. Here, it's the first.

---

# ▶ SLIDE 10 — Section divider: Causal Structure and DAGs

Which coefficients mean what you think they mean.

---

# ▶ SLIDE 11 — Directed Acyclic Graphs

Everything so far has been about *estimating* a coefficient well. Now: what does it mean?

A **DAG** is a picture of what you believe causes what. Nodes are variables, arrows run from cause
to effect, and acyclic means nothing causes itself — no arrow path ever leads back to where it
started.

Crucially, you draw it **before** the model, from domain knowledge. And its purpose is specific:
it tells you **which variables belong in the regression.**

*(Point at the figure. Walk it left to right.)*

Here's ours. Price on the left, units sold on the right, and the thick blue arrow between them is
`b_price` — the one effect we want.

*(Point at the two red boxes.)*

Now the two red boxes. The **series** — which store, which category — sets the price level and the
volume: food is cheap and sells in volume, hobbies cost more and sell less. And **time**: over these
five years prices drifted up and volumes grew. Each red box has an arrow into price *and* an arrow
into units.

*(Point at the confounder definition on the right.)*

That's the definition of a **confounder**: it causes both your predictor and your outcome. You
**must** control for it — otherwise its influence leaks into your coefficient and you attribute
someone else's effect to your variable. That's why the model on slide 5 has one intercept per
series and a trend: each red box is a term in the model.

*(Point at the gray box.)*

SNAP and events are in the model too, but look at their arrows: into units only. They explain
demand, which tightens the estimate, but they aren't confounders. Leave them out and `b_price`
would be noisier, not wrong.

**[pause]**

*(Point at the collider definition.)*

A **collider** is the opposite shape: caused by both. And controlling for a collider **creates** a
spurious association that was not there to begin with.

There isn't one in our picture, but it's easy to build one. Revenue is price times units, so both
arrows point into it. Hold revenue fixed in the regression and price and units are forced to move
in opposite directions — you'd get an elasticity of exactly minus one, manufactured entirely by the
control.

*(Point at the consequence.)*

That's why "add every control you have" is bad advice, and it surprises people.

More controls is not safer. Conditioning on a collider actively manufactures a relationship out of
nothing. So which variables to include is a question about **causal structure** — and no amount of
model fit, no cross-validation score, no information criterion will answer it. Two models can fit
identically and mean opposite things.

*(Point at the muted caveat, and deliver it plainly.)*

**[pause]**

One honest limit. Whether a given DAG is *right* is a substantive question about your business,
not a statistical one. This course uses DAGs to reason about **what a coefficient means**.
Estimating causal effects credibly — instruments, natural experiments, all of that — is a separate
subject and a separate course.

---

# ▶ SLIDE 12 — Confounding, Measured

Let's make that concrete on our own data, because the size of the effect is startling.

Two things move price and units together in our panel, and neither of them is price causing
demand. They're the two red boxes from the DAG.

*(Point at the left panel.)*

The first is the **category**. Each color is a category, and each is its own cloud. Food is cheap
and sells in volume; hobby and household items cost more and sell less. Fit one line through all of
it and you get a slope of about **minus two point eight** — but look at what that line is doing. It
is joining the clouds. It measures the difference between food and hobbies, not what happens when a
price changes.

*(Point at the middle panel.)*

So subtract each series' own mean, which removes the category. And the slope **flips sign**: plus
one point seven. Higher prices, more units. That's the second confounder, **time**. The shading is
the year — light is 2011, dark is 2016. Over those five years prices drifted up by a percent or two
a year, and volumes grew about ten percent a year. Within any one series the later weeks are both
dearer and busier. So time is an arrow into price *and* into units: a confounder exactly like the
category.

*(Point at the right panel.)*

Remove the trend as well, and what's left is the price response: a slope of about **minus point
four nine**. That is what the model with series effects and a trend estimates — minus point four
nine three, HDI from minus point five nine six to minus point three eight seven.

**[pause]**

*(Now the punchline. Slow down.)*

The uncontrolled estimate, minus two point eight zero five, is nearly **six times too large**. It
claims demand collapses when price rises — a one percent price increase costing almost three
percent of volume.

And here's the part I want you to remember for the rest of your career.

It is **not noisier**. Its interval runs from minus two eight-five-six to minus two
seven-five-six — a width of about point one. The controlled one is roughly point two wide.

**The wrong answer has an interval half the width of the right one.**

*(Point at the warning box.)*

**[pause]**

It is confidently, precisely wrong. And nothing in the output flags it.

A tight credible interval says the model is **sure**. It says nothing whatsoever about whether the
model is answering **your question**.

Bayesian machinery does not protect you from omitting a confounder. Neither does more data — more
data would tighten that wrong interval further. Only the DAG protects you, and the DAG is drawn
before you fit anything. And notice that it took both arrows: control for the category but forget
time, and you get the wrong *sign*.

---

# ▶ SLIDE 13 — Scenario Analysis with Posterior Samples

Here's the payoff for having a distribution instead of a number.

Because the posterior is samples, you can push a business scenario through it and get back a
**distribution of outcomes**.

*(Point at the code.)*

Three lines. Pull the elasticity samples. Apply the price change to each one. Report the mean and
the interval.

**[pause]**

With an elasticity near minus point four nine three, a five percent price rise implies roughly a
**two point four percent fall in units** — with ninety-four percent of the posterior between minus
two point nine and minus one point nine percent.

So you can report the pessimistic end, which is what a planner actually wants. Not "demand will
fall about two percent" but "demand will fall about two percent, and it's unlikely to be worse
than three."

*(Point at the decision paragraph.)*

**[pause]**

And then the decision follows directly. Revenue is price times units, so a five percent price rise
multiplies revenue by one point oh five to the power one plus the elasticity. That is above one
exactly when the elasticity is above minus one — when demand is inelastic.

And slide 8 already told us every posterior draw is above minus one. So revenue rises in every
draw: by about **two and a half percent** on average, with probability **one point zero zero**.

*(Beat.)*

Notice how the two slides fit together. The elasticity's exact size is uncertain — we couldn't even
say whether a one percent rise costs more than half a percent of volume. But the revenue question
only cares which side of minus one we're on, and on that the posterior is unanimous. A decision can
be settled even when the parameter behind it is not pinned down, and the posterior shows you which
decisions those are.

*(Muted note.)*

With one caveat, and it's the one from slide 11. This is valid only to the extent the **DAG** is.
It's a causal claim about intervening on price, and it inherits every assumption in that diagram.

---

# ▶ SLIDE 14 — Section divider: Course Synthesis and Next Steps

From classical time series to Bayesian hierarchical models: a complete forecasting toolkit.

---

# ▶ SLIDE 15 — Course Method Map

*(Walk the table at a steady pace. Students will photograph this slide.)*

Everything we've built, on four axes: what data it takes, what uncertainty it gives you, how
interpretable it is, and what it's good at.

ETS and ARIMA — univariate series, confidence intervals, highly interpretable, fast and automatic.
And remember the benchmark beat every one of them in Homework 1, which is the first thing this
table should remind you of.

GAMs and Prophet — series plus predictors, limited uncertainty, interpretable, good at nonlinear
and interpretable together.

Decision trees — tabular, no uncertainty, but a rule-based explanation anyone can read.

Random forests — robust, minimal tuning, and the out-of-bag error comes free. That is an error
estimate, not an interval.

XGBoost — also Lecture 5. It captures interactions, and it scored seven eighty-one on our board.

Ridge and LASSO — Lecture 6. Linear, interpretable, and the best number on our scoreboard: seven
forty-four. The simplest model in the table beat the boosted trees.

Feedforward nets and CNNs — flexible nonlinear fits, fast, opaque.

LSTMs and Transformers — sequences and long-range dependencies, very low interpretability.

Bayesian — any data, **full posterior**, highly interpretable, and strongest exactly where samples
are small.

**[pause]**

*(Point at the key box.)*

**No single method dominates.** That's the honest summary of eleven weeks, and it's what the
scoreboard has been telling you all semester.

Choose on four things: your data type and size, whether you need uncertainty quantified, whether
someone needs to understand the model, and how fast you need it deployed.

And in practice, **combine**. Use whichever model wins your scoreboard for the point forecast — on
ours that was LASSO, ahead of XGBoost — plus a Bayesian model for the intervals, because they're
honest. You are not obliged to pick one row of that table.

---

# ▶ SLIDE 16 — Lecture 12: Key Takeaways

One. Bayesian linear regression places priors on coefficients and returns a full posterior for
each one. Its mode is Ridge under a Normal prior and LASSO under a Laplace prior.

Two. Posterior coefficient plots give you the mean, the HDI, the probability the coefficient
exceeds any threshold you care about, and how much of it sits inside a ROPE — richer than a
p-value, and answering the question people actually ask.

Three. Draw the **DAG** before you specify the regression. Control for the confounders — here that
meant both the series and time — and never for a collider.

Four. Scenario analysis with posterior samples: push any counterfactual through the posterior draws
and get a full distribution for that scenario, not a point projection.

Five. No single method is best for all forecasting problems. Build intuition for when to reach for
each one, and pair an accurate point forecast with Bayesian intervals.

**[pause]**

*(Close the course.)*

And the one to carry out of this room, which is really the lesson of the whole semester.

The uncontrolled elasticity was six times too large, with an interval **half the width** of the
correct one. Precision is not accuracy. A confident model is not a correct model. And the thing
that caught it was not a diagnostic, not a score, not a cross-validation fold — it was a picture
you draw before fitting anything, from what you know about the business.

That habit is the most transferable thing in this course.

---

# ▶ SLIDE 17 — References

*(Advance and close. No narration needed.)*
