# Lecture 10 — Recording Script

**ECON 8310: Business Forecasting · Bayesian Statistics I — Foundations**

Deck: `Slides/Lecture10_BayesianFoundations.pdf` (20 pages) · Measured runtime: see the timing guide

---

## How to use this document

- **`▶ SLIDE n — Title`** marks where to advance. The number is the PDF page.
- *Italic parentheticals* are stage directions. **[pause]** means stop for a beat.
- This lecture changes the *question*, not the accuracy. Slide 5 is where students decide whether
  Bayesian methods are worth their attention — spend time there and use the meeting framing, not
  the philosophy.
- Slide 16 introduces the three diagnostics that every remaining assignment requires. Be explicit
  that they are non-negotiable.

---

# ▶ SLIDE 1 — Title page

Lecture 10: Bayesian Statistics, part one — Foundations.

For nine weeks we've been in a competition. Different models, one test set, and the winner is
whoever has the lowest RMSE. That framing has been useful, and today I want to step outside it.

**[pause]**

Because every model so far has returned the same *object*: one number per period. ETS, ARIMA,
random forests, XGBoost, the RNN — a point forecast. And there are questions a business asks that
a point forecast cannot answer, no matter how accurate it is.

Today is about changing the deliverable. Not "what will demand be," but "what is the *distribution*
of demand, and what is the probability I stock out if I order four hundred units."

That's a different question, and it needs different machinery.

---

# ▶ SLIDE 2 — Lecture Outline

Five parts. Why you'd think this way at all — and I'll make that argument in business terms, not
philosophical ones. Then Bayes' theorem, which is one equation and simpler than its reputation.
Then priors, which is the part people object to. Then PyMC and, critically, how to tell whether
the answer you got is trustworthy. And takeaways.

---

# ▶ SLIDE 3 — Section divider: Why Bayesian Thinking?

A different answer to the question of what a forecast should tell you.

---

# ▶ SLIDE 4 — Frequentist vs. Bayesian: What Is Random?

The whole difference between the two frameworks is which thing you treat as uncertain. Everything
else follows.

*(Walk the table row by row.)*

**The parameter.** For a frequentist, it's a fixed unknown constant — there is one true value, you
just don't know it. For a Bayesian, it's a quantity you are uncertain about, and you describe that
uncertainty with a distribution.

**The data.** For a frequentist, the data is random — your sample is one draw from many possible
samples. For a Bayesian, the data is *fixed*. It's what you observed. It's sitting on your disk.

**The answer.** A point estimate and a standard error, versus a full distribution over values.

**Prior belief.** No formal place for it, versus stated explicitly and updated.

**[pause]**

Everything else follows from that first row, so let me make it concrete.

If the parameter is fixed and the data is random, then all you can describe is how your *estimate*
would vary across hypothetical repeated samples that you never actually took. If the parameter is
uncertain, you can describe your uncertainty about *it*, directly.

And that second thing is almost always what someone actually asked you.

**[pause]**

I want to head off a misconception, because it costs people a lot of confusion later.

These are not rival theories of probability where one is correct and the other is a mistake. They
are different questions, and each framework answers its own question well. Frequentist methods
give you guarantees about *procedures* repeated over the long run, which is exactly right for
quality control on a production line.

Bayesian methods give you a statement about *this* parameter given *this* data, which is what you
need when you have one dataset and one decision to make on Thursday.

Most forecasting sits in that second category. That's the reason for these three lectures — not
that the last nine weeks were wrong.

---

# ▶ SLIDE 5 — What This Buys You in a Meeting

*(This is the persuasive slide. Slow down and use it.)*

Here's the sharpest version of the difference.

A ninety-five percent **confidence** interval of point-oh-two to point-oh-eight does **not** mean
there's a ninety-five percent chance the parameter lies in that range. It means intervals
constructed this way would contain the true value ninety-five percent of the time across repeated
samples.

That is a statement about the *procedure*, not about your parameter.

A ninety-five percent **credible** interval of point-oh-two to point-oh-eight *does* mean there is
a ninety-five percent probability the parameter lies in that range, given your model and your data.

**[pause]**

*(Point at the key box.)*

Now — I know that sounds like a technicality that only statisticians care about. It isn't.

It is the difference between a sentence you can say out loud to a decision-maker and a sentence
you have to keep correcting. Everyone in every meeting you have ever been in interprets a
confidence interval as a credible interval. The Bayesian version is the one that means what
everybody already thinks it means.

**[pause]**

And there's a second, larger payoff. Because the answer is a full distribution, you can interrogate
it directly.

*What is the probability the promotion lifts demand by more than five percent?* Count the posterior
samples above five percent. *What's the chance we stock out if we order four hundred units?* Count
the samples above four hundred.

Those are the questions a planner actually has. A point estimate with a standard error does not
answer them — not because it's inaccurate, but because it isn't that kind of object.

*(Point at the muted footnote.)*

The cost is that you must state a prior, and defend it. That's the rest of the lecture.

---

# ▶ SLIDE 6 — Section divider: Bayes' Theorem

One equation, and it is a bookkeeping rule rather than a philosophy.

---

# ▶ SLIDE 7 — Bayes' Theorem

*(Point at the definition box, then read it as English.)*

Here's the equation. Posterior equals likelihood times prior, over the evidence.

Read it as a sentence, and it stops being intimidating: **what I believe after seeing the data**
equals **how well each parameter value explains the data**, times **what I believed beforehand**,
rescaled so it sums to one.

That's it. It's a bookkeeping rule for updating beliefs in light of evidence.

**[pause]**

The denominator — p of D, the evidence — is only that rescaling. It doesn't change the *shape* of
the distribution, only its normalization. So in practice people write the working form:

**Posterior is proportional to likelihood times prior.**

That proportionality is what we'll actually use, and it's why the denominator being uncomputable
turns out not to matter — which is the subject of slide 13.

*(Point at the last paragraph.)*

**[pause]**

One reassurance, since some of you are looking at that equation with concern.

**You will not compute this by hand.** This course never asks you to. There is no integral on any
homework in this course.

What you are responsible for is three things: choosing a prior you can defend, checking that the
sampler actually worked, and reading the posterior correctly. Those are judgment skills, and they
are the ones that matter in practice.

**[pause]**

And it's worth registering that the equation itself is not controversial. It's a theorem — it
follows from the definition of conditional probability in two lines, and no statistician of any
persuasion disputes it.

What was historically contested is whether you're entitled to put a probability distribution on a
*parameter* at all, as opposed to on data. That's a philosophical argument, it ran for most of the
twentieth century, and it is not one we're going to settle in this room.

What settled it in practice was computation. These methods were mathematically fine and
practically impossible until MCMC and cheap computers arrived in the 1990s. The philosophy didn't
change; the feasibility did.

---

# ▶ SLIDE 8 — The Update, in Pictures

Let's make it concrete with a case you could actually meet. Estimating a conversion rate — theta —
with a Beta five, forty-five prior, and then thirty successes in two hundred trials.

*(Point at the blue curve.)*

The **prior**, in blue, is what you thought before you looked. Its mean is point-one-zero — that's
the dotted line. Notice the peak sits a little left of the mean, at about point-oh-eight, because
the curve has a long right tail. When I say this prior is centered on point-one, I mean its mean.

*(Point at the dashed curve.)*

The **likelihood**, dashed, comes from the data alone. Thirty out of two hundred, so it peaks at
point-one-five.

*(Point at the red curve.)*

The **posterior**, in red, is what you get when you multiply those two curves point by point and
rescale. It lands *between* them, and it is taller and narrower than either, because it carries the
information of both.

**[pause]**

*(Point at the lever.)*

Now the lever underneath, because it is the whole mechanism in one picture. The posterior mean is a
balance point. Put the prior's point-one-zero on one side with a weight of fifty — that's how many
observations the prior is worth. Put the data's point-one-five on the other with a weight of two
hundred. The beam balances at **point-one-four**. Exactly: fifty times point-one, plus two hundred
times point-one-five, over two hundred fifty.

The squares are drawn with area proportional to weight, so the data's square is four times the
prior's. Between the two, much nearer the data — because two hundred observations outweigh a prior
worth fifty.

**[pause]**

*(Point at the right-hand panel.)*

And the panel on the right shows what happens as data accumulates. Keep the same fifteen percent
rate and change only how many trials it came from. Three out of twenty barely moves you:
point-one-one-four. Thirty out of two hundred gets you to point-one-four. Three hundred out of two
thousand, point-one-four-nine — the prior has almost vanished.

That's the whole mechanism, and it behaves the way you'd want. More data, and the posterior moves
toward it. Little data, and the prior does most of the work — which is a feature when the prior is
defensible, and a problem when it isn't.

---

# ▶ SLIDE 9 — Section divider: Prior Distributions

The part you have to justify, and the part people worry about.

---

# ▶ SLIDE 10 — Priors You Will Actually Use

A prior is just a distribution over a parameter, chosen to match what that parameter *can* be.

*(Walk the table.)*

**Beta**, on the interval zero to one — for rates and proportions. Conversion rates,
click-through, defect rates.

**Normal**, on all the reals — for regression coefficients, trends, anything that can legitimately
be negative.

**Exponential**, on zero to infinity — for standard deviations, waiting times, anything that must
be positive.

**[pause]**

*(Point at the support paragraph — this is the practical bit.)*

And the first thing to get right is the **support** — the set of values the distribution allows.

Put a Normal prior on a standard deviation and you have allowed negative standard deviations,
which are not a modeling approximation, they're impossible. The sampler will struggle, and the
model isn't so much wrong as **incoherent**.

That's the most common beginner error in Bayesian modeling, and it's entirely avoidable: match
the support to the parameter.

*(Point at the footnote.)*

One arithmetic note worth internalizing. Beta of five and forty-five has a mean of five over fifty
— point-one. And it carries about as much weight as fifty prior observations. That's a useful way
to think about prior strength: how many observations is this worth?

**[pause]**

That question — *how many observations is my prior worth?* — is the most useful one to ask when
you're choosing, and it turns an abstract decision into a concrete one.

If you have two hundred data points and your prior is worth fifty, the data will dominate but the
prior will still pull. If your prior is worth five thousand, you have effectively decided the
answer in advance and the data is decoration.

And notice the asymmetry that creates. With plenty of data, the prior barely matters and people
argue about it anyway. With very little data, the prior matters enormously — and that is exactly
the situation where Bayesian methods are most useful, and where you owe the most careful
justification.

---

# ▶ SLIDE 11 — Choosing a Prior You Can Defend

Now the objection. The standard criticism of Bayesian work is that the prior is arbitrary — that
you can get any answer you like by choosing the right one.

The standard answer is that you should not be choosing arbitrary priors.

*(Walk the three categories.)*

**Weakly informative** is your default, and it's the one you'll use in this course. It rules out
the absurd while staying agnostic about the plausible. A Normal zero-one on a standardized
coefficient says "probably not larger than about two," which is nearly always true and nearly
never controversial.

**Informative** priors encode real evidence — last year's conversion rate, a published elasticity.
Perfectly defensible *when you can name the source*.

**Flat**, or so-called uninformative, sounds neutral. It often isn't, it can be improper, and it
is a bad reflex choice. Avoid it.

**[pause]**

*(Point at the closing paragraph. This is the argument to make with conviction.)*

And here's the honest framing of the whole objection.

A prior is an assumption **stated in public**, in a form where it can be argued with. Somebody can
read your model, disagree with your prior, change it, and re-run.

Frequentist analyses contain assumptions too — functional form, which variables you included,
which you dropped, the sampling scheme, what you did with outliers. They are just not written down
as distributions, so they're harder to see and harder to contest.

The Bayesian prior isn't an extra assumption. It's an assumption made visible.

---

# ▶ SLIDE 12 — Section divider: Posterior Inference with PyMC

Sampling, the code, and how to tell whether the answer is trustworthy.

---

# ▶ SLIDE 13 — MCMC: The Big Idea

Back to that denominator. For any model more complicated than a textbook example, it cannot be
computed — it's an integral over every possible parameter combination.

So we don't compute the posterior. We **draw samples from it**, and then work entirely with the
samples.

*(Tell the picture as a story.)*

Here's the picture I want you to hold. You are exploring a hilly landscape in the dark, where
height corresponds to posterior probability. You take a step. If the ground is higher, you move
there. If it's lower, you usually move back — usually, not always.

Do that thousands of times and you spend most of your time on the high ground, proportionally to
how high it is.

*(Point at the footnote.)*

PyMC's actual sampler is NUTS, which takes far smarter steps than the random walk I described. The
intuition is unchanged.

**[pause]**

One consequence of the sampling approach is worth flagging now, because it will shape how you plan
your work.

This is **slow**. Not slow like XGBoost is slow — slow in a different category. A random forest on
our panel fits in seconds. A Bayesian model on the same data can take minutes, and a poorly
specified one can take much longer.

You are not solving an optimization problem and stopping. You are wandering around a distribution,
thousands of steps, four times over.

That has a practical implication for the remaining assignments: start them early, and test your
model on a small subset before you run the full thing. Discovering a specification error forty
minutes into sampling is a genuinely miserable experience, and one you can avoid.

---

# ▶ SLIDE 14 — The Samples Are the Answer

And here is the trick that makes it all work: **the record of where you walked is the answer.**

A histogram of the positions you visited has the shape of the posterior. You never computed the
posterior. You just wandered around in it and wrote down where you'd been.

*(Point at the key box.)*

So everything downstream is counting. Posterior mean is the average of your visited values.
Credible interval is a percentile of them. The probability that a parameter exceeds a threshold is
the fraction of samples above it.

**Counted, not derived.** That's why Bayesian output can answer arbitrary questions — you're just
querying a list of numbers.

*(Point at the last line.)*

Remember the planner from slide 5, asking about stocking out at four hundred units? Simulate demand
from the posterior draws, and that question becomes one line of code: the share of simulated demand
above four hundred.

---

# ▶ SLIDE 15 — PyMC: A First Model

Here's the conversion-rate problem, in six lines.

A Beta prior with alpha five and beta forty-five — that's our point-one prior worth fifty
observations. A Binomial likelihood: two hundred trials, thirty observed successes. Then
`pm.sample`, and you're done.

**[pause]**

*(This framing is worth stating explicitly — it's the conceptual shift.)*

Notice what you actually wrote. You did not specify an estimation procedure. You did not choose an
optimizer or a loss function.

You described **how the data could have been generated**: here's what I believe about the rate,
and here's how observations arise given a rate. That's a *story about the world*.

You write the story; the sampler does the inference. That is a genuinely different way of working
from everything else in this course, where you chose a fitting algorithm.

*(Point at the arguments.)*

Two arguments to understand. `tune=1000` discards the first thousand steps, while the sampler is
still finding the high ground — those steps aren't from the posterior yet. And `chains=4` runs
four *independent* walks from different starting points.

That second one seems wasteful. It's what makes the next slide possible.

Think about why four independent walks are worth the compute. If a single chain wanders into some
corner of the parameter space and gets stuck there, it will happily report a confident, tight,
completely wrong posterior — and from inside that one chain, there is no way to tell.

Run four from different starting points and ask whether they agree. If they do, that's evidence
they all found the same distribution. If they don't, you know something is wrong even though no
individual chain looks troubled.

It's the same logic as the ensemble in Lecture 5, applied to diagnosis rather than prediction:
independent attempts agreeing is evidence; a single confident attempt is not.

---

# ▶ SLIDE 16 — Did the Sampler Actually Work?

*(Emphasize this slide. Every remaining assignment requires these three numbers.)*

MCMC can fail **quietly**. It always returns samples. Whether those samples came from your
posterior is an entirely separate question, and nothing raises an exception if they didn't.

So there are three checks, and this course requires all three on every model from here on.

*(Walk the table.)*

**R-hat**, and you want it below one-point-oh-one. It compares the four chains: did independent
walks, started in different places, end up describing the same distribution? If they didn't,
nothing else on the output matters.

**ESS** — effective sample size — and you want it above four hundred. MCMC draws are correlated
with each other, so eight thousand draws may carry only as much information as two hundred
independent ones. ESS tells you how many you effectively have.

**Divergences**, and you want exactly zero. A divergence means the sampler hit terrain it could not
navigate — typically a sharp funnel. Even a handful means parts of the posterior were explored
wrongly. We'll meet this for real in Lecture 11, twice.

**[pause]**

*(Point at the code line. This is a practical trap.)*

One practical note that will save you confusion. Call `az.summary` with **both**
`ci_kind="hdi"` and `ci_prob=0.94`.

ArviZ now defaults to an eighty-nine percent *equal-tailed* interval, not the HDI. If you don't
pass those arguments you will get a different interval than the one you think you're reading, and
than the one on these slides.

On our simple model: R-hat one-point-oh-oh, ESS three thousand six hundred eighty-six, zero
divergences. **Report all three, every time.** A posterior mean without its diagnostics is not a
result.

---

# ▶ SLIDE 17 — Prior Predictive Checks

One more habit, and it's the one most often skipped.

**Before you touch the data**, simulate from the prior alone. Two lines: `sample_prior_predictive`,
then plot it.

What you're doing is asking your model to hallucinate datasets, using only your priors, and then
looking at whether those datasets could plausibly have come from your business.

**[pause]**

*(Give the three failure signatures concretely.)*

Three things to look for.

**Impossible values** — negative sales, probabilities above one. That means the prior is
misspecified, not merely wide. Fix the support.

**Absurd ranges** — simulated weekly demand spanning zero to a billion units. That prior is too
vague, and vague is not the same as neutral.

**Identical simulations** — every simulated dataset looking the same. Now the prior is too tight
and the data won't be able to move it.

*(Point at the key box.)*

**[pause]**

This is **model debugging that costs two lines**, and it catches errors before you've spent an hour
sampling. It's required in Homework 6, and I'd encourage you to make it reflexive.

The general principle is one you've met in other forms all semester: check the thing that could
silently be wrong, before it costs you.

---

# ▶ SLIDE 18 — Section divider: Key Takeaways

Bayesian inference: beliefs in, beliefs out. Data updates your prior into a posterior.

---

# ▶ SLIDE 19 — Lecture 10: Key Takeaways

One. Bayesian inference treats parameters as uncertain. The posterior is what you believe after
seeing the data.

Two. Bayes' theorem: posterior is proportional to likelihood times prior. It lands between the two,
nearer whichever carries more weight — the lever from slide 8.

Three. Priors: match the support to the parameter first, and use weakly informative ones by
default.

Four. MCMC, specifically NUTS in PyMC, draws samples from the posterior, and every summary you want
— mean, HDI, the probability a parameter exceeds a threshold — is counted from them. Counted, not
derived.

Five. Check before you trust: R-hat, effective sample size and divergences after you sample, and a
prior predictive check before you fit. Every time.

**[pause]**

And the one that isn't numbered, which is really the point of the hour: **this doesn't make your
forecast more accurate.** It changes what the forecast *is*, from a number into a distribution —
and that's what lets you answer a question about probability.

Next time: Bayesian time series. Trend and seasonality with priors attached — and a sampler
failure that looks exactly like success.

---

# ▶ SLIDE 20 — References

*(Advance and close. No narration needed.)*
