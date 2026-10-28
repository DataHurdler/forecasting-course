# Lecture 10 — Recording Script

**ECON 8310: Business Forecasting · Bayesian Statistics I — Foundations**

Deck: `Slides/Lecture10_BayesianFoundations.pdf` (22 pages) · Measured runtime: see the timing guide

---

## How to use this document

- **`▶ SLIDE n — Title`** marks where to advance. The number is the PDF page.
- *Italic parentheticals* are stage directions. **[pause]** means stop for a beat.
- This lecture changes the *question*, not the accuracy. Slide 5 is where students decide whether
  Bayesian methods are worth their attention — spend time there and use the meeting framing, not
  the philosophy.
- Slide 17 introduces the three diagnostics that every remaining assignment requires. Be explicit
  that they are non-negotiable.

---

# ▶ SLIDE 1 — Title page

This is Lecture 10, Bayesian Statistics Part One, and today is about the foundations.

For ten weeks we've been running a competition. We fit different models, we score them on one test
set, and the winner is whichever one has the lowest RMSE. That framing has served us well, and
today I want to step outside it.

**[pause]**

Here's why. Every model so far has returned the same kind of *object*, which is one number per
period. ETS, ARIMA, random forests, XGBoost, the recurrent networks and last week's Transformer all
hand you a point forecast. And there are questions a business asks that a point forecast can't
answer, no matter how accurate it is.

So today is about changing the deliverable. The question stops being "what will demand be?" and
becomes "what is the *distribution* of demand, and what's the probability that I stock out if I
order four hundred units?"

That's a different question, and it needs different machinery.

---

# ▶ SLIDE 2 — Lecture Outline

We'll take this in five parts. First, why you'd think this way at all, and I'll make that argument
in business terms rather than philosophical ones. Second, Bayes' theorem, which is a single
equation and a lot simpler than its reputation. Third, priors, which are the part people object
to. Fourth, PyMC and, just as important, how to tell whether the answer you got is trustworthy.
And we'll finish with the takeaways.

---

# ▶ SLIDE 3 — Section divider: Why Bayesian Thinking?

This first section is about a different answer to the question of what a forecast should tell you.

---

# ▶ SLIDE 4 — Frequentist vs. Bayesian: What Is Random?

The whole difference between the two frameworks comes down to which thing you treat as uncertain.
Everything else follows from that.

*(Walk the table row by row.)*

Start with the parameter. For a frequentist, it's a fixed but unknown constant. There's one true
value, and you just don't know it. For a Bayesian, it's a quantity you're uncertain about, and you
describe that uncertainty with a distribution.

Next comes the data. For a frequentist, the data is random: your sample is one draw out of many
samples you could have gotten. For a Bayesian, the data is *fixed*. It's what you observed, and
it's sitting on your disk.

Then there's the answer. The frequentist gives you a point estimate and a standard error, and the
Bayesian gives you a full distribution over values.

And last, there's prior belief. The frequentist framework has no formal place for it, while the
Bayesian one states it explicitly and then updates it.

**[pause]**

Everything else follows from that first row, so let me make it concrete.

If the parameter is fixed and the data is random, then all you can describe is how your *estimate*
would vary across hypothetical repeated samples, samples you never actually took. If the parameter
is uncertain, you can describe your uncertainty about *it*, directly.

And that second thing is almost always what someone actually asked you.

I want to head off a misconception now, because it causes a lot of confusion later.

These aren't rival theories of probability where one is right and the other is a mistake. They
answer different questions, and each framework answers its own question well. Frequentist methods
give you guarantees about *procedures* repeated over the long run, which is exactly right for
quality control on a production line.

Bayesian methods give you a statement about *this* parameter, given *this* data, which is what you
need when you have one dataset and one decision to make on Thursday.

Most forecasting sits in that second category. That's the reason for this lecture and the Bayesian
lectures that follow it. It isn't that the last ten weeks were wrong.

---

# ▶ SLIDE 5 — What This Buys You in a Meeting

*(This is the persuasive slide. Slow down and use it.)*

Here's the sharpest version of the difference.

Suppose you report a ninety-five percent **confidence** interval that runs from zero point zero two
to zero point zero eight. That does *not* mean there's a ninety-five percent chance the parameter
lies in that range. It means that intervals built this way would contain the true value
ninety-five percent of the time, across repeated samples.

That's a statement about the *procedure*, not about your parameter.

A ninety-five percent **credible** interval from zero point zero two to zero point zero eight
*does* mean there's a ninety-five percent probability that the parameter lies in that range, given
your model and your data.

**[pause]**

*(Point at the key box.)*

Now, I know that sounds like a technicality only statisticians care about. It isn't.

It's the difference between a sentence you can say out loud to a decision-maker and a sentence you
have to keep correcting. Everyone in every meeting you've ever been in reads a confidence interval
as if it were a credible interval. The Bayesian version is the one that means what everybody
already thinks it means.

And there's a second, larger payoff. Because the answer is a full distribution, you can question
it directly.

What's the probability that the promotion lifts demand by more than five percent? Count the
posterior samples above five percent. What's the chance we stock out if we order four hundred
units? Count the samples above four hundred.

Those are the questions a planner actually has. A point estimate with a standard error doesn't
answer them, and that isn't because it's inaccurate. It's because it isn't that kind of object.

**[pause]**

*(Point at the muted footnote.)*

The cost is that you have to state a prior, and defend it. That's what the rest of the lecture is
about.

---

# ▶ SLIDE 6 — Section divider: Bayes' Theorem

The next section is one equation, and it's a bookkeeping rule rather than a philosophy.

---

# ▶ SLIDE 7 — Bayes' Theorem

*(Point at the definition box, then read it as English.)*

Here's the equation. Theta is the unknown, whatever you're uncertain about, and D is the data. The
posterior equals the likelihood times the prior, divided by the evidence.

Read it as a sentence, and it stops being intimidating. **What I believe after seeing the data**
equals how well each parameter value explains the data, times what I believed beforehand, rescaled
so that it all sums to one.

That's all it is: a bookkeeping rule for updating beliefs in light of evidence.

The denominator, p of D, which is called the evidence, is only that rescaling. It doesn't change
the *shape* of the distribution, just its scale. So in practice people write the working form,
which says that the posterior is proportional to the likelihood times the prior.

That proportionality is what we'll actually use. It's also why it turns out not to matter that the
denominator can't be computed, and that's the subject of slide fourteen.

**[pause]**

*(Point at the last paragraph.)*

Here's one reassurance, since some of you are looking at that equation with concern. **You won't
compute this by hand.** This course never asks you to, and there's no integral on any homework.

What you're responsible for is three things: choosing a prior you can defend, checking that the
sampler actually worked, and reading the posterior correctly. Those are judgment skills, and
they're the ones that matter in practice.

**[pause]**

It's also worth knowing that the equation itself isn't controversial. It's a theorem. It follows
from the definition of conditional probability in two lines, and no statistician of any school
disputes it.

What was historically contested is whether you're entitled to put a probability distribution on a
*parameter* at all, rather than only on data. That's a philosophical argument. It ran for most of
the twentieth century, and we're not going to settle it in this room.

What settled it in practice was computation. These methods were mathematically sound and
practically impossible until MCMC and cheap computers arrived in the nineteen-nineties. The
philosophy didn't change, but the feasibility did.

---

# ▶ SLIDE 8 — Bayes' Theorem, Drawn

*(Point at the square before saying anything about the equation.)*

That equation looks abstract, so let me draw it with our own data. These are two hundred
seventy-seven weeks of FOODS sales at store CA_1, the same weeks you'll use in the lab.

I'll ask two yes-or-no questions about each week. The first is whether it was a **SNAP week**. SNAP
is the federal food-assistance program, and in California the benefit days are the first ten days
of every month. A week with five or more of those days is what the lab calls a high-SNAP week. The
second question is whether it was a strong week, meaning one whose units beat the median for its
own calendar year. Comparing each week with its own year takes out the growth, so a week in twenty
eleven competes with twenty eleven, not with twenty fifteen.

*(Trace the vertical split.)*

The whole square has an area of one, and it stands for all two hundred seventy-seven weeks. First,
split it left and right by how common SNAP weeks are. Seventy-two of the weeks are SNAP weeks,
about a quarter, so the left column is zero point two six wide. That width is the prior: how likely
a SNAP week is before you know anything about sales.

*(Trace the horizontal split in each column.)*

Now split each column by how often its weeks were strong. In the SNAP column, fifty-five of the
seventy-two were strong, which is zero point seven six of the height. That's the likelihood, how
probable a strong week is if it's a SNAP week. In the other column, eighty-two of two hundred five
were strong, which is zero point four zero.

Every block's area is now a joint probability, its width times its height. The crimson block is
zero point two six times zero point seven six, about a fifth of the square, and it holds fifty-five
weeks. So likelihood times prior is an area.

**[pause]**

*(Point at the strip under the square.)*

Now read the square the other way. I tell you a week was strong, and I ask whether it was a SNAP
week. Now you're inside the strong area only, crimson plus pink. That's a hundred thirty-seven of
the two hundred seventy-seven weeks, or zero point four nine of the square, and that area is the
evidence, p of D. The strip underneath lays the two blocks end to end, so you can compare them by
length instead of guessing at areas.

Fifty-five of those hundred thirty-seven weeks are crimson, which is zero point four zero. That's
the posterior.

*(Point at the formula, then down the table.)*

And that's the equation, term by term. Zero point seven six times zero point two six, divided by
zero point four nine, is zero point four zero. I said you won't compute Bayes' theorem by hand. With
two yes-or-no questions, it isn't computing at all. It's counting.

Here's the part worth remembering. Seventy-six percent of SNAP weeks are strong, but only forty
percent of strong weeks are SNAP weeks. Those numbers answer different questions, and people swap
them all the time.

Why are they so different? Look at the widths. The SNAP column is narrow, because SNAP weeks are
only a quarter of all weeks. So even though most of that column is crimson, its block is smaller
than the pink block sitting in the wide column. **The prior is the width.** Make SNAP weeks rarer,
and the posterior falls, even if the likelihood stays exactly where it is.

*(Point at the muted footnote.)*

There's one trap on this slide, which is that the number zero point four zero appears twice. The
gray one, on the right edge, is the share of other weeks that were strong, eighty-two of two hundred
five. The crimson one is the posterior. They match by coincidence, and nothing in the theorem makes
them equal.

**[pause]**

Notice what the evidence did. It divided fifty-five and eighty-two by the same hundred
thirty-seven. The ratio between them didn't change. Dividing only made them add up to one, giving
zero point four and zero point six. That's what I meant on the last slide by rescaling rather than
reshaping. With two yes-or-no answers, the evidence is a single sum you can count. With a parameter
that can take any value, like the conversion rate on the next slide, it becomes an integral, and
that's why the rest of this lecture samples instead.

*(If there's time, add the two caveats.)*

I have two honest caveats about the data. The first is that twenty eleven and twenty sixteen are
partial years. The data start at the end of January twenty eleven and stop in May twenty sixteen,
so those two medians are taken over fewer weeks. The pattern doesn't lean on them, though. In every
one of the six years, SNAP weeks were strong more often than other weeks.

The second caveat is that the definition is a choice. If "strong" meant "beat the same week last
year," SNAP would tell you almost nothing. Fifty-eight percent of SNAP weeks beat last year,
against fifty-seven percent of other weeks. The reason is that last year's same week was usually a
SNAP week too, eighty-five percent of the time, so the comparison cancels out the SNAP lift.
Deciding what counts as the event is part of the model.

---

# ▶ SLIDE 9 — The Update, in Pictures

The square had a yes-or-no theta. Most parameters are numbers instead, so let's take one. Theta is
now a conversion rate, with a Beta five, forty-five prior, and then the data arrive: thirty
successes in two hundred trials. You can read a Beta prior as a count you already have. Beta five,
forty-five is as if you'd seen five successes in fifty trials before the data arrived.

*(Point at the blue curve.)*

The prior, in blue, is what you thought before you looked. Its mean is zero point one zero, and
that's the dotted line. Notice that the peak sits a little to the left of the mean, at about zero
point zero eight, because the curve has a long right tail. So when I say this prior is centered on
zero point one, I mean its mean.

*(Point at the dashed curve.)*

The likelihood, the dashed curve, comes from the data alone. That's thirty out of two hundred, so it
peaks at zero point one five.

*(Point at the red curve.)*

The posterior, in red, is what you get when you multiply those two curves point by point and
rescale. It lands *between* them, and it's taller and narrower than either one, because it carries
the information of both.

**[pause]**

*(Point at the lever.)*

Now look at the lever underneath, because it's the whole mechanism in one picture. The posterior
mean is a balance point. Put the prior's zero point one zero on one side, with a weight of fifty,
because that's how many observations the prior is worth. Put the data's zero point one five on the
other side, with a weight of two hundred. The beam balances at **zero point one four**, exactly.
That's fifty times zero point one, plus two hundred times zero point one five, all divided by two
hundred fifty.

The squares are drawn with their area proportional to the weight, so the data's square is four
times the size of the prior's. The balance point sits between the two, much nearer the data,
because two hundred observations outweigh a prior worth fifty.

**[pause]**

*(Point at the right-hand panel.)*

The panel on the right shows what happens as data accumulates. Keep the same fifteen percent rate,
and change only how many trials it came from. Three out of twenty barely moves you, to zero point
one one four. Thirty out of two hundred gets you to zero point one four. Three hundred out of two
thousand gets you to zero point one four nine, and by then the prior has almost vanished.

That's the whole mechanism, and it behaves the way you'd want. With more data, the posterior moves
toward the data. With little data, the prior does most of the work, which is a feature when the
prior is defensible and a problem when it isn't.

---

# ▶ SLIDE 10 — Section divider: Prior Distributions

The next section is about priors, which are the part you have to justify and the part people worry
about.

---

# ▶ SLIDE 11 — Priors You Will Actually Use

A prior is just a distribution over a parameter, chosen to match the values that parameter *can*
take.

*(Walk the first three panels, left to right. Each curve is drawn over exactly the values it allows.)*

The first is the Beta, which lives on the interval from zero to one. It's for rates and
proportions, like conversion rates, click-through rates and defect rates. This one is our Beta five,
forty-five, bunched up near zero point one.

The second is the Normal, which covers all the real numbers. It's for regression coefficients,
trends, and anything else that can legitimately be negative.

The third is the Exponential, which runs from zero to infinity. It's for standard deviations,
waiting times, and anything else that has to be positive. Notice that it simply stops at zero.

**[pause]**

*(Point at the right-hand panel. This is the practical part.)*

The first thing to get right is the **support**, which is the set of values the distribution
allows.

The dashed curve is a Normal zero, one, put on a standard deviation, sigma. Everything to the left of
the line at zero is shaded red, because a negative standard deviation isn't a modeling
approximation. It's impossible. And that red area is exactly half the prior. Half of what you said
you believed is about values that can't exist. The sampler will struggle, and the model isn't so
much wrong as incoherent.

*(Trace the arrow under the curves.)*

The fix is the solid blue curve. Take that impossible half and fold it over onto the positive side.
You get the same shape at twice the height, with all of it on values of sigma at or above zero.
That's the HalfNormal, and you'll see it again on the standard deviations of the hierarchical model
in Lecture 11, Part 2, which is two weeks from now.

That's the most common beginner error in Bayesian modeling, and it's entirely avoidable. Match the
support to the parameter.

*(Back to the first panel.)*

Here's one piece of arithmetic worth internalizing. Beta five, forty-five has a mean of five over
fifty, which is zero point one. And it carries about as much weight as fifty prior observations.
That's a useful way to think about the strength of a prior: how many observations is it worth?

**[pause]**

That question, how many observations your prior is worth, is the most useful one to ask when you're
choosing a prior, because it turns an abstract decision into a concrete one.

If you have two hundred data points and your prior is worth fifty, the data will dominate, but the
prior will still pull. If your prior is worth five thousand, you've effectively decided the answer
in advance, and the data is decoration.

And notice the asymmetry that creates. With plenty of data, the prior barely matters, and people
argue about it anyway. With very little data, the prior matters enormously, and that's exactly the
situation where Bayesian methods are most useful and where you owe the most careful justification.

---

# ▶ SLIDE 12 — Choosing a Prior You Can Defend

Now for the objection. The standard criticism of Bayesian work is that the prior is arbitrary, and
that you can get any answer you like by choosing the right one.

The standard answer is that you shouldn't be choosing arbitrary priors.

*(Walk the three categories.)*

There are three kinds to know. The first kind, weakly informative, is your default, and it's the
one you'll use in this course. A weakly informative prior rules out the absurd while staying
agnostic about the plausible. A Normal zero, one on a standardized coefficient says the coefficient
is "probably not larger than about two," which is nearly always true and nearly never controversial.

Informative priors encode real evidence, like last year's conversion rate or a published
elasticity. They're perfectly defensible *when you can name the source*.

A flat prior, sometimes called uninformative, sounds neutral. It often isn't, and it can be
improper, which means it isn't a real probability distribution at all. It's a bad reflex choice, so
avoid it.

**[pause]**

*(Point at the closing paragraph. This is the argument to make with conviction.)*

And here's the honest framing of the whole objection.

A prior is an assumption **stated in public**, in a form where people can argue with it. Somebody
can read your model, disagree with your prior, change it, and run the analysis again.

Frequentist analyses contain assumptions too: the functional form, which variables you included and
which you dropped, the sampling scheme, and what you did with outliers. They just aren't written
down as distributions, so they're harder to see and harder to contest.

So the Bayesian prior isn't an extra assumption. It's an assumption made visible.

---

# ▶ SLIDE 13 — Section divider: Posterior Inference with PyMC

This section covers sampling, the code, and how to tell whether the answer is trustworthy.

---

# ▶ SLIDE 14 — MCMC: The Big Idea

Let's go back to that denominator. For any model more complicated than a textbook example, it can't
be computed, because it's an integral over every possible combination of parameter values.

So we don't compute the posterior. We **draw samples from it**, and then we work entirely with the
samples.

*(Tell the picture as a story.)*

Here's the picture I want you to hold on to. You're exploring a hilly landscape in the dark, where
height stands for posterior probability. You take a step. If the ground is higher, you move there.
If it's lower, you usually move back, though not always.

*(Point at the left panel.)*

The left panel is that landscape for our conversion rate, where the height is likelihood times
prior. You're standing at zero point one five, which is the black stick. Someone proposes a step to
zero point one eight, the blue stick. It's lower, so you move with a probability equal to the ratio
of the two heights, which is about zero point two three. You roll the dice, and most of the time
you stay put.

**[pause]**

Here's why the uncomputable denominator stops mattering. The true posterior height at each point is
likelihood times prior, *divided by p of D*. Both sticks are divided by the same p of D, so when you
take their ratio, it cancels. The walk only ever needs heights you *can* compute.

*(Point at the right panel.)*

On the right are four of these walks, started deliberately far apart: one near zero, one at zero
point eight, and two in between. Within about sixty-five steps, every one of them has found the same
band around zero point one four, and from then on you can't tell them apart.

The gray region is the warm-up. Those first thousand steps get thrown away, because the early ones
still remember where they started. What's kept is two thousand steps per chain, which makes eight
thousand draws.

Do that long enough, and you spend time at each height in proportion to how high it is.

*(Point at the footnote.)*

This is the textbook random-walk sampler, which is called Metropolis. PyMC's actual sampler is
called NUTS, and it takes far smarter steps, but the intuition is the same.

**[pause]**

One consequence of the sampling approach is worth flagging now, because it'll shape how you plan
your work.

This is **slow**, and not slow the way XGBoost is slow. It's a different category of slow. A random
forest on our panel fits in seconds. A Bayesian model on the same data can take minutes, and a
poorly specified one can take much longer.

You're not solving an optimization problem and stopping. You're wandering around a distribution,
for thousands of steps, four times over.

That has a practical implication for the remaining assignments. Start them early, and test your
model on a small subset before you run the full thing. Discovering a specification error forty
minutes into sampling is a genuinely miserable experience, and it's one you can avoid.

---

# ▶ SLIDE 15 — The Samples Are the Answer

And here's the trick that makes it all work: **the record of where you walked is the answer.**

*(Point at the histogram.)*

These bars are the eight thousand kept draws from the previous slide, and nothing more. The black
curve is the exact Beta thirty-five, two hundred fifteen, which we can only write down because this
example is simple. The bars follow the curve. You never computed the posterior. You just wandered
around in it and wrote down where you'd been.

*(Point at the red bars.)*

Now let's ask a question. What's the probability that the conversion rate is above fifteen percent?
Count the draws to the right of the dashed line, and you get zero point three three. The exact
answer is zero point three one. That's close, and it gets closer with more draws.

Why only close? Because neighboring steps of a random walk are correlated. So eight thousand draws
here carry about as much information as seven hundred eighty independent ones. NUTS, on the same
model, gets more than three thousand. That's what smarter steps buy you, and it's the effective
sample size you'll meet on slide seventeen.

*(Point at the key box.)*

So everything downstream is counting. The posterior mean is the average of the values you visited.
A credible interval comes from their percentiles. And the probability that a parameter exceeds a
threshold is the fraction of samples above it.

Every summary is **counted, not derived**. That's why Bayesian output can answer arbitrary
questions. You're just querying a list of numbers.

*(Point at the last line.)*

Remember the planner from slide five, who asked about stocking out at four hundred units? Simulate
demand from the posterior draws, and that question becomes one line of code: the share of simulated
demand above four hundred.

---

# ▶ SLIDE 16 — PyMC: A First Model

Here's the conversion-rate problem, in six lines.

There's a Beta prior with alpha five and beta forty-five, which is our prior centered on zero point
one and worth fifty observations. Then there's a Binomial likelihood, with two hundred trials and
thirty observed successes. Then you call `pm.sample`, and you're done.

**[pause]**

*(This framing is worth stating explicitly. It's the conceptual shift.)*

Notice what you actually wrote. You didn't specify an estimation procedure, and you didn't choose an
optimizer or a loss function.

You described **how the data could have been generated**. Here's what I believe about the rate, and
here's how observations arise, given a rate. That's a *story about the world*.

You write the story, and the sampler does the inference. That's a genuinely different way of
working from everything else in this course, where you chose a fitting algorithm.

*(Point at the arguments.)*

There are two arguments to understand. The first, `tune=1000`, discards the first thousand steps,
while the sampler is still finding the high ground. Those steps aren't from the posterior yet, and
they're the gray band on slide fourteen. The second, `chains=4`, runs four *independent* walks from
different starting points.

That second one seems wasteful, but it's what makes the next slide possible.

Think about why four independent walks are worth the compute. If a single chain wanders into some
corner of the parameter space and gets stuck there, it will happily report a confident, tight
posterior that's completely wrong, and from inside that one chain, there's no way to tell.

So run four chains from different starting points, and ask whether they agree. If they do, that's
evidence they all found the same distribution. If they don't, you know something is wrong, even
though no individual chain looks troubled.

It's the same logic as the random forests in Lecture 5, where many independently grown trees were
averaged, applied here to diagnosis instead of prediction. Independent attempts that agree are
evidence, and a single confident attempt is not.

---

# ▶ SLIDE 17 — Did the Sampler Actually Work?

*(Emphasize this slide. Every remaining assignment requires these three numbers.)*

MCMC can fail **quietly**. It always returns samples. Whether those samples came from your
posterior is an entirely separate question, and nothing raises an exception if they didn't.

So there are three checks, and this course requires all three on every model from here on.

*(Walk the table.)*

The first is R-hat, and you want it below one point zero one. It compares the four chains, and it
asks whether independent walks, started in different places, ended up describing the same
distribution. If they didn't, nothing else in the output matters.

The second is ESS, the effective sample size, and you want it above four hundred. MCMC draws are
correlated with each other, so eight thousand draws may carry only as much information as two
hundred independent ones. ESS tells you how many you effectively have.

The third is divergences, and you want exactly zero. A divergence means the sampler hit terrain it
couldn't navigate, typically a sharp funnel. Even a handful means parts of the posterior were
explored poorly. We'll meet divergences for real in Lecture 11, Part 2, on hierarchical models, two
weeks from now. Next week's model, in Lecture 11, Part 1, fails the other way, the way only R-hat
and the effective sample size can catch.

**[pause]**

*(Point at the code line. This is a practical trap.)*

Here's a practical note that will save you some confusion. Call `az.summary` with **both**
`ci_kind="hdi"` and `ci_prob=0.94`.

ArviZ now defaults to an eighty-nine percent *equal-tailed* interval, not the HDI. If you don't pass
those arguments, you'll get a different interval from the one you think you're reading, and a
different one from the interval on these slides.

The HDI, the highest-density interval, is the narrowest interval that holds ninety-four percent of
the draws. So why ninety-four, when slide five said ninety-five? Nothing special happens at either
one. The level is a choice, so say which one you used.

On our simple model, R-hat is one point zero zero, the ESS is three thousand two hundred ninety, and
there are zero divergences. **Report all three, every time.** A posterior mean without its
diagnostics is not a result.

---

# ▶ SLIDE 18 — Prior Predictive Checks

There's one more habit, and it's the one most often skipped.

Before you touch the data, you **simulate from the prior alone**. It takes two lines: you call
`sample_prior_predictive`, and then you plot the result.

What you're doing is asking your model to hallucinate datasets using only your priors, and then
looking at whether those datasets could plausibly have come from your business.

**[pause]**

*(Point at the paragraph.)*

The first thing to look for is impossible values, like negative sales or probabilities above one.
Those mean the prior is misspecified, not merely wide. Fix the support, the way we did on slide
eleven.

*(Point at the key box.)*

This is **model debugging that costs two lines**, and it catches errors before you've spent an hour
sampling. It's required in Homework 6, and I'd encourage you to make it a reflex.

The general principle is one you've met in other forms all semester: check the thing that could be
silently wrong, before it costs you.

---

# ▶ SLIDE 19 — Three Priors, Three Predictions

Here's what that check looks like on our conversion-rate model. Before any data arrive, each prior
makes a prediction: out of two hundred trials, how many successes will there be?

*(Point at the top row.)*

The top row is the flat prior, Beta one, one. Every count from zero to two hundred gets the same
probability, one in two hundred one. So it thinks a hundred ninety conversions out of two hundred is
exactly as plausible as twenty. Nobody who has run a website believes that. That's the signature of
a prior that's **too vague**, the same thing as simulated weekly demand that spans zero to a billion
units, and it's a reminder that vague isn't the same as neutral.

*(Point at the bottom row.)*

Now look at Beta five hundred, forty-five hundred. It has the same center, zero point one, but it's
worth five thousand observations. It has already decided that theta is zero point one before a
single trial, so ninety percent of its predictions fall between thirteen and twenty-seven successes.
That's a prior that's **too tight**, and two hundred trials of data won't be able to move it much.

Notice that the bars don't collapse onto a single count, though. Even with theta pinned down
exactly, two hundred coin flips at ten percent still scatter. That spread is binomial noise, and no
prior can remove it.

*(Point at the middle row.)*

Beta five, forty-five sits in between. Ninety percent of its predictions fall between seven and
thirty-seven. It rules out the absurd and leaves room for the data to speak.

**[pause]**

Here's one aside, for after the fact. When the thirty successes did arrive, the tight prior had
given thirty or more less than a two percent chance. The middle prior gave it about fifteen percent.
A prior predictive check is how you find that out *before* you're embarrassed by it.

---

# ▶ SLIDE 20 — Section divider: Key Takeaways

Bayesian inference is beliefs in and beliefs out, and the data is what updates your prior into a
posterior. Let's pull the lecture together.

---

# ▶ SLIDE 21 — Lecture 10: Key Takeaways

Here are the main points.

First, Bayesian inference treats parameters as uncertain, and the posterior is what you believe
after seeing the data.

Second, Bayes' theorem says the posterior is proportional to the likelihood times the prior. It
lands between the two, nearer whichever one carries more weight, which is the lever from slide nine.

Third, for priors, match the support to the parameter first, and use weakly informative priors by
default.

Fourth, MCMC, which in PyMC means NUTS, draws samples from the posterior. Every summary you want,
whether it's the mean, the HDI or the probability that a parameter exceeds a threshold, is counted
from those samples, not derived.

And last, check before you trust. That means R-hat, the effective sample size and divergences after
you sample, and a prior predictive check before you fit, every time.

**[pause]**

There's one more point that isn't on the list, and it's really the point of the hour: **this
doesn't make your forecast more accurate.** It changes what the forecast *is*, from a number into a
distribution, and that's what lets you answer a question about probability.

Next time is Bayesian Statistics Part Two, starting with Bayesian time series. We'll fit trend and
seasonality with priors attached, and you'll see a sampler failure that looks exactly like success.
The week after that, the same machinery becomes a hierarchical model.

---

# ▶ SLIDE 22 — References

*(Advance and close. No narration is needed.)*
