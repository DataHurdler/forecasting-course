# Lab 10 — Recording Script

**ECON 8310: Business Forecasting · Priors, Posteriors, and Checking the Sampler**

Lab: `Labs/Lecture10_lab.qmd` (5 steps) · Measured runtime: **~20 minutes** of narration
(the in-room version is budgeted at 35)

---

## How to use this document

- **`▶ STEP n — Title`** matches the lab's own headings exactly.
- *Italic parentheticals* are stage directions. **[pause]** is a beat; **[STOP — learner works]**
  is where you tell the viewer to pause the video and do something.
- `random_seed=42` on every `pm.sample`, so the numbers reproduce. MCMC is stochastic, so the last
  digit can still move on a different platform — say "about" where the lab says a third decimal.
- Each fit is about ten seconds; Step 5 runs three. No dead air worth planning around.

### This model is conjugate, and I checked the sampler against the exact answer

Beta-Binomial has a closed form, so every number here can be verified without MCMC. I did:

| | MCMC (the lab) | Exact (conjugate) |
|---|---|---|
| posterior mean | 0.2628 | **0.2633** |
| P(θ > 0.30) | 0.080 | **0.084** |
| P(0.24 < θ < 0.28) | 0.558 | **0.553** |
| 94% HDI | [0.2132, 0.3098] | **[0.2144, 0.3129]** |

Everything agrees to within Monte Carlo error. That is worth one sentence on camera — it is the
only lab in the course where you can *prove* the sampler is telling the truth, and it buys
credibility for the four labs after it where you cannot.

### One display gotcha that looks like a contradiction

Step 2's table prints the 94% HDI as **[0.22, 0.31]**. Step 4 prints the same interval as
**[0.2132, 0.3098]**. Those are the same numbers: `az.summary` rounds to **two decimals by
default** (`round_to=2`), and the `.round(4)` in the lab cannot undo it. Pass
`az.summary(..., round_to=4)` if you want the digits.

A sharp student will spot the mismatch and think something is wrong. Name it before they ask.

### In-room language that needs replacing

Step 4's "say the first line out loud as a sentence a manager would understand", Step 5's "your
turn", and the three closing **Discuss** blocks.

---

# ▶ OPENING

*(Show the rendered lab, at the top of the document.)*

This is Lab 10, and today the shape of the work changes.

Every model in this course so far has returned a number. You fit it, you score it, and you put it
on the scoreboard. Today's model returns a **distribution**, and the work shifts to match. You'll
spend less time fitting, and more time deciding what you believe before you look, and then checking
whether the machinery actually worked.

**[pause]**

The question is deliberately small, so that the workflow stays visible. What fraction of weeks at
CA\_1 FOODS are high-SNAP weeks? There's one parameter, and there's no forecasting. If the question
were hard, you'd spend the session on the question instead of on the method, and today the method
is the point.

There are five steps, and they're the five steps of every Bayesian analysis you'll ever do. First
you state a prior, and then you **check it before looking at the data**. Then you fit the model,
you verify the sampler, and finally you ask the posterior the questions no point estimate can
answer.

---

# ▶ SETUP

*(Run the setup cell.)*

```
CA_1 FOODS: 72 high-SNAP weeks out of 277 (26% observed)
```

That's seventy-two weeks out of two hundred seventy-seven, or twenty-six percent. Hold on to that
number, because the whole lab is about how different beliefs collide with it.

*(Show the ArviZ callout.)*

Read this callout, because otherwise it will bite you in Homework 6. **ArviZ changed its default
interval.** `az.summary` now reports an eighty-nine percent *equal-tailed* interval. This course
uses the highest density interval, which is the shortest interval that contains the probability
you ask for, so you have to request it every time, with `ci_kind="hdi"` and `ci_prob=0.94`.

For a symmetric posterior, the two nearly coincide. Worked out exactly for this model, the
equal-tailed ninety-four percent interval runs from about zero point two one six to zero point
three one four, and the HDI runs from about zero point two one four to zero point three one three,
so here it hardly matters. For a **skewed** posterior they come apart, and the HDI is the honest
summary. Get in the habit now, while the stakes are low.

---

# ▶ STEP 1 — Check the prior before you use it

*(Run the prior predictive.)*

This is the step almost everyone skips, and it takes about four seconds.

Before fitting anything, we ask what the prior **alone** implies about the data. Our prior is a
Beta two, two, which is weakly informative and centered at one half. We draw five hundred values of
theta from it, simulate a count of high-SNAP weeks from each one, and look at what comes out.

```
prior predictive spans 8 to 276 weeks
```

*(Point at the histogram.)*

**[pause]**

The simulated counts run from eight to two hundred seventy-six, out of two hundred seventy-seven.
So the prior considers essentially every possible answer plausible, from almost never to almost
always.

The observed value, seventy-two, sits comfortably inside that range. So the prior is **not wrong**,
and nothing here is going to blow up.

But look at what it's willing to entertain. "Two hundred seventy-six of two hundred seventy-seven
weeks are high-SNAP weeks" isn't an uncertain belief. It's a false one. SNAP benefits are paid out
on a **published calendar**. We aren't genuinely ignorant about this rate, and a Beta two, two
pretends we are.

That's the difference between a prior that's *defensible* and one that's *lazy*, and the prior
predictive check is what makes the difference visible. It costs one function call, and it's the
single most valuable habit in this lecture.

---

# ▶ STEP 2 — Fit it

*(Run the sampler.)*

```
         mean     sd hdi94_lb hdi94_ub ess_bulk ess_tail r_hat
theta  0.2628  0.026     0.22     0.31     3686     5175  1.00
```

The posterior mean is **zero point two six two eight**.

**[pause]**

Now put three numbers side by side, because the way they're arranged is the lesson. The prior mean
was zero point five. The observed proportion is zero point two five nine nine. And the posterior
mean is zero point two six two eight.

The posterior sits between the other two, as it has to, but look at *where* it sits. It's three
thousandths away from the data and nearly a quarter away from the prior. Two hundred seventy-seven
observations against a Beta two, two prior isn't a fair fight, and the data wins almost completely.

That's what it means, concretely, for the likelihood to dominate, and it answers half the
objections people raise about Bayesian methods before they've ever run one.

*(Point at the interval columns.)*

The ninety-four percent HDI reads zero point two two to zero point three one here. It shows two
decimals because that's how `az.summary` rounds by default. Step 4 prints the same interval more
precisely, so it's the same numbers with a different display.

---

# ▶ STEP 3 — Did the sampler work?

*(Run the diagnostics and the trace plot.)*

```
R-hat        1.001    (want < 1.01)
ESS (bulk)   3687      (want > 400)
divergences  0        (want 0)
```

Here's the thing to remember about MCMC: it **always** returns samples. It never fails with an
error just because the answer is wrong. Whether those samples came from the posterior you asked for
is a separate question, and you're expected to answer it every single time, before you read any
result.

By now that sentence should sound familiar. It's the same shape as the forgotten `zero_grad()` in
Lab 7, the reshape where a transpose belonged in Lab 8, and the missing positional encoding in
last week's lab. The code runs, the output is plausible, and nothing tells you. Bayesian workflow
just has better instruments for catching it.

**[pause]**

There are three numbers to read. R-hat is one point zero zero one, comfortably under one point zero
one, so the four chains agree with each other. The effective sample size is three thousand six
hundred eighty-seven, far above the rule of thumb of four hundred, so the samples aren't so
autocorrelated that you have less information than it looks. And there are zero divergences, so the
sampler never had to give up on a step.

*(Point at the trace plot.)*

Now look at the picture. On the right, four chains overlap in a flat band, with no drift and no
stuck stretches. That's what convergence looks like, and it's worth memorizing the *shape*, so that
a bad one jumps out at you. On the left, four densities lie on top of each other.

If those chains had separated, R-hat would exceed one point zero one, and **nothing downstream
would mean anything**: not the mean, not the interval, and not the probabilities in the next step.

---

# ▶ STEP 4 — Ask the posterior a question

*(Run the probability block.)*

The posterior is a set of samples, so any question about theta is answered by **counting**, which
is the part that feels almost too easy.

```
P(theta > 0.30)          = 0.080
P(0.24 < theta < 0.28)   = 0.558
94% HDI                  = [0.2132 0.3098]
```

**[STOP — learner works]** — *(replaces "say it out loud as a sentence a manager would
understand")*

Pause the video, and write the first line as a sentence you could say in a meeting. Then try to
write the same kind of sentence from a confidence interval.

*(Resume.)*

The first one is easy: "There's an eight percent chance that more than thirty percent of weeks are
high-SNAP weeks." That's a claim about the world, in the units of the question.

The second one is impossible, and this is the part worth being precise about rather than glib. A
frequentist ninety-four percent interval is a statement about the **procedure**. It says that
intervals built this way cover the true value ninety-four percent of the time. It isn't a statement
about theta, and "there's a ninety-four percent chance that theta is in this interval" is exactly
the misreading every textbook warns against and every practitioner makes anyway.

The Bayesian version means what people already thought the frequentist version meant. That's the
practical appeal, and you pay for it with the prior.

---

# ▶ STEP 5 — How much did the prior matter?

*(Run the three-prior comparison.)*

Now for the honest objection, the one you should expect from any audience: **the prior is
arbitrary, so the answer is arbitrary.** Let's test that rather than argue about it.

There are three priors. The first is weak and centered at zero point five. The second is informed
and centered at zero point two five, which is roughly right. And the third is centered at **zero
point nine one**, which isn't a mild disagreement. It's a belief that nine weeks in ten are
high-SNAP weeks, and it's deliberately, badly wrong.

```
                                 prior  prior_mean  posterior_mean
         Beta(2,2) — weak, centered 0.5       0.500           0.265
   Beta(5,15) — informed, centered 0.25       0.250           0.259
Beta(20,2) — badly wrong, centered 0.91       0.909           0.309
```

**[pause]**

The prior means span zero point two five to zero point nine one, a range of **zero point six six**.
The posterior means span zero point two five nine to zero point three zero nine, a range of **zero
point zero five**. So the data compressed the disagreement by a factor of thirteen.

*(Show the callout.)*

Now let me be careful, because this is the step where it's tempting to oversell, and the callout's
title says it precisely: outweighed, not erased.

The badly wrong prior did **not** land in the same place as the others. It landed at zero point
three zero nine. The weak prior's ninety-four percent interval, the one Step 4 printed, runs up to
about zero point three one. So the wrong prior's answer sits *just inside* the upper edge of what
the honest analysis considered plausible. It's inside the range, but right at its boundary. And if
you work out the wrong prior's own interval exactly, which this model lets you do, it runs from
about zero point two five eight to zero point three five eight. That barely contains the observed
proportion of zero point two six zero, right at its lower edge.

So the correct claim isn't that the prior didn't matter. It's this: **with two hundred seventy-seven
observations, even an absurd prior gets dragged to the edge of the right answer rather than away
from it.** That's a strong result, and it doesn't need embellishment.

**[pause]**

But the edge isn't the middle. Go back to the manager's question from Step 4. Under the weak prior,
the chance that more than thirty percent of weeks are high-SNAP weeks was eight percent. Under the
wrong prior, it's about sixty-one percent, from the same data. The mean shifted by four hundredths,
and the answer to the question you'd actually be asked goes from "unlikely" to "more likely than
not," because the threshold sits right where the mean moved. So check the prior's influence on the
question you'll be asked, not only on the mean.

*(Point at the condition in the callout.)*

And the result comes with conditions. The one to land hardest is that it holds **when you have
enough data.** With eight observations instead of two hundred seventy-seven, that third prior would
still be sitting in your answer, plain to see.

**[STOP — learner works]**

Now it's your turn. Pause the video, re-run the badly wrong prior against only the **first ten
weeks**, and report how far the posterior moves. There are two blanks to fill in, and they're the
alpha and beta of the Beta twenty, two.

*(Resume.)*

That situation, with a handful of observations for each unit and a prior that therefore still
matters, isn't a contrived exercise. It's Lecture 11, Part 2, two weeks from now, where ten series
have very different amounts of data and the thin ones borrow strength from the thick ones.
Everything you just did by hand, the hierarchy will do automatically.

---

# ▶ BEFORE YOU LEAVE

**[STOP — learner works]** — *(replaces the in-room "Discuss")*

Here are three questions to work through on your own before you go.

The first is about the prior. Your Step 5 table says that a deliberately wrong prior moved the
posterior mean by only about zero point zero four, and yet it moves the probability that theta is
above zero point three zero from zero point zero eight to about zero point six one. Under what
circumstances would you *not* be reassured by a small shift in the mean? Name two, and be specific:
one about sample size, and one about what the prior is a prior *over*.

**[pause]**

You already have the sample-size answer from Step 5. With only a handful of observations, the prior
would still be sitting in the answer. The second answer is the more interesting half. What we had
today was a prior on a single bounded rate. A prior on a variance, or on a hierarchical scale
parameter, can dominate a posterior even with far more data than two hundred seventy-seven
observations, and Lecture 11, Part 2 has an example.

The second question is about the check. You verified R-hat, the effective sample size and
divergences before reading any result. Name the equivalent check you've been running all semester
on non-Bayesian models, and say which of the two is easier to skip.

**[pause]**

The answer is the held-out benchmark. And the honest reply is that the Bayesian check is the easier
one to skip, because MCMC hands you a tidy summary table that looks authoritative whether or not the
sampler converged.

The third question looks ahead to Homework 6. Part 1 of that homework asks you to run a prior
predictive check and to propose something better than a Beta two, two for a rate you know is around
a quarter. Choose your alpha and beta, and defend the **strength** of the prior, not just its
center. A Beta five, fifteen and a Beta fifty, one hundred fifty have the same mean, and they say
very different things about how sure you are.

Solutions for the coded parts go up on Canvas after class. Next week, in Lecture 11, Part 1,
we'll point the same machinery at a time series.

---

# Appendix — expected output

`random_seed=42` throughout; ~10 s per fit, three more in Step 5. **Conjugate, so every value is
verifiable in closed form** — the exact column below is `Beta(2+72, 2+205)` etc.

| Quantity | MCMC | Exact |
|---|---|---|
| Data | 72 high-SNAP weeks / 277 (**0.2599**) | — |
| Prior predictive span | 8 → 276 weeks | — |
| Posterior mean | 0.2628 | **0.2633** |
| Posterior sd | 0.026 | — |
| 94% HDI | [0.2132, 0.3098] | **[0.2144, 0.3129]** |
| 94% equal-tailed (for contrast) | — | [0.2155, 0.3141] |
| R-hat / ESS bulk / divergences | 1.001 · 3,687 · **0** | — |
| P(θ > 0.30) | 0.080 | **0.084** |
| P(0.24 < θ < 0.28) | 0.558 | **0.553** |
| Beta(2,2) → posterior mean | 0.265 | 0.2633 · HDI [0.2144, 0.3129] |
| Beta(5,15) → posterior mean | 0.259 | 0.2593 · HDI [0.2119, 0.3072] |
| Beta(20,2) → posterior mean | 0.309 | 0.3077 · HDI [0.2579, 0.3580] |
| P(θ > 0.30), Beta(2,2) vs Beta(20,2) | 0.080 vs not printed | **0.084 vs 0.608** |
| Prior-mean spread → posterior spread | 0.66 → **0.05** (13× compression) | — |

**Three things to know before recording.**

*Say that the sampler was checked against the exact answer.* This is the only lab where the
closed form exists, agreement is within Monte Carlo error on all four quantities, and stating it
buys trust for Lectures 11 and 12, where no such check is available.

*Do not oversell Step 5.* **Corrected 2026-10-19:** the callout used to be titled "With enough
data the prior stops mattering" and said "All three posteriors land close together." That was
more than a little generous: 0.309 against 0.259 is a 19% relative difference, the wrong prior's
posterior mean sits at the **upper edge** (0.309) of the weak prior's interval (top 0.313), not in
the middle of it, and it moves P(θ > 0.30) from 0.084 to 0.608 (exact). The callout now says
"outweighed, not erased" and gives that probability. The precise claim — an absurd prior gets
dragged to the *edge* of the right answer rather than away from it — is stronger *because* it is
exact, and it sets up the condition that Lecture 11 Part 2 depends on.

*The `az.summary` rounding will be noticed.* [0.22, 0.31] in Step 2 versus [0.2132, 0.3098] in
Step 4 is `round_to=2`, not a disagreement. Name it in passing at Step 2 so it never becomes a
question.
