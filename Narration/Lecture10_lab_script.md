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

*(Screen: rendered lab, top of document.)*

Lab 10, and the shape of the work changes today.

Every model in this course so far returned a number. Fit it, score it, put it on the scoreboard.
Today's model returns a **distribution**, and the labour moves accordingly — less time fitting,
more time deciding what you believe before you look, and then checking whether the machinery
actually worked.

**[pause]**

The question is deliberately small so the workflow stays visible: **what fraction of weeks at
CA\_1 FOODS are high-SNAP weeks?** One parameter. No forecasting. If the question were hard you
would spend the session on the question instead of on the method, and the method is the point
today.

Five steps, and they are the five steps of every Bayesian analysis you will ever do: state a
prior, **check it before looking at the data**, fit, verify the sampler, then ask the posterior
questions no point estimate can answer.

---

# ▶ SETUP

*(Screen: run setup.)*

```
CA_1 FOODS: 72 high-SNAP weeks out of 277 (26% observed)
```

Seventy-two of two hundred seventy-seven. Twenty-six percent. Hold that number — the whole lab is
about how different beliefs collide with it.

*(Screen: the ArviZ callout.)*

Read this callout, because it will bite you in Homework 6 otherwise. **ArviZ changed its default
interval.** `az.summary` now reports an 89% *equal-tailed* interval. This course uses the
**highest density interval** — the shortest interval containing the mass — so you must ask for it
every time: `ci_kind="hdi", ci_prob=0.94`.

For a symmetric posterior the two nearly coincide; here the equal-tailed 94% interval is
[0.2155, 0.3141] against an HDI of [0.2144, 0.3129], so it hardly matters. For a **skewed**
posterior they diverge, and the HDI is the honest summary. Get in the habit now, while the stakes
are low.

---

# ▶ STEP 1 — Check the prior before you use it

*(Screen: run the prior predictive.)*

Here is the step almost everyone skips, and it costs about four seconds.

Before fitting anything, we ask what the prior **alone** implies about the data. `Beta(2,2)` —
weakly informative, centered at one half. Draw five hundred values of θ from it, simulate a count
of high-SNAP weeks from each, and look at what comes out.

```
prior predictive spans 8 to 276 weeks
```

*(Point at the histogram.)*

**[pause]**

Eight to two hundred seventy-six. Out of 277. The prior considers essentially every possible
answer plausible — from almost never to almost always.

The observed value, 72, sits comfortably inside that. So the prior is **not wrong.** Nothing here
is going to blow up.

But look at what it is willing to entertain. "Two hundred seventy-six of two hundred seventy-seven
weeks are high-SNAP weeks" is not an uncertain belief, it is a false one. **SNAP is disbursed on a
published calendar.** We are not genuinely ignorant about this rate, and `Beta(2,2)` pretends we
are.

That is the difference between a prior that is *defensible* and one that is *lazy*, and the prior
predictive check is what makes the difference visible. It costs one function call, and it is the
single highest-value habit in this lecture.

---

# ▶ STEP 2 — Fit it

*(Screen: run the sampler.)*

```
         mean     sd hdi94_lb hdi94_ub ess_bulk ess_tail r_hat
theta  0.2628  0.026     0.22     0.31     3686     5175  1.00
```

Posterior mean **0.2628**.

**[pause]**

Now put three numbers side by side, because their arrangement is the lesson. The prior mean was
**0.5**. The observed proportion is **0.2599**. The posterior mean is **0.2628**.

The posterior sits between them, as it must — but look at *where* between them. It is three
thousandths from the data and nearly a quarter away from the prior. Two hundred seventy-seven
observations against a `Beta(2,2)` prior is not a fair fight, and the data wins almost totally.

That is what "the likelihood dominates" means, concretely, and it is the answer to half the
objections people raise about Bayesian methods before they have run one.

*(Note the interval.)*

The 94% HDI reads [0.22, 0.31] here — two decimals, because that is `az.summary`'s default
rounding. Step 4 prints the same interval more precisely. Same numbers, different display.

---

# ▶ STEP 3 — Did the sampler work?

*(Screen: run the diagnostics and the trace.)*

```
R-hat        1.001    (want < 1.01)
ESS (bulk)   3687      (want > 400)
divergences  0        (want 0)
```

**MCMC always returns samples.** It never fails with an error because the answer is wrong. Whether
those samples came from the posterior you asked for is a **separate question**, and you are
expected to answer it every single time before reading any result.

By now that sentence should sound familiar. It is the same shape as `zero_grad()`, as the
transposed axis, as the missing positional encoding — the code runs, the output is plausible, and
nothing tells you. Bayesian workflow just has better instruments for catching it.

**[pause]**

Three numbers. **R-hat** at 1.001, comfortably under 1.01 — the four chains agree with each other.
**Effective sample size** at 3,687, far above the 400 rule of thumb — the samples are not so
autocorrelated that you have less information than it looks. **Divergences** at zero — the sampler
never had to give up on a step.

*(Point at the trace plot.)*

And the picture. On the right, four chains overlapping in a flat band, no drift, no stuck
stretches — that is what convergence looks like, and it is worth memorizing the *shape* so a bad
one jumps out. On the left, four densities lying on top of each other.

If those chains had separated, R-hat would exceed 1.01 and **nothing downstream would mean
anything** — not the mean, not the interval, not the probabilities in the next step.

---

# ▶ STEP 4 — Ask the posterior a question

*(Screen: run the probability block.)*

The posterior is a set of samples. So any question about θ is answered by **counting**, which is
the part that feels almost too easy.

```
P(theta > 0.30)          = 0.080
P(0.24 < theta < 0.28)   = 0.558
94% HDI                  = [0.2132 0.3098]
```

**[STOP — learner works]** — *(replaces "say it out loud as a sentence a manager would
understand")*

Pause and write the first line as a sentence you could say in a meeting. Then try to write the
same sentence from a confidence interval.

*(Resume.)*

The first one is easy: *"There is an eight percent chance that more than thirty percent of weeks
are high-SNAP weeks."* A claim about the world, in the units of the question.

The second is impossible, and this is the part worth being precise about rather than glib. A
frequentist 94% interval is a statement about the **procedure** — that intervals built this way
cover the true value 94% of the time. It is not a statement about θ, and "there is a 94% chance θ
is in this interval" is exactly the misreading every textbook warns against and every practitioner
makes anyway.

The Bayesian version means what people already thought the frequentist version meant. That is the
practical appeal, and it is bought with the prior.

---

# ▶ STEP 5 — How much did the prior matter?

*(Screen: run the three-prior comparison.)*

Now the honest objection, the one you should expect from any audience: **the prior is arbitrary,
so the answer is arbitrary.** Test it rather than argue about it.

Three priors. Weak and centered at 0.5. Informed and centered at 0.25 — roughly right. And one
centered at **0.91**, which is not a mild disagreement, it is a belief that nine weeks in ten are
high-SNAP weeks. Deliberately, badly wrong.

```
                                 prior  prior_mean  posterior_mean
         Beta(2,2) — weak, centered 0.5       0.500           0.265
   Beta(5,15) — informed, centered 0.25       0.250           0.259
Beta(20,2) — badly wrong, centered 0.91       0.909           0.309
```

**[pause]**

The prior means span 0.25 to 0.91 — a range of **0.66**. The posterior means span 0.259 to 0.309 —
a range of **0.05**. The data compressed the disagreement by a factor of thirteen.

*(Screen: the callout.)*

**[pause]**

Now let me be careful, because this is the step where it is tempting to oversell, and the
callout's title says it precisely: outweighed, not erased.

The badly-wrong prior did **not** land in the same place as the others. It landed at 0.309, and
the weak prior's 94% interval runs up to 0.313. So the wrong prior's answer sits *just inside* the
upper edge of what the honest analysis considered plausible — inside the range, but at the very
boundary of it. And its own interval, [0.258, 0.358], barely contains the observed proportion of
0.260 at its lower edge.

So the correct claim is not "the prior did not matter." It is: **with 277 observations, even an
absurd prior gets dragged to the edge of the right answer rather than away from it.** That is a
strong result and it does not need embellishment.

**[pause]**

And the edge is not the middle. Go back to the manager's question from Step 4. Under the weak
prior, the chance that more than thirty percent of weeks are high-SNAP weeks was eight percent.
Under the wrong prior it is about sixty-one percent. Same data. A shift of four hundredths in the
mean, and the answer to the question you would actually be asked goes from "unlikely" to "more
likely than not," because the threshold sits right where the mean moved. So check the prior's
influence on the question you will be asked, not only on the mean.

*(Point at the condition in the callout.)*

And it comes with conditions. The first is the sentence to land hardest on: it holds **when you
have enough data.** With eight observations instead of 277, that third prior would still be sitting in
your answer, plainly visible.

**[STOP — learner works]**

That is your turn. Re-run the badly-wrong prior against only the **first ten weeks** and report
how far the posterior moves. Fill in the two blanks — they are the alpha and beta of the
`Beta(20,2)`.

*(Resume.)*

And that situation — a handful of observations per unit, and a prior that therefore still matters
— is not a contrived exercise. It is Lecture 11 Part 2, where ten series have very different
amounts of data and the thin ones borrow strength from the thick ones. Everything you just did by
hand, the hierarchy will do automatically.

---

# ▶ BEFORE YOU LEAVE

**[STOP — learner works]** — *(replaces the in-room "Discuss")*

Three questions for the board.

**The prior.** Your Step 5 table says a deliberately wrong prior moved the posterior mean by only
about 0.04, yet it moves P(θ > 0.30) from 0.08 to about 0.61. Under what circumstances would you
*not* be reassured by a small shift in the mean? Name two, and be specific — one about sample
size, one about what the prior is a prior *over*.

**[pause]** — *the second one is the interesting half: this was a prior on a single bounded rate.
A prior on a variance, or on a hierarchical scale parameter, can dominate a posterior with far
more data than 277 observations. Lecture 11 Part 2 has an example.*

**The check.** You verified R-hat, ESS and divergences before reading any result. Name the
equivalent check you have been running all semester on non-Bayesian models — and say which of the
two is easier to skip.

**[pause]** — *the answer is the held-out benchmark, and the honest reply is that the Bayesian one
is easier to skip, because MCMC hands you a tidy summary table that looks authoritative whether or
not it converged.*

**For Homework 6.** Part 1 asks you to run a prior predictive check and propose something better
than `Beta(2,2)` for a rate you know is around a quarter. Choose your alpha and beta — and defend
the **strength**, not just the center. `Beta(5,15)` and `Beta(50,150)` have the same mean and say
very different things about how sure you are.

Solutions for the coded parts go up on Canvas after the deadline. Next week: the same machinery
pointed at a time series.

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
