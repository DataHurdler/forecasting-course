# Lecture 11 Part 2 — Recording Script

**ECON 8310: Business Forecasting · Bayesian Statistics II — Hierarchical Models**

Deck: `Slides/Lecture11_Part2_Hierarchical.pdf` (20 pages) · Measured runtime: see the timing guide

---

## How to use this document

- **`▶ SLIDE n — Title`** marks where to advance. The number is the PDF page.
- *Italic parentheticals* are stage directions. **[pause]** means stop for a beat.
- Slides 4 and 5 come *before* any modeling on purpose. The exchangeability question is the one
  students will skip, and the lab is built to punish skipping it. Do not rush them.
- Slide 12's lesson is that divergences fired while $\hat{R}$ and ESS both passed. That inverts
  what students took from Lecture 11 Part 1, so say it explicitly.

---

# ▶ SLIDE 1 — Title page

Lecture 11, Part 2: Bayesian Statistics II — Hierarchical Models.

Last week we fitted one series and got a full predictive distribution out of it. Today we have
thirty series, and they are not equally informative — some have years of history, and in the real
world some are new stores with a handful of usable weeks.

That imbalance is the situation partial pooling exists for, and it's genuinely one of the most
useful ideas in applied statistics.

**[pause]**

But before any of that, there's a question you have to answer first — and it's not a statistical
question at all. Getting it wrong produces a model that fits cleanly, passes every diagnostic, and
gives you a confidently wrong answer.

That question is where we start.

---

# ▶ SLIDE 2 — Lecture Outline

Five parts. Which series may be pooled — the question before the model. Then partial pooling
itself, which is one extra line of code. Then fitting it, and a trap that catches nearly everyone
the first time. Then reading the result, checked against data the model never saw. And takeaways.

---

# ▶ SLIDE 3 — Section divider: Which Series May Be Pooled

Before you pool anything, ask what the groups have in common.

---

# ▶ SLIDE 4 — The Question

Here's a concrete business question, and you've met it before. **Does demand rise in weeks when
SNAP benefits are disbursed, and by how much — store by store?**

Homework 3's memo had you decide which store-categories to stock up for a SNAP week rather than
applying one national uplift. That question *assumed* the per-series differences were real. Today
we check.

**[pause]**

*(Now the pre-model check. This is the slide that makes the lecture work.)*

But one question comes before the model, and it costs nothing to ask.

SNAP is a **food** assistance program. So let's measure the raw SNAP-week effect — no model, just
the difference in mean log units — in all thirty series, grouped by category.

*(Walk the table.)*

FOODS, ten series: plus point-oh-eight-nine, with a spread across stores of point-oh-five-eight.

HOBBIES, ten series: plus point-oh-oh-six.

HOUSEHOLD, ten series: plus point-oh-one-one.

**[pause]**

Look at that first column. FOODS responds. The other two barely move — and that is not a surprise
once you say it out loud, because SNAP is food assistance. You could have predicted the sign of
that table before computing it.

**These are not thirty versions of the same thing.** Pooling all thirty would be telling the model
they are interchangeable, and the model would believe you.

**[pause]**

And I want to stress how cheap that check was. Three group-bys and a subtraction. No model, no
sampler, no priors. Two minutes of work.

That is characteristic of the most valuable checks in this course — the leakage checks, the
benchmark, the planted noise control in Homework 3. They are nearly free, they happen before the
modeling, and they are the ones people skip because they don't feel like real analysis.

---

# ▶ SLIDE 5 — Exchangeability: the Assumption Under Every Hierarchy

So let's name the assumption properly, because it sits under every hierarchical model you will
ever build.

*(Point at the definition box.)*

Groups are **exchangeable** when, *before seeing the data*, you have no reason to expect one to
differ from another in a particular direction. Swapping their labels would not change what you
believe.

That's the test. Not "are they similar?" — but "would relabelling them change my expectations?"

**[pause]**

That distinction matters, so let me push on it. Exchangeable does **not** mean identical. The ten
FOODS stores certainly differ — different neighborhoods, different sizes, different customers.
Exchangeability doesn't deny that.

It says something weaker and more useful: that before seeing data, you have no basis for ordering
them. You cannot say *in advance* that the Texas store's SNAP effect should exceed California's.

That's a claim about your knowledge, not about the world. Which is why it's a judgment call you
have to make and defend — and why no diagnostic can make it for you.

**[pause]**

Apply it. The ten FOODS series **qualify**: same category, ten different stores, and you genuinely
could not say in advance which store's SNAP effect should be largest. Swap the store labels and
you'd believe exactly the same things.

FOODS against HOBBIES does **not** qualify. You know before looking that a food-assistance
program moves food. There's a direction, and you knew it in advance.

*(Point at the setup.)*

So for today we work with the **ten FOODS series** — the defensible set — and we deliberately thin
**five** of them down to eight weeks of data. Five data-rich, five data-poor. That imbalance is the
whole point of the exercise.

**[pause]**

*(Point at the muted footnote, and sell the lab.)*

And in the lab, we pool all thirty anyway — on purpose — and measure exactly what it costs. I'd
encourage you to do that one carefully. It is the most instructive forty minutes of the semester,
because the wrong model looks *completely fine* until you check it against held-out data.

---

# ▶ SLIDE 6 — Complete Pooling and No Pooling

Within that exchangeable set, there are two obvious things you could do, and each fails in a
different direction.

*(Walk the table.)*

**Complete pooling**: one SNAP effect for all ten series. One parameter. This denies that stores
differ at all, so a genuinely different store gets averaged away into the crowd.

**No pooling**: a separate effect per series, estimated independently — `shape=10`. Now a series
with eight weeks of data gets an estimate built from eight weeks of data. That's mostly noise,
reported as a finding.

**[pause]**

*(This callback is the one that lands. Deliver it deliberately.)*

And here's the thing — **you have used both of these already, without naming them.**

Homework 3's random forest pooled all thirty series into one model with store dummies. That's
close to complete pooling.

Homework 1 fitted ARIMA to one series at a time, completely independently. That's no pooling.

You made that choice twice, and both times you made it by **convenience** — by what the library
made easy. Partial pooling turns it into a modeling decision, with a parameter, and lets the data
settle where between the two extremes you should sit.

**[pause]**

It's worth noticing that this is a recurring shape in the course. Something you had been deciding
implicitly, by tooling default, gets promoted into an explicit parameter with a posterior.

Lecture 6 did it for which predictors survive — you were choosing by intuition, then LASSO made
it a penalty. Lecture 11 Part 1 did it for how fast the level may move. Today it's how much
groups share.

Each time, the gain is the same: the decision becomes visible, arguable, and estimable rather than
buried in a default nobody wrote down.

---

# ▶ SLIDE 7 — Section divider: Partial Pooling

Let each series have its own effect — drawn from a shared distribution the data also estimates.

---

# ▶ SLIDE 8 — The Hierarchy Is One Extra Line

Here is the whole idea, and it really is one line.

*(Point at the equation.)*

Give each series its own effect, b-j. But then say **where those effects come from**: b-j is drawn
from a Normal with mean mu-b and standard deviation sigma-b.

mu-b is the typical SNAP effect across FOODS. sigma-b is **how much stores genuinely differ**.

And critically — both of those are *estimated from the data*, not assumed by you.

**[pause]**

*(Walk the three-row table, because it shows the mechanism.)*

Think about the two extremes.

If sigma-b goes to zero, the model has concluded stores don't differ, and every b-j collapses onto
mu-b. That reproduces complete pooling.

If sigma-b is large, the model has concluded they differ a lot, and each b-j is free to sit
wherever its own data says. That approaches no pooling.

And in between — which is where you almost always end up — each b-j is pulled toward mu-b, and
**how far depends on how much data that series has.**

*(Point at the last paragraph.)*

**[pause]**

That last row is the whole idea, so let me say it plainly. A series with two hundred seventy-seven
weeks barely moves, because its own estimate is already trustworthy. A series with eight weeks
moves a long way, because its own estimate carries almost no information.

**Nobody chose those weights.** There is no threshold, no rule, no "if n is less than thirty."
They fall out of estimating sigma-b.

---

# ▶ SLIDE 9 — Why This Is the Right Answer, Not a Compromise

I want to address an objection that's probably forming, because it's a reasonable one.

Partial pooling sounds like **splitting the difference** between two bad options. A fudge. Meeting
in the middle because you can't decide.

It is better than that, and the reason is worth stating.

**[pause]**

When a series has eight observations, its unpooled estimate is mostly sampling noise. Pulling it
toward the group mean is not a compromise — it is the **correct response to a noisy measurement**.

You already believe this in another form. You would not conclude a coin is biased after three
flips, even if all three came up heads. You'd pull your estimate toward one-half, because three
flips is not much evidence. That's shrinkage, and you do it instinctively.

*(Point at the key box, and slow down for the callback.)*

**[pause]**

And this is **regularization, for the sixth time this semester.**

A hierarchical prior shrinks per-group estimates toward a common value exactly as Ridge shrinks
coefficients toward zero. Same operation, different name.

The smoothing penalty in Lecture 3. Cost-complexity pruning in Lecture 4. `reg_lambda` in Lecture
5. Ridge and LASSO in Lecture 6. Weight decay in Lecture 7. And now the hierarchical prior.

*(Beat.)*

But this version has something none of the other five had. In every previous case, you chose the
shrinkage strength by cross-validation — you tried values and picked one. Here, sigma-b is
**estimated**, as part of the model, with its own posterior and its own uncertainty.

That makes it the cleanest version of the idea in the course.

---

# ▶ SLIDE 10 — Section divider: Fitting It, and the Trap

The obvious code does not work, and the reason is geometry rather than statistics.

---

# ▶ SLIDE 11 — The Model in PyMC

Here it is in PyMC. Nine lines.

mu-b and sd-b are the group-level parameters — the typical effect and how much stores differ. Then
z, a vector of standard Normals. Then b is constructed as mu-b plus z times sd-b.

Then a per-series intercept, observation noise, and the likelihood.

*(Point at the indexing.)*

The pattern to internalize is `b[idx]`. `idx` maps each row to its series number, so `b[idx]` picks
out the right effect for every observation. Every hierarchical model in PyMC is written that way —
once you can read that line, you can read most of the applied Bayesian literature.

**[pause]**

*(Point at the closing paragraph.)*

And notice what is **not** in that code.

There is no decision about how much to pool. No weighting scheme. No threshold for "enough data."
No rule that says series with fewer than twenty observations get shrunk by half.

All of that comes out of estimating one parameter, sigma-b. That's the elegance of it.

**[pause]**

One practical note on priors, since we're looking at the code. `sd_b` gets a HalfNormal with scale
point-two-five — positive-only, because a standard deviation must be, and concentrated on small
values because we're working in log units where an effect of point-two-five is already very large.

That prior is doing real work. With only ten groups, there is not much information about how much
groups differ, so the data cannot pin sigma-b down on its own. A vague prior here lets it wander
large, which weakens the pooling — you'd drift toward the unpooled answer.

This is the general situation with group-level scale parameters: they are the hardest thing in the
model to estimate, and the place where your prior matters most.

---

# ▶ SLIDE 12 — The Trap: Write It the Obvious Way and It Breaks

Now, why did I write `b` as `mu_b + z * sd_b` instead of the obvious thing?

The obvious thing is `pm.Normal("b", mu_b, sd_b, shape=J)` — which reads *exactly* like the
equation on slide 8. It is the natural translation. Everybody writes it that way first.

And on our data it produces this.

*(Walk the table.)*

Divergences: **forty-three**, where we want zero. The non-centered version: zero.

R-hat: one-point-oh-oh-three for both. **Passes**, comfortably.

ESS: nine hundred sixty-six for the centered fit. **Passes** too — more than twice the threshold.

**[pause]**

*(This is the point of the slide. Take it slowly.)*

Look at those three rows again, because only one of them objects.

If you checked R-hat and ESS — the two numbers Lecture 10 taught you to check — you would have
shipped this model. Both are clean. **Forty-three divergences against a threshold of exactly
zero** is the only alarm, and it is the one to trust. There is no version of "close to zero" that
gets you to forty-three.

So the rule to carry forward: on hierarchical models, **divergences fire first and least
ambiguously.** Do not wait for R-hat to tell you something is wrong. It may never say so.

*(Point at the muted line.)*

**[pause]**

One honest caveat about the exact numbers. Divergence counts are not stable quantities: change the
seed, or even the machine, and they move. Reading the same data back from a CSV file, which changes
the last binary digit of a few hundred values, moves this one from forty-three to forty-nine. Across ten more seeds, the centered form gave anywhere from fifteen to
two hundred eighty-five divergences, and the non-centered form from zero to eleven. What does not
move is the pattern: the centered form diverged on every one of the eleven runs, while R-hat or
ESS objected on only two of them.

So if you run the lab and see a different count, that is expected. If you see the centered form
diverge and the non-centered one barely at all, you have reproduced the finding.

---

# ▶ SLIDE 13 — Why It Breaks: a Funnel

Why does it happen? This picture is the answer.

*(Point at the left panel.)*

Every gray dot is one posterior draw from the centered fit, for one thin series, CA_1. Across: how
far its SNAP effect sits from the group mean, b-j minus mu-b. Up: sigma-b, on a log scale.

The dashed lines are what the prior says. Given sigma-b, b-j sits within about two sigma-b of
mu-b. So when sigma-b is large the band is wide, and when sigma-b is small the band pinches to
nothing. On a log axis that is a **funnel**.

**[pause]**

*(Point at the crimson dots.)*

The crimson dots are the forty-three divergences. Look where they are: **in the neck.** Thirty-nine
of the forty-three sit in the bottom tenth of the sigma-b values the sampler visited.

And look at the dotted line. The centered chains **never got below it** — never below a sigma-b of
about point-oh-one-one. The sampler tunes its step size for the wide part of the funnel, then tries
to get into the neck with steps that are far too big, fails, and reports a divergence. So it never
explores the neck at all.

*(Point at the right panel.)*

**[pause]**

Now the non-centered fit, on the same vertical axis. Across, z-j — the same quantity, divided by
sigma-b. That is all non-centering is: b-j equals mu-b plus sigma-b times z-j is a **change of
coordinates**. Dividing each height by its own width turns the funnel into a column. Under the
prior, z-j has spread one at every height — the dashed lines are straight.

And the sampler walks it. Zero divergences — and it goes below the dotted line: about three and a
half percent of its draws sit in the region the centered chains never reached. Thirty draws go
lower even than the bottom of this axis.

*(Beat.)*

One precision. The funnel-and-column statement is exact for the **prior**. The posterior adds the
data, which narrows things a little, but as you can see the draws follow the prior's shape
closely. Same model. Same posterior. Different coordinates — and only one set of them is walkable.

This is BMCP section four-point-six-point-one, and it is the single most common failure in applied
hierarchical modeling.

**[pause]**

*(A practical note, because students will reach for the wrong fix.)*

The instinct when you see divergences is to raise `target_accept` — force smaller steps. Sometimes
that helps a little. Here it helps partly: at point-nine-five the centered fit still gives fourteen
divergences instead of forty-three, and it runs slower for the privilege.

Reparameterizing is the actual fix. It is not free — on this machine the non-centered fit took a
little longer per run than the centered one — but it removes the problem instead of tiptoeing
around it.

So the habit: when a hierarchical model diverges, your first move is to check whether you wrote it
centered — not to tune the sampler.

---

# ▶ SLIDE 14 — Section divider: Reading the Result

Shrinkage, measured — and checked against data the model never saw.

---

# ▶ SLIDE 15 — Shrinkage, Measured

Now let's see what the pooling actually did. One row per series. The dot is each series' own
estimate — the unpooled fit. The arrow goes to where the hierarchy put it.

*(Point at the crimson rows.)*

The **thin** series, eight weeks each, in crimson. Look at the length of those arrows. CA_1 starts
at point-two-nine and ends just right of the dashed line, which is mu-b, point-oh-seven-eight.
WI_2 starts at point-three-four and travels even farther. Even TX_1, which started *below* the
group mean, gets pulled up toward it. Shrinkage moves estimates both ways — always toward mu-b.

*(Point at the gray rows.)*

The **full** series, two hundred seventy-seven weeks, in gray. Short arrows. TX_2 barely moves at
all — that tick is its whole journey.

**[pause]**

The average thin arrow is point-one-two-zero long; the average full arrow, point-oh-one-five. So the
data-poor series moved **eight point one times further** toward the group mean than the data-rich
ones. And nobody told the model which series were thin. It inferred that entirely from how much
each series had to say.

*(Point at the spread numbers. This is the subtle part.)*

**[pause]**

Now look at where the arrowheads land, because this is what separates partial pooling from just
averaging everything.

The thin series' apparent spread of point-one-oh-eight nearly **vanishes** — down to point-oh-one.
But the full series **keep most of theirs** — point-oh-five-five down to point-oh-three-seven. WI_3
still sits well right of the others after pooling.

The hierarchy did not flatten everything. It discarded spread that eight weeks could not support,
and kept spread that two hundred seventy-seven weeks *could*.

*(Point at the muted note.)*

There is a clean way to see the rule it followed. Each pooled estimate lands, approximately, at a
precision-weighted point between mu-b and the series' own estimate. A thin series' own estimate is
very imprecise, so it keeps only about a tenth of it and goes about ninety percent of the way to
mu-b. A full series keeps about three-quarters of its own. I say "approximately" because mu-b and
sigma-b are not known numbers here — the model averages over its uncertainty about them — but a
plain weighted average lands within about point-oh-one of every arrowhead.

*(Beat.)*

That's a strong claim, and you should be skeptical of it. So the next slide tests it against data
the model never saw.

---

# ▶ SLIDE 16 — Checked Against Data the Model Never Saw

Here's the nice thing about having thinned those series ourselves: **the weeks we removed still
exist.** So we can compute the SNAP effect that actually occurred in them, and grade both models
against it. That is the ring on each crimson row.

*(Walk the crimson rows — this is the payoff.)*

CA_1. Unpooled said point-two-eight-six. The hierarchy said point-oh-nine-seven. The truth in the
held-out weeks: **point-oh-eight-seven** — right where the arrow landed.

CA_3. The arrow runs from point-one-oh-five to point-oh-eight-two, and the truth is point-one-oh-two.
This is the one series where the unpooled estimate was closer — by a hair.

TX_1. Unpooled point-oh-five-two, pooled point-oh-seven-five, truth point-oh-eight-two. The arrow
pointed the right way and nearly reached the ring.

TX_3. From point-two-one-nine to point-oh-nine-one; truth point-oh-seven-two.

WI_2. From point-three-three-eight all the way to point-one-oh-three — but the ring sits at
point-two-oh-seven, in the middle of the arrow. The hierarchy pulled it **past** its truth. This is
the honest cost of shrinkage: WI_2 really is a high-response store, and eight weeks were not
enough to convince the model. Even so, the pooled estimate is closer than the unpooled one.

**[pause]**

*(Now the summary line.)*

RMSE against the truth: unpooled point-one-two-six. Hierarchical **point-oh-four-eight.** The
pooled error is **two point six times smaller**, and it beat the unpooled fit on four of the five
series.

And notice the character of the unpooled errors. They were not merely noisy — they were
**confidently wrong**. Point-three-three-eight is a specific, precise-looking claim that a store's
SNAP uplift is forty percent. It came from eight weeks of data, and the truth was about
twenty-three percent.

*(Muted note.)*

Borrowing strength is not a figure of speech. The thin series were estimated better **because the
other five existed.**

---

# ▶ SLIDE 17 — What the Model Actually Concluded

So what's the business answer?

The estimated hierarchy: mu-b of point-oh-seven-eight, with a ninety-four percent HDI from about
point-oh-two to point-one-five. And sigma-b of point-oh-six-five.

In words: a SNAP week lifts FOODS units by about **eight percent**, and stores genuinely differ
around that by about **seven percentage points**.

**[pause]**

Both of those are answers, and the second one is the one no point estimate would ever have given
you. "How much do stores differ?" is a question with a number attached now.

*(Point at the warning box.)*

But be careful what you conclude from it.

This does **not** license store-by-store stocking. An analyst reading the thin stores' own
estimates would report uplifts from five percent to forty percent and build a differentiated
stocking plan on it. The held-out weeks say most of that range was noise.

But — and this is the part that requires care — sigma-b of point-oh-six-five says it is **not all**
noise either. Stores do differ. Just far less than eight weeks of data made it look.

**[pause]**

*(Land the recommendation.)*

So the defensible plan is a three-part sentence: plan on an eight percent FOODS uplift everywhere;
allow for real but modest store variation around it; and do not act on any individual store's own
estimate until that store has the weeks to support one.

That's a more nuanced answer than either extreme, it's backed by held-out data, and it is exactly
what the hierarchy was for.

---

# ▶ SLIDE 18 — Section divider: Key Takeaways

What hierarchy buys, and what it costs.

---

# ▶ SLIDE 19 — Lecture 11 Part 2: Key Takeaways

One, and it's first for a reason. **Ask what may be pooled before you pool it.** Groups must be
exchangeable. FOODS and HOBBIES are not, and no amount of good sampling fixes a hierarchy built
over the wrong set.

Two. Complete pooling denies groups differ; no pooling lets a group with eight observations report
noise as a finding. You've used both without naming them.

Three. Partial pooling gives each group its own parameter drawn from a shared distribution, and
estimates how much groups genuinely differ.

Four. Shrinkage is automatic and proportionate. Our thin series moved **eight point one times**
further toward the group mean than data-rich ones, with nobody specifying that — and on held-out
weeks their error was **two point six times** smaller.

Five. This is regularization for the sixth time, and the only version where the shrinkage strength
is estimated rather than tuned.

And six. **Write it non-centered.** The obvious form gave forty-three divergences; the
reparameterized form gave none. Same model, a change of coordinates — and divergences were the only
alarm that fired, against a threshold that allows no argument.

**[pause]**

Next time: Bayesian linear regression. Coefficients as distributions, what you can ask them — and
a preprocessing step you have used all semester that will silently destroy the answer.

---

# ▶ SLIDE 20 — References

*(Advance and close. No narration needed.)*
