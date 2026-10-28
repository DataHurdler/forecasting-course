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

This is Lecture 11, Part 2, Bayesian Statistics Two, and today is about hierarchical models.

Last week, in Part 1, on Bayesian time series, we fitted one series and got a full predictive
distribution out of it. Today we have thirty series, and they aren't equally informative. Some
have years of history, and in the real world some would be new stores with only a handful of
usable weeks.

That imbalance is exactly the situation partial pooling exists for, and it's genuinely one of the
most useful ideas in applied statistics.

**[pause]**

But before any of that, there's a question you have to answer first, and it isn't a statistical
question at all. If you get it wrong, you end up with a model that fits cleanly, passes every
diagnostic, and gives you a confidently wrong answer.

That question is where we start.

---

# ▶ SLIDE 2 — Lecture Outline

We'll take this in five parts. First, which series may be pooled at all, which is the question
that comes before the model. Second, partial pooling itself, which turns out to be one extra line
of code. Third, fitting it, along with a trap that catches nearly everyone the first time. Fourth,
reading the result, and checking it against data the model never saw. And we'll finish with the
takeaways.

---

# ▶ SLIDE 3 — Section divider: Which Series May Be Pooled

This first section makes one point: before you pool anything, ask what the groups have in common.

---

# ▶ SLIDE 4 — The Question

Here's a concrete business question, and you've met it before. **Does demand rise in weeks when
SNAP benefits are paid out, and by how much, store by store?**

The memo in Homework 3, the one on trees and random forests, asked how your model could help
decide which store-category combinations to stock up for a SNAP week, rather than applying one
national uplift. That question *assumed* the differences between series were real. Today we check.

*(Point at the figure. This is the slide that makes the lecture work.)*

But one question comes before the model, and it costs nothing to ask. SNAP is a **food**
assistance program. So let's measure the raw SNAP-week effect in all thirty series, over all two
hundred seventy-seven weeks of each one. There's no model here. It's just the mean log units in
high-SNAP weeks minus the mean in the other weeks.

*(Point at the axis first.)*

Each dot is one series, meaning one store and one category. There are three rows, with ten stores
in each. The dashed line at zero means no effect at all, and the black bar in each row is that
category's mean.

*(Point at the two gray rows.)*

Start at the bottom. HOBBIES and HOUSEHOLD sit in a tight huddle right on the zero line. Their
means are plus zero point zero zero six and plus zero point zero one one, and the spread across
stores is about a hundredth. A few stores even come out slightly negative.

*(Point at the blue row.)*

Now look at FOODS. Every one of the ten dots is to the right of zero, and the mean is plus zero
point zero eight nine, which is roughly a nine percent lift. And look how wide that row is. It
runs from California 2, at about plus zero point zero one, to Wisconsin 2, at about plus zero
point two one. The spread across stores is zero point zero five eight, which is four to six times
what the other two rows show.

**[pause]**

So FOODS responds, and the other two barely move. That's no surprise once you say it out loud,
because SNAP is food assistance. You could have predicted which row would sit apart before
computing anything.

Notice what that wide blue row is, too. It's the spread that a hierarchy over the FOODS stores has
to explain. Some of it is stores genuinely differing, and some of it is noise, because even two
hundred seventy-seven weeks is a finite sample. Splitting those two apart is exactly the job we'll
hand the model in a few slides.

So **these are not thirty versions of the same thing.** Pooling all thirty would tell the model
they're interchangeable, and the model would believe you.

**[pause]**

I want to stress how cheap that check was. It took three group-bys and a subtraction, with no
model, no sampler and no priors. That's two minutes of work.

That's typical of the most valuable checks in this course: the leakage checks, the benchmark, and
the planted noise control in Homework 3. They're nearly free, they happen before the modeling, and
they're the ones people skip because they don't feel like real analysis.

---

# ▶ SLIDE 5 — Exchangeability: the Assumption Under Every Hierarchy

So let's name the assumption properly, because it sits under every hierarchical model you'll ever
build.

*(Point at the definition box.)*

Groups are **exchangeable** when, *before seeing the data*, you have no reason to expect one to
differ from another in a particular direction. Swapping their labels wouldn't change what you
believe.

That's the test. The question isn't "are they similar?" It's "would relabeling them change what I
expect?"

**[pause]**

That distinction matters, so let me push on it. Exchangeable does **not** mean identical. The ten
FOODS stores certainly differ. They have different neighborhoods, different sizes and different
customers, and exchangeability doesn't deny any of that.

It says something weaker and more useful: before seeing the data, you have no basis for ranking
them. You can't say *in advance* that the Texas store's SNAP effect should be bigger than
California's.

That's a claim about your knowledge, not about the world. That's why it's a judgment call you have
to make and defend, and it's why no diagnostic can make it for you.

Now let's apply it. The ten FOODS series **qualify**. They're the same category in ten different
stores, and you genuinely couldn't say in advance which store's SNAP effect should be largest.
Swap the store labels, and you'd believe exactly the same things.

FOODS against HOBBIES does **not** qualify. You know before looking that a food-assistance program
moves food. There's a direction, and you knew it in advance.

*(Point at the line that begins "Today".)*

So for today we work with the **ten FOODS series**, which is the set we can defend, and we
deliberately thin five of them down to eight weeks of data. That leaves five series rich in data
and five poor in it, and that imbalance is the whole point of the exercise.

**[pause]**

*(Point at the muted footnote, and sell the lab.)*

In the lab, we pool all thirty anyway, on purpose, and measure exactly what it costs. I'd
encourage you to do that one carefully. It's the most instructive forty-five minutes of the
semester, because the wrong model looks *completely fine* until you check it against held-out
data.

---

# ▶ SLIDE 6 — Complete Pooling and No Pooling

Within that exchangeable set, there are two obvious things you could do, and each one fails in a
different direction.

*(Walk the table.)*

The first is complete pooling, which means one SNAP effect for all ten series. In the code that's
a single b, drawn from one Normal, so there's one parameter. This denies that stores differ at
all, so a store that really is different gets averaged away into the crowd.

The second is no pooling, which means a separate effect for each series, estimated independently.
In the code that's shape equals ten. Now a series with eight weeks of data gets an estimate built
from eight weeks of data. That's mostly noise, reported as a finding.

**[pause]**

*(Deliver this callback deliberately. It's the one that lands.)*

And here's the thing: **you've already used both of these, without naming them.**

The random forest in Homework 3 pooled all thirty series into one model, with store dummies.
That's close to complete pooling.

Homework 1 fitted ARIMA to one series at a time, completely independently. That's no pooling.

So you've made this choice twice, and both times you made it by **convenience**, by whatever the
library made easy. Partial pooling turns it into a modeling decision, with a parameter, and it
lets the data settle where you should sit between the two extremes.

It's worth noticing that this is a recurring shape in the course. Something you'd been deciding
implicitly, by a tooling default, gets promoted into an explicit parameter with a posterior.

Lecture 6, on regularization, did it for which predictors survive. You'd been choosing them by
intuition, and then LASSO turned that choice into a penalty. Part 1, last week, did it for how
fast the level is allowed to move. Today it's how much the groups share.

**[pause]**

Each time, the gain is the same. The decision becomes visible, arguable and estimable, instead of
being buried in a default nobody wrote down.

---

# ▶ SLIDE 7 — Section divider: Partial Pooling

In this section, each series gets its own effect, drawn from a shared distribution that the data
also estimates.

---

# ▶ SLIDE 8 — The Hierarchy Is One Extra Line

Here's the whole idea, and it really is one line.

*(Point at the equation.)*

Give each series its own effect, b-j. But then say **where those effects come from**: b-j is drawn
from a Normal with mean mu-b and standard deviation sigma-b, for j from one to ten.

mu-b is the typical SNAP effect across FOODS, and sigma-b is **how much stores genuinely differ**.
And crucially, both of those are *estimated from the data*, not assumed by you.

**[pause]**

*(Walk the three-row table, because it shows the mechanism.)*

Think about the two extremes first.

If sigma-b goes to zero, the model has concluded that stores don't differ, and every b-j collapses
onto mu-b. That reproduces complete pooling.

If sigma-b is large, the model has concluded that they differ a lot, and each b-j is free to sit
wherever its own data puts it. That approaches no pooling.

And in between, which is where you almost always end up, each b-j is pulled toward mu-b, and **how
far it's pulled depends on how much data that series has.**

*(Point at the last paragraph.)*

**[pause]**

That last row is the whole idea, so let me say it plainly. A series with two hundred seventy-seven
weeks barely moves, because its own estimate is already trustworthy. A series with eight weeks
moves a long way, because its own estimate carries almost no information.

And **nobody chose those weights.** There's no threshold, no rule, nothing like "if n is less than
thirty." The weights fall out of estimating sigma-b.

---

# ▶ SLIDE 9 — Why This Is the Right Answer, Not a Compromise

I want to address an objection that's probably forming, because it's a reasonable one.

Partial pooling sounds like **splitting the difference** between two bad options. It sounds like a
fudge, like meeting in the middle because you can't decide.

It's better than that, and the reason is worth stating.

**[pause]**

When a series has eight observations, its unpooled estimate is mostly sampling noise. Pulling it
toward the group mean isn't a compromise. It's the **correct response to a noisy measurement**.

You already believe this in another form. You wouldn't conclude that a coin is biased after three
flips, even if all three came up heads. You'd pull your estimate back toward one half, because
three flips aren't much evidence. That's shrinkage, and you do it by instinct.

*(Point at the key box, and slow down for the callback.)*

And this is **regularization, for the sixth time this semester.**

A hierarchical prior shrinks each group's estimate toward a common value, exactly as Ridge shrinks
coefficients toward zero. It's the same operation under a different name.

We met the smoothing penalty in Lecture 3, on GAMs, then cost-complexity pruning in Lecture 4,
`reg_lambda` in Lecture 5, Ridge and LASSO in Lecture 6, and weight decay in Lecture 7. And now
there's the hierarchical prior.

**[pause]**

But this version has something none of the other five had. In every earlier case, you chose the
shrinkage strength by cross-validation: you tried values and picked one. Here, sigma-b is
**estimated** as part of the model, with its own posterior and its own uncertainty.

That makes it the cleanest version of the idea in the course.

---

# ▶ SLIDE 10 — Section divider: Fitting It, and the Trap

This section is about fitting the model, and here's the trap: the obvious code doesn't work, and
the reason is geometry rather than statistics.

---

# ▶ SLIDE 11 — The Model in PyMC

Here it is in PyMC, in eight lines.

mu-b and sd-b are the group-level parameters, meaning the typical effect and how much stores
differ. Then comes z, a vector of standard Normals, and b is built as mu-b plus z times sd-b.

After that come a per-series intercept, the observation noise and the likelihood.

*(Point at the indexing.)*

The pattern to internalize is `b[idx]`. J is the number of series, which is ten here, and `idx`
maps each row to its series number, so `b[idx]` picks out the right effect for every observation.
Every hierarchical model in PyMC is written that way, and once you can read that line, you can read
most of the applied Bayesian literature.

**[pause]**

*(Point at the closing paragraph.)*

And notice what is **not** in that code.

There's no decision about how much to pool. There's no weighting scheme, no threshold for "enough
data," and no rule that says series with fewer than twenty observations get shrunk by half.

All of that comes out of estimating one parameter, sigma-b. That's the elegance of it.

**[pause]**

Here's one practical note on priors, since we're looking at the code. `sd_b` gets a HalfNormal
prior with a scale of zero point two five. It's positive only, because a standard deviation has to
be, and it's concentrated on small values, because we're working in log units, where an effect of
zero point two five is already very large.

That prior is doing real work. With only ten groups, there isn't much information about how much
the groups differ, so the data can't pin sigma-b down on its own. A vague prior here would let it
wander large, which weakens the pooling, and you'd drift toward the unpooled answer.

That's the general situation with group-level scale parameters. They're the hardest thing in the
model to estimate, and they're where your prior matters most.

---

# ▶ SLIDE 12 — The Trap: Write It the Obvious Way and It Breaks

Now, why did I write b as mu-b plus z times sd-b, instead of the obvious thing?

The obvious thing is to declare b directly as a Normal, with mean mu-b, standard deviation sd-b and
shape J. That's called the *centered* form, and it reads *exactly* like the equation on slide
eight. It's the natural translation, and everybody writes it that way first.

On our data, it produces this.

*(Walk the table.)*

The centered fit has **forty-three** divergences, where we want zero. The non-centered version has
zero.

R-hat is one point zero zero three for both, so it **passes**, comfortably.

The effective sample size is nine hundred sixty-six for the centered fit and one thousand seven
hundred nineteen for the non-centered one. That **passes** too, since even the centered number is
more than twice the threshold of four hundred.

**[pause]**

*(This is the point of the slide. Take it slowly.)*

Look at those three rows again, because only one of them objects.

Lecture 10, two weeks ago, on Bayesian foundations, gave you three numbers to check, and it
promised you'd meet divergences for real in this lecture. Here they are. If you'd checked only the
other two, R-hat and the effective sample size, you would have shipped this model, because both
are clean. **Forty-three divergences against a threshold of exactly zero** is the only alarm, and
it's the one to trust. There's no version of "close to zero" that gets you to forty-three.

Notice, too, that this is the mirror image of Part 1, last week. There, R-hat and the effective
sample size failed while the divergences stayed at zero. Today the divergences are the only thing
that objects.

So the rule to carry forward is this: on hierarchical models, **divergences fire first, and least
ambiguously.** Don't wait for R-hat to tell you something is wrong, because it may never say so.

*(Point at the muted line.)*

**[pause]**

Here's one honest caveat about the exact numbers. Divergence counts aren't stable quantities.
Change the seed, or even the machine, and they move. Reading the same data back from a CSV file,
which changes the last binary digit of a few hundred values, moves this count from forty-three to
forty-nine. Across ten more seeds, the centered form gave anywhere from fifteen to two hundred
eighty-five divergences, and the non-centered form gave from zero to eleven. What doesn't move is
the pattern. The centered form diverged on every one of the eleven runs, while R-hat or the
effective sample size objected on only two of them.

So if you run the lab and see a different count, that's expected. If you see the centered form
diverge and the non-centered form barely diverge at all, you've reproduced the finding.

---

# ▶ SLIDE 13 — Why It Breaks: a Funnel

Why does it happen? This picture is the answer.

*(Point at the left panel.)*

Every gray dot is one posterior draw from the centered fit, for one thin series, CA_1. Across the
bottom is how far its SNAP effect sits from the group mean, which is b-j minus mu-b. Up the side is
sigma-b, on a log scale.

The dashed lines show what the prior says. Given sigma-b, b-j sits within about two sigma-b of
mu-b. So when sigma-b is large the band is wide, and when sigma-b is small the band pinches almost
to nothing. On a log axis, that shape is a **funnel**.

**[pause]**

*(Point at the crimson dots.)*

The crimson dots are the forty-three divergences. Look where they are: **in the neck.** Thirty-nine
of the forty-three sit in the bottom tenth of the sigma-b values the sampler visited.

Now look at the dotted line. The centered chains **never got below it**, which means they never
went below a sigma-b of about zero point zero one one. The sampler tunes its step size for the wide
part of the funnel, and then it tries to get into the neck with steps that are far too big. It
fails, and it reports a divergence. So it never explores the neck at all.

*(Point at the right panel.)*

Now here's the non-centered fit, on the same vertical axis. Across the bottom is z-j, which is the
same quantity divided by sigma-b. That's all non-centering is. Writing b-j as mu-b plus sigma-b
times z-j is a **change of coordinates**. Dividing each height by its own width turns the funnel
into a column. Under the prior, z-j has a spread of one at every height, so the dashed lines are
straight.

And the sampler walks it. There are zero divergences, and it goes below the dotted line: about
three and a half percent of its draws sit in the region the centered chains never reached. Thirty
draws go even lower than the bottom of this axis.

**[pause]**

I should be precise about one thing. What I said about the funnel and the column is exact for the
**prior**. The posterior adds the data, which narrows things a little, but as you can see, the
draws follow the prior's shape closely. It's the same model and the same posterior, in different
coordinates, and only one set of coordinates is walkable.

This is section four point six point one of Martin, Kumar and Lao, the book cited on the slide,
and it's one of the most common failures in applied hierarchical modeling.

*(A practical note, because students will reach for the wrong fix.)*

When you see divergences, the instinct is to raise `target_accept`, which forces smaller steps.
Sometimes that helps a little. Here it helps partly: at zero point nine five, the centered fit
still gives fourteen divergences instead of forty-three, and it runs slower for the privilege.

Reparameterizing is the real fix. It isn't free, and on this machine the non-centered fit took a
little longer per run than the centered one. But it removes the problem instead of tiptoeing
around it.

So here's the habit. When a hierarchical model diverges, your first move is to check whether you
wrote it centered, not to tune the sampler.

---

# ▶ SLIDE 14 — Section divider: Reading the Result

This section reads the result. We'll measure the shrinkage, and then check it against data the
model never saw.

---

# ▶ SLIDE 15 — Shrinkage, Measured

So let's see what the pooling actually did. There's one row per series. The dot is each series'
own estimate, from the unpooled fit, and the arrow goes to where the hierarchy put it.

*(Point at the crimson rows.)*

The **thin** series, with eight weeks each, are in crimson. Look at the length of those arrows.
CA_1 starts at zero point two nine and ends just right of the dashed line, which is mu-b, at zero
point zero seven eight. WI_2 starts at zero point three four and travels even farther. Even TX_1,
which started *below* the group mean, gets pulled up toward it. Shrinkage moves estimates in both
directions, but always toward mu-b.

*(Point at the gray rows.)*

The **full** series, with two hundred seventy-seven weeks each, are in gray, and their arrows are
short. TX_2 barely moves at all. That little tick is its whole journey.

**[pause]**

The average thin arrow is zero point one two zero long, and the average full arrow is zero point
zero one five. So the data-poor series moved **eight point one times further** toward the group
mean than the data-rich ones. And nobody told the model which series were thin. It inferred that
entirely from how much each series had to say.

*(Point at the spread numbers. This is the subtle part.)*

Now look at where the arrowheads land, because this is what separates partial pooling from just
averaging everything.

The thin series' apparent spread of zero point one zero eight nearly **vanishes**, down to zero
point zero one. But the full series **keep most of theirs**, going from zero point zero five five
down to zero point zero three seven. WI_3 still sits well to the right of the others after pooling.

The hierarchy didn't flatten everything. It discarded spread that eight weeks couldn't support,
and it kept spread that two hundred seventy-seven weeks *could*.

**[pause]**

*(Point at the muted note.)*

There's a clean way to see the rule it followed. Each pooled estimate lands, approximately, at a
precision-weighted point between mu-b and the series' own estimate. A thin series' own estimate is
very imprecise, so it keeps only about a tenth of it, and it goes about ninety percent of the way
to mu-b. A full series keeps about three-quarters of its own. I say "approximately" because mu-b
and sigma-b aren't known numbers here, and the model averages over its uncertainty about them.
Even so, a plain weighted average lands within about zero point zero one of every arrowhead.

That's a strong claim, and you should be skeptical of it. So the next slide tests it against data
the model never saw.

---

# ▶ SLIDE 16 — Checked Against Data the Model Never Saw

Here's the nice thing about having thinned those series ourselves: **the weeks we removed still
exist.** They're the weeks we thinned from CA_1, CA_3, TX_1, TX_3 and WI_2. So we can compute the
SNAP effect that actually occurred in them, and grade both models against it. That's the ring on
each crimson row.

*(Walk the crimson rows. This is the payoff.)*

Start with CA_1. The unpooled fit said zero point two eight six, and the hierarchy said zero point
zero nine seven. The truth in the held-out weeks is **zero point zero eight seven**, right where
the arrow landed.

For CA_3, the arrow runs from zero point one zero five to zero point zero eight two, and the truth
is zero point one zero two. This is the one series where the unpooled estimate was closer, by a
hair.

TX_1 went from zero point zero five two unpooled to zero point zero seven five pooled, against a
truth of zero point zero eight two. The arrow pointed the right way and nearly reached the ring.

TX_3 went from zero point two one nine to zero point zero nine one, and its truth is zero point
zero seven two.

WI_2 went from zero point three three eight all the way down to zero point one zero three, but its
ring sits at zero point two zero seven, in the middle of the arrow. The hierarchy pulled it
**past** its truth. That's the honest cost of shrinkage. WI_2 really is a high-response store, and
eight weeks weren't enough to convince the model. Even so, the pooled estimate ended up closer than
the unpooled one.

**[pause]**

*(Now the summary line.)*

The RMSE against the truth is zero point one two six unpooled, and **zero point zero four eight**
for the hierarchy. The pooled error is **two point six times smaller**, and pooling beat the
unpooled fit on four of the five series.

Notice the character of the unpooled errors, too. They weren't merely noisy. They were **confidently
wrong**. Zero point three three eight is a specific, precise-looking claim that a store's SNAP
uplift is forty percent. It came from eight weeks of data, and the truth was about twenty-three
percent.

*(Point at the muted note.)*

So borrowing strength is literal here. The thin series were estimated better **because the other
five existed.**

---

# ▶ SLIDE 17 — What the Model Actually Concluded

So what's the business answer?

The estimated hierarchy gives a mu-b of zero point zero seven eight, with a ninety-four percent HDI
from about zero point zero two to zero point one five, and a sigma-b of zero point zero six five.

In words, a SNAP week lifts FOODS units by about **eight percent**, and stores genuinely differ
around that by about **seven percentage points**.

**[pause]**

Both of those are answers, and the second is the one no point estimate would ever have given you.
"How much do stores differ?" is now a question with a number attached.

*(Point at the warning box.)*

But be careful what you conclude from it.

This does **not** license store-by-store stocking. An analyst reading the thin stores' own
estimates would report uplifts from five percent to forty percent, and build a differentiated
stocking plan on them. The held-out weeks say that most of that range was noise.

But, and this is the part that takes care, a sigma-b of zero point zero six five says it is **not
all** noise either. Stores do differ, just far less than eight weeks of data made it look.

**[pause]**

*(Land the recommendation.)*

So the defensible plan comes in three parts. Plan on an eight percent FOODS uplift everywhere.
Allow for real but modest store variation around it. And don't act on any individual store's own
estimate until that store has the weeks to support one.

That's a more nuanced answer than either extreme, it's backed by held-out data, and it's exactly
what the hierarchy was for.

---

# ▶ SLIDE 18 — Section divider: Key Takeaways

This last section sums up what a hierarchy buys you, and what it costs.

---

# ▶ SLIDE 19 — Lecture 11, Part 2: Key Takeaways

There are six takeaways, and the first one comes first for a reason. **Ask what may be pooled
before you pool it.** The groups must be exchangeable. FOODS and HOBBIES aren't, and no amount of
good sampling fixes a hierarchy built over the wrong set.

Second, complete pooling denies that groups differ, and no pooling lets a group with eight
observations report noise as a finding. You've used both without naming them.

Third, partial pooling gives each group its own parameter, drawn from a shared distribution, and
it estimates how much the groups genuinely differ.

Fourth, shrinkage is automatic and proportionate. Our thin series moved **eight point one times**
further toward the group mean than the data-rich ones, with nobody specifying that, and on held-out
weeks their error was **two point six times** smaller.

Fifth, this is regularization for the sixth time, and it's the only version where the shrinkage
strength is estimated rather than tuned.

And the last one: **write it non-centered.** The obvious form gave forty-three divergences, and
the reparameterized form gave none. It's the same model with a change of coordinates, and
divergences were the only alarm that fired, against a threshold that allows no argument.

**[pause]**

Next week, in Lecture 12, we move on to Bayesian linear regression: coefficients as distributions,
and what you can ask them. You'll also meet a preprocessing step you've used all semester, one that
will silently destroy the answer.

---

# ▶ SLIDE 20 — References

*(Advance and close. No narration needed.)*
