# Lab 12 — Recording Script

**ECON 8310: Business Forecasting · Elasticity — What the Coefficient Means, and What It Costs to Get Wrong**

Lab: `Labs/Lecture12_lab.qmd` (4 steps) · Measured runtime: **~25 minutes** of narration
(the in-room version is budgeted at 45)

**This is the last lab of the course.** Week 15 is synthesis with no lab, so this recording is
also where the practical half of the semester ends. Worth one sentence at the close.

---

## How to use this document

- **`▶ STEP n — Title`** matches the lab's own headings exactly.
- *Italic parentheticals* are stage directions. **[pause]** is a beat; **[STOP — learner works]**
  is where you tell the viewer to pause the video and do something.
- `random_seed=42`, three fits, most under 30 seconds.

### Numbers aligned with the deck (re-checked 2026-10-13)

Every number below is the lab's own output as last rendered (`random_seed=42`). The Lecture 12
figure script re-ran Steps 1, 2 and 4 on 2026-10-13 and got the same numbers, plus the new
$P(\text{elasticity} < -1) = 0.000$. An earlier version of this script quoted a different run (controlled
$-0.4921$, ESS 2,736, $P(\text{elasticity} < -0.5) = 0.446$). Lecture 12 now carries the lab's
reproducible numbers: naive $-2.805$, controlled $-0.493$ (ESS 3,097), $P(\text{elasticity} < -0.5)
= 0.46$, and the 5% scenario at roughly $-2.4\%$ of units and $+2.5\%$ of revenue.

**The elastic / inelastic boundary is $-1$, not $-0.5$.** Earlier versions of this lab and of
Lecture 12 treated $-0.5$ as the boundary. Step 4 now prints $P(\text{elasticity} < -1)$, which is
0.000: demand is inelastic in every draw, and that is *why* revenue rises in every draw. The
$-0.5$ question survives as a finer size question — does a 1% rise cost more than half a percent of
volume? — and it is the one the data leave open.

### In-room language that needs replacing

Step 2's, Step 3's and Step 4's "your turn" blocks, and the four closing **Discuss** blocks.

---

# ▶ OPENING

*(Show the rendered lab, at the top of the document.)*

This is Lab 12, the last lab of the course, and it's about a single number: a **price elasticity**,
which is the percent change in demand per percent change in price. Of everything in this course,
this is the number most likely to be acted on directly by somebody who will never read your code.

Today you'll estimate one, and then you'll find **three separate ways it can be wrong** while every
diagnostic still says the model is fine.

**[pause]**

Here's the part I want you to carry through the recording. Only one of the three is a statistical
failure. One is a failure of causal structure, where the model answers a different question from
the one you asked. And one is a **preprocessing convenience** you've used in nearly every lab since
Lecture 6, on regularization. It quietly changes what the number means, and it tells nobody.

None of the three can be detected by R-hat, by the effective sample size, or by divergences.

---

# ▶ SETUP

*(Run the setup cell.)*

```
8,310 rows | 30 series | sd(log price) = 0.2444
```

Both units and price are in logs, and that's what makes the price coefficient an elasticity, percent
per percent. Say that sentence carefully now, because Step 3 is entirely about the moment it stops
being true.

*(Point at the printed standard deviation.)*

And note **zero point two four four four**, the standard deviation of log price across the whole
panel. Keep it in view. In Step 3 a different, much smaller number takes its place, and the gap
between them is where the damage comes from.

---

# ▶ STEP 1 — The naive elasticity

*(Run the naive fit.)*

This first model regresses log units on log price, SNAP and events, and nothing else.

```
Naive
           mean  hdi94_lb  hdi94_ub   r_hat   ess_bulk
b_price -2.8048   -2.8556   -2.7559  1.0018  2321.0850
```

**[pause]**

That's an elasticity of **minus two point eight**. A one percent price rise costs almost three
percent of volume. That's a dramatic, highly elastic result, and it would change how a category is
priced.

Now look at the diagnostics. R-hat is one point zero zero two, and the effective sample size is two
thousand three hundred twenty-one. The interval runs from minus two point eight six to minus two
point seven six, a width of one tenth. This model is *certain*.

It's also badly wrong, and the useful part is that we knew that before fitting it.

The DAG from Lecture 12, this week's lecture, says why. Our thirty series are different
**categories** in different stores, and categories differ systematically in both their typical
price and their typical volume. Hobby items cost more per unit, and they sell fewer of them. So
series identity is an arrow into price **and** an arrow into units, and that's the textbook shape of
a confounder.

Comparing a hobby item with a food item isn't measuring a price response. It's measuring the
difference between two product categories, and calling it an elasticity.

---

# ▶ STEP 2 — Control for the confounder

*(Run the controlled fit and the comparison.)*

Now we add an intercept for each series, and a trend. The intercepts are what confine the model to
price variation inside a series over time, which is the only variation a manager could actually
create. We also center log price *within* each series. That leaves the coefficient exactly where it
was. What it improves is the sampler's job.

Why the trend? Because time is a second confounder. Prices and volumes both rose over these five
years, so within any one series the later weeks are dearer *and* busier. If you leave the trend out,
the slope within a series comes out *positive*, at about plus one point seven. The confounding slide
in this week's lecture shows it.

```
Controlled
           mean  hdi94_lb  hdi94_ub
b_price -0.4931   -0.5963   -0.3872
```

```
  naive       -2.805   width of 94% HDI 0.099
  controlled  -0.493   width of 94% HDI 0.208
  the naive estimate is 5.7x too large
```

*(Let this sit, because the callout depends on it.)*

**[pause]**

So the price coefficient is minus zero point four nine rather than minus two point eight. The naive
estimate was **five point seven times too large**.

Now here's the line that matters, and it's in the second column. The naive interval is **zero point
zero nine nine** wide. The correct one is **zero point two zero eight**, about twice as wide.

*(Show the callout.)*

So the wrong model wasn't noisy. It was **more precise than the right one**, by a factor of two,
while being nearly six times off. It was confidently, tightly, precisely wrong.

**[pause]**

That's worth dwelling on, because it reverses an instinct almost everybody has. A narrow credible
interval feels like a good sign. All it says is that the model is sure about the answer to the
question it was actually asked. The naive model was asked how hobby items differ from food items,
and the data answer that very precisely indeed.

No diagnostic in ArviZ will ever tell you the question was wrong. Only the DAG does, and the DAG is
drawn before any code is written.

**[STOP — learner works]**

Pause the video, and answer the question in the cell. The naive estimate is too *negative*. Which
arrow in the DAG makes the bias run that way, rather than the other way?

---

# ▶ STEP 3 — The preprocessing step that silently destroys the answer

*(Show the framing paragraph, then run the cell.)*

You've standardized predictors in nearly every lab since Lecture 6. It makes a Normal zero, one prior
mean the same thing for every variable, and it helps samplers. It's a genuinely good habit, and I'm
not about to tell you to abandon it.

Now do it to log price here.

```
  unstandardized  b_price = -0.4931
  standardized    b_price = -0.0225
  sd(centered log price)   = 0.0455
  ratio                   = 21.9362
```

The two fits agree perfectly, because they're the same model. Both have clean diagnostics, and both
have intervals that exclude zero. But the coefficient went from minus zero point four nine to
**minus zero point zero two two**, a factor of nearly twenty-two.

Nothing is broken. Standardizing divided log price by its standard deviation, so the coefficient is
now the change in log units per **standard deviation of log price**, rather than per percent. It's
the same model and the same fit, in a different unit, and the number no longer means "elasticity."

*(Point at 0.0455.)*

Now look at what it was divided by: **zero point zero four five five**.

Remember that the setup printed zero point two four four four for log price across the whole panel?
This isn't that number. This is the standard deviation of log price **within** a series, after
centering, and it's five times smaller, because prices move very little inside a single store and
category over time.

*(This is the subtle part. Say it slowly.)*

**[pause]**

And here's the irony worth naming. The number we divided by is small **because of a good decision
we made in Step 2.** The series intercepts removed the confounding. Centering within series, which we
did for the sampler, also stripped the spread between series out of log price, and it left only the
small part within each series. The smaller the number you divide by, the larger the distortion. So
**a sound modeling choice amplified the preprocessing error.**

*(Run the scenario block.)*

```
  correct (elasticity)   5% price rise -> units -2.38%  revenue +2.50%
  standardized           5% price rise -> units -0.11%  revenue +4.88%
```

*(This is the payoff. Let it land.)*

**[pause]**

Read those two lines as advice given to a person.

Using the standardized coefficient, a manager is told that a five percent price rise costs about
**a tenth of a percent** of volume, which is essentially free. The truth is **minus two point three
eight percent**, more than twenty times larger. And the revenue projection nearly doubles, from plus
two point five percent to plus four point nine percent.

That's a real decision, made from a real model, and the model passed every check.

*(Show the callout.)*

The model was right, the sampler was right, and the diagnostics were right. The error came in
through a **preprocessing convenience three cells earlier**, and it survived all the way to a
stocking decision, because nothing downstream carries units. A float is a float. Minus zero point
zero two two five looks exactly as authoritative as minus zero point four nine three one.

**[STOP — learner works]**

Pause the video, and fill in the blank in the cell. To recover the elasticity from the standardized
fit, what do you multiply it by?

*(Resume.)*

You multiply by the ratio the cell printed, about twenty-one point nine four, which is roughly one
over zero point zero four five five. Standardizing divided log price by its standard deviation, and
that multiplied the coefficient by the same standard deviation. So to undo it, you divide the
coefficient by zero point zero four five five, which is the same as multiplying it by twenty-one
point nine four. Do that to minus zero point zero two two five, and you're back at an elasticity of
about minus zero point four nine.

And the habit that prevents all this is the one line to take from the lab. **Write down what one
unit of each variable is before you rescale it, and check that the coefficient still answers the
question you asked.** Keep standardizing. Just know what you're holding afterwards.

---

# ▶ STEP 4 — Ask the posterior a business question

*(Run the probability block.)*

Now we go back to the correct fit. The posterior is samples, so any question is answered by
counting.

```
  P(elasticity < 0)     = 1.000
  P(elasticity < -1)    = 0.000
  P(elasticity < -0.5)  = 0.460
```

**[pause]**

The first line settles the direction completely. Every posterior sample is negative, so demand
falls when price rises, with no doubt whatsoever.

The second line settles the question a pricing manager asks first. Minus one is the boundary between
**elastic** and **inelastic** demand. Below it, a price rise loses revenue, and above it, a price
rise gains revenue. Not one posterior sample is below minus one. **Demand is inelastic in every
draw.**

The third line is a finer question about size: does a one percent price rise cost more than half a
percent of volume? The posterior puts that at **zero point four six zero**, within a few points of a
coin flip. So the honest report is that demand is certainly inelastic, but the data can't settle
*how* inelastic. The usual p-value, tested against zero, would have given you only the direction,
and the direction was never in question.

*(Point at the histogram, then run the scenario.)*

```
  5% price rise:  units -2.38%  [-2.87, -1.87]
                  revenue +2.50%  [+1.99, +3.03]
  P(revenue rises) = 1.000
```

**[pause]**

Now for the resolution, which is the most useful thing in the lab.

Revenue rises with a probability of **one point zero zero zero**, across every posterior sample,
and the second line of the earlier block says why. A five percent rise multiplies revenue by one
point zero five, raised to the power of one plus the elasticity. That's above one exactly when the
elasticity is above minus one, and every draw is. The finer question about size is genuinely
unresolved, but the decision question isn't.

Only a posterior lets you say both. A point estimate forces you to pick one number and then argue
about which side of the line it falls on. The distribution lets you answer the question that was
actually asked, which is whether we should raise the price, with a probability attached, while
being honest that a different question remains open.

**[STOP — learner works]**

Pause the video, and fill in the two strings in the cell. The first is your two-sentence
recommendation to the manager, with a probability attached. The second names one arrow in the DAG
that, if it's wrong, invalidates that recommendation.

---

# ▶ BEFORE YOU LEAVE

**[STOP — learner works]** — *(replaces the in-room "Discuss")*

Here are four questions to work through on your own before you go, and the last one closes the
course.

The first is about the three failures. You met three ways an elasticity analysis goes wrong: an
omitted confounder, a rescaled coefficient, and a question about size that the data can't settle.
Which of the three would any ArviZ diagnostic have caught? And what does your answer tell you about
what diagnostics are for?

**[pause]**

The answer is none of them, and it's worth sitting with. Diagnostics verify that the computation did
what the code said. Whether the code said the right thing is entirely up to you.

The second question is about the habit. Standardizing is good practice, and you should keep doing
it. State the rule that lets you keep the habit without repeating Step 3.

The third question is about the decision. Advise the manager on the five percent rise using only the
output of Step 4. Then say what you'd want measured before a **twenty percent** rise, and why
extrapolating this elasticity that far is a different kind of claim.

**[pause]**

Here's the answer I'd steer you toward. The elasticity was estimated from the price variation that
actually occurred within series, and the standard deviation of that variation is zero point zero
four five five, only a few percent. A twenty percent move is far outside the range the data have
ever seen. That's extrapolation, not estimation, and it's the same limit you met in Lab 4, on
decision trees, when a tree couldn't predict beyond its training range.

The last question is about the course. You've now estimated this kind of quantity with OLS, with
XGBoost and with a Bayesian regression. Only one of them naturally answers the question of how
likely it is that revenue rises. Explain what it is about the Bayesian output that makes that
question cheap, and name one forecasting task from this course where you still wouldn't reach for
it.

*(Closing. This is the last lab of the semester.)*

And that's the last lab. It's been fourteen weeks, with one main dataset, a dozen model families,
and a scoreboard that ends with a forty-six-coefficient linear model at the top.

The through-line was never which method wins. It was that every one of these models will return a
confident number whether or not it deserves one. There was a tree in Lab 4 that couldn't
extrapolate, and a network in Lab 7 whose result moved two hundred thirty RMSE with the seed. There
was a forecast interval two weeks ago that was honest and useless, and a hierarchy last week that
pooled series that never belonged together. And today there was a coefficient that stopped being an
elasticity three cells before anybody read it.

None of those announced themselves. Every one of them was caught by checking something the model
couldn't check for itself. That habit is the course.

Solutions go up on Canvas after class. See you next week, in Week 15, for the synthesis.

---

# Appendix — expected output

`random_seed=42`; three fits, whole lab ≈ 2 minutes.

| Quantity | Value |
|---|---|
| Panel | 8,310 rows · 30 series · **sd(log price) = 0.2444** |
| **Naive** `b_price` | **−2.8048** · HDI [−2.8556, −2.7559] · R-hat 1.002 · ESS 2,321 |
| **Controlled** `b_price` | **−0.4931** · HDI [−0.5963, −0.3872] · R-hat 1.002 · ESS 3,097 |
| Bias | naive is **5.7× too large** |
| Interval widths | naive **0.099** vs controlled **0.208** — the wrong model is *twice as precise* |
| Standardized `b_price` | **−0.0225** |
| sd(**centered** log price) | **0.0455** — five times smaller than the panel-wide 0.2444 |
| Rescaling ratio | **21.94** ( = 1 / 0.0455 ) |
| 5% rise, correct | units **−2.38%** [−2.87, −1.87] · revenue **+2.50%** [+1.99, +3.03] |
| 5% rise, standardized | units **−0.11%** · revenue +4.88% — **21× understated** |
| P(elasticity < 0) | **1.000** |
| P(elasticity < −1) | **0.000** — inelastic in every draw |
| P(elasticity < −0.5) | **0.460** — near a coin flip |
| P(revenue rises at +5%) | **1.000** |

**Three things to know before recording.**

*Step 3's irony is the best thing in the lab, and it is easy to skip.* The standard deviation we
divide by is small (0.0455) **because** Step 2 centered within series. (The series intercepts are
what remove the confounding; centering leaves the coefficient unchanged and helps the sampler.)
A sound modeling decision is what makes the preprocessing error enormous. Say it explicitly;
it is the kind of interaction students never anticipate.

*Do not let Step 2's headline be "the estimate changed."* The headline is that the **wrong model
had the tighter interval** — 0.099 against 0.208. Precision is not accuracy, and a narrow interval
is not evidence of anything except that the model is sure about whatever question it was given.

*Step 4's probabilities must be delivered together.* P(elasticity < −1) = 0.000 is why
P(revenue rises) = 1.000: the revenue question only asks which side of −1 the elasticity is on. The
finer P(elasticity < −0.5) = 0.460 is unresolved. Students hear 0.46 and conclude the analysis
failed. The point is that a posterior can leave one question open while closing the one that
matters — which no point estimate can do. And say plainly that the boundary is −1: earlier
versions of this lab put it at −0.5.
