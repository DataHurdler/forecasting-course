# Lab 12 — Recording Script

**ECON 8310: Business Forecasting · Elasticity — What the Coefficient Means, and What It Costs to Get Wrong**

Lab: `Labs/Lecture12_lab.qmd` (4 steps) · Measured runtime: **~25 minutes** of narration
(the in-room version is budgeted at 45)

**This is the last lab of the course.** Week 16 is synthesis with no lab, so this recording is
also where the practical half of the semester ends. Worth one sentence at the close.

---

## How to use this document

- **`▶ STEP n — Title`** matches the lab's own headings exactly.
- *Italic parentheticals* are stage directions. **[pause]** is a beat; **[STOP — learner works]**
  is where you tell the viewer to pause the video and do something.
- `random_seed=42`, four fits, most under 30 seconds.

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

*(Screen: rendered lab, top of document.)*

Lab 12, the last one. And it is about a single number: a **price elasticity** — the percent change
in demand per percent change in price. Of everything in this course, this is the number most
likely to be acted on directly by somebody who will never read your code.

Today you will estimate one, and then find **three separate ways it can be wrong** while every
diagnostic still says the model is fine.

**[pause]**

And here is the part I want you carrying into the recording. Only one of the three is a
statistical failure. One is a causal-structure failure — the model answers a different question
from the one you asked. And one is a **preprocessing convenience** you have used in nearly every
lab since Lecture 6, which quietly changes what the number means and tells nobody.

None of the three is detectable by R-hat, ESS, or divergences.

---

# ▶ SETUP

*(Screen: run setup.)*

```
8,310 rows | 30 series | sd(log price) = 0.2444
```

Both units and price are in logs, and that is what makes the price coefficient an elasticity —
percent per percent. Say that sentence carefully now, because Step 3 is entirely about the moment
it stops being true.

*(Point at the printed standard deviation.)*

And note **0.2444**, the standard deviation of log price across the whole panel. Keep it in view.
In Step 3 a different, much smaller number takes its place, and the gap between them is where the
damage comes from.

---

# ▶ STEP 1 — The naive elasticity

*(Screen: run the naive fit.)*

Log units on log price, SNAP, and events. Nothing else.

```
Naive
           mean  hdi94_lb  hdi94_ub   r_hat   ess_bulk
b_price -2.8048   -2.8556   -2.7559  1.0018  2321.0850
```

**[pause]**

An elasticity of **minus 2.8**. A one percent price rise costs almost three percent of volume.
That is a dramatic, highly elastic result — and it would change how a category is priced.

Look at the diagnostics. R-hat 1.002. ESS 2,321. The interval runs from −2.86 to −2.76 — a width
of one tenth. This model is *certain*.

It is also badly wrong, and the useful part is that we knew that before fitting it.

*(Beat.)*

Lecture 12's DAG says why. Our thirty series are different **categories** in different stores, and
categories differ systematically in both typical price and typical volume — hobby items cost more
per unit and sell fewer of them. So series identity is an arrow into price **and** an arrow into
units. That is the textbook shape of a confounder.

Comparing a hobby item to a food item is not measuring a price response. It is measuring the
difference between two product categories, and calling it elasticity.

---

# ▶ STEP 2 — Control for the confounder

*(Screen: run the controlled fit and the comparison.)*

Add per-series intercepts and a trend, and center log price *within* each series — so the model
uses only price variation inside a series over time, which is the only variation a manager could
actually create.

Why the trend? Because time is a second confounder. Prices and volumes both rose over these five
years, so within any one series the later weeks are dearer *and* busier. Leave the trend out and
the within-series slope comes out *positive* — about plus 1.7. Lecture 12's confounding slide shows
it.

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

**[pause]** — *let this sit; the callout depends on it.*

Minus 0.49 rather than minus 2.8. The naive estimate was **5.7 times too large**.

Now the line that matters, and it is the second column. The naive interval is **0.099** wide. The
correct one is **0.208** — twice as wide.

*(Screen: the callout.)*

So the wrong model was not noisy. It was **more precise than the right one**, by a factor of two,
while being nearly six times off. It was confidently, tightly, precisely wrong.

**[pause]**

That is worth dwelling on, because it inverts an instinct almost everybody has. A narrow credible
interval feels like a good sign. All it says is that the model is sure about the answer to the
question it was actually asked — and the naive model was asked "how do hobby items differ from
food items?", which the data answers very precisely indeed.

No diagnostic in ArviZ will ever tell you the question was wrong. Only the DAG does, and the DAG
is drawn before any code is written.

**[STOP — learner works]**

Answer the question in the cell: the naive estimate is too *negative*. Which arrow in the DAG
makes the bias run that way rather than the other?

---

# ▶ STEP 3 — The preprocessing step that silently destroys the answer

*(Screen: the framing paragraph, then run.)*

You have standardized predictors in nearly every lab since Lecture 6. It makes `Normal(0,1)`
priors mean the same thing across variables, it helps samplers, and it is a genuinely good habit
that I am not about to tell you to abandon.

Do it to log price here.

```
  unstandardized  b_price = -0.4931
  standardized    b_price = -0.0225
  sd(centered log price)   = 0.0455
  ratio                   = 21.9362
```

**[pause]**

The two fits agree perfectly — they are the same model, and both have clean diagnostics and
intervals excluding zero. But the coefficient went from −0.49 to **−0.022**, a factor of nearly
twenty-two.

Nothing is broken. Standardizing divided log price by its standard deviation, so the coefficient
is now the change in log units per **standard deviation of log price** rather than per percent.
Same model, same fit, different unit — and the number no longer means "elasticity."

*(Point at 0.0455.)*

Now look at what it was divided by. **0.0455.**

Remember the setup printed 0.2444 for log price across the whole panel? This is not that number.
This is the standard deviation of log price **within** a series, after centering — and it is five
times smaller, because prices move very little inside a single store-category over time.

**[pause]** — *this is the subtle part; say it slowly.*

And here is the irony worth naming. That within-series variation is small **because of the good
decision we made in Step 2.** Centering within series is what removed the confounding. It is also
what makes the standard deviation tiny, and the smaller the number you divide by, the larger the
distortion. **The correct modeling choice amplified the preprocessing error.**

*(Run the scenario block.)*

```
  correct (elasticity)   5% price rise -> units -2.38%  revenue +2.50%
  standardized           5% price rise -> units -0.11%  revenue +4.88%
```

**[pause]** — *this is the payoff. Let it land.*

Read those two lines as advice given to a person.

Using the standardized coefficient, a manager is told that a 5% price rise costs about **a tenth
of a percent** of volume — essentially free. The truth is **−2.38%**, more than twenty times
larger. And the revenue projection nearly doubles, from +2.5% to +4.9%.

That is a real decision, made from a real model, that passed every check.

*(Screen: the callout.)*

The model was right. The sampler was right. The diagnostics were right. The error entered through
a **preprocessing convenience three cells earlier**, and it survived all the way to a stocking
decision because nothing downstream carries units. A float is a float. `-0.0225` looks exactly as
authoritative as `-0.4931`.

**[STOP — learner works]**

Fill in the blank: to recover the elasticity from the standardized fit, what do you multiply by?
*(The standard deviation you divided by — 0.0455 — which is the same as dividing by 21.94.)*

*(Resume.)*

And the habit that prevents this, which is the one line to take from the lab: **write down what
one unit of each variable is before you rescale it, and check that the coefficient still answers
the question you asked.** Keep standardizing. Just know what you are holding afterwards.

---

# ▶ STEP 4 — Ask the posterior a business question

*(Screen: run the probability block.)*

Back to the correct fit. The posterior is samples, so any question is answered by counting.

```
  P(elasticity < 0)     = 1.000
  P(elasticity < -1)    = 0.000
  P(elasticity < -0.5)  = 0.460
```

**[pause]**

The first line settles the direction completely. Every posterior sample is negative — demand falls
when price rises, with no doubt whatsoever.

The second line settles the question a pricing manager asks first. Minus one is the boundary
between **elastic** and **inelastic** demand: below it, a price rise loses revenue; above it, a price
rise gains revenue. Not one posterior sample is below minus one. **Demand is inelastic in every
draw.**

The third line is a finer question about size: does a one percent price rise cost more than half a
percent of volume? The posterior puts us at **0.460** — within a few points of a coin flip. So the
honest report is: inelastic, certainly; *how* inelastic, the data cannot settle. A significance
test here would have announced the direction — which was never in question — and said nothing
about either size question.

*(Point at the histogram, then run the scenario.)*

```
  5% price rise:  units -2.38%  [-2.87, -1.87]
                  revenue +2.50%  [+1.99, +3.03]
  P(revenue rises) = 1.000
```

**[pause]**

And now the resolution, which is the most useful thing in the lab.

**Revenue rises with probability 1.000** — across every posterior sample — and the second line
above says why. A five percent rise multiplies revenue by 1.05 to the power one plus the
elasticity, which is above one exactly when the elasticity is above minus one, and every draw is.
The finer size question is genuinely unresolved; the decision question is not.

Only a posterior lets you say both. A point estimate forces you to pick one number and then argue
about which side of the line it falls on. The distribution lets you answer the question that was
actually asked — *should we raise the price?* — with a probability attached, while being honest
that a different question remains open.

**[STOP — learner works]**

Write the two strings: your two-sentence recommendation to the manager with a probability
attached, and one DAG arrow that, if wrong, invalidates it.

---

# ▶ BEFORE YOU LEAVE

**[STOP — learner works]** — *(replaces the in-room "Discuss")*

Four questions, and the last one closes the course.

**The three failures.** You met three ways an elasticity analysis goes wrong — an omitted
confounder, a rescaled coefficient, and a size question the data cannot settle. Which would any
ArviZ diagnostic have caught?

**[pause]** — *none of them. That is the answer, and it is worth sitting with: diagnostics verify
that the computation did what the code said. Everything about whether the code said the right
thing is yours.*

**The habit.** Standardizing is good practice and you should keep doing it. State the rule that
lets you keep the habit without repeating Step 3.

**The decision.** Advise on the 5% rise using only Step 4's output. Then say what you would want
measured before a **20%** rise — and why extrapolating this elasticity that far is a different
kind of claim.

**[pause]** — *the answer worth steering to: the elasticity was estimated from the price variation
that actually occurred within series, and the standard deviation of that is 0.0455 — a few percent.
A 20% move is far outside the range the data has ever seen. That is extrapolation, not estimation,
and it is the same limit you met in Lab 4 when a tree could not predict beyond its training range.*

**The course.** You have now estimated this kind of quantity with OLS, with XGBoost, and with a
Bayesian regression. Only one of them naturally answers "how likely is it that revenue rises?"
Explain what about the Bayesian output makes that question cheap — and name one forecasting task
from this course where you would still not reach for it.

**[pause]**

*(Closing — this is the last lab of the semester.)*

And that is the last lab. Fifteen weeks, one dataset, a dozen model families, and a scoreboard
that ends with a 46-coefficient linear model at the top.

The through-line was never which method wins. It was that every one of these models will return a
confident number whether or not it deserves one — a tree that cannot extrapolate, a network whose
result moves 230 RMSE with the seed, a forecast interval that is honest and useless, a hierarchy
pooling series that never belonged together, and today a coefficient that stopped being an
elasticity three cells before anybody read it.

None of those announced themselves. Every one of them was caught by holding something out and
checking. That habit is the course.

Solutions go up on Canvas after the deadline. See you in Week 16 for the synthesis.

---

# Appendix — expected output

`random_seed=42`; four fits, whole lab ≈ 2 minutes.

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

*Step 3's irony is the best thing in the lab, and it is easy to skip.* The within-series standard
deviation is small (0.0455) **because** Step 2 centered within series to remove the confounding.
The correct modeling decision is what makes the preprocessing error enormous. Say it explicitly;
it is the kind of interaction students never anticipate.

*Do not let Step 2's headline be "the estimate changed."* The headline is that the **wrong model
had the tighter interval** — 0.100 against 0.207. Precision is not accuracy, and a narrow interval
is not evidence of anything except that the model is sure about whatever question it was given.

*Step 4's probabilities must be delivered together.* P(elasticity < −1) = 0.000 is why
P(revenue rises) = 1.000: the revenue question only asks which side of −1 the elasticity is on. The
finer P(elasticity < −0.5) = 0.460 is unresolved. Students hear 0.46 and conclude the analysis
failed. The point is that a posterior can leave one question open while closing the one that
matters — which no point estimate can do. And say plainly that the boundary is −1: earlier
versions of this lab put it at −0.5.
