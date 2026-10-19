# Lecture 5 — Recording Script

**ECON 8310: Business Forecasting · Tree Ensembles: Random Forests & Boosted Trees**

Deck: `Slides/Lecture05_TreeEnsembles.pdf` (27 pages) · Measured runtime: see the timing guide

---

## How to use this document

- **`▶ SLIDE n — Title`** marks where to advance. The number is the PDF page.
- *Italic parentheticals* are stage directions. **[pause]** means stop for a beat.
- This is the longest deck in the course. The timing guide flags what compresses.

---

# ▶ SLIDE 1 — Title page

Lecture 5: Tree Ensembles.

Last week ended on a complaint. A single decision tree is unstable — change a few training
points and the whole structure changes. I called that a fatal weakness.

Today we do something that should feel slightly perverse. We are going to take that
instability and turn it into the source of the method's strength.

**[pause]**

Two families, and they attack opposite problems. **Bagging and random forests** attack
variance, by averaging many trees built in parallel. **Boosting** attacks bias, by building
trees sequentially where each one fixes what the last one got wrong.

By the end you'll have the method that, on most tabular business data, is simply the best
thing available.

---

# ▶ SLIDE 2 — Lecture Outline

Five parts. From single trees to ensembles. Bagging. Random forests. Feature importance. Then
gradient boosting and XGBoost.

---

# ▶ SLIDE 3 — Section divider: From Single Trees to Ensembles

Part one, and it starts with an observation that has nothing to do with statistics.

---

# ▶ SLIDE 4 — The Wisdom of Crowds

In 1906, at a county fair, eight hundred people guessed the weight of an ox.

Francis Galton collected the guesses expecting to show that the crowd was foolish. Individually
they were — almost nobody was close.

But the **average of all eight hundred guesses** came within one pound of the true weight.

**[pause]**

Why does that work? Because the errors were *independent*. Some people guessed high, some low,
for unrelated reasons, and averaging cancelled them.

Now transfer that to models. A single decision tree is noisy — it splits somewhere slightly
arbitrary and everything downstream inherits that arbitrariness. But if you train many trees on
different versions of the data, each is noisy in a *different* way. Average them and the noise
largely cancels while the signal, which they agree on, survives.

**[pause]**

I want you to notice the word doing all the work: **independent**.

Galton's crowd worked because the guesses were unrelated. If eight hundred people had all
overheard the same wrong estimate, averaging would have gained nothing.

Hold onto that. The entire difference between bagging and random forests — the whole reason
random forests exist — turns on that one word.

---

# ▶ SLIDE 5 — Section divider: Bagging

Part two. Bootstrap aggregating.

---

# ▶ SLIDE 6 — Bagging: The Big Idea

Three steps.

Draw **B bootstrap samples** — sample your training data with replacement, same size as the
original. Because it's with replacement, each sample gets some observations twice and misses
others entirely. That's the point: each sample is a slightly different view of the data.

Fit one deep tree on each. Deep — we're not pruning. We *want* each tree to be low-bias and
high-variance, because averaging is about to handle the variance.

Then average all B predictions.

**[pause]**

Now the equation, which is the most important thing on this slide.

Call one tree's variance sigma squared, and rho the correlation between two trees' predictions.
The variance of the averaged forecast is **rho sigma squared, plus one-minus-rho over B, sigma
squared**.

Two terms. Look at what happens as B — the number of trees — grows.

The second term has B in the denominator, so it shrinks toward zero. That's why more trees
never hurts.

The first term has no B in it at all. It does not move.

---

# ▶ SLIDE 7 — More Trees, or Less Alike Trees

Here is that equation drawn. I've divided through by sigma squared, so one tree sits at one on
the vertical axis.

*(Point at the axes.)*

Across the bottom, the number of trees you average, on a log scale, from one to five hundred.
Up the side, the variance of the average.

This is a schematic. Nothing here is measured, because rho is a property of how the trees vary
across possible training sets, and one forest can't measure its own rho. I picked three values
to show the shape.

**[pause]**

*(Trace the orange curve.)*

Each value of rho draws its own curve. Take rho of one half, the orange one. Every curve starts
at one, because one tree is one tree. Add trees and you slide along the curve — the black arrow
— fast at first, then flattening onto the dashed line at one half. That dashed line is the
**floor, rho sigma squared**. Past a hundred trees or so you're buying almost nothing.

*(Point at the gray curve, then the red arrow.)*

Now the gray curve, rho of point nine. Highly correlated trees. Its floor is point nine:
average a thousand of them and you have barely improved on one. You've grown a thousand trees
and reproduced one. That's Galton's crowd overhearing each other.

The red arrow is the only way down: make the trees **less alike**, and you drop to a lower
curve with a lower floor. Rho of point one, the blue curve, ends near a tenth of one tree's
variance.

**[pause]**

So there are two dials, and they do different things. More trees moves you along a curve.
Decorrelation moves you to a different curve. And it's exactly the second dial that the next
section turns.

---

# ▶ SLIDE 8 — Out-of-Bag (OOB) Error: Free Cross-Validation

A pleasant side effect of bootstrapping.

Each bootstrap sample draws n observations with replacement, so some get picked more than once
and some not at all. Work through the probability and you find each sample contains about
**sixty-three percent** of the distinct training points.

Which means roughly **thirty-seven percent are left out** of any given tree. Those are the
out-of-bag observations.

**[pause]**

Here's the trick. For observation *i*, collect predictions only from the trees where *i* was
out-of-bag. Average those. Compute the RMSE across all observations.

Every one of those predictions came from a tree that **never saw that observation**. So the OOB
error is an honest out-of-sample estimate — and you got it without setting aside a validation
set or refitting anything. It falls out of the bootstrapping you were doing anyway.

In sklearn it's one argument: `oob_score=True`.

**[pause]**

One caution, and it's the one that matters for this course. Two, really — and the second is the
important one.

First, the size. You may hear that OOB is unreliable on small data because each row is
out-of-bag for too few trees. That's not the reason. Each row sits out about thirty-seven percent
of the trees whatever the sample size is — with five hundred trees, about a hundred eighty of
them. What a small sample lacks is *rows*: the OOB error is an average over n rows, and a forest
on CA_1 food sales trains on a hundred seventy-three weeks. A hundred seventy-three errors make a
noisy average. That part is ordinary sampling noise.

**[pause]**

Second, and this is the one that bites. Look at who does the scoring.

*(Point at the figure.)*

This is the real forest from later in the lecture — the weekly CA_1 forest, five hundred trees —
and these are its actual bootstrap draws. Each row is one tree; I'm showing twenty-four of them.
Each column is one training week, in date order, from early 2012 to May 2015. Gray means that
week was in the tree's bootstrap sample. White means it was out of bag.

*(Point at the boxed column.)*

Pick one week: the boxed column, the week of thirty December 2013. Eight of these twenty-four
trees left it out — the ones marked with red triangles. Those eight trees are the ones that score
it.

Now look to the right of the box, along those eight rows. Everything red is a week *after*
thirty December that the scoring tree trained on. Across the whole forest, the trees that score a
week still trained on about sixty-three percent of the weeks that came after it — the same share
as any bootstrap sample, because leaving one week out says nothing about the others.

So OOB is answering "how well can the model fill in a missing week inside the years it has seen?"
— not "how well will it forecast next month?" Those are different questions, and only the second
one is your job.

So on this data, use proper walk-forward cross-validation. Lab 5 shows you the two numbers side by
side and asks you to account for the gap.

---

# ▶ SLIDE 9 — Section divider: Random Forests

Part three. Decorrelating the trees.

---

# ▶ SLIDE 10 — The Problem Bagging Does Not Fully Solve

Back to the floor.

Why would trees end up correlated? Because of the greedy algorithm from last week. At the root,
CART picks the single best split. If one feature is clearly the strongest predictor, then
*every* tree picks it, and every tree starts the same way.

And in forecasting, one feature is often dominant. On our weekly data it's the four-week lag —
that was the root of last week's tree. On the daily data in Lab 5, it's the day of the week.

**[pause]**

The Random Forest fix, from Breiman in 2001, is almost absurdly simple. At each split, don't
consider all p features. Consider a random subset of **m**, and take the best split among
*those*. That's the whole method.

*(Point at the left panel.)*

Here is what that does, measured on Lab 5's data. The left panel is three hundred bagged trees,
every feature available at every split. For each tree I asked one question: what did it split on
first? Two hundred ninety-eight of the three hundred chose day of the week. The other two chose the
four-week lag. Three hundred trees, one opening move.

*(Point at the right panel.)*

The right panel is a random forest — three of the ten features offered at each split. Now day of
the week opens only seventy-seven trees. The four-week lag opens seventy-two, last week fifty-six,
and so on down. The structures genuinely differ. That's the decorrelation, made visible.

**[pause]**

Now the honest part — read the numbers in the corners.

The random forest's trees are individually *worse*: four hundred seventeen RMSE on average,
against three hundred ninety-eight for the bagged trees. That's the price — many of them weren't
allowed to look at day of the week first. And averaging does help them more: four seventeen down to
two fifty-four, against three ninety-eight down to two forty-eight.

But not enough. The bagged ensemble wins, two forty-eight to two fifty-four.

That is not a failure of the idea. It's the idea's boundary condition. You deliberately make each
tree worse in order to make the ensemble better — and that trade pays when you have many partly
redundant features and no single dominant one. With ten features and one of them carrying most of
the signal, the handicap costs more than the independence buys. Don't expect Homework 3 to rescue
the textbook rule: its forest has twenty-seven features, but last week's sales dominate there the
way day of the week dominates here, and the square-root setting loses there too.

---

# ▶ SLIDE 11 — Random Forest Hyperparameters

Five parameters, and the striking thing is how few need attention.

`n_estimators` — number of trees. More is better and plateaus around five hundred. There is no
overfitting risk from adding trees; that's the first term of the variance equation, which
doesn't care.

`max_features` — **this is the one that matters.** It's m from the previous slide. It is the
dial controlling rho, and therefore the dial controlling the floor. Lower means more
decorrelated trees.

And here is a trap. For a *regression* forest, scikit-learn's default is one point zero — every
feature at every split. That is plain bagging, not a random forest. The `'sqrt'` you'll see in
textbooks and tutorials is the *classifier's* default. So if you never set `max_features`, the
"random forest" you fit has no feature randomness at all. Set it yourself, and tune it.

`max_depth` — leave it at None. We want deep trees.

`min_samples_leaf` — raise it to about five for smoother predictions.

`oob_score` — set True if your dataset is large enough for it to mean anything.

**[pause]**

Start with five hundred trees and `min_samples_leaf` of five, and try a few values of
`max_features` — `sqrt`, and a larger fraction like zero point six — by walk-forward CV.

And note the practical consequence: essentially one parameter needs real tuning. Compare that
to what's coming with XGBoost, where you'll be juggling six. A large part of why random forests
are the right default is that they are very hard to misconfigure.

Set `random_state=42`. A forest is random twice over — in the bootstrap samples *and* in the
feature subsets — so without a seed your results move between runs and you'll waste an hour
wondering why.

---

# ▶ SLIDE 12 — Section divider: Feature Importance

Part four. Which predictors matter?

---

# ▶ SLIDE 13 — Random Forest Feature Importance

Mean decrease in impurity. For each feature, sum the variance reduction across every split
where it was used, weighted by how many observations passed through that node. Do that in every
tree, average across the trees, and normalize to one.

This is the forest from the next slide, on our weekly CA_1 food sales. The ranking comes out:
**four-week lag at zero point two eight**, last week at zero point two one, **SNAP benefit days at
zero point one seven**, and the same week last year at zero point one five — with price further
down.

Compare last week's single tree, which gave the four-week lag zero point six two. The forest
spreads the credit — partly because it averages many trees, and partly because each split was only
allowed to see two of the eight features, so other features got their turn.

**[pause]**

That SNAP result is worth pausing on. Nobody told the model that food sales spike when benefits
are disbursed. It found it, and ranked it above last year's sales and well above price. That is a real business finding, and no
linear specification would have surfaced it unless someone already suspected it.

**[pause]**

But read this as a screening device, not an explanation, because **MDI is biased**.

The bias is toward features with many possible split points. A continuous variable like price
offers hundreds of thresholds to try. A binary flag offers one. The continuous variable gets
more chances to look useful — not because it carries more signal, but because it had more
opportunities.

`permutation_importance` is the honest alternative: shuffle one column and measure how much
accuracy falls. If shuffling it doesn't hurt, it wasn't doing anything. That's a direct
measurement rather than an accounting artifact.

Use MDI to narrow forty candidates to eight. Use permutation importance before you put a
ranking in front of anyone who'll act on it.

---

# ▶ SLIDE 14 — Random Forest in Python

The code is four lines, and the results table is the interesting part.

On CA_1 food sales, fifty-two week test: seasonal naive at **1,660**. A single deep tree at
**1,456**. The random forest at **1,115** — that's the template on this slide, five hundred trees,
square-root features, at least five weeks per leaf.

**[pause]**

Two things.

The forest cuts RMSE about **twenty-three percent** against the single tree. That's the variance
reduction, bought by averaging — same algorithm, same features, five hundred of them instead of
one.

And both tree models **beat the seasonal naive benchmark on this series**. In Homework 1
exponential smoothing and ARIMA couldn't. But before you credit the trees, check what each one was
asked to do.

**[pause]**

The forest forecasts **one week ahead**. Every test week it is handed last week's actual sales,
and the three weeks before that. The ETS and ARIMA numbers from Lectures 1 and 2 forecast the
whole test year from the end of training, without seeing a single test-year actual. Those are
different jobs.

Give the classical methods the same one-week job and they beat sixteen sixty too: simple
exponential smoothing scores about thirteen hundred, an ARIMA about twelve sixty. And a forest
handed only the series' own lags — no price, no SNAP, no events — scores twelve thirty-one.

So of the five hundred forty-five units the forest gains over seasonal naive, about four hundred
thirty come from seeing last week's actual sales, before any price or SNAP column. The features
add the last hundred or so, and the algorithm on its own adds little: at the same horizon, on the
same history, the forest and the ARIMA are within a few percent of each other.

That's the real lesson of the second half of this course. Ask what information a model was given,
and for how far ahead, before you ask how clever it is.

---

# ▶ SLIDE 15 — Section divider: Gradient Boosting

Part five, and we change direction entirely.

---

# ▶ SLIDE 16 — Boosting vs. Bagging: The Core Difference

Bagging builds trees **in parallel** and independently. Each tries to predict y directly.
Averaging reduces variance. And critically — adding more trees never reduces bias. If your
trees are systematically wrong in the same direction, averaging preserves that exactly.

Boosting builds trees **sequentially**. Each new tree fits the *errors* of everything built so
far. That reduces bias as well as variance — and it can overfit, which bagging essentially
cannot.

**[pause]**

The intuition on the slide is the clearest version I know.

Forecast one says fifty thousand dollars. The actual is sixty thousand. The error is plus ten
thousand.

Now, forecast two does not try to predict sales at all. It tries to predict **that error**. It
says: "the first model was short by about ten thousand."

Add them together: fifty plus ten is sixty.

**[pause]**

Each stage corrects what remains. And notice this fixes something averaging never touches — if
every bagged tree under-predicts December by the same amount, the average under-predicts
December. Boosting sees that residual pattern and builds a tree specifically for it.

Bagging attacks variance. Boosting attacks bias. That sentence is the whole slide.

---

# ▶ SLIDE 17 — The Gradient Boosting Algorithm

Formally, three steps in a loop.

Start with the mean. Then repeatedly: compute the residuals, fit a shallow tree to those
residuals, and add a fraction of that tree to the running prediction.

**[pause]**

Two design choices worth understanding.

**The trees are deliberately shallow** — depth three to six. That's the opposite of bagging,
where we grew them deep. Here each tree only needs to capture a small piece of the remaining
error, so a small tree is enough. A deep tree would fit the residual noise, and the next
iteration would be trying to correct noise.

**The learning rate eta is the main dial.** After fitting a tree to the residuals, you don't add
the whole thing — you add eta times it, where eta might be zero point zero five.

Which means you deliberately under-correct at every step.

**[pause]**

Why on earth would you do that?

Because each tree is fitted to the residuals of one particular sample, so part of what it sees
is noise. Take the full correction and you bake that noise in. Take five percent of it, five
hundred times, and the noise averages out while the signal accumulates.

The reliable recipe is: small eta, many trees, and **early stopping**. Without early stopping,
boosting will keep reducing training error until it has fitted your noise perfectly. Unlike
bagging, this method genuinely can overfit — and it will if you let it.

---

# ▶ SLIDE 18 — Boosting, Tree by Tree

Here is that loop running, on data where you can see every step.

To draw a fit you need one feature, so I've borrowed Lecture 3's electricity data: weekday load in
the eastern US, against the day of the year. Every gray dot is one weekday between 2002 and 2018.
The shape is the one you met in Lecture 3, a W — high in winter, a dip in spring, the big summer peak
from air conditioning, a dip in autumn.

The dashed line is where boosting starts: the mean, F zero.

**[pause]**

*(Left panel.)*

After **one tree**. A depth-three tree fitted to the residuals from the mean, and we add
zero point three of it. The red curve has barely left the dashed line — a step up for summer, a
step down for spring and autumn. Training RMSE three point nine gigawatts.

*(Second panel.)*

After **two trees**. The second tree was fitted to what the first one left over — the residuals.
The summer step got taller, the dips got deeper. Three point five three.

*(Third panel.)*

After **five**. You can see the W now. Three point oh seven.

*(Fourth panel.)*

After **fifty**. The curve follows the data closely, and it has started to wiggle. RMSE two
point eight nine.

**[pause]**

Here is how to read that last number. The best *any* curve in day of the year could do on these
data — the average load on each calendar day — has an RMSE of two point eight six. Fifty trees
are within three hundredths of that. Everything left is variation the day of the year cannot
explain: weather, holidays, one year being hotter than another. Keep adding trees and boosting
starts fitting that noise — those wiggles near July are the beginning of it.

One honest note. I used a learning rate of zero point three, the top of the range on the
previous slide, so that four panels would show visible progress. At zero point zero five the
same picture takes about six times as many trees.

---

# ▶ SLIDE 19 — Section divider: XGBoost

The version that won everything.

---

# ▶ SLIDE 20 — XGBoost: Three Advances

Three improvements over vanilla gradient boosting.

**The Newton step.** Vanilla boosting uses the gradient — the direction of steepest descent.
XGBoost also uses the second derivative, the curvature, to size each leaf's value. Be precise about
when that matters. For squared error — what we use, and XGBoost's default for regression — the
curvature is the same everywhere, so the Newton step is just the ordinary fit to the residuals;
nothing changes. It earns its keep on losses that bend unevenly, like log-loss for classification
or a Poisson loss for counts.

**Explicit regularization.** This is the important one. XGBoost puts penalties directly in the
objective — gamma on the number of leaves, lambda on the magnitude of the leaf weights. So
overfitting is controlled by the loss function itself, not only by stopping early.

**Column subsampling.** Borrowed straight from random forests: sample features per tree and per
split. The decorrelation idea from slide ten, imported into boosting.

**[pause]**

Notice what that second advance actually is.

`reg_lambda` penalizes the sum of squared leaf weights. That is an **L2 penalty** — it is Ridge
regression, applied to leaf weights instead of regression coefficients. `reg_alpha` penalizes their
absolute values: that's the **L1 penalty**, LASSO. And `gamma` is different from both: it charges
a fixed price for every leaf, however small its weight. That's a count penalty — exactly Lecture
4's `ccp_alpha`, the price per leaf in cost-complexity pruning.

You are seeing next week's lecture, one week early, wearing different clothes.

That's not a coincidence. Regularization is not a linear-model topic. It is how *every* flexible
model is kept honest, and you'll meet it a fourth time in Lecture 7 as weight decay and dropout.

**[pause]**

XGBoost dominated Kaggle tabular competitions from 2014 to 2018. LightGBM is faster on large
data; CatBoost handles categorical features natively. The ideas are the same.

---

# ▶ SLIDE 21 — XGBoost: Key Hyperparameters

Six parameters, against the random forest's one that mattered. This is the cost of the extra
accuracy.

Number of trees, five hundred to two thousand — high, because you're pairing it with early
stopping. Learning rate zero point zero one to zero point one. Depth three to six, shallow.
Row subsample and column subsample around zero point eight. And `reg_lambda`, the Ridge penalty
on leaf weights.

**[pause]**

The two that interact most are `learning_rate` and `n_estimators`, and they trade off directly.
Halve the learning rate and you need roughly twice the trees. Small eta with many trees is
almost always better than large eta with few — you're just paying for it in compute.

Practical starting point: learning rate zero point zero five, depth four, a thousand trees with
early stopping on validation RMSE. Then tune depth and subsample by `TimeSeriesSplit`.

---

# ▶ SLIDE 22 — XGBoost in Python

The code, and one argument that matters more than it looks.

`eval_set` is what makes `early_stopping_rounds` work. XGBoost watches RMSE on that set and
halts when it stops improving. That's how you get away with setting a thousand trees — you're
not committing to a thousand, you're setting a ceiling and letting the data decide.

**[pause]**

And the validation set has to respect time. Build it with `TimeSeriesSplit`, hold out the final
stretch as a true test set, and never let a random split choose which weeks the model sees.

If you use a random split here, early stopping will pick the iteration that best predicts weeks
the model has effectively already seen. You'll stop too late, ship an overfitted model, and your
diagnostics will look excellent.

**[pause]**

One practical note that will save you an evening. On macOS, `xgboost` needs the OpenMP runtime.
If you get an error about `libomp.dylib` not loading, run `brew install libomp`. The Python
package installs fine and then fails at import, which makes it look like a code problem when
it's a system library problem.

---

# ▶ SLIDE 23 — Section divider: Key Takeaways

Let's pull it together.

---

# ▶ SLIDE 24 — Tree Methods: Which One When

Three rows.

Decision tree — single tree, highly interpretable, minimal tuning, baseline accuracy.
Random forest — parallel, medium interpretability, low tuning, good accuracy.
XGBoost — sequential, low interpretability, higher tuning, best accuracy.

**[pause]**

My advice is that **random forest is the better default**, and I want to be clear about why,
because "XGBoost is best" is what the accuracy column says.

A random forest is robust, needs almost no tuning, and hands you OOB error for free. It is
genuinely hard to make one catastrophically wrong.

Move to XGBoost when you can afford to tune and you need the last few points. And budget for
that tuning honestly — an untuned XGBoost frequently loses to a default random forest. The
accuracy in that column is the accuracy of a *well-tuned* XGBoost, and that word is carrying
weight.

---

# ▶ SLIDE 25 — Why No Tree Can Extrapolate

One limit applies to every row of that table, and it deserves its own picture.

A forest's prediction is a weighted average of the training targets — every leaf is an average of
training values, and the forest averages the leaves. A weighted average can never leave the range
of the things it averages. So a forest can never predict above the highest value it was trained on,
or below the lowest. Not "rarely does." Cannot.

*(Point at the chart.)*

This is US retail sales, monthly. I trained a forest on 1993 through 2019 — the shaded part — and
then asked it for every month since, one month ahead, *giving it the true value of last month and
the month before*. The most generous setup there is.

The highest month in its training data was about four hundred fifty billion dollars: the dashed
line. Sales went through it in 2020 and kept going, to six hundred sixty-five billion by mid-2026.
Ninety-four percent of the months since have been above the line. The forest's forecast — the red
line — never crosses it.

**[pause]**

Boosted trees fail the same way for a slightly different reason: each tree is flat beyond its last
split, so the sum is flat too.

So the classical methods did not become obsolete today. If your series trends and you need to
forecast beyond its historical range, an ARIMA with drift will do something a forest simply
cannot. Or model the *change* instead of the level, so the target stays inside its range.

---

# ▶ SLIDE 26 — Lecture 5: Key Takeaways

Eight things, one line each.

**One.** Ensembles work because independent errors cancel. The word independent is load-bearing.

**Two.** Bagging averages B trees on bootstrap samples. Variance falls as B grows but floors at
rho sigma squared.

**Three.** Random forests cut rho by sampling features at each split — a trade that pays when no
single feature dominates, and didn't in Lab 5.

**Four.** OOB error is free, but on a time series it scores filling gaps, not forecasting.

**Five.** Boosting inverts the idea. Trees built sequentially, each fitting what remains.
Bagging attacks variance; boosting attacks bias.

**Six.** No tree can extrapolate beyond its training range.

**Seven.** Tune by cross-validation, always with `TimeSeriesSplit`, and pair boosting with early
stopping.

**Eight.** The forest beat naive on its inputs, not its algorithm. The jump from the seasonal
naive benchmark to the random forest was large — sixteen sixty down to eleven fifteen — and most of
it came from being handed last week's actual sales, one week ahead. Remember that when you're
tempted to reach for a more complicated model instead of a better predictor.

---

# ▶ SLIDE 27 — References

Reading is ISLP section 8.2 — bagging, random forests, and boosting — plus the XGBoost
"Introduction to Boosted Trees" tutorial, which derives the regularized objective we waved at
on slide 20.

Next week: Regularization and Model Selection. We take the penalty idea that showed up inside
XGBoost and give it a lecture of its own.

See you then.

---

## Timing guide

| Segment | Slides | Target |
|---|---|---:|
| Opening & wisdom of crowds | 1–4 | ~4 min |
| Bagging & OOB | 5–8 | ~7 min |
| Random forests | 9–11 | ~6 min |
| Feature importance & Python | 12–14 | ~6 min |
| Boosting & XGBoost | 15–22 | ~10 min |
| Close | 23–27 | ~5 min |

This is the longest deck in the course. If you must compress, slides 11 and 21 — the two
hyperparameter tables — survive trimming best. Do **not** compress slides 7, 10, or 16: the
variance floor, the decorrelation fix, and the bagging-versus-boosting distinction are what the
whole lecture is built on.
