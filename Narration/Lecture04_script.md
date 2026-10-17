# Lecture 4 — Recording Script

**ECON 8310: Business Forecasting · Decision Trees**

Deck: `Slides/Lecture04_DecisionTrees.pdf` (21 pages) · Measured runtime: **~19 minutes** (17–21 depending on pace)

---

## How to use this document

- **`▶ SLIDE n — Title`** marks where to advance. The number is the PDF page.
- Written to be **spoken**. *Italic parentheticals* are stage directions.
- **[pause]** means actually stop for a beat.
- Change anything that doesn't sound like you.

---

# ▶ SLIDE 1 — Title page

Lecture 4: Decision Trees.

Today the course changes character. Everything in the first three weeks came out of statistics
— exponential smoothing, ARIMA, GAMs. We specified a structure, estimated its parameters, and
read the coefficients.

From here on we're doing machine learning. And the mental shift is this: instead of specifying
a functional form and estimating it, we hand an algorithm a pile of features and let it
discover the structure itself.

**[pause]**

You gain something real. Trees capture interactions and threshold effects automatically,
without you having to know they're there.

You give up something real too. Last week I told you a GAM is additive by construction and
therefore explainable, and that the price of that was no interactions. Today we buy the
interactions back — and start paying in interpretability.

---

# ▶ SLIDE 2 — Lecture Outline

Four parts. Why we go beyond linear models. How a tree actually works. Implementing one in
Python. And where a single tree falls short — which sets up the next two weeks.

---

# ▶ SLIDE 3 — Section divider: Why Go Beyond Linear Models?

*(Divider.)*

Part one. What linear models can't represent.

---

# ▶ SLIDE 4 — The Limits of Linear Models

Three things a constant slope cannot do.

**Interactions.** The effect of advertising during the holiday season isn't the same as its
effect in March. A linear model can only assign advertising one slope. You can add an
interaction term by hand — but only if you already suspected it was there.

**Threshold effects.** Spending might be flat while unemployment sits above five percent, then
move sharply once it drops below. That's a kink. A straight line has no kinks.

**Asymmetry.** Extreme values pull coefficients equally hard in both directions, whether or
not the underlying relationship is symmetric.

**[pause]**

The retail example on the slide ties them together. Monthly sales depend on last year's
same-month sales, on whether it's December, and on the unemployment rate. And the December
effect is *larger* in a strong economy — that's an interaction between seasonality and the
macro environment.

A linear model can represent each of those three variables. It cannot represent the fact that
they modify each other.

**Decision trees capture nonlinearity and interactions automatically.** That's the promise.
The rest of the lecture is about the mechanism and the cost.

---

# ▶ SLIDE 5 — The Bias–Variance Tradeoff

Before the tree itself, the framework we'll use to judge every model from here to December.

Expected mean squared error decomposes into three pieces: bias squared, variance, and
irreducible noise.

**Bias** is systematic error — the model is too simple to represent the truth. A straight line
through a curved relationship has bias, and collecting more data won't fix it.

**Variance** is sensitivity to the particular sample you happened to draw. A model with high
variance would look quite different if you'd collected a different year of data.

**Sigma squared** is irreducible noise. And I want to be precise here, because this gets
misstated: sigma squared is a property of the data-generating process, not of your sample. It
is not "the error you'd remove with better data." More observations shrink variance. Nothing
shrinks sigma squared.

**[pause]**

Now let me show you the two failure modes, on our own data, before we've even defined a tree.

*(Point at the figure.)*

One test week: the week of the fourteenth of December, twenty fifteen. The store actually sold
eighteen thousand two hundred seventy-nine units — that's the dashed line.

I took the hundred seventy-three training weeks, drew a bootstrap resample — same size, with
replacement — and fitted a model. Then again, twenty times. Each dot is what one of those twenty
models forecast for this one week.

**The red row is twenty stumps** — trees that ask a single question. They agree with each other:
the spread is about three hundred forty units. But they agree on the wrong answer. Their average,
the black diamond, misses by about twelve hundred eighty. Too simple to see what's special about
this week. That's **high bias**: bad even on average, and — this is the diagnostic — bad on the
training data too.

**The blue row is twenty fully grown trees**, grown until every training week has its own leaf.
Their average lands almost exactly on the truth — off by twenty-eight units. But look at the
scatter: from fifteen thousand to twenty-one thousand. A spread of about seventeen hundred
fifty. Whichever resample you happened to draw decides your forecast. That's **high variance**:
great on the data it memorized, unreliable on anything new.

**[pause]**

One honest caveat, and it's why the figure says "miss of the average" and not "bias." The
actual eighteen thousand two hundred seventy-nine includes this week's own noise — the sigma
squared term. So the miss is bias *plus* one draw of noise. I picked this week because it shows
the pattern cleanly; across all fifty-two test weeks the ordering is the same — the stumps' average
misses by more, about fourteen hundred forty against eleven hundred, and the deep trees spread
more, about twelve hundred against seven hundred sixty.

**[pause]**

Concretely, from work you've already done. In Homework 1 the undamped Holt model had **low
bias** — it fit the training data's upward trend faithfully — and **high variance** in the
sense that matters here: it was exquisitely sensitive to the trend estimated from that
particular window, and when the trend flattened it was catastrophically wrong. The seasonal
naive benchmark had more bias and far less sensitivity, and it won.

That's the tradeoff, and it wasn't a story about trees. It's the frame for everything.

The goal is the complexity that balances them, and you can only find it out-of-sample. Which
is why Lecture 1 spent so long on that rule.

---

# ▶ SLIDE 6 — Section divider: How Decision Trees Work

Part two. The mechanism — partition the feature space, predict the mean in each region.

---

# ▶ SLIDE 7 — What Is a Decision Tree?

Here's the whole idea in one picture — and it's a real one. This is the tree a computer actually
grows on our weekly CA_1 food sales, stopped at two levels so we can read it.

A tree asks a series of yes/no questions about the features and arrives at a prediction.

*(Walk the diagram.)*

Start at the top. Did the store sell twenty thousand six hundred and three units or fewer four
weeks ago? If yes, go left, and there's a second question about the same lag: eighteen thousand
six hundred or fewer? Yes, and the forecast is about eighteen thousand four hundred. No, and it's
about twenty thousand one hundred.

If the first answer was no — four weeks ago was a strong week — go right, and now the tree asks a
different question: did the same week last year sell more than twenty-one thousand seven
hundred? If not, forecast about twenty-one thousand four hundred. If so, about twenty-three
thousand.

**[pause]**

Each green box is just an average. The "twenty-seven weeks" under the first one means
twenty-seven training weeks landed there, and the forecast is their mean. Every prediction is
the answer at the bottom of one path.

Notice the right branch asks about last year and the left branch doesn't. Hold onto that; the
next slide draws it.

**[pause]**

Four properties worth naming.

It's **non-parametric** — no functional form is assumed anywhere. It's **nonlinear**, and
notice it captures a threshold effect natively, which is what a linear model couldn't do.
It's **interpretable**, because you can trace the exact path. And it's **scale-invariant** —
no standardization needed, because the tree only ever asks "is this value above or below a
threshold," and that question doesn't care about units.

That last one will matter in Lecture 6, where standardization becomes mandatory.

And the warning in the last line is the whole reason Lectures 5 and 6 exist: a single deep tree
overfits severely.

---

# ▶ SLIDE 8 — The Same Tree, Drawn as Rectangles

Same tree, different picture. And this picture is what a tree really is.

*(Point at the axes.)*

The tree used two features, so we can draw every training week as a point in a plane. Across the
bottom, what the store sold four weeks earlier. Up the side, what it sold the same week last
year. A hundred seventy-three white dots, one per training week.

*(Trace the cuts.)*

Each question in the flowchart is a straight cut. The root — four weeks ago at or below twenty
thousand six hundred and three — is the solid vertical line. The left branch's second question,
eighteen thousand five hundred ninety-eight, is the dashed vertical line on the left. And the
right branch's question about last year is the dashed horizontal line — but notice it only runs
across the right half.

**[pause]**

So four rectangles, and they're the four green boxes from the flowchart. The darker the shading,
the higher the forecast. Every week that falls inside a rectangle gets that rectangle's average:
eighteen thousand four hundred, twenty thousand one hundred, twenty-one thousand four hundred,
twenty-three thousand.

Two things to see.

First, the cuts are always parallel to an axis. A tree never draws a diagonal line. It can only
ask "is this one feature above or below a number?" — so its whole world is rectangles, and its
forecast is flat inside each one.

Second, the horizontal cut stops at the vertical one. Last year's sales split the right half and
leave the left half alone. That's an **interaction**: last year matters, but only when recent
demand is already high. Nobody told the tree to look for that. A linear model would need you to
write the interaction term yourself; the tree found it by cutting.

---

# ▶ SLIDE 9 — The CART Algorithm (Regression)

How does the tree decide what to ask?

CART builds greedily, top to bottom. At each node it searches over **every feature and every
possible threshold**, and keeps the split with the smallest total squared error — the squared
deviations from the mean on the left, plus the same on the right.

Hold onto the word **greedily** — the next slide is entirely about what it costs.

**[pause]**

The picture on the right is that search, done on our data for the very first split.

*(Point at the top panel.)*

Every dot is one of the hundred seventy-three training weeks. Across the bottom, what the store
sold four weeks earlier; up the side, what it sold that week. The dashed line is a cut. Weeks to
its left are red, weeks to its right are blue, and each side gets one prediction: its mean, the
thick horizontal line. The thin vertical sticks are the errors — how far each week sits from its
side's mean.

The algorithm squares every stick and adds them up. That total is the number it is trying to
make small.

*(Point at the bottom panel.)*

Now slide the cut. The bottom panel is that total, for every place the cut could go. With no
split at all the total is about six hundred sixty million — that's the dotted line. Wherever you
put the cut, the total drops. It drops furthest with the cut at twenty thousand six hundred and
three: four hundred thirty-one million.

**[pause]**

Six hundred sixty-one million down to four hundred thirty-one. **One yes/no question removed
thirty-five percent of the squared error.** Eighty-nine weeks on the left, averaging about
nineteen thousand six hundred; eighty-four on the right, averaging about twenty-one thousand
nine hundred.

The algorithm did this for every feature, not just this one, and the four-week lag won. That's
the root of the tree two slides back.

Then it repeats the whole search inside each group, and again inside those — until the groups get
too small or you hit the depth cap. The prediction at a leaf is just the mean of the training
weeks that landed there.

For classification, replace squared error with Gini impurity or entropy. Same algorithm,
different measure of how mixed a region is.

---

# ▶ SLIDE 10 — Greedy Is Not Optimal — and That Matters

I want to slow down on "greedy," because it is the single most important word on the previous
slide and it is easy to hear it as a criticism when it is really a description.

CART is **exhaustive at each node**. It really does try every feature and every cut point — no
shortcuts, no sampling. So it is not lazy.

What it never does is **look ahead**. It evaluates one split, commits to it, and moves down.

**[pause]**

The table is a made-up illustration, to show what that costs. Suppose splitting on price removes
forty percent of the
variance right now. Splitting on promotion removes thirty-eight — slightly worse. But *because*
of how promotion divides the data, it sets up two further splits that would together remove
another fifty percent.

Forty-plus-twelve, versus thirty-eight-plus-fifty. Fifty-two against eighty-eight.

CART takes the forty. Not because it weighed the options and chose badly — because it has no
mechanism to see the fifty coming at all.

**[pause]**

So the tree you fit is **not the optimal tree**. It is the result of a sequence of locally
optimal decisions, and those are genuinely different objects.

The obvious question is why we tolerate that. The answer is that finding the truly optimal tree
is computationally intractable for any realistic number of features — the search space explodes
combinatorially. So the greedy approximation isn't a shortcut somebody settled for; it's what
makes trees usable at all.

Two consequences to carry forward, and both matter for the next three weeks.

First, **a tree can miss structure that is really there** — not from lack of data, but from the
order in which it asked its questions.

Second, and this is the one that pays off next week: a small change in the data can send the
whole greedy sequence down a different path. That is the instability we turn to right now.

---

# ▶ SLIDE 11 — Depth Controls the Bias–Variance Tradeoff

Now connect the tree back to slide 5.

A **depth-one stump** asks one question. High bias, low variance — the red row on slide 5.

A **full tree** keeps splitting until every training week has its own leaf. It reproduces the
training data exactly: zero *training* error. That is not zero bias — it's memorization — and
the variance is high. The blue row on slide 5.

*(Point at the figure.)*

Here's the whole range on our weekly data. Across the bottom, the depth cap, from one up to
twelve, and then a fully grown tree with a hundred seventy-three leaves. The gray line is the
error on the training weeks; the red line is the error on the test year.

The gray line does what it must: it falls with every extra level and hits **zero** for the full
tree. A deeper tree can always fit the sample better.

The red line doesn't follow it. It drops from the stump to depth three — about twelve hundred
thirteen — and then stops improving. Past depth three it wanders between about thirteen hundred
fifty and fifteen hundred sixty. The gap between the two lines is overfitting, drawn to scale.

**[pause]**

Why does it stop at three? Because we only have a hundred seventy-three weeks. Every split halves
the data, and after three levels a leaf holds about twenty weeks. Split further and you're
fitting the noise in a handful of weeks. With more data, deeper trees pay off — you'll see that in
the lab, where the daily data has over a thousand rows.

And one warning about how to read this. Do not pick depth three *from this picture*. That red line
is the test year, and choosing a setting by looking at the test year is exactly the leak we keep
forbidding. Read the shape — falling training error, flat test error — not the minimum. The depth
itself gets chosen by `TimeSeriesSplit` on the training weeks.

**[pause]**

So: here is why trees are unstable, which matters enormously next week.

Change one training observation. If that changes the split at the root — and it easily can,
because the algorithm is picking the single best threshold — then *every prediction below it*
changes. Not slightly. The whole structure downstream is different.

That's not a flaw in the implementation. It's inherent to a greedy hierarchical partition. And
it is precisely the property that averaging will exploit in Lecture 5.

Tune with `max_depth` and `min_samples_leaf`, selected by `TimeSeriesSplit` cross-validation.

---

# ▶ SLIDE 12 — Two Ways to Stop a Tree Overfitting

`max_depth` is the blunt instrument — it caps the tree in advance, before you know how deep it
needed to go. Two other controls do the job better, and they usually get one line each in a
table, which is not enough.

**`min_samples_leaf`** is the one I'd reach for first. Leave it at the default of one and the
tree may build a leaf holding a single observation, whose "prediction" is that one observation's
value. That is memorization with extra steps. Set it to ten and every prediction is an average
over at least ten weeks — smoother, and far more honest about what the data supports.

**`ccp_alpha`** works the other way around. Grow the tree out fully, *then* charge a penalty per
leaf and cut back the branches that don't pay for themselves. Larger alpha, smaller tree.

Whichever you use, choose its value by cross-validation — which for us means `TimeSeriesSplit` —
and never by looking at the test year.

---

# ▶ SLIDE 13 — Cost-Complexity Pruning: A Line Touching the Curve

Pruning has a picture, and it's worth two minutes, because the same picture comes back twice in
this course.

Cost-complexity pruning minimizes the fit — the sum of squared errors — plus a price alpha for
every leaf.

*(Point at the black curve.)*

Each black dot is a pruned version of one fully grown tree on our weekly data. Across the bottom,
how many leaves it keeps; up the side, its training error. One leaf — no splits at all — is the
dot at the top left, six hundred sixty million. Every leaf you add lowers the error, so the curve
runs down to the right, flattening as it goes. The full tree, off the right edge, has a hundred
seventy-three leaves and zero training error: one leaf per week, pure memorization.

**[pause]**

Now the geometry. Take a straight line with slope minus alpha and push it up from below until it
touches the curve. The first dot it touches is the tree that alpha picks.

*(Point at the red line.)*

At thirty-four million per leaf — the red line — it touches at four leaves. That four-leaf tree is
exactly the flowchart from slide 7.

*(Point at the blue line.)*

Cheapen leaves to fifteen million — the blue line, flatter — and it touches further right, at
nine leaves. A bigger alpha means a steeper line, which means a smaller tree. That's all alpha
does.

**[pause]**

The hollow circles are trees capped by `max_depth` at three and at four. They sit *above* the
curve: for the same number of leaves, pruning found a lower training error. A depth cap cuts every
branch at the same level whether it was earning its keep or not.

**[pause]**

You have seen this before. That is the GAM smoothing penalty from Lecture 3 with a different
symbol on it — fit against complexity, one parameter tuning the trade. And you'll meet it a third
time in Lecture 6, as Ridge and LASSO, where the curve becomes a diamond and a disc.

Three model families that look nothing alike, one idea underneath. I'd rather you remember the
idea than the three implementations, because the idea is what transfers.

One technical footnote on the slide: scikit-learn measures `ccp_alpha` per observation, so its
number is this alpha divided by the number of training rows.

---

# ▶ SLIDE 14 — Section divider: Implementing Decision Trees in Python

Part three. One class, a few hyperparameters.

---

# ▶ SLIDE 15 — Decision Tree Regression in Python

The code is short.

Import `DecisionTreeRegressor`. Take the weekly features — the lags, SNAP days, events and price.
Instantiate with `max_depth=5`,
`min_samples_leaf=10`, and `random_state=42`. Fit. Predict. That template is exactly what produced
the importance table on the next slide.

The table gives you the four hyperparameters worth knowing. Two of them — `min_samples_leaf`
and `ccp_alpha` — we just spent a slide on. `max_depth` caps the depth. And `max_features` does
feature subsampling: remember that one, because it becomes the central idea next week.

**[pause]**

And the line at the bottom, which you've now heard in every lecture: cross-validate with
`TimeSeriesSplit`, never `KFold`. Random folds leak the future into the training set.

I'll keep saying it because it's the error that survives longest. It doesn't produce a crash
or a warning — it produces a number that looks better than the truth.

---

# ▶ SLIDE 16 — Feature Importance

A tree will tell you which variables it found useful.

Impurity-based importance sums, for each feature, the total variance reduction across every
split where that feature was used, weighted by how many observations passed through. Then
normalize so it sums to one.

On our weekly Walmart data a depth-five tree ranks them like this: the **four-week lag**
dominates at about zero point six two, then the **same week last year** at zero point one five,
then **last week** at zero point one three, and then **SNAP benefit days** at zero point zero
five and **price** at zero point zero four.

Notice how concentrated that is — one feature takes almost two-thirds of the total. That is
characteristic of a single tree, and it is a symptom of the greediness we discussed: the root
split commits to one feature and everything below inherits that choice. Next week you'll see
the same table from a random forest, and the mass is spread much more evenly.

Read that as a screening device. It says which variables the tree leaned on — a fast way to
narrow forty candidate predictors down to eight worth thinking about.

And notice SNAP appearing at all, because it's the kind of thing this method is good for.
Nobody specified that food sales respond to benefit disbursement days. The tree found it, and
ranked it level with price. That's a real business finding no linear specification would have
surfaced unless someone already thought to look.

**[pause]**

But be careful, because impurity importance is **biased toward high-cardinality features**.

Look at our own table. `avg_price` is continuous — it offers hundreds of distinct split points,
so the algorithm gets hundreds of chances to find one that happens to help. `snap_days` takes a
handful of integer values, and `event_days` fewer still. Price gets more opportunities to look
useful purely because it has more thresholds to try, not because it carries more signal.

Which means the gap between SNAP at zero point zero five and price at zero point zero four is not
something I'd read anything into at all — and if anything, the bias runs in price's favor.

Permutation importance is the honest alternative: shuffle one column, and measure how much
accuracy falls. Slower, and worth it whenever the ranking will inform a decision. We'll use it
properly next week.

---

# ▶ SLIDE 17 — Interpreting and Communicating Tree Predictions

This is the tree's real advantage, and it's worth dwelling on because you lose it steadily from
here.

You can explain **why** a specific prediction came out the way it did.

*(Read the manager quote from the slide.)*

Notice what that explanation is: it's just the right-hand path of the flowchart from slide 7,
read aloud. Four weeks ago the store sold more than twenty thousand six hundred units. The same
week last year sold more than twenty-one thousand seven hundred. The twenty-five training weeks
where both held averaged about twenty-three thousand units.

No coefficients. No "holding all else constant." A sequence of conditions and an average.

**[pause]**

`plot_tree` or `export_text` will produce this for you.

And here's why it matters commercially. A manager who can see the splits can tell you when the
model is wrong — "that threshold is picking up the year we had the fire, not a real pattern."
You cannot get that feedback from a model nobody can read. It's the cheapest source of domain
knowledge you have, and every method after this one makes it harder to access.

---

# ▶ SLIDE 18 — Section divider: Limitations and Preview

Part four. Where a single tree falls short.

---

# ▶ SLIDE 19 — When a Single Tree Falls Short

Strengths on the left, and they're real: interpretable, no scaling needed, captures
nonlinearity, handles missing values, fast to train.

Weaknesses on the right, and the first one is fatal.

**High variance.** We covered the mechanism on slide 10 — small data changes give a very
different tree.

**Flat predictions within each leaf.** Every observation in a leaf gets the identical forecast.
The prediction surface is a step function.

**Cannot extrapolate beyond the training range.** This one deserves a beat. A tree predicts the
mean of a leaf, and no leaf can contain values it never saw. So on a trending series, a tree
will flatten out exactly when you need it to keep climbing. If your sales have grown every year
and you ask a tree to forecast next year, it will predict something inside the range it already
knows.

Put a number on it. CA_1 food sales grew about twenty-nine percent over five years. Train a
tree on the first four and ask it for year five, and the highest value it can possibly output
is the highest leaf mean it learned — a number from the *old*, lower range. It will
systematically under-forecast, and no amount of tuning fixes it, because the model has no
representation of "up."

An ARIMA with a drift term has no such problem. It will happily forecast values it has never
seen.

That limitation never goes away. Random forests have it. XGBoost has it. It's a property of
trees, and it's why the classical methods from weeks one and two don't simply become obsolete
today.

**[pause]**

And the ensemble solution, which is next week: average many trees trained on different
bootstrap samples. Variance falls sharply while the feature-importance interpretability
survives — typically twenty to forty percent better RMSE than a single tree.

---

# ▶ SLIDE 20 — Lecture 4: Key Takeaways

Six things.

**One.** Linear models cannot capture interactions or threshold effects. Trees can, without
your specifying them in advance.

**Two.** Depth sets the bias–variance trade: too shallow underfits, too deep overfits.

**Three.** CART takes the split with the least squared error — slide the cut, total the squared
sticks. The prediction is the mean of the leaf.

**Four.** Stop overfitting with `min_samples_leaf` or `ccp_alpha` — a line of slope minus alpha
touching the curve — tuned by `TimeSeriesSplit`.

**Five.** Trees are interpretable — you can trace the exact path to any prediction.

**Six.** Single trees have high variance. Ensembles solve this, and that's Lecture 5.

**[pause]**

If you take one thing: **a tree is a step function that cannot extrapolate.** Everything good
and everything bad about today follows from that sentence.

---

# ▶ SLIDE 21 — References

Reading is ISLP section 8.1 — the basics of decision trees. That's the chapter we'll live in
for the next two weeks as well.

Next time: Tree Ensembles. We take this unstable, high-variance model and turn its instability
into an advantage.

See you then.

---

## Timing guide

| Segment | Slides | Target |
|---|---|---:|
| Opening & why beyond linear | 1–5 | ~5 min |
| How trees work | 6–13 | ~12 min |
| Python & interpretation | 14–17 | ~5 min |
| Limitations & close | 18–21 | ~2 min |
| **Total** | | **~24 min** |

If you're compressing, slide 15's hyperparameter table survives trimming — slides 12 and 13 have
already covered the two parameters that matter. Do **not** compress slides 10, 11, or 19: the greedy
approximation, the instability it produces, and the extrapolation limit are the three things
Lectures 5 and 6 are built on.
