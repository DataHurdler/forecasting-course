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

This is Lecture 4, on decision trees.

Today the course changes character. Everything in the first three weeks came out of statistics:
exponential smoothing, ARIMA, and generalized additive models. We specified a structure, estimated
its parameters, and read the coefficients.

From here on we're doing machine learning, and the mental shift is this. Instead of specifying a
functional form and estimating it, we hand an algorithm a pile of features and let it discover the
structure itself.

**[pause]**

You gain something real. Trees capture interactions and threshold effects automatically, without
your having to know they're there.

You give up something real, too. Last week, in Lecture 3 on generalized additive models, I told
you that a GAM is additive by construction and therefore explainable, and that the price of that
was no interactions. Today we buy the interactions back, and we start paying for them in
interpretability.

---

# ▶ SLIDE 2 — Lecture Outline

We'll take this in four parts. First, why we go beyond linear models at all. Second, how a tree
actually works. Third, how to implement one in Python. And last, where a single tree falls short,
which sets up the next two weeks.

---

# ▶ SLIDE 3 — Section divider: Why Go Beyond Linear Models?

The first part is about what a linear model can't represent. Relationships in real data are often
nonlinear, they involve interactions, and they shift with context.

---

# ▶ SLIDE 4 — The Limits of Linear Models

There are three things a constant slope cannot do.

The first is interactions. The effect of advertising during the holiday season isn't the same as
its effect in March, but a linear model can only give advertising one slope. You can add an
interaction term by hand, but only if you already suspected it was there.

The second is threshold effects. Spending might stay flat while unemployment sits above five
percent, and then move sharply once it drops below. That's a kink, and a straight line has no
kinks.

The third is asymmetry. A constant slope means that a rise and a fall of the same size move the
forecast by the same amount, whether or not the underlying relationship is symmetric. Customers may
punish a price rise harder than they reward a price cut, and a straight line has no way to say so.

**[pause]**

The retail example on the slide ties all three together. Monthly sales depend on last year's sales
in the same month, which is the trend, on whether it's December, which is the seasonality, and on
the unemployment rate, which stands in for economic conditions. And the December effect is
*larger* in a strong economy. That's an interaction between the season and the economy.

A linear model can represent each of those three variables. What it can't represent is the fact
that they modify each other.

Decision trees capture nonlinearity and interactions **automatically**, without your having to
specify them in advance. That's the promise. The rest of the lecture is about the mechanism and the
cost.

---

# ▶ SLIDE 5 — The Bias–Variance Tradeoff

Before we get to the tree itself, here's the framework we'll use to judge every model from now
until December.

Expected mean squared error breaks into three pieces: bias squared, variance, and irreducible
noise.

Bias is systematic error. It means the model is too simple to represent the truth. A straight line
through a curved relationship has bias, and collecting more data won't fix it.

Variance is sensitivity to the particular sample you happened to draw. A model with high variance
would look quite different if you'd collected a different year of data.

Sigma squared is the irreducible noise, and I want to be precise here, because this one gets
misstated. Sigma squared is a property of the process that generates the data, not of your sample.
It is not "the error you'd remove with better data." More observations shrink the variance.
Nothing shrinks sigma squared.

**[pause]**

Now let me show you the two ways a model fails, on our own data, before we've even defined a tree.

*(Point at the figure.)*

The data is weekly CA_1 FOODS sales, and we're looking at one test week, the week of the
fourteenth of December, twenty fifteen. The store actually sold eighteen thousand two hundred
seventy-nine units that week, and that's the dashed line.

I took the hundred seventy-three training weeks, drew a bootstrap resample, which means the same
number of weeks drawn with replacement, and fitted a model. Then I did it again, twenty times in
all. Each dot is what one of those twenty models forecast for this one week.

The red row is twenty stumps, which are trees that ask a single question. They agree with each
other, and the spread is only about three hundred forty units. But they agree on the wrong answer.
Their average, the black diamond, misses by about twelve hundred eighty. They're too simple to see
what's special about this week. That's **high bias**. The model is bad even on average, and here's
the diagnostic: it's bad on the training data, too.

The blue row is twenty fully grown trees, grown until every training week has its own leaf. Their
average lands almost exactly on the truth, off by just twenty-eight units. But look at the scatter.
It runs from fifteen thousand to twenty-one thousand, a spread of about seventeen hundred fifty.
Whichever resample you happened to draw decides your forecast. That's **high variance**: the model
is great on the data it memorized, and unreliable on anything new.

**[pause]**

There's one honest caveat, and it's why the figure says "miss of the average" and not "bias." The
actual eighteen thousand two hundred seventy-nine includes this week's own noise, the sigma squared
term. So the miss is bias *plus* one draw of noise. I picked this week because it shows the pattern
cleanly, but the ordering holds across all fifty-two test weeks. The stumps' average misses by
more, about fourteen hundred forty against eleven hundred, and the deep trees spread more, about
twelve hundred against seven hundred sixty.

Here's a concrete case from work you've already done. In Homework 1, the undamped Holt model had
**low bias**, because it fit the upward trend in the training data faithfully. And it had high
variance in the sense that matters here. It was exquisitely sensitive to the trend estimated from
that particular window, and when the trend flattened, it was catastrophically wrong. The seasonal
naive benchmark had more bias and far less sensitivity, and it won.

So that's the tradeoff, and it was never a story about trees. It's the frame for everything.

The goal is the complexity that balances the two, and you can only find it out of sample. That's
why Lecture 1, on forecast evaluation, spent so long on that rule.

---

# ▶ SLIDE 6 — Section divider: How Decision Trees Work

The second part is the mechanism. A tree partitions the feature space into rectangles, and it
predicts the mean in each one.

---

# ▶ SLIDE 7 — What Is a Decision Tree?

Here's the whole idea in one picture, and it's a real one. This is the tree a computer actually
grows on our weekly CA_1 food sales, stopped at two levels so that we can read it. It was fitted on
the hundred seventy-three weeks before the test year.

A tree asks a series of yes-or-no questions about the features and arrives at a prediction.

*(Walk through the diagram.)*

Start at the top. Did the store sell twenty thousand six hundred and three units or fewer four
weeks ago? If yes, go left, and there's a second question about the same lag: was it eighteen
thousand five hundred ninety-eight or fewer? If yes, the forecast is about eighteen thousand four
hundred. If no, it's about twenty thousand one hundred.

If the first answer was no, meaning that four weeks ago was a strong week, go right, and now the
tree asks a different question. Did the same week last year sell more than twenty-one thousand six
hundred ninety-two units? If not, the forecast is about twenty-one thousand four hundred. If so,
it's about twenty-three thousand.

**[pause]**

Each green box is just an average. The "twenty-seven weeks" under the first one means that
twenty-seven training weeks landed there, and the forecast is their mean. Every prediction is the
answer at the bottom of one path.

Notice that the right branch asks about last year and the left branch doesn't. Hold on to that,
because the next slide draws it.

**[pause]**

There are four properties worth naming.

A tree is **non-parametric**, which means no functional form is assumed anywhere. It's nonlinear,
and notice that it captures a threshold effect natively, which is exactly what a linear model
couldn't do. It's interpretable, because you can trace the exact path. And it's scale-invariant,
so no standardization is needed. The tree only ever asks whether a value is above or below a
threshold, and that question doesn't care about units.

That last one will matter in Lecture 6, on regularization, where standardization becomes
mandatory.

And the warning in the last line is the whole reason the next lecture exists: a single deep tree
overfits severely. Ensembles, which are Lecture 5, are the fix.

---

# ▶ SLIDE 8 — The Same Tree, Drawn as Rectangles

This is the same tree in a different picture, and this picture is what a tree really is.

*(Point at the axes.)*

The tree used two features, so we can draw every training week as a point in a plane. Across the
bottom is what the store sold four weeks earlier, and up the side is what it sold in the same week
last year. There are a hundred seventy-three white dots, one for each training week.

*(Trace the cuts.)*

Each question in the flowchart is a straight cut. The root, four weeks ago at or below twenty
thousand six hundred and three, is the solid vertical line. The left branch's second question, at
eighteen thousand five hundred ninety-eight, is the dashed vertical line on the left. And the right
branch's question about last year is the dashed horizontal line, but notice that it only runs
across the right half.

**[pause]**

So there are four rectangles, and they're the four green boxes from the flowchart. The darker the
shading, the higher the forecast. Every week that falls inside a rectangle gets that rectangle's
average: eighteen thousand four hundred, twenty thousand one hundred, twenty-one thousand four
hundred, or twenty-three thousand.

There are two things to see here.

First, the cuts are always parallel to an axis. A tree never draws a diagonal line. All it can ask
is whether one feature is above or below a number, so its whole world is rectangles, and its
forecast is flat inside each one.

Second, the horizontal cut stops at the vertical one. Last year's sales split the right half and
leave the left half alone. That's an **interaction**: last year matters, but only when recent
demand is already high. Nobody told the tree to look for that. A linear model would need you to
write the interaction term yourself, and the tree found it just by cutting.

---

# ▶ SLIDE 9 — The CART Algorithm (Regression)

So how does the tree decide what to ask?

The algorithm is called CART, which is short for classification and regression trees, and it
builds the tree greedily, from the top down. At each node it searches over **every feature and
every possible threshold**, and it keeps the split with the smallest total squared error. That's
the squared deviations from the mean on the left, plus the same thing on the right.

Hold on to the word "greedily," because the next slide is entirely about what it costs.

**[pause]**

The picture on the right is that search, done on our data for the very first split.

*(Point at the top panel.)*

Every dot is one of the hundred seventy-three training weeks. Across the bottom is what the store
sold four weeks earlier, and up the side is what it sold that week. The dashed line is a cut. Weeks
to its left are red, weeks to its right are blue, and each side gets one prediction, its mean,
which is the thick horizontal line. The thin vertical sticks are the errors, meaning how far each
week sits from its side's mean.

The algorithm squares every stick and adds them up. That total is the number it's trying to make
small.

*(Point at the bottom panel.)*

Now slide the cut. The bottom panel shows that total for every place the cut could go. With no
split at all, the total is about six hundred sixty-one million, and that's the dotted line.
Wherever you put the cut, the total drops. It drops furthest with the cut at twenty thousand six
hundred and three, where it reaches four hundred thirty-one million.

**[pause]**

So the total goes from six hundred sixty-one million down to four hundred thirty-one million.
**One yes-or-no question removed thirty-five percent of the squared error.** That leaves
eighty-nine weeks on the left, averaging about nineteen thousand six hundred, and eighty-four on
the right, averaging about twenty-one thousand nine hundred.

The algorithm did this for every feature, not just this one, and the four-week lag won. That's the
root of the tree from two slides back.

Then it repeats the whole search inside each group, and again inside those, until the groups get
too small or you hit the depth cap. The prediction at a leaf is just the mean of the training weeks
that landed there.

For classification, you replace squared error with Gini impurity or entropy. It's the same
algorithm, with a different measure of how mixed a region is.

---

# ▶ SLIDE 10 — Greedy Is Not Optimal — and That Matters

I want to slow down on "greedy," because it's the single most important word on the previous slide,
and it's easy to hear it as a criticism when it's really a description.

CART is **exhaustive at each node**. It really does try every feature and every cut point, with no
shortcuts and no sampling. So it isn't lazy.

What it never does is look ahead. It evaluates one split, commits to it, and moves down.

**[pause]**

The table is a made-up illustration, to show what that costs. Suppose splitting on price removes
forty percent of the variance right now, and only sets up another twelve percent below it.
Splitting on promotion removes thirty-eight percent, which is slightly worse. But *because* of how
promotion divides the data, it sets up two further splits that together remove another fifty
percent.

So it's forty plus twelve against thirty-eight plus fifty, which comes to fifty-two percent against
eighty-eight.

CART takes the forty. It isn't that it weighed the options and chose badly. It has no way to see
the fifty coming at all, because it scores one split at a time.

**[pause]**

So the tree you fit is **not the optimal tree**. It's the result of a sequence of locally best
decisions, and those are genuinely different things.

The obvious question is why we tolerate that. The answer is that finding the truly optimal tree is
computationally intractable for any realistic number of features, because the search space explodes
combinatorially. So the greedy approximation isn't a shortcut somebody settled for. It's what makes
trees usable at all.

There are two consequences to carry forward, and both matter for the next three weeks.

First, a tree can **miss structure that's really there**, not from a lack of data, but because of
the order in which it asked its questions.

Second, and this is the one that pays off next week, a small change in the data can send the whole
greedy sequence down a different path. That's the instability we turn to right now.

---

# ▶ SLIDE 11 — Depth Controls the Bias–Variance Tradeoff

Now let's connect the tree back to slide five, on the bias–variance tradeoff.

A depth-one stump asks one question. It has high bias and low variance, like the red row on slide
five.

A full tree keeps splitting until every training week has its own leaf. It reproduces the training
data exactly, so it has zero *training* error. That isn't zero bias, it's memorization, and the
variance is high. That's the blue row on slide five.

*(Point at the figure.)*

Here's the whole range on our weekly data. Across the bottom is the depth cap, from one up to
twelve, and then a fully grown tree with a hundred seventy-three leaves. The gray line is the error
on the training weeks, and the red line is the error on the test year.

The gray line does what it must. It falls with every extra level and hits **zero** for the full
tree, because a deeper tree can always fit the sample better.

The red line doesn't follow it. It drops from the stump to depth three, where it's about twelve
hundred thirteen, against sixteen sixty for seasonal naive, and then it stops improving. There's one
thing to know about that comparison: these trees forecast one week ahead, and they're handed the
true lags every week. Past depth three, the test error wanders between about thirteen hundred fifty
and fifteen hundred sixty. The gap between the two lines is overfitting, drawn to scale.

**[pause]**

Why does it stop at three? Because we only have a hundred seventy-three weeks. Every split halves
the data, and after three levels a leaf holds about twenty weeks. Split further, and you're fitting
the noise in a handful of weeks. With more data, deeper trees pay off, and you'll see that in the
lab, where the daily data has over a thousand rows.

There's also a warning about how to read this. Don't pick depth three *from this picture*. That red
line is the test year, and choosing a setting by looking at the test year is exactly the leak we
keep forbidding. Read the shape, which is falling training error and flat test error, and not the
minimum. The depth itself gets chosen by `TimeSeriesSplit` on the training weeks.

Now here's why trees are unstable, which matters enormously next week.

Change one training observation. If that changes the split at the root, and it easily can, because
the algorithm is picking the single best threshold, then *every prediction below it* changes. And
not slightly, either. The whole structure downstream is different.

That isn't a flaw in the implementation. It's built into a greedy, hierarchical partition. And it's
precisely the property that averaging will exploit in Lecture 5, on ensembles.

So you tune with `max_depth` and `min_samples_leaf`, selected by `TimeSeriesSplit`
cross-validation.

---

# ▶ SLIDE 12 — Two Ways to Stop a Tree Overfitting

`max_depth` is the blunt instrument. It caps the tree in advance, before you know how deep it
needed to go. Two other controls do the job better, and they usually get one line each in a table,
which isn't enough.

The first is `min_samples_leaf`, and it's the one I'd reach for first. It stops the tree early.
Leave it at the default of one, and the tree may build a leaf holding a single observation, whose
"prediction" is just that one observation's value. That's memorization with extra steps. Set it
somewhere between five and ten, and every prediction is an average over a real group of weeks.
That's smoother, and far more honest about what the data supports.

The second is `ccp_alpha`, and it works the other way around: it prunes afterward. You grow the
tree out fully, *then* charge a penalty, alpha, for each leaf, and cut back the branches that don't
pay for themselves. A larger alpha gives you a smaller tree.

Whichever one you use, choose its value by cross-validation, which for us means `TimeSeriesSplit`,
and never by looking at the test year.

---

# ▶ SLIDE 13 — Cost-Complexity Pruning: A Line Touching the Curve

Pruning has a picture, and it's worth two minutes, because the same picture shows up in two other
places in this course.

Cost-complexity pruning minimizes the fit, which is the sum of squared errors, plus a price alpha
for every leaf.

*(Point at the black curve.)*

Each black dot is a pruned version of one fully grown tree on our weekly data, and each one is the
best pruned tree for some range of alpha. Across the bottom is how many leaves it keeps, and up the
side is its training error. One leaf, which means no splits at all, is the dot at the top left, at
six hundred sixty-one million. Every leaf you add lowers the error, so the curve runs down to the
right, flattening as it goes. The full tree, off the right edge, has a hundred seventy-three leaves
and zero training error. That's one leaf per week, which is pure memorization.

**[pause]**

Now for the geometry. Take a straight line with a slope of minus alpha, and push it up from below
until it touches the curve. The first dot it touches is the tree that alpha picks.

*(Point at the red line.)*

At thirty-four million per leaf, which is the red line, it touches at four leaves. That four-leaf
tree is exactly the flowchart from slide seven.

*(Point at the blue line.)*

Make leaves cheaper, at fifteen million each, and the line gets flatter. That's the blue line, and
it touches further to the right, at nine leaves. A bigger alpha means a steeper line, and a steeper
line means a smaller tree. That's all alpha does.

The hollow circles are trees capped by `max_depth` at three and at four. They sit *above* the
curve, so for the same number of leaves, pruning found a lower training error. A depth cap cuts
every branch at the same level, whether it was earning its keep or not.

**[pause]**

You have seen this before. It's the GAM smoothing penalty from Lecture 3 with a different symbol on
it: fit against complexity, with one parameter tuning the tradeoff. And you'll meet it a third time
in Lecture 6, as Ridge and LASSO, where the curve becomes a diamond and a disc.

That's three model families that look nothing alike, with one idea underneath. I'd rather you
remember the idea than the three implementations, because the idea is what transfers. And alpha
itself is chosen by cross-validation.

There's one technical footnote on the slide. Scikit-learn measures `ccp_alpha` per observation, so
its number is this alpha divided by the number of training rows.

---

# ▶ SLIDE 14 — Section divider: Implementing Decision Trees in Python

The third part is the implementation. It's one class and a few hyperparameters, and the results
are interpretable.

---

# ▶ SLIDE 15 — Decision Tree Regression in Python

The code is short.

You import `DecisionTreeRegressor`, and you take the weekly features, which are the lags, SNAP
days, events and price. You instantiate the tree with `max_depth=5`, `min_samples_leaf=10` and
`random_state=42`. Then you fit, and you predict. That template is exactly what produced the
importance table on the next slide.

The table gives you the four hyperparameters worth knowing. We just spent a slide on two of them,
`min_samples_leaf` and `ccp_alpha`. Then `max_depth` caps the depth, and `max_features` does
feature subsampling. Remember that last one, because it becomes the central idea next week.

**[pause]**

And then there's the line at the bottom, which you've now heard in every lecture: cross-validate
with `TimeSeriesSplit`, never `KFold`. Random folds leak the future into the training set.

I'll keep saying it, because it's the error that survives longest. It doesn't produce a crash or a
warning. It produces a number that looks better than the truth.

---

# ▶ SLIDE 16 — Feature Importance

A tree will also tell you which variables it found useful.

Impurity-based importance works like this. For each feature, it sums the total reduction in
variance across every split where that feature was used, weighted by how many observations passed
through. Then it normalizes the results so they add up to one.

On our weekly Walmart data, a depth-five tree ranks the features like this. The four-week lag
dominates, at about zero point six two. Then comes the same week last year at zero point one five,
then last week at zero point one three, and then SNAP benefit days at zero point zero five and price
at zero point zero four.

Notice how concentrated that is. One feature takes almost two-thirds of the total. That's
characteristic of a single tree, and it's a symptom of the greediness we discussed: the root split
commits to one feature, and everything below inherits that choice. Next week you'll see the same
table from a random forest, and there the mass is spread much more evenly.

Read it as a screening device. It tells you which variables the tree leaned on, and it's a fast way
to narrow forty candidate predictors down to eight worth thinking about.

And notice that SNAP appears at all, because it's the kind of thing this method is good for. Nobody
specified that food sales respond to the days when benefits are paid out. The tree found it, and it
ranked it level with price. That's a real business finding, and no linear specification would have
surfaced it unless someone had already thought to look.

**[pause]**

But be careful, because impurity importance is **biased toward features with many split points**,
which in the jargon are called high-cardinality features.

Look at our own table. `avg_price` is continuous, so it offers hundreds of distinct split points,
and the algorithm gets hundreds of chances to find one that happens to help. `snap_days` takes a
handful of whole-number values, and `event_days` takes fewer still. Price gets more opportunities to
look useful purely because it has more thresholds to try, not because it carries more signal.

That means I wouldn't read anything at all into the gap between SNAP at zero point zero five and
price at zero point zero four. If anything, the bias runs in price's favor.

Permutation importance is the honest alternative. You shuffle one column and measure how much the
accuracy falls. It's slower, and it's worth it whenever the ranking will inform a decision. We'll
use it properly next week, in Lecture 5.

---

# ▶ SLIDE 17 — Interpreting and Communicating Tree Predictions

This is the tree's real advantage, and it's worth dwelling on, because you lose it steadily from
here on.

You can explain **why** a specific prediction came out the way it did.

*(Point at the manager quote.)*

Here's how that explanation sounds when you give it to a manager. "Our model forecasts twenty-two
thousand nine hundred eighty-eight units for this week, for three reasons. Four weeks ago, the store
sold more than twenty thousand six hundred and three units, so recent demand is high. The same week
last year sold more than twenty-one thousand six hundred ninety-two, so the season is strong. And
the twenty-five training weeks where both of those held averaged twenty-two thousand nine hundred
eighty-eight units."

Notice what that explanation is. It's just the right-hand path of the flowchart from slide seven,
read aloud. There are no coefficients and no "holding all else constant." It's a sequence of
conditions and an average.

**[pause]**

Scikit-learn's `plot_tree` or `export_text` will produce the path for you.

And here's why it matters commercially, because this is where business buy-in comes from. A manager
who can see the splits can check that the tree is splitting on variables they consider sensible,
and can tell you when the model is wrong: "That threshold is picking up the year we had the fire,
not a real pattern." You can't get that feedback from a model nobody can read. It's the cheapest
source of domain knowledge you have, and every method after this one makes it harder to get at.

---

# ▶ SLIDE 18 — Section divider: Limitations and Preview

The fourth part is about where a single tree falls short. On its own, a tree is a weak learner, and
the preview is that combining hundreds of them builds a strong one.

---

# ▶ SLIDE 19 — When a Single Tree Falls Short

The strengths are on the left, and they're real. A tree is interpretable, it needs no feature
scaling, it captures nonlinear relationships, it handles missing values well, and it's fast to
train.

The weaknesses are on the right, and the first one is fatal.

The first weakness is high variance. We covered the mechanism on slides ten and eleven: a small
change in the data gives a very different tree.

The second is flat predictions within each leaf. Every observation in a leaf gets the identical
forecast, so the prediction surface is a step function.

The third is that a tree cannot extrapolate beyond the training range, and this one deserves a
moment. A tree predicts the mean of a leaf, and no leaf can contain values it never saw. So on a
trending series, a tree will flatten out exactly when you need it to keep climbing. If your sales
have grown every year and you ask a tree to forecast next year, it will predict something inside
the range it already knows.

Let's be honest about where that bites. CA_1 food sales grew about twenty-nine percent over five
years, but almost all of that growth came between the first year and the second, and since then the
store has plateaued. Train a tree on the first four years and ask it for the fifth, and not one of
the fifty-two weeks rises above the training maximum. So on this series, the ceiling never binds.
Next week you'll see a series where it does. That's US retail sales, where a forest trained through
twenty nineteen never forecasts above four hundred fifty billion dollars, while actual sales climb
to six hundred sixty-five billion. On a series like that, a tree systematically under-forecasts,
and no amount of tuning fixes it, because the model has no representation of "up."

An ARIMA with a drift term has no such problem. It will happily forecast values it has never seen.

That limitation never goes away. Random forests have it, and XGBoost has it. It's a property of
trees, and it's why the classical methods from Lectures 1 and 2, exponential smoothing and ARIMA,
don't simply become obsolete today.

The last weakness on the list is that a tree is sensitive to outliers in a leaf, because each
leaf's forecast is a plain average of the weeks inside it.

**[pause]**

And the ensemble solution, which is next week, is to average many trees, each fitted to a different
bootstrap sample. The variance falls sharply, the feature importance survives, and the result is
typically twenty to forty percent lower RMSE than a single tree.

---

# ▶ SLIDE 20 — Lecture 4: Key Takeaways

There are six things to take away from today.

First, linear models can't capture interactions or threshold effects. Trees can, without your
having to specify them in advance.

Second, depth sets the bias–variance tradeoff. Too shallow underfits, and too deep overfits.

Third, CART takes the split with the least squared error. You slide the cut and total the squared
sticks, and the prediction is the mean of the leaf.

Fourth, you stop overfitting with `min_samples_leaf` or with `ccp_alpha`, which is the line of
slope minus alpha touching the curve, and you tune either one with `TimeSeriesSplit`.

Fifth, trees are interpretable. You can trace the exact path to any prediction.

And the last one: a tree is a step function, so it cannot extrapolate, and a single tree has high
variance. Ensembles solve the variance, and that's Lecture 5. They do not solve the extrapolation.

**[pause]**

If you take only one thing away, take this: **a tree is a step function that cannot extrapolate.**
Everything good and everything bad about today follows from that sentence.

---

# ▶ SLIDE 21 — References

The reading is section eight point one of ISLP, An Introduction to Statistical Learning with
Applications in Python, which covers the basics of decision trees. That's the chapter we'll live in
for the next two weeks as well.

Next time is Lecture 5, on tree ensembles: variance reduction through bootstrap aggregation, and
sequential error correction with boosting. We'll take this unstable, high-variance model and turn
its instability into an advantage.

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
