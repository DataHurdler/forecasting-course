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

This is Lecture 5, on tree ensembles.

Lecture 4, last week's lecture on decision trees, ended on a complaint. A single decision tree is
unstable. Change a few training points, and the whole structure of the tree changes. I called
that a fatal weakness.

Today we'll do something that should feel slightly perverse. We'll take that
instability and turn it into the source of the method's strength.

**[pause]**

There are two families of ensemble, and they attack opposite problems. Bagging and random
forests attack **variance**, by averaging many trees that are built in parallel. Boosting attacks
**bias**, by building trees in sequence, where each one fixes what the last one got wrong.

By the end, you'll have the method that, on most tabular business data, is simply
the best thing available.

---

# ▶ SLIDE 2 — Lecture Outline

We'll take this in five parts. First, the step from a single tree to an ensemble. Second,
bagging, and third, random forests. Fourth, feature importance. And fifth, gradient boosting and
XGBoost. The last section on the outline pulls it all together.

---

# ▶ SLIDE 3 — Section divider: From Single Trees to Ensembles

Part one starts with an observation that has nothing to do with statistics: averaging many
imperfect models can do better than any one of them.

---

# ▶ SLIDE 4 — The Wisdom of Crowds

In nineteen oh-six, at a county fair, eight hundred people guessed the weight of an ox.

Francis Galton collected the guesses, expecting to show that the crowd was foolish. Individually,
they were. Almost nobody was close.

But the **average** of all eight hundred guesses came within one pound of the true weight.

**[pause]**

Why does that work? Because the errors were *independent*. Some people guessed high and some
guessed low, for unrelated reasons, and averaging canceled them out.

Now carry that over to models. A single decision tree is noisy. It splits somewhere slightly
arbitrary, and everything further down the tree inherits that arbitrariness. But if you train
many trees on different versions of the data, each one is noisy in a *different* way. Average
them, and the noise largely cancels, while the signal, which they all agree on, survives.

**[pause]**

I want you to notice the word that's doing all the work here, and it's **independent**.

Galton's crowd worked because the guesses were unrelated. If all eight hundred people had
overheard the same wrong estimate, averaging would have gained nothing.

Hold on to that. The entire difference between bagging and random forests, the whole reason
random forests exist, turns on that one word.

---

# ▶ SLIDE 5 — Section divider: Bagging

Part two is bagging, which is short for bootstrap aggregating.

---

# ▶ SLIDE 6 — Bagging: The Big Idea

Bagging comes from Leo Breiman, in nineteen ninety-six, and it's aimed squarely at the problem
Lecture 4 left us with: one tree has high variance. It takes three steps.

First, draw **B bootstrap samples**. That means sampling your training data with replacement,
with each sample the same size as the original. Because it's with replacement, each sample gets
some observations twice and misses others entirely. That's the point. Each sample is a slightly
different view of the data.

Second, fit one deep tree to each sample. I do mean deep: we're not pruning. We *want* each tree
to have low bias and high variance, because averaging is about to take care of the variance.

Third, average all B predictions.

**[pause]**

Now for the equation, which is the most important thing on this slide.

Call one tree's variance sigma squared, and call the correlation between two trees' predictions
rho. Then the variance of the averaged forecast is rho times sigma squared, plus one minus rho,
over B, times sigma squared.

There are two terms. Look at what happens to each one as B, the number of trees, grows.

The second term has B in the denominator, so it shrinks toward zero. That's why more trees never
hurt.

The first term has no B in it at all. It doesn't move.

---

# ▶ SLIDE 7 — More Trees, or Less Alike Trees

Here's that equation, drawn. I've divided through by sigma squared, so one tree sits
at one on the vertical axis.

*(Point at the axes.)*

Across the bottom is the number of trees you average, on a log scale, from one to five hundred.
Up the side is the variance of the average.

This is a schematic, and nothing in it is measured. Rho is a property of how the trees vary
across all the training sets you might have drawn, and one forest can't measure its own rho. So
I picked three values to show the shape.

**[pause]**

*(Trace the orange curve.)*

Each value of rho draws its own curve. Take rho of one half, the orange one. Every curve starts
at one, because one tree is one tree. As you add trees, you slide along the curve, following the
black arrow. You move fast at first, and then the curve flattens onto the dashed line at one
half. That dashed line is the **floor**, rho times sigma squared. Past a hundred trees or so,
you're buying almost nothing.

*(Point at the gray curve, then at the red arrow.)*

Now look at the gray curve, where rho is zero point nine. These trees are highly correlated, and
their floor is at zero point nine. Average a thousand of them, and you've barely improved on one.
You've grown a thousand trees and reproduced one. That's Galton's crowd, overhearing each other.

The red arrow is the only way down. Make the trees **less alike**, and you drop to a lower curve
with a lower floor. The blue curve, with rho of zero point one, ends near a tenth of one tree's
variance.

So there are two dials, and they do different things. More trees move you along a curve.
Decorrelation moves you to a different curve. And it's exactly that second dial that the next
section turns.

---

# ▶ SLIDE 8 — Out-of-Bag (OOB) Error: Free Cross-Validation

Bootstrapping has a pleasant side effect.

Each bootstrap sample draws n observations with replacement, so some get picked more than once
and some aren't picked at all. If you work through the probability, you find that each sample
contains about **sixty-three percent** of the distinct training points.

That means roughly thirty-seven percent are left out of any given tree. Those are the
out-of-bag observations, or O-O-B for short.

Here's the trick. For observation i, collect predictions only from the trees where i was out of
bag, and average those. Then compute the RMSE across all the observations.

Every one of those predictions came from a tree that **never saw** that observation. So the
O-O-B error is an honest out-of-sample estimate, and you got it without setting aside a
validation set or refitting anything. It falls out of the bootstrapping you were doing anyway.

In scikit-learn, it's one argument: `oob_score=True`.

**[pause]**

There's a caution, though. In fact there are two, and the second is the one that matters for
this course.

The first is about size. You may hear that O-O-B is unreliable on small data because each row is
out of bag for too few trees. That's not the reason. Each row sits out about thirty-seven percent
of the trees, whatever the sample size is. With five hundred trees, that's about a hundred eighty
of them. What a small sample lacks is *rows*. The O-O-B error is an average over n rows, and a
forest on CA_1 food sales trains on a hundred seventy-three weeks. A hundred seventy-three errors
make a noisy average, but that part is ordinary sampling noise.

The second caution is the one that bites. Look at who does the scoring.

*(Point at the figure.)*

This is the real forest from later in the lecture, the weekly CA_1 forest with five hundred trees,
and these are its actual bootstrap draws. Each row is one tree, and I'm showing twenty-four of
them. Each column is one training week, in date order, from early twenty twelve to May twenty
fifteen. Gray means that week was in the tree's bootstrap sample, and white means it was out of
bag.

*(Point at the boxed column.)*

Pick one week, the boxed column, which is the week of the thirtieth of December, twenty thirteen.
Eight of these twenty-four trees left it out, and they're the ones marked with red triangles.
Those eight trees are the ones that score it.

Now look to the right of the box, along those eight rows. Everything in red is a week *after* the
thirtieth of December that the scoring tree trained on. Across the whole forest, the trees that
score a week still trained on about sixty-three percent of the weeks that came after it. That's
the same share as any bootstrap sample, because leaving one week out says nothing about the
others.

**[pause]**

So O-O-B is answering the question, "How well can the model fill in a missing week inside the
years it has already seen?" It isn't answering, "How well will it forecast next month?" Those are
different questions, and only the second one is your job.

So on this data, use proper walk-forward cross-validation. Lab 5 shows you the two numbers side
by side, and asks you to account for the gap.

---

# ▶ SLIDE 9 — Section divider: Random Forests

Part three is about decorrelating the trees, by restricting each split to a random subset of the
features.

---

# ▶ SLIDE 10 — The Problem Bagging Does Not Fully Solve

Let's go back to the floor.

Why would trees end up correlated? Because of the greedy algorithm from Lecture 4, last week. At
the root, CART picks the single best split. If one feature is clearly the strongest predictor,
then *every* tree picks it, and every tree starts the same way.

And in forecasting, one feature is often dominant. On our weekly data it's the four-week lag,
which was the root of last week's tree. On the daily data in Lab 5, it's the day of the week.

**[pause]**

The random forest fix, from Leo Breiman in two thousand one, is almost absurdly simple. At each
split, don't consider all p features. Consider a random subset of **m** of them, and take the
best split among *those*. That's the whole method.

*(Point at the left panel.)*

Here's what that does, measured on Lab 5's data. The left panel is three hundred bagged trees,
with every feature available at every split. For each tree, I asked one question: what did it
split on first? Two hundred ninety-eight of the three hundred chose the day of the week, and the
other two chose the four-week lag. That's three hundred trees with one opening move.

*(Point at the right panel.)*

The right panel is a random forest, with three of the ten features offered at each split. Now the
day of the week opens only seventy-seven trees. The four-week lag opens seventy-two, last week's
sales open fifty-six, and so on down. The structures genuinely differ, and that's the
decorrelation, made visible.

**[pause]**

Now for the honest part, so read the numbers in the corners.

The random forest's trees are individually *worse*: four hundred seventeen RMSE on average,
against three hundred ninety-eight for the bagged trees. That's the price, because many of them
weren't allowed to look at the day of the week first. And averaging does help them more. It takes
them from four seventeen down to two fifty-four, against three ninety-eight down to two
forty-eight.

But it doesn't help enough. The bagged ensemble wins, two forty-eight to two fifty-four.

That isn't a failure of the idea. It's the idea's boundary condition. You deliberately make each
tree worse in order to make the ensemble better, and that trade pays when you have many partly
redundant features and no single dominant one. With ten features, and one of them carrying most
of the signal, the handicap costs more than the independence buys. And don't expect Homework 3 to
rescue the textbook rule. Its forest has twenty-seven features, but last week's sales dominate
there the way the day of the week dominates here, and the square-root setting loses there too.

---

# ▶ SLIDE 11 — Random Forest Hyperparameters

This table has five parameters, and the striking thing is how few of them need attention.

The first is `n_estimators`, the number of trees, with a default of a hundred. More is better,
and the gain levels off at around five hundred. Adding trees carries no risk of overfitting,
because more trees only shrink the second term of the variance equation.

The second is `max_features`, and **this is the one that matters**. It's the m from the previous
slide. It's the dial that controls rho, and so it's the dial that controls the floor. A lower
value means more decorrelated trees.

And here's a trap. For a *regression* forest, scikit-learn's default is one point zero, which
means every feature at every split. That's plain bagging, not a random forest. The `sqrt` setting
you'll see in textbooks and tutorials is the *classifier's* default. So if you never set
`max_features`, the "random forest" you fit has no feature randomness at all. Set it yourself,
and tune it.

The third is `max_depth`. Leave it at None, because we want deep trees. The only reason to limit
it is to save memory.

The fourth is `min_samples_leaf`. Its default is one, and raising it to about five gives you
smoother predictions.

The last is `oob_score`. Setting it to True gives you the free estimate from slide eight, and it's
worth turning on if your data set is large enough for it to mean anything.

**[pause]**

So start with five hundred trees and a `min_samples_leaf` of five, and try a few values of
`max_features`, such as the square root and a larger fraction like zero point six, by
walk-forward cross-validation.

And notice the practical consequence: only one parameter needs real tuning. Compare that with
what's coming in XGBoost, where you'll be juggling six. A large part of why random forests are the
right default is that they're very hard to misconfigure.

Finally, set `random_state=42`. A forest is random twice over, in the bootstrap samples *and* in
the feature subsets, so without a seed your results move between runs, and you'll waste an hour
wondering why.

---

# ▶ SLIDE 12 — Section divider: Feature Importance

Part four asks which predictors matter most, and a random forest gives a data-driven answer.

---

# ▶ SLIDE 13 — Random Forest Feature Importance

The standard measure is called mean decrease in impurity, or M-D-I. For each feature, you add up
the impurity reduction, which for a regression tree is the drop in variance, across every split
where that feature was used, weighted by how many observations passed through that node. You do
that in every tree, average across the trees, and normalize so that the
importances add up to one.

This is the forest from the next slide, on our weekly CA_1 food sales. The ranking comes out with
the **four-week lag** at zero point two eight, last week at zero point two one, SNAP benefit days
at zero point one seven, and the same week last year at zero point one five. The two-week and
three-week lags follow at zero point zero nine and zero point zero six, and price is further down,
below everything on the table.

Compare that with Lecture 4's single tree, last week, which gave the four-week lag zero point six
two. The forest spreads the credit around. Partly that's because it averages many trees, and
partly it's because each split was only allowed to see two of the eight features, so other
features got their turn.

**[pause]**

That SNAP result is worth pausing on. Nobody told the model that food sales spike when benefits
are paid out. It found that on its own, and ranked it above last year's sales and well above
price. That's a real business finding, and no linear specification would have surfaced it unless
someone already suspected it.

But read this table as a screening device, not an explanation, because **M-D-I is biased**.

The bias is toward features with many possible split points. A continuous variable like price
offers hundreds of thresholds to try, while a binary flag offers one. The continuous variable gets
more chances to look useful, not because it carries more signal, but because it had more
opportunities.

Permutation importance is the honest alternative. You shuffle one column and measure how much the
accuracy falls. If shuffling it doesn't hurt, the feature wasn't doing anything. That's a direct
measurement, rather than an accounting artifact.

So use M-D-I to narrow forty candidates down to eight. Then use permutation importance before you
put a ranking in front of anyone who'll act on it.

---

# ▶ SLIDE 14 — Random Forest in Python

The code takes four lines, and the results table is the interesting part.

On CA_1 food sales, with a fifty-two-week test, seasonal naive scores **sixteen sixty**. A single
deep tree scores fourteen fifty-six. And the random forest scores **eleven fifteen**. That's the
template on this slide: five hundred trees, square-root features, and at least five weeks in each
leaf.

**[pause]**

Two things stand out. The first is that the forest cuts RMSE by about twenty-three percent against
the single tree. That's the variance reduction, bought by averaging. It's the same algorithm and
the same features, with five hundred trees instead of one.

The second is that both tree models beat the seasonal naive benchmark on this series. In Homework
1, back in the first two weeks, exponential smoothing and ARIMA couldn't. But before you give the
trees the credit, check what each one was asked to do.

**[pause]**

The forest forecasts **one week ahead**. Every test week, it's handed last week's actual sales, and
the three weeks before that. The ETS and ARIMA numbers from Lectures 1 and 2 forecast the whole
test year from the end of training, without seeing a single actual from the test year. Those are
different jobs.

Give the classical methods the same one-week job, and they beat sixteen sixty too. Simple
exponential smoothing scores about thirteen hundred, and an ARIMA scores about twelve sixty. And a
forest that's handed only the series' own lags, with no price, no SNAP and no events, scores
twelve thirty-one.

So of the five hundred forty-five units the forest gains over seasonal naive, about four hundred
thirty come from seeing last week's actual sales, before any price or SNAP column. The features
add the last hundred or so. And the algorithm on its own adds little: at the same horizon, on the
same history, the forest and the ARIMA are within a few percent of each other.

That's the real lesson of the second half of this course. Ask what information a model was given,
and how far ahead it was forecasting, before you ask how clever it is.

---

# ▶ SLIDE 15 — Section divider: Gradient Boosting

Part five changes direction entirely. Instead of averaging independent trees, we fit each new tree
to the errors of all the trees before it.

---

# ▶ SLIDE 16 — Boosting vs. Bagging: The Core Difference

Bagging builds its trees **in parallel**, independently of one another, and each tree tries to
predict y directly. Averaging reduces variance. And critically, adding more trees never reduces
bias. If your trees are systematically wrong in the same direction, averaging preserves that error
exactly.

Boosting builds its trees **in sequence**. Each new tree fits the *errors* of everything built so
far. That reduces bias as well as variance, and it also means boosting can overfit, if its trees
are too deep or there are too many of them. That's something bagging essentially can't do.

**[pause]**

The intuition on the slide is the clearest version of this I know.

Forecast one says fifty thousand dollars. The actual is sixty thousand, so the error is plus ten
thousand.

Now, forecast two doesn't try to predict sales at all. It tries to predict **that error**. It says,
"The first model was short by about ten thousand."

Add them together, and fifty plus ten is sixty.

**[pause]**

Each stage corrects what remains. And notice that this fixes something averaging never touches. If
every bagged tree under-predicts December by the same amount, the average under-predicts December.
Boosting sees that pattern in the residuals and builds a tree specifically for it.

So bagging attacks variance, and boosting attacks bias. That one sentence is the whole slide.

---

# ▶ SLIDE 17 — The Gradient Boosting Algorithm

Formally, the algorithm is three steps in a loop.

You start with the mean. Then, over and over, you compute the residuals, fit a shallow tree to
those residuals, and add a fraction of that tree to the running prediction.

Two design choices here are worth understanding.

The first is that the trees are **deliberately shallow**, somewhere between depth three and depth
six. That's the opposite of bagging, where we grew them deep. Here each tree only needs to capture
a small piece of the remaining error, so a small tree is enough. A deep tree would fit the noise in
the residuals, and then the next iteration would be trying to correct noise.

The second is that the learning rate, eta, is the main dial. After fitting a tree to the residuals,
you don't add the whole thing. You add eta times it, where eta might be zero point zero five. The
slide gives the range: somewhere between zero point zero one and zero point one is slow and robust,
and above zero point three you risk overshooting.

That means you deliberately under-correct at every step.

**[pause]**

Why on earth would you do that?

Because each tree is fitted to the residuals of one particular sample, so part of what it sees is
noise. Take the full correction, and you bake that noise in. Take five percent of it, five hundred
times over, and the noise averages out while the signal accumulates.

**[pause]**

The reliable recipe is a small eta, many trees, and **early stopping**. Without early stopping,
boosting will keep reducing the training error until it has fitted your noise perfectly. Unlike
bagging, this method genuinely can overfit, and it will if you let it.

---

# ▶ SLIDE 18 — Boosting, Tree by Tree

Here's that loop running, on data where you can see every step.

To draw a fit, you need a single feature, so I've borrowed the electricity data from Lecture 3, on
generalized additive models. It's weekday load in the eastern US, plotted against the day of the
year. Every gray dot is one weekday between two thousand two and twenty eighteen. The shape is the
one you met in Lecture 3, a W. Load is high in winter, dips in spring, climbs to the big summer
peak from air conditioning, and dips again in the fall.

The dashed line is where boosting starts, at the mean, F-zero.

**[pause]**

*(Point at the left panel.)*

This is after **one tree**. It's a depth-three tree fitted to the residuals from the mean, and we
add zero point three of it. The red curve has barely left the dashed line. There's a step up for
summer, and a step down for spring and fall. The training RMSE is three point nine gigawatts.

*(Point at the second panel.)*

This is after **two trees**. The second tree was fitted to what the first one left over, the
residuals. The summer step got taller, and the dips got deeper. The RMSE is now three point
five three.

*(Point at the third panel.)*

After **five** trees, you can see the W, and the RMSE is three point oh seven.

*(Point at the fourth panel.)*

After **fifty**, the curve follows the data closely, and it has started to wiggle. The RMSE is two
point eight nine.

**[pause]**

Here's how to read that last number. The best that *any* curve in the day of the year could do on
these data, which is the average load on each calendar day, has an RMSE of two point eight six.
Fifty trees are within three hundredths of that. Everything that's left is variation the day of the
year can't explain: weather, holidays, and one year being hotter than another. If you keep adding
trees, boosting starts fitting that noise, and those wiggles near July are the beginning of it.

One honest note. I used a learning rate of zero point three, right at the edge of the previous
slide's warning about overshooting, so that four panels would show visible progress. At zero point
zero five, the same picture takes about six times as many trees.

---

# ▶ SLIDE 19 — Section divider: XGBoost

Next comes XGBoost, the version of boosting that won everything. It makes three improvements on
plain gradient boosting: a second-order step, explicit regularization, and column subsampling.

---

# ▶ SLIDE 20 — XGBoost: Three Advances

The first is the **Newton step**. Plain boosting uses the gradient, which is the direction of
steepest descent. XGBoost also uses the second derivative, the curvature, to size the value in each
leaf. Be precise about when that matters. For squared error, which is what we use and is XGBoost's
default for regression, the curvature is the same everywhere. So the Newton step is just the
ordinary fit to the residuals, and nothing changes. It earns its keep on losses that bend unevenly,
like log-loss for classification, or a Poisson loss for counts.

The second is **explicit regularization**, and this is the important one. XGBoost puts penalties
directly into the objective: gamma on the number of leaves, and lambda on the size of the leaf
weights. So overfitting is controlled by the loss function itself, not only by stopping early.

The third is **column subsampling**, borrowed straight from random forests. It samples features for
each tree and for each split. That's the decorrelation idea from slide ten, imported into boosting.

**[pause]**

Notice what that second advance actually is.

The parameter `reg_lambda` penalizes the sum of the squared leaf weights. That's an L-two penalty.
It's Ridge regression, applied to leaf weights instead of regression coefficients. The parameter
`reg_alpha` penalizes their absolute values, and that's the L-one penalty, LASSO. And `gamma` is
different from both. It charges a fixed price for every leaf, however small that leaf's weight is.
That's a penalty on the count, and it's exactly Lecture 4's `ccp_alpha`, the price per leaf in
cost-complexity pruning.

So you're seeing next week's lecture, Lecture 6, one week early, wearing different clothes.

That's not a coincidence. Regularization isn't a topic for linear models alone. It's how *every*
flexible model is kept honest, and you'll meet it again in Lecture 7, on neural networks,
as weight decay and dropout.

**[pause]**

XGBoost dominated the tabular competitions on Kaggle from twenty fourteen to twenty eighteen.
LightGBM is faster on large data, and CatBoost handles categorical features natively, but the ideas
are the same.

---

# ▶ SLIDE 21 — XGBoost: Key Hyperparameters

This table has six parameters, against the random forest's one that mattered. That's the cost of
the extra accuracy.

The number of trees runs from five hundred to two thousand. That's high, because you're pairing it
with early stopping. The learning rate runs from zero point zero one to zero point one. The depth
runs from three to six, which is shallow. The row subsample and the column subsample both sit
between zero point seven and zero point nine. And `reg_lambda`, the Ridge penalty on the leaf
weights, is typically left at one.

**[pause]**

The two that interact most are `learning_rate` and `n_estimators`, and they trade off directly.
Halve the learning rate, and you need roughly twice the trees. A small eta with many trees is almost
always better than a large eta with few. You just pay for it in compute.

A practical starting point is a learning rate of zero point zero five, a depth of four, and a
thousand trees with early stopping on validation RMSE. Then tune the depth and the subsample with
`TimeSeriesSplit`.

---

# ▶ SLIDE 22 — XGBoost in Python

Here's the code, and one argument in it matters more than it looks.

It's `eval_set`, and it's what makes `early_stopping_rounds` work. XGBoost watches RMSE
on that validation set and halts when it stops improving, here after fifty rounds without a
gain. That's how you get away with setting a thousand trees. You're not committing to a
thousand. You're setting a ceiling and letting the data decide.

**[pause]**

And the validation set has to respect time. Build it with `TimeSeriesSplit`, hold out the final
stretch as a true test set, and never let a random split choose which weeks the model sees.

If you use a random split here, early stopping will pick the iteration that best predicts weeks the
model has effectively already seen. You'll stop too late, ship an overfitted model, and your
diagnostics will look excellent.

**[pause]**

One practical note will save you an evening. On macOS, `xgboost` needs the OpenMP runtime. If you
get an error saying that `libomp.dylib` won't load, run `brew install libomp`. The Python package
installs fine and then fails when you import it, which makes it look like a problem in your code
when it's really a problem with a system library.

---

# ▶ SLIDE 23 — Section divider: Key Takeaways

Let's pull it together. Bagging reduces variance, boosting reduces bias, and between them they
cover most of what tabular forecasting needs.

---

# ▶ SLIDE 24 — Tree Methods: Which One When

The table has three rows.

A decision tree is a single tree. It's highly interpretable and needs little tuning, and its
accuracy is the baseline. A random forest trains in parallel, with medium interpretability, little
tuning and good accuracy. XGBoost trains in sequence, with low interpretability, more tuning, and
the best accuracy of the three.

**[pause]**

My advice is that the **random forest** is the better default, and I want to be clear about why,
because the accuracy column says XGBoost is best.

A random forest is robust, it needs almost no tuning, and it hands you O-O-B error for free. It's
genuinely hard to make one catastrophically wrong.

Move to XGBoost when you can afford to tune and you need the last few points. And budget for that
tuning honestly, because an untuned XGBoost often loses to a default random forest. The accuracy in
that column is the accuracy of a *well-tuned* XGBoost, and that word is carrying a lot of weight.

---

# ▶ SLIDE 25 — Why No Tree Can Extrapolate

There's one limit that applies to every row of that table, and it deserves its own picture.

A forest's prediction is a weighted average of the training targets. Every leaf is an average of
training values, and the forest averages the leaves. A weighted average can never leave the range
of the things it averages. So a forest can never predict above the highest value it was trained on,
or below the lowest. It isn't that it rarely does. It can't.

*(Point at the chart.)*

This is US retail sales, monthly. I trained a forest on nineteen ninety-three through twenty
nineteen, which is the shaded part. Then I asked it for every month since, one month ahead, *giving
it the true values* for last month and the month before. That's the most generous setup there is.

The highest month in its training data was about four hundred fifty billion dollars, and that's the
dashed line. Sales went through it in twenty twenty and kept going, to six hundred sixty-five
billion by the middle of twenty twenty-six. Ninety-four percent of the months since have been above
the line. The forest's forecast, the red line, never crosses it.

**[pause]**

Boosted trees fail the same way, for a slightly different reason. Each tree is flat beyond its last
split, so the sum is flat too.

So the classical methods didn't become obsolete today. If your series trends, and you need to
forecast beyond its historical range, an ARIMA with drift will do something a forest simply can't.
Or you can model the *change* instead of the level, so that the target stays inside its range.

---

# ▶ SLIDE 26 — Lecture 5: Key Takeaways

There are eight takeaways, and each one fits in a sentence or two.

First, ensembles work because independent errors cancel, and the word "independent" is carrying the
load.

Second, bagging averages B trees grown on bootstrap samples. The variance falls as B grows, but it
hits a floor at rho times sigma squared.

Third, random forests cut rho by sampling features at each split. That trade pays when no single
feature dominates, and in Lab 5, where one feature did dominate, it didn't pay.

The fourth is that O-O-B error is free, but on a time series it scores how well you fill gaps, not
how well you forecast.

Fifth, boosting turns the idea around. Its trees are built in sequence, and each one fits what
remains. Bagging attacks variance, and boosting attacks bias.

Sixth, no tree can extrapolate beyond its training range.

Seventh, tune by cross-validation, always with `TimeSeriesSplit`, and pair boosting with early
stopping.

And the last one is that the forest beat naive because of its inputs, not its algorithm. The jump
from the seasonal naive benchmark to the random forest was large, from sixteen sixty down to eleven
fifteen, and most of it came from being handed last week's actual sales, one week ahead. Remember
that when you're tempted to reach for a more complicated model instead of a better predictor.

---

# ▶ SLIDE 27 — References

The slide lists the four sources cited today: Breiman's papers on bagging and on random forests,
Molnar's book on interpretable machine learning, and the XGBoost paper by Chen and Guestrin.

For reading, take section eight point two of I-S-L-P, which covers bagging, random forests and
boosting. Add the XGBoost tutorial
called "Introduction to Boosted Trees," which derives the regularized objective we waved at on slide
twenty.

Next week is Lecture 6, on regularization and model selection. We'll take the penalty idea that
showed up inside XGBoost and give it a lecture of its own.

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
