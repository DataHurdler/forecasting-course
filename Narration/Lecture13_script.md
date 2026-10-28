# Lecture 13 — Recording Script

**ECON 8310: Business Forecasting · Synthesis: What the Results Taught Us**

Deck: `Slides/Lecture13_Synthesis.pdf` (22 pages) · Measured runtime: see the timing guide

---

## How to use this document

- **`▶ SLIDE n — Title`** marks where to advance. The number is the PDF page.
- *Italic parentheticals* are stage directions. **[pause]** means stop for a beat.
- Week 15 on the syllabus. **Deliberately has no lab** — the session is this synthesis followed
  by project work time, so budget for questions about their projects.
- Slides 14 and 15 are the heart of it. The split — bugs in the code versus a correct program
  answering the wrong question — is the most durable thing in the hour.

---

# ▶ SLIDE 1 — Title page

This is Lecture 13, the synthesis, and its subject is what the results taught us.

Twelve lectures, four families of method and one dataset have left us with a scoreboard that
almost nobody would have predicted in Week 1.

Today isn't new material. It's the session where the fourteen weeks become one argument, and
where I want to hand you the handful of things that will still be true when every library in this
course has been replaced.

---

# ▶ SLIDE 2 — Lecture Outline

We'll take this in five parts. First, what we built, and how the four parts of the course relate
to each other. Second, what the results actually taught us, which comes down to three lessons,
each one paid for with a measurement. Third, how to choose a method. Fourth, how to do the work
honestly, which is the longest section. And last, where to go next.

---

# ▶ SLIDE 3 — Section divider: What We Built

This first section looks back at what we built: fourteen weeks, four families of method, and one
dataset carried through all of them.

---

# ▶ SLIDE 4 — The Arc of the Course

*(Walk the table, then land the reframing underneath it.)*

The course came in four parts. The first was classical time series, Lectures 1 to 3, where the
idea was to **specify a structure**, meaning level, trend and season, and then estimate it. The
second was trees, Lectures 4 to 6, where we stopped specifying. You handed an algorithm features
and let it find the structure. The third was deep learning, Lectures 7 to 9, where we let the
architecture encode what kind of structure to expect. And the fourth was Bayesian, Lectures 10 to
12, where we changed the deliverable to a distribution instead of a number.

**[pause]**

Now, read down that last column again, because there's something there.

This course isn't really four toolkits. It's four answers to a single question: **where does the
knowledge in a forecast come from?**

It can come from you, encoded in the structure of the model. It can come from the data, through
a search procedure. It can come from the built-in assumptions of an architecture. Or it can come
from a prior, stated openly and then updated.

That framing is what survives when the libraries change. Every method you meet for the rest of
your career sits somewhere on that spectrum, and knowing where it sits tells you what it will be
good at and where it will fail.

---

# ▶ SLIDE 5 — The Scoreboard

*(Picture first, then the table. Take your time; this is the evidence for everything after it.)*

Here is every model in this course on the same M5 weekly panel, with the same thirty series and
the same held-out block.

*(Point at the dots.)*

There's one axis, and it starts at zero. Each dot is one model's test RMSE. The blue dots came
from Lectures 4 to 6, on trees and regularization. The red dots came from Lectures 7 to 9, on
neural networks. And the dashed line on the right is the seasonal-naive benchmark, which simply
repeats the same week last year. It scores twenty-one fifty-two.

**[pause]**

Look at the shape before you read a single number. Seven models, built over seven weeks with four
completely different philosophies, all land inside that gray band, between seven forty-four and
one thousand thirty-two. That's a spread of under three hundred. And every one of them sits more
than eleven hundred away from the benchmark.

*(Now the table, which says where each number came from.)*

The LASSO with forty-six engineered features scores **seven forty-four**, from Homework 4, on
XGBoost and regularization. XGBoost scores seven eighty-one, also from Homework 4. That number
came from an earlier version of xgboost, and the version pinned for this course gives seven
ninety-two, which still leaves it in second place. The vanilla RNN scores eight-oh-five, and the
random forest from Homework 3 scores eight ninety-nine. The Transformer encoder scores nine
ninety, the LSTM scores nine ninety-eight, and the 1D CNN scores one thousand thirty-two.

There's one honest footnote on that list. The three recurrent and attention models all read
fifty-two weeks of history. The CNN number comes from Lab 8, the lab for the CNN lecture, where
it read twenty-six weeks, and the table says so. Lab 8 also found that handing the CNN a longer
window made it worse, not better, so the label is a caveat, not an excuse.

**[pause]**

*(Now the reading.)*

That ordering isn't the one anyone predicts in Week 1.

A linear model with **forty-six coefficients** is at the top. It sits above gradient boosting,
and above every neural architecture we built.

And the two architectures with the most attention and money behind them, the Transformer and the
LSTM, finished **fifth and sixth of eight**, behind a plain RNN with a fraction of their
parameters.

Now, notice the last clause on the slide: at the training budget each was measured at. Lecture 9
Part 1, on RNNs and LSTMs, showed why that matters. With twenty-six-week windows, if you give the
LSTM a hundred epochs instead of thirty, it catches the RNN, and the two tie. So the RNN's real
edge is speed. That's a different claim from saying the RNN is a better model, and the slide is
careful to make only the first one.

I want to be careful about what the rest of the ranking does and doesn't mean, too, because the
wrong lesson is easy to take. It doesn't mean Transformers are overrated in general. They
reorganized an entire field. It means they're **mismatched to this problem**, and the rest of
this lecture is about learning to see that mismatch in advance, rather than after nine hours of
fitting.

---

# ▶ SLIDE 6 — Section divider: What the Results Taught

The next section covers three lessons, and each one came out of a measurement rather than out of
anyone's opinion.

---

# ▶ SLIDE 7 — Lesson 1: The Benchmark Is a Real Competitor

Let's start with the lesson that should make you the most uncomfortable.

In Homework 1, on exponential smoothing and ARIMA, you forecast the whole test year from one
starting point. And the seasonal-naive benchmark, which is the same week last year and one line of
code, **beat every exponential smoothing and ARIMA model the assignment asked you to fit.**

That wasn't a trick question or a rigged setup. It's simply what the data supports at that
horizon, and the reason is something you can count. Holt-Winters had to estimate fifty-two
seasonal indices from four point three annual cycles of training data, which is about four
observations per parameter. Seasonal naive estimates nothing. And the level it carries barely
moves. Taking each calendar year against the one before, the change was plus three point seven
percent, then plus one point three, then minus four point four, then plus two point nine. Those
are the years since twenty twelve. The year before that, from twenty eleven to twenty twelve, the
series jumped almost twenty-nine percent, and that's why I say "since twenty twelve." The claim
is about the years the benchmark actually had to copy.

**[pause]**

*(Point at the figure.)*

Every bar in the figure is the same calculation, and the slide calls it skill. It's one minus the
model's RMSE divided by the seasonal naive's RMSE, on the same series and the same test period. A
bar to the right of zero beats the benchmark, and a bar to the left of zero loses to it. Each
group also has its **own** benchmark: sixteen hundred sixty units a week in the top group, four
hundred five a day in the middle one, and twenty-one fifty-two a week across thirty series at the
bottom. So read the signs and the rough sizes, and never compare one group's RMSE with another's.

*(Point at the top group.)*

The top group is Homework 1, on CA_1 FOODS. Holt-Winters comes in at minus eight percent, and SES
at minus six. ARIMA two-one-zero, the best of the seven models the assignment asked for, is at
minus three. Only three of the seven are drawn, and the rest sit further left. Damped Holt is at
minus sixteen, the other two ARIMAs are at minus seven and minus twelve, and undamped Holt is at
minus ninety, which would have squashed every other bar on the slide.

Under them is the Bayesian structural model from Lecture 11 Part 1, on Bayesian time series, with
the same series, the same test year and the same benchmark. It comes in at **plus four percent**.
It crosses the line, but not by much.

*(Point at the middle group.)*

The middle group is Homework 2, on GAMs, which used daily data and forecast a whole year ahead.
Prophet with the SNAP and event regressors comes in at **plus twelve**, and pyGAM at plus nine.
On the year-ahead questions in this figure, twelve percent is the best margin anyone got.

*(Point at the bottom group.)*

Then the bars get long. The random forest from Homework 3 is at plus fifty-eight, and the LASSO
with forty-six engineered features from Homework 4 is at **plus sixty-five**. But read the group's
header before you read its bars. These models see last week's sales and forecast **one week**
ahead, not a year. If you ask Homework 1's own ARIMA that easier question, refitting it every week
as in the walk-forward part of the assignment, then on the same test year it goes from minus three
to plus twenty-six. And on this panel, last year is a weaker guide. The seasonal naive misses by
twenty-three percent of a typical week here, against eight percent on CA_1 FOODS. So the long bars
come partly from an easier question and partly from a weaker benchmark, and that's exactly why
each group gets its own zero line.

*(If someone asks.)*

You might wonder about two Holt-Winters variants the assignment didn't ask for, one with a damped
trend and one with no trend at all. Both of them do beat the benchmark, by about six and seven
percent. Both stop extrapolating the trend and keep the annual pattern, which is the slowdown from
Homework 1 again. A student who fitted one and said so was right.

**[pause]**

*(Point at the key box.)*

So, compute the benchmark first, every time.

It costs one line of code, it's often close, and it's the only number that tells you whether the
rest of your work was worth doing.

*(Then make it about them.)*

This is also the cheapest professional habit you can carry out of this course, and the one most
likely to set you apart early in a job. A forecast presented without a benchmark is a number
nobody can evaluate, and that includes you, even if you're the one who built it.

If you take one thing from fourteen weeks, take that.

---

# ▶ SLIDE 8 — Lesson 2: More Sophisticated Did Not Mean Better

Four times in this course, the more advanced method failed to win. And each time, we could **name
the reason**, which is the part that matters.

*(Walk the table.)*

The first time, the LSTM lost to a plain RNN. It lost at three window lengths, at a fixed
thirty-epoch budget. At twenty-six weeks, given a hundred epochs, the two tie, so even the honest
version of this result is "no gain," not "the LSTM is worse." Gating fixes long-range forgetting,
and ninety-one percent of this panel's variance is between series, so there's very little to
forget. The LSTM had four times the parameters, and they bought no gain.

The second time, the Transformer lost to an RNN. It had sixty-seven thousand three hundred
twenty-nine parameters, about fifteen times as many as the RNN. Its advantages are advantages
**at scale**, and our data isn't scale. We trained on five thousand one hundred sixty windows, each
fifty-two weeks long, but neighboring windows overlap by ninety-eight percent. Together they hold only about a
hundred effectively independent blocks of fifty-two weeks, which is roughly a hundred years of
data.

The third time, the CNNs lost to the LASSO, because their twenty-six-week window held no annual
lag, and the LASSO was handed one as a column.

And the last time, tuning bought nothing measurable: nine fits produced a three-unit change.

*(Be precise here, because three units is not a result.)*

Both of those numbers came from a stochastic fit, because the assignment sets `subsample` and
`colsample_bytree` to zero point eight. Fixing `random_state` at forty-two makes each number
reproducible, but it does **not** make their difference meaningful. Two different hyperparameter
settings at the same seed draw entirely different subsamples. If you vary only the seed, at one
fixed configuration on this panel, the spread is **twenty to ninety** in RMSE. A three-unit gap
fits inside that many times over.

So the lesson isn't that tuning hurt. It's that nine fits bought nothing you could measure:
cross-validation improved, and the test score didn't. XGBoost's defaults are well chosen for
tabular data, and a grid search that improves cross-validation doesn't have to improve the test
score.

**[pause]**

*(Point at the disclaimer, and mean it.)*

None of this means the losing method is bad. Every one of them solves a problem this data doesn't
have. LSTMs earn their gates on long chains of dependency, and Transformers earn their parameters
on large corpora. Those are real capabilities, and they simply weren't needed here.

---

# ▶ SLIDE 9 — Lesson 2, Continued: A Problem This Data Does Not Have

*(Two panels from Lecture 9 Part 1, with the same data and the same numbers.)*

Take the first row of that table and look at it, because "a problem this data does not have" is a
claim you can draw.

*(Point at the left panel.)*

This is the panel the recurrent networks were trained on, with all thirty store-and-category
series on one axis. Foods are up top, household is in the middle, and hobbies run along the
bottom. The dashed line is the grand mean, at about eight thousand units a week. The big
differences are between the lines, not along them.

*(Point at the triangle.)*

The triangle tells you how big those differences are. Each week's distance from the grand mean
splits into two pieces, the series' own level and that week's wiggle around it, and the squares
add exactly. It's Pythagoras, for variances. The between-series side is about six thousand eight
hundred, the within-series side is about two thousand one hundred, and the total is about seven
thousand one hundred. Between squared over total squared comes to **ninety-one percent**.

So what does this panel ask a model to remember? Mostly, it asks which series the model is
looking at, and the last few weeks tell it that. There's very little for a gate to keep. The
Transformer's version of the same story is the size of the data. Its parameters pay off on large
corpora, and we had about a hundred independent years.

*(Point at the key box.)*

Match the method to the **structure of the problem**, not to the recency of the paper.

And notice the standard I'm holding these results to. It isn't "the fancy model lost, so fancy
models are bad." It's "the fancy model lost, and we can say precisely which of its assumptions
didn't apply." That second statement is worth something. The first is just cynicism.

---

# ▶ SLIDE 10 — Lesson 3: One Idea Appeared Six Times

*(This is the intellectual spine of the course. Deliver it slowly.)*

Here's the lesson I most want you to leave with.

*(Walk the table.)*

In Lecture 3, on GAMs, it was the smoothing penalty, which penalizes the wiggliness of a spline.
In Lecture 4, on decision trees, it was cost-complexity pruning, which penalizes the number of
leaves. In Lecture 5, on tree ensembles, it was `reg_lambda` and `gamma`, which penalize the leaf
weights and the leaf count. In Lecture 6, on regularization, it was Ridge, LASSO and Elastic Net,
which penalize the size of the coefficients. In Lecture 7, on neural networks, it was weight decay
and dropout, which penalize the network's weights. And in Lecture 11 Part 2, on hierarchical
models, it was the hierarchical prior, which penalizes distance from the group mean.

**[pause]**

That's six model families that look nothing like each other: splines, trees, boosted trees,
linear models, neural networks and a Bayesian hierarchy.

Underneath all of them is one bargain: **fit against complexity, with a single parameter setting
the exchange rate.**

*(Point at the distinction.)*

Five of those six tune that parameter by cross-validation, which means you try values and pick
one. The sixth, the hierarchical prior, **estimates** it, with a posterior and its own
uncertainty. That makes it the cleanest version of the idea in the course.

**[pause]**

*(Land the transferability.)*

This is what to remember when the tools change. And they will change. Most of the specific
libraries in this course will look dated within a decade.

The trade between fit and complexity won't. When you meet a method none of us have heard of yet,
one of the first useful questions to ask is where its complexity penalty is, and what sets it.

---

# ▶ SLIDE 11 — Section divider: Choosing a Method

Next comes a decision guide, built from what we actually measured.

---

# ▶ SLIDE 12 — Which Method, When

*(A practical slide. Walk it briskly; students will photograph it.)*

If you have one series with clear seasonality and a short history, start with ETS or ARIMA, and
check the benchmark.

If you have several seasonal cycles at once, start with a GAM or Prophet.

If you have many engineered features in tabular data, start with **regularized regression**, and
then try XGBoost.

If you have many parallel series with shared drivers, start with gradient boosting, using
identifiers for the series.

If you have genuinely long dependencies and lots of data, reach for recurrent or attention
architectures.

If you have few observations per group, use a hierarchical Bayesian model, with partial pooling.

And if the decision needs a probability, go Bayesian, for the predictive distribution.

**[pause]**

*(Two notes on the table.)*

There are two things to notice about that list.

"Regularized regression first" isn't a concession, or an option for beginners. It **won our
largest comparison**, and its coefficients can be read aloud in a meeting. Start there, and make
the complicated model prove it deserves to replace it.

And the last row is the only one where the **question changes**, rather than the accuracy.
Everything above it is a horse race on RMSE. That row is about needing a different kind of
answer, which is what Lectures 10 through 12, the Bayesian part of the course, were really for.

---

# ▶ SLIDE 13 — Section divider: Doing It Honestly

This section is about doing the work honestly, because the errors in this field don't announce
themselves.

---

# ▶ SLIDE 14 — Errors That Fail Silently I: the Pipeline

Every error on this slide produces a confident, plausible, **wrong** answer, with no error
message. And every one of them is a bug in the code.

*(Point at the picture, left half first.)*

Start with the first one, because it's the one you can see. On the left is walk-forward
validation, which is `TimeSeriesSplit`. It's the picture from Lecture 1, on forecast evaluation,
drawn three times over. Gray is training, red is the test block, and every fold trains only on
weeks *before* the block it's scored on. The dotted boxes are weeks that fold never touches.

*(Point at the right half.)*

On the right is plain `KFold`. It cuts the timeline into folds and holds each one out in turn.
Now, look at the hatched boxes. Those are weeks *after* the test block, sitting in the training
set. The model learns from the future and is then graded on the past. Your cross-validation error
looks excellent, and your forecast fails. And notice the bottom row, where the last fold happens
to be honest. One clean fold proves nothing about the others.

*(Walk the rest of the table.)*

The second error is scaling outside the fold, which lets validation statistics leak into
training. It's a smaller effect, but it's the same class of error.

The third is leaving rolling features unshifted. A window that includes the current week leaks
the target into its own predictor.

The last is taking MDI importance at face value. In Homework 3, on trees and random forests,
**pure noise ranked seventeenth of twenty-nine**, above a real feature, purely because it had many
distinct values.

**[pause]**

*(Point at the last line.)*

The common thread is that the failure isn't a crash. It's a number that looks fine.

That's exactly why the discipline has to be **procedural**. You split by time, you scale inside
the fold, you shift the window, and you plant a control.

You do those things because they're on a list, not because something broke. Nothing is ever going
to break.

---

# ▶ SLIDE 15 — Errors That Fail Silently II: the Code Was Fine

*(This is the most important slide in the deck. Slow right down.)*

Now comes the harder category, and it's the one that's far more common in professional work.

Here, **nothing is wrong with your program.** The model fits, the sampler converges, and every
diagnostic passes. And the answer is still wrong.

*(Walk the four rows deliberately.)*

The first row is pooling groups that are not exchangeable. The lab for Lecture 11 Part 2, on
hierarchical models, pooled ten FOODS series with twenty series that ignore SNAP. The shrinkage
was textbook and the diagnostics were clean, and it reported an effect of plus zero point zero
one one, where the held-out weeks said plus zero point zero eight nine.

The second row is standardizing a coefficient that you then interpret. In Lecture 12, on Bayesian
regression, if you divide log price by its standard deviation, `b_price` stops being an
elasticity. The fit is identical, the diagnostics are identical, and the five percent price
scenario moves by a factor of **twenty-two**.

The third row is omitting a confounder. The naive elasticity was six times too large, and it came
*with an interval half the width* of the correct one. Precision is not accuracy.

The fourth row is ignoring R-hat and divergences. Our centered fits in Lecture 11 looked fine, and
they weren't. In Part 1, on Bayesian time series, the centered structural model handed back a
forecast with an R-hat of one point oh two two. In Part 2, the centered hierarchy passed R-hat and
ESS, and it failed only on its forty-three divergences.

**[pause]**

*(Now the line the whole deck is built toward.)*

Look at that list, and ask which of these a diagnostic would have caught.

**Only the last one would.**

The first three were caught by held-out data, by a DAG drawn before the model, and by asking what
one unit of a variable actually means. None of those is something a library will run for you.
There's no `check_exchangeability()` function, and there's no warning when your coefficient stops
being an elasticity.

That's the difference between knowing how to call a method and knowing what it assumes. Fourteen
weeks of this course have been aimed at the second one.

---

# ▶ SLIDE 16 — The Day Every Model in This Course Fails

*(Figure up. This is the last new result of the course, so give it room.)*

There's one more failure, and it isn't on the previous slide's list, because it isn't a mistake
anybody made. It's a limit.

Every twenty-fifth of December, the stores close, and recorded sales are **exactly zero**. That
happens five times in our sample, against a daily average of about twenty-eight hundred.

*(Point at the left panel.)*

This is CA_1 FOODS, daily, from the eighteenth of December to the first of January, with four
Christmases, twenty twelve through twenty fifteen, laid on top of each other. The black lines are
what actually sold. Every one of them dives to zero on the twenty-fifth, at that black dot at the
bottom.

The dashed lines are the seasonal-naive forecast, which is the same weekday one year earlier. On
the twenty-fifth they sit at twenty-two thirty, twenty forty-five, twenty-four eighty-five and
twenty-three sixty-five. The red bars are the errors. Each one is a hundred percent of the
forecast, every time, on a date that was known years in advance.

*(Point at the hollow circles.)*

Now, look at what the dashed lines do the day before. They dive to zero too. That's last year's
closure, replayed, because fifty-two weeks back from the twenty-fourth lands on last year's
Christmas. In twenty twelve it lands on the twenty-third, because twenty twelve was a leap year.
So the benchmark pays twice: once for not knowing the store is closed, and once for thinking it's
closed when it isn't. That echo costs another seven percent of the squared error on its own.

**[pause]**

*(Point at the right panel.)*

Here's the picture that makes this more than an anecdote. A squared error is an area. The small
gray square is an average day, and its side is the benchmark's RMSE over the whole evaluation
window, which is five thirty-seven. Each red square's side is one Christmas error. Count the gray
squares that fit inside each red one, and you get seventeen, fifteen, twenty-one and nineteen. On
average, **one closed day weighs as much as eighteen ordinary days**.

That's why four days, which are **one quarter of one percent** of the evaluation window, carry
**four point six percent** of the seasonal naive's squared error. That's eighteen times their
share. Drop those four days out of fifteen hundred seventy-seven, and the benchmark's RMSE falls
from five thirty-seven to five twenty-five, without changing a single thing about the model.

So why did nobody catch it? It wasn't for want of information.

The `is_event` flag fires on **a hundred and fifty-eight** days in this data, and **five** of them
are closures. Every model in this course saw that flag: the trees that split on it, the networks
that took it as a channel, and the Bayesian regressions that gave it a coefficient. And the flag
lumps a locked front door together with a hundred and fifty-three Super Bowls and Easters.

The feature couldn't express the difference, so no model could learn it.

**[pause]**

*(Point at the key box.)*

That's the honest shape of this failure. It was knowable years ahead. Nothing was missing from the
world, but something was missing from the **encoding**. The model wasn't misspecified. It was asked
a question in a vocabulary that had no word for "closed."

The harder case is a shock that nobody could have known about at all. On the sixteenth of April,
twenty sixteen, a magnitude seven point eight earthquake struck Ecuador, and sales at a large
grocery chain there spiked for weeks as people bought water and relief supplies. A forecast made on
the fifteenth, by any method in this course, is catastrophically wrong for the following month.
That's not because the model was bad. It's because the future stopped resembling the past, which
is the one assumption every method here shares.

That chain's sales data is available if your project wants a real structural break rather than a
simulated one.

Knowing where the whole toolkit stops working is part of knowing how to use it.

---

# ▶ SLIDE 17 — What a Defensible Forecast Report Contains

So what does good work actually look like when you hand it to someone?

*(Walk the table.)*

It starts with a benchmark that you computed, not one you cited, and every model is measured
against it.

It uses a time-aware split, either walk-forward or a held-out final block, and never a random one.

It has more than one evaluation, because a single test split can flatter a model. Homework 4
ranked the models the same way twice, and that's why that ranking is believable.

It reports the result that went against you, including the tuning that didn't help and the model
you expected to win.

And it gives an interpretation in the units of the business, rather than of the loss function.

**[pause]**

*(Point at the fourth row.)*

That fourth row is the one that distinguishes professional work, and it's the one that feels
counterintuitive when you're junior.

A report where **every** result confirms the author's expectation isn't a report anyone should
act on. Either the author got lucky, or they stopped looking, or they aren't telling you
everything. Reporting what went against you is what makes the rest of it credible.

*(Point at the last paragraph.)*

And you've practiced this all semester, whether or not it felt like it. The assignments were built
so that the obvious answer is frequently the wrong one, and the credit was for **noticing**, not
for getting the flattering number.

---

# ▶ SLIDE 18 — Where You Do This Yourself

That brings us to the final project, because it asks for exactly the report on the previous slide,
on data of your choosing.

*(Walk the requirements.)*

First, it needs three methods, from at least two parts of the course, and the four parts are time
series, trees, deep learning and Bayesian. Three variations on one family don't satisfy this. The
comparison has to be between approaches that differ *in kind*.

Second, it needs a benchmark on top of those three. The benchmark doesn't count toward the three,
and it isn't optional. Lesson 1, from earlier today, is why.

Third, it needs walk-forward validation for every comparison. That means no random split, and no
single split done once.

Fourth, it needs a recommendation that a non-specialist can act on, in the units of the business,
with the uncertainty attached.

And last, it needs a methods reflection: which model won, and what it was about *your* data that
made it win. This is where the course is actually assessed.

**[pause]**

*(Point at the last row and make the standard explicit.)*

That last row isn't a formality, and it's where the grade actually lives.

Every scoreboard in this course came with a **reason**. The LSTM lost because gating solves a
problem this data doesn't have. The CNN lost because its twenty-six-week window held no annual
lag. The Transformer lost because its advantages are advantages at scale.

A project that reports a ranking without a reason has done the computation, but not the work.

---

# ▶ SLIDE 19 — Section divider: Where to Go Next

To finish, here's what we left out, and what to read next.

---

# ▶ SLIDE 20 — What This Course Did Not Cover

Here are some honest gaps, so that you know what you don't know.

*(Walk the table.)*

The first is hierarchical reconciliation, which means making store forecasts add up to regional
totals. It's genuinely different from the hierarchy in Lecture 11 Part 2, and Chapter 11 of
Forecasting: Principles and Practice covers it.

The second is intermittent demand, meaning series that are mostly zeros. That's the territory of
Croston's method and its relatives. It's very common with spare parts and slow-moving products,
and none of our methods handle it well.

The third is structural breaks, which is what happens when the future stops resembling the past,
and every method fails together.

The fourth is causal inference. The DAGs told us that confounders exist, but estimating causal
effects credibly is a separate course.

The fifth is foundation models, which are pretrained forecasters applied zero-shot, meaning to
series they were never trained on. That area is moving fast, and Chapter 15 of FPP-Py, the Python
edition of the same textbook, is a starting point.

**[pause]**

*(Point at the photo.)*

You've seen this photo before. It was the input image for the convolution filters in Lecture 8,
on CNNs. It's a grocery aisle in Washington, DC, on the eighteenth of March, twenty twenty, five
days after the national emergency was declared. Lecture 8 promised we'd come back to it. The
shelves are empty because demand jumped overnight, and every forecast for that aisle was built on
weeks that had stopped being any guide.

*(Point at the third row, and close hard on it.)*

The third row is the one most likely to matter in your career, so let me be explicit about why.

**Every model in this course assumes the future resembles the past.** Every single one does. And
when that stops being true, because of a pandemic, a competitor entering the market, a supply
shock or a change in pricing, the model doesn't tell you.

Worse, **the benchmark fails at exactly the same moment**, so your usual safety net goes with it.
Your forecast is wrong, your comparison is wrong, and everything still looks internally
consistent.

Detecting that early is a field of its own, and it's worth knowing that the field exists.

---

# ▶ SLIDE 21 — Closing

*(Seven points, at a steady pace. This is the last thing they hear.)*

Let me leave you with seven points.

First, **compute the benchmark** before anything else. It's often close, and it's the only number
that tells you whether your work was worth doing.

Second, **match the method to the problem**, not to the date the paper was published. The most
advanced model on our data, the Transformer, finished fifth.

Third, remember that **one idea recurs**: fit against complexity, with a parameter setting the
rate. We saw it six times, under six names.

Fourth, **split by time, always.** The leakage errors in this field don't crash. They return a
better-looking number.

Fifth, remember that **diagnostics check the model you wrote, not whether it was the right one.**
Convergence, R-hat and ESS all passed on the two worst answers this course produced. Held-out data
caught one of them, and a DAG drawn before the model caught the other.

Sixth, **report what went against you.** It's the part that makes the rest credible.

And the last one is to **know what your forecast is for.** A point estimate and a predictive
distribution answer different questions, and only one of them supports a decision under
uncertainty.

**[pause]**

*(The last line. Say it plainly and stop.)*

The libraries will change, but the habits are the transferable part.

Thank you, and good luck with the project.

---

# ▶ SLIDE 22 — References

*(Advance and close. No narration needed.)*
