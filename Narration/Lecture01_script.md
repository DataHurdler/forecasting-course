# Lecture 1 — Recording Script

**ECON 8310: Business Forecasting · Introduction, Exponential Smoothing & Forecast Evaluation**

Deck: `Slides/Lecture01_ETS_Eval.pdf` (22 pages) · Measured runtime: **~33 minutes** (31–38 depending on pace)

---

## How to use this document

- **`▶ SLIDE n — Title`** marks where to advance. The number is the PDF page, so you can
  confirm you are in the right place at a glance.
- Text is written to be **spoken**, not read silently. Short sentences. Contractions are fine.
- *Italic parentheticals* are stage directions — do not read them aloud.
- Where you see **[pause]**, actually stop for a beat. On video, a one-second silence reads
  as confidence; rushing reads as nervousness.
- Nothing here is sacred. If a sentence doesn't sound like you, change it.

---

# ▶ SLIDE 1 — Title page

Welcome to ECON 8310, Business Forecasting.

I'm Zijun Luo, and over the next twelve lectures we're going to build up a toolkit that
takes you from the classical statistical methods — the ones that have been running supply
chains since the 1950s — all the way through machine learning and modern Bayesian inference.

Today is Lecture 1. We have three jobs. First, establish what forecasting actually is and
why it matters commercially. Second, build our first real family of models: exponential
smoothing. And third — and I'd argue this is the most important part of the whole course —
learn how to tell whether a forecast is any good.

That third one sounds like the boring administrative part. It isn't. It's the part that
separates people who produce forecasts from people whose forecasts get trusted.

**[pause]**

Let's look at where we're going.

---

# ▶ SLIDE 2 — Lecture Outline

Four parts today.

We start with why forecasting matters — the business framing, the notation, and the
benchmarks you must beat before anyone should take your model seriously.

Then exponential smoothing: simple exponential smoothing, Holt's method, Holt-Winters, and
the ETS framework that ties them all together.

Then forecast evaluation — the accuracy metrics and the one rule you must never break.

And we'll close with a roadmap for the semester.

---

# ▶ SLIDE 3 — Section divider: Why Forecasting Matters

*(This is a divider slide. Read the line, then move on fairly quickly — don't linger.)*

Part one. Why forecasting matters.

The framing I want you to carry all semester is this: every business decision made today
rests on some expectation about tomorrow. Forecasting is just the discipline of making that
expectation explicit, and checkable.

---

# ▶ SLIDE 4 — Forecasting Is Everywhere in Business

Look at this table. Four domains, and in each one, a question somebody is actually being
paid to answer.

Operations and supply chain. How much inventory should Walmart order for Black Friday?
How many nurses does a hospital need next Tuesday night? Get the first one wrong and you
either have empty shelves on the busiest day of the year, or a warehouse full of unsold
goods in January. Get the second one wrong and you either burn out your staff or you pay
for nurses standing around.

Finance and risk. Will credit card defaults rise next quarter? What will the Fed Funds rate
be in six months? Banks hold capital against these answers. Regulators check the answers.

Marketing and strategy. How many units will this new product sell in year one? What's the
lifetime value of this customer cohort? These numbers decide whether a product launches
at all.

And macro and public policy. Will GDP grow by two percent or three percent next year?
What will unemployment be in Q4? A one-point difference here moves billions in tax revenue.

**[pause]**

Here's what I want you to notice. Not one of these is a question about statistics. Every
single one is a *decision* waiting on a number. Somebody has to place an order, set a
capital reserve, approve a launch, write a budget — and they're stuck until the number
arrives.

That's the line at the bottom of the slide, and it's the sentence I'd most like you to
remember from today. A decision is only as good as the forecast it rests on.

Which cuts both ways. A brilliant forecast attached to no decision is an academic exercise.
And a decision resting on a lazy forecast is just a guess wearing a suit.

---

# ▶ SLIDE 5 — The Basic Setup

Let's get precise. Some notation, and we'll use it for the rest of the semester.

A **time series** is an ordered sequence of observations, y-one through y-T, where t
indexes time. The word doing the work in that definition is *ordered*. Order is what makes
this different from every other dataset you've worked with. If I shuffle the rows of a
customer database, I've lost nothing. If I shuffle a time series, I've destroyed it.

Four pieces of notation.

y-sub-t is the observed value at time t. Straightforward.

h is the **forecast horizon** — how many periods ahead we're looking.

y-hat, subscript t-plus-h given t, is our forecast of the value at time t-plus-h, made
using only the information we have at time t. That vertical bar matters enormously. It says:
here is what I knew when I made this call. We'll come back to that bar repeatedly.

And e-sub-t is the **forecast error** — what actually happened minus what we said would
happen.

Concretely: forecasting monthly retail sales, h equals one is next month, and h equals
twelve is a year out. And the problem gets harder as h grows, because uncertainty compounds.
Next month is mostly determined by conditions that already exist. Twelve months out, the
economy could turn, a competitor could enter, your supply chain could break.

**[pause]**

Now the line at the bottom is the theoretical anchor for the whole course.

Under squared-error loss, the optimal forecast is the **conditional expectation** — the
expected value of y at time t-plus-h, given everything we know at time t.

Two things worth flagging. First, "under squared-error loss." That qualifier is doing real
work. If you care about squared errors, the conditional mean is optimal. Change the loss
function and the optimal forecast changes — under absolute error, for instance, you'd want
the conditional *median*. Optimal is always optimal *with respect to something*.

Second, that script-F is the information set. Every model we build this semester is
essentially a different guess about how to turn an information set into a conditional
expectation. Exponential smoothing does it one way. Neural networks do it another. But
that's the target, all twelve lectures.

---

# ▶ SLIDE 6 — Components of a Time Series

Before modeling anything, it helps to know what you're looking at. The classical
decomposition says any series is the sum of four parts.

Trend — the long-run direction. Is this growing, shrinking, or flat over years?

Seasonality — the regular, calendar-driven pattern. December retail. Summer electricity.
Monday morning traffic. The key word is *regular*: it repeats on a known cycle.

Cycle — medium-run swings. Business cycles. These are the ones people confuse with
seasonality, so let me separate them clearly. Seasonality has a fixed, known period. Cycles
don't. A recession doesn't arrive every forty-eight months on schedule.

And irregular — the random shocks. Everything left over.

Which parts matter depends entirely on the series. Retail sales are trend plus seasonality.
Quarterly GDP is trend plus cycle. Daily stock returns are mostly irregular — which is why
predicting them is so hard. Energy demand has all four.

**[pause]**

Now read the warning box with me, because this is a mistake I see constantly.

Ignoring seasonality causes large, *systematic* errors. Not random ones — systematic. A
non-seasonal model on monthly retail data will consistently under-forecast December and
over-forecast January. Every single year.

That word "systematic" is why this matters. Random error averages out. Systematic error
doesn't. If you're wrong by the same amount in the same direction every December, you will
be short of stock in the same month every year, and no amount of data collection fixes it.
The model is structurally wrong.

---

# ▶ SLIDE 7 — Always Beat a Benchmark First

This slide is a professional standard, and I want to be blunt about it.

Before you present any forecast model to anyone, check that it beats a naive benchmark. A
model that can't beat the baseline has zero value. Not "a little value." Zero.

Four standard benchmarks in the table.

The **naive** forecast: tomorrow equals today. y-hat equals y-sub-t. Sounds stupid. It is
extremely hard to beat for asset prices, because if you could reliably beat it, you'd be
rich rather than in this class.

The **seasonal naive**: this December equals last December. Use it for any series with a
season, and it is a genuinely tough benchmark for retail.

The **mean**: forecast the historical average. Use when there's no trend and no season.

And **drift**: take the last value and extend the average historical slope. Use when there's
trend but no season.

**[pause]**

Now the line at the bottom. This is not a formality.

In the M4 Competition — one of the big open forecasting competitions, a hundred thousand
series — many machine learning models *failed to beat* simple exponential smoothing.

Sit with that. Sophisticated methods, real researchers, and they lost to a method from the
1950s that you'll learn in about ten minutes.

There are two lessons. One: complexity is not accuracy. A model isn't better because it's
harder to explain. Two: the benchmark is how you find out. Without it, you have no idea
whether your fancy model is adding anything at all.

So: benchmark first, always. And if your model loses to seasonal naive, that is a finding,
not a failure. Report it.

---

# ▶ SLIDE 8 — Section divider: Exponential Smoothing

Part two. Exponential smoothing.

The core intuition is one sentence: recent observations should carry more weight than old
ones.

---

# ▶ SLIDE 9 — The Problem with Equal Weights

Here's the problem we're solving.

Most classical methods treat every historical observation equally. Every data point gets the
same vote. That's fine if the world is stable — but what if it changes?

Take a concrete case. A retailer renovates a store, and sales shift. If your model is
equal-weighting three years of pre-renovation data, it will keep forecasting the old store —
wrong for months, in the same direction, until the new data finally outweighs the old.

**Exponential weighting** fixes this. An observation k weeks old gets weight alpha times
one-minus-alpha to the k. Each step back in time, the weight shrinks by the same factor.

*(Point at the left panel first.)*

These are real weights from the store you'll use in the lab: two aisles of Walmart's store
TX_1, with alpha fitted to each. Each bar is the weight on one past week. The vertical axis is
a log scale — each gridline is ten times the one below — and on a log scale, shrinking by the
same factor every week is a straight staircase.

On the left, household goods, alpha zero point six two: last week alone gets sixty-two percent
of the vote. On the right, food, alpha zero point two one: a much gentler staircase.

The dashed line is equal weighting: one over two hundred twenty-five, the number of training
weeks. On household goods, every week older than five gets *less* than an equal share; on
food, every week older than sixteen.

**[pause]**

Now the triangles. Think of the bars as weights on a plank. The triangle is where the plank
balances — the average age of the data the forecast is built from. Household goods: zero point
six weeks, essentially last week. Food: three point seven weeks. Equal weighting would balance
a hundred and twelve weeks back — more than two years in the past.

The weights never hit exactly zero, and they sum to one, so this is a proper weighted average.
Alpha is the forgetting rate. High alpha, short memory: you react fast and forget fast. Low
alpha, long memory: stable, but slow to notice change. Same store, two aisles, very different
memories.

Hold onto alpha. It's about to become the single parameter of our first model.

---

# ▶ SLIDE 10 — Simple Exponential Smoothing (SES)

Here's the model. One equation, written two ways.

First: the new level is alpha times the current observation plus one-minus-alpha times the
previous level. Second — the same equation rearranged, and the one to picture — the new level
is the old level plus alpha times the *gap* between what just happened, y-sub-t, and what the
old level said. Every week, the level closes the fraction alpha of the gap. Alpha is a learning
rate.

*(Point at the right-hand panel.)*

That's the picture on the right: one real week, the first week of July 2013, food sales at
Walmart's store CA_1. Sales jumped four thousand eight hundred and sixty-two units above the
old level. With alpha at zero point three two, the level moved up one thousand five hundred and
thirty-four — about a third of the gap — and kept two thirds of its old belief.

And the forecast is just y-hat equals ell-T, for every horizon h. Flat.

**[pause]**

*(Now the main panel.)*

Two years of weekly food sales in gray, then the test year, shaded. Three levels run through
it. Orange, alpha zero point eight, closes eighty percent of every gap — it chases every wiggle,
almost a copy of the data. Blue, alpha zero point one, closes ten percent: smooth, and slow —
see how long it takes to come down after the 2013 peak. Red is the alpha statsmodels picked,
zero point three two, by minimizing next-week squared errors in the training data.

Then the test year starts, and each level runs flat. That flatness is the model's honest
admission of its limits: SES has no concept of trend or season. We'll fix trend on the next
slide.

One twist worth knowing. On this test year the three flat lines land within about thirty units
of RMSE of each other, and the fitted one isn't the best. Alpha is fitted to next-week errors,
not a whole year ahead.

**[pause]**

And here's the elegant part. SES *nests both benchmarks* from slide seven. Set alpha to
exactly one, and the level becomes y-sub-t — that's the naive forecast. Let alpha go to
zero, and you get the historical mean.

So SES isn't a competitor to those benchmarks. It's a dial that spans them, and estimating
alpha is letting the data pick the point on that dial.

Use it when there's no trend and no seasonality.

---

# ▶ SLIDE 11 — Holt's Method: Adding Trend

SES gives a flat forecast. So if the series trends, we need more. Holt's insight, from 1957,
is to track two things instead of one: the level, and the slope.

Three equations in the box. The **level** equation looks like SES, except it compares against
the old level *plus* the old trend: the series was expected to move. The **trend** equation is
the same smoothing idea applied to the slope, controlled by beta-star. And the **forecast** is
level plus h times trend — a sloped line instead of a flat one.

Read ell-t as where the series *is* right now, and b-sub-t as how fast it's moving. Position
and velocity.

**[pause]**

Now the caution, and this one has cost people real money. That forecast is a straight line
extended forever. Nothing grows without limit.

*(Point at the chart.)*

Here's Holt fitted to the same food series. Today's level is twenty-one thousand two hundred
seventy-two units, and the fitted slope is seventy units a week. The red line is the
undamped forecast: plus seventy, every week, forever — by the end of a year, three thousand
six hundred more units a week than the store sells today. On the real test year, that model's
RMSE is three thousand four hundred and twenty-eight, about double SES's.

The fix is the **damped trend**, in blue. Each week ahead adds phi times the previous week's
rise. Here phi is zero point eight five, an illustrative value. The first week adds about sixty
units instead of seventy, the next about fifty, then forty-three: a staircase whose steps
shrink by the same factor each week.

**[pause]**

And a staircase like that has a top. Add up all the steps, forever, and you never pass the
dashed line: today's level plus five point seven weeks' worth of slope. That's the ceiling,
ell-T plus b-T times phi over one-minus-phi. Instead of adding thirty-six hundred units over a
year, the damped forecast adds at most about four hundred.

Empirically, damped trend is one of the strongest performers in forecasting competitions.
Use it for anything beyond about six periods. In the lab you'll fit phi yourself — and I'll ask
you to look hard at where it lands.

---

# ▶ SLIDE 12 — Holt-Winters: Level, Trend, and Seasonality

One more component. Holt-Winters adds seasonality, so now we're tracking three things.

The equations look intimidating, so let me just point at the structure. Level, trend, and
now a **seasonal** equation with a third smoothing parameter, gamma. And notice the
subscript: s-sub-t-minus-m, where m is the seasonal period — twelve for monthly data, four
for quarterly, seven for daily data with a weekly cycle. We're looking back one full season.

In the level equation, we subtract the seasonal component before updating. We de-seasonalize
first, then update. Then the forecast adds the season back in.

**[pause]**

Now the comparison — and this is a real decision you'll have to make.

**Additive** seasonality: swings are constant in size. December adds the same amount every
year, whether the store is big or small.

**Multiplicative** seasonality: swings grow with the level. December is plus thirty
*percent*, not plus a fixed number of dollars.

How do you tell? Look at the plot. If the seasonal swings get visibly bigger as the series
grows, it's multiplicative.

US retail sales are the textbook case. The December spike grows every year as overall sales
grow — so it's multiplicative. Retail and tourism usually are.

One practical constraint at the bottom: you need at least two full seasons of data to
initialize. For monthly data, that's twenty-four observations minimum. With eighteen months
of history you cannot fit a seasonal model, no matter how much you want to.

---

# ▶ SLIDE 13 — The ETS Framework: A Unified View

Now let's tie it together, because it may feel like we've collected three unrelated
techniques. We haven't. They're one family.

Hyndman and Athanasopoulos — the authors of our textbook — unify every exponential smoothing
method as a **state space model**, described by three letters: Error, Trend, Seasonal.

The **Error** can be additive or multiplicative. The **Trend** can be none, additive, or
damped. The **Seasonal** can be none, additive, or multiplicative.

Two times three times three is eighteen combinations. Three of them — additive errors with
multiplicative seasonality — are numerically unstable, because the state equations end up
dividing by numbers that can get close to zero. So the textbook drops them, which leaves
fifteen. And every method from the last three slides is one of them.

ETS(A,N,N) — additive error, no trend, no season — that's simple exponential smoothing.

ETS(A,A,N) is Holt linear. ETS(A,A-damped,N) is Holt with a damped trend.

ETS(A,A,A) is Holt-Winters additive. And Holt-Winters multiplicative is ETS(M,A,M) —
multiplicative errors to go with the multiplicative season, which keeps it out of the
unstable three.

**[pause]**

So they weren't three tricks. They were four points in a structured space, and now you can
see the whole space at once — including combinations we never discussed.

And here's the practical payoff. **Automatic selection.** You don't have to agonize over
which of the fifteen to use. Fit all fifteen, and pick the one with the minimum AIC. In
Python, the ETSModel class in statsmodels fits any one of them; the search itself is a short
loop over the fifteen that you write, and the AIC comes back with every fit.

Which raises a fair question: if the computer picks the model, why did we walk through the
equations? Because when the automatic selection returns something strange — a multiplicative
trend on a series that shouldn't have one — you need to recognize that and know why it's
wrong. Automation is a labor-saving device, not a substitute for judgment.

---

# ▶ SLIDE 14 — Section divider: Forecast Evaluation

Part three. Forecast evaluation.

And the line on this slide is one I mean literally: forecasting without evaluation is not
science. It's storytelling.

---

# ▶ SLIDE 15 — How Do We Measure Forecast Accuracy?

We have out-of-sample errors — actual minus forecast, one for each week of the test year. How
do we summarize fifty-two of them into one number?

Two classic answers, on the left. MAE is the mean absolute error: average the sizes of the
misses. RMSE is the root mean squared error: square each miss, average the squares, take the
square root. Both come out in the same units as your data — here, units of food per week.
Which makes them easy to explain to a manager: "we're off by about fourteen hundred units a
week."

*(Point at the left panel of the figure.)*

Here's the picture that tells them apart. Each test week's miss is drawn as a square, with
the size of the miss as its side. All fifty-two squares share a corner. The left panel is
SES on the food series; the right panel is seasonal naive, the same week last year.

The dashed square has the *average side*. That's MAE. The filled square has the *average
area* — its area is the average of the fifty-two squares' areas — and its side is RMSE.

**[pause]**

Now look at what that does. SES and seasonal naive have almost the same dashed square: MAE of
thirteen seventy-six against thirteen sixty-five. On MAE, they tie. But SES has a few huge
squares out at the edge — the biggest is the week of Christmas 2015, when sales fell four
thousand six hundred units below its flat forecast. Area grows with the square of the side, so
those few big squares dominate the average area. Five weeks carry forty percent of all of
SES's squared error. And so the filled square is bigger: RMSE of seventeen fifty-four against
sixteen sixty. On RMSE, seasonal naive wins.

And notice the filled square always contains the dashed one. An average of squares can never
be smaller than the square of the average. RMSE is always at least as big as MAE, and the gap
between them tells you how lumpy the misses are.

**[pause]**

So which should you use? It depends on your cost function. If big misses are especially
costly — a stockout, an understaffed shift, a blown capital reserve — prefer RMSE, because it
reflects that asymmetry. If all errors hurt roughly in proportion to size, MAE is more
representative.

That's a business judgment, not a statistical one. Ask what the error actually costs.

---

# ▶ SLIDE 16 — Comparing Across Series: MAPE and MASE

Both of those numbers come in units — units of food, dollars, patients. That means you can
only compare models on the *same* series. An RMSE of four hundred means nothing next to an
RMSE of two, if one series sells forty units a week and the other forty thousand.

To compare *across* series, you scale the errors. Two ways.

MAPE is mean absolute *percentage* error: each miss as a percentage of the actual value. Easy
to explain. But it breaks down when the actual value is near zero — divide by a tiny number
and one week can blow up the whole average. Slow-moving products, with weeks of zero sales,
make MAPE useless.

MASE is mean absolute *scaled* error. The top is the MAE you already know. The bottom is the
yardstick: how far off seasonal naive was, on average, on the *training* data — predicting
each week from the same week a year earlier.

**[pause]**

So MASE below one means your errors are smaller than seasonal naive's typical miss in the
past. Above one, you're doing worse than that — and you should say so. It's built on the
benchmark idea from slide seven, it's the standard in the M-Competitions, and it has a
property the others lack: the number carries its own yardstick. An RMSE of eight hundred
needs context. A MASE of one point two says you missed by more than last year's value used to.

One honest caution, from our own food series. Both SES *and* seasonal naive score about zero
point five nine there. Why? Because the yardstick was set in the training years, and in 2012
the store's food sales jumped twenty-nine percent — so last year was a terrible guess that
year, and the yardstick is generous. MASE below one does not mean you beat seasonal naive on
*this* test year. To compare two models on the same series, compare their MASEs to each
other.

---

# ▶ SLIDE 17 — The Golden Rule: Always Evaluate Out-of-Sample

If you take one operational rule from this lecture, take this one.

**Never evaluate a forecast model on the data used to fit it.**

In-sample fit measures how well the model memorized the past. It says almost nothing about
how well it predicts the future. And the two can point in opposite directions — a model
complex enough to fit your history perfectly will usually forecast terribly, because it has
learned the noise along with the signal.

The diagram shows the correct setup. Training set up to time T. Then the forecast origin.
Then the test set beyond it. The model sees only the training data. We judge it only on the
test data. The split is *chronological* — the test set is the future.

**[pause]**

Now the warning box, and this is the single most common technical error I see.

**Never randomly shuffle time series observations.**

In scikit-learn, use `TimeSeriesSplit`, never `KFold`. `KFold` is the default in most
tutorials, and it shuffles. On a time series, that puts future observations into the
training set and past observations into validation. The model gets to peek at the answer.

This is called data leakage, and the reason it's dangerous is that it doesn't look like an
error. Your cross-validation scores get *better*. The model looks excellent. Then it goes
into production and fails, and nobody can work out why — because the bug was in the
evaluation, not the model.

So: `TimeSeriesSplit`. Every time. This will come back in every remaining lecture.

---

# ▶ SLIDE 18 — Section divider: Course Roadmap

Last part. Let's look at where the semester goes.

---

# ▶ SLIDE 19 — ECON 8310: Four-Part Structure

Twelve lectures, four parts.

**Part one, time series models.** Today's lecture, then ARIMA and VAR in Lecture 2, then
generalized additive models in Lecture 3. This is the classical econometric toolkit.

**Part two, tree-based models.** Decision trees, then tree ensembles — random forests and
boosted trees together — then regularization and model selection. This is where we cross
into machine learning, and where feature engineering starts to matter more than the model.

**Part three, deep learning.** Neural networks in PyTorch, CNN architectures, then RNNs,
LSTMs and Transformers. We build these from the components up, not just by calling a library.

**Part four, Bayesian methods.** Foundations, then time series and hierarchical models, then
Bayesian linear regression. This is where we stop producing a single number and start
producing a full distribution — which is often what the decision actually needed.

**[pause]**

Python throughout. Statsmodels, scikit-learn, xgboost, PyTorch, prophet, pyGAM, and pymc.

One note on sequencing: the parts get harder, but not uniformly. If you come from
econometrics, part one will feel comfortable and part four will be a genuine shift in
thinking. If you come from data science, the reverse. Whichever you are, don't coast through
the familiar half — the methods build on each other, and falling behind in one week makes
the next one harder.

---

# ▶ SLIDE 20 — Tools and What You Will Be Able to Do

Quickly, the stack. Statsmodels for the classical time series models. Scikit-learn for
machine learning models, pipelines, and cross-validation. XGBoost for gradient boosting.
PyTorch for neural networks. Prophet and pyGAM for additive models. And pymc for Bayesian
work.

Get these installed before the next class. There's a test script in the course repository —
run it, and if something fails, email me now rather than the night before Assignment 1.

**[pause]**

And here's what I'm actually promising. By the end of the semester you'll be able to choose
the right method for a given business problem — which is mostly about knowing what each one
assumes. Implement and tune it in Python. Evaluate it rigorously out-of-sample. Interpret
the result for someone non-technical. Use AI tools effectively to speed the work up. And
avoid the most common mistakes, several of which we've already named today.

Notice that only one of those six is about writing code. The code is the easy part now. The
judgment is what you're here for.

---

# ▶ SLIDE 21 — Lecture 1: Key Takeaways

Five things to carry out of today.

**One.** Forecasting underpins every forward-looking business decision, and under
squared-error loss the optimal point forecast is the conditional expectation.

**Two.** Always beat a benchmark first. Naive, seasonal naive, mean, drift. If you can't
beat them, you have nothing — and you should say so.

**Three.** Exponential smoothing assigns geometrically declining weights to past
observations. Alpha controls how fast old data is forgotten.

**Four.** SES gives you level. Holt adds trend. Holt-Winters adds seasonality. The ETS
framework unifies all fifteen variants, and AIC selects among them automatically.

**Five.** Evaluate out-of-sample only, and never shuffle time series data. RMSE punishes
large errors; MASE scales by seasonal naive's misses, so it compares across series.

**[pause]**

If you remember only two of those, make it number two and number five. Benchmark first,
evaluate honestly. Every method in the remaining eleven lectures is judged by those two
standards.

---

# ▶ SLIDE 22 — References

Readings for this week are in the syllabus. For today, that's chapters one, two, five, and
eight of *Forecasting: Principles and Practice* — either edition works for this material,
the chapter numbers are identical.

Next time: ARIMA and VAR. We'll relax the idea that the past enters only through a smoothed
level, and start modeling the correlation structure of the series directly.

See you then.

---

## Timing guide

Measured from the script (2026-10-13, after the figures were added): **4,646 spoken words**.
At 140 wpm — a normal lecture pace — that is **~33 minutes**; 38 if you read deliberately, 31
if you push.

| Segment | Slides | Words | Runtime |
|---|---|---:|---:|
| Opening & why forecasting matters | 1–7 | 1,391 | ~10 min |
| Exponential smoothing | 8–13 | 1,685 | ~12 min |
| Forecast evaluation | 14–17 | 973 | ~7 min |
| Roadmap & close | 18–22 | 597 | ~4 min |
| **Total** | | **4,646** | **~33 min** |

Content slides run 65–165 seconds each; section dividers 4–17 seconds.

If you're running long, slides 4 and 20 compress most easily — the domain examples and the
tool list both survive trimming. Do **not** compress slides 7, 15, 16, or 17; those carry the
standards the rest of the course depends on.

**If you need this longer**, the script is deliberately lean and there are three honest ways
to extend it, in rough order of value:

1. **Work a numeric example on screen.** Slide 10 turns the SES crank for one real week;
   slides 11–12 (Holt, Holt-Winters) state their update equations without turning one.
   Walking through three periods of a Holt update by hand adds 4–6 minutes and is where the
   intuition actually lands.
2. **Live-code the ETS fit.** Slide 13 mentions `ETSModel` in one sentence. Fitting it to a
   real series on camera, and showing what automatic selection picks, adds 6–10 minutes.
3. **Demonstrate the leakage failure.** Slide 17 asserts that `KFold` flatters a model.
   Showing the inflated CV score next to the honest one makes the warning land far harder
   than stating it. Adds 4–5 minutes.

All three add *demonstration*, not more talking — which is the right way to lengthen a
video lecture.
