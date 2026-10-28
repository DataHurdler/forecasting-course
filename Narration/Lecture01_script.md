# Lecture 1 — Recording Script

**ECON 8310: Business Forecasting · Introduction, Exponential Smoothing & Forecast Evaluation**

Deck: `Slides/Lecture01_ETS_Eval.pdf` (22 pages) · Measured runtime: **~36 minutes** (33–41 depending on pace)

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

I'm Zijun Luo. Over twelve method lectures and a closing synthesis, we're going to build up a
toolkit that takes you from the classical statistical methods, the ones that have been running
supply chains since the nineteen fifties, all the way through machine learning and modern Bayesian
inference.

This is Lecture 1, and today we have three jobs. The first is to establish what forecasting
actually is, and why it matters commercially. The second is to build our first real family of
models, which is exponential smoothing. And the third, which I'd argue is the most important part
of the whole course, is to learn how to tell whether a forecast is any good.

That third job sounds like the boring administrative part, but it isn't. It's the part that
separates people who produce forecasts from people whose forecasts get trusted.

**[pause]**

So let's look at where we're going today.

---

# ▶ SLIDE 2 — Lecture Outline

We'll take today in four parts. We start with why forecasting matters: the business framing, the
notation we'll use all semester, and the benchmarks you must beat before anyone should take your
model seriously.

Second comes exponential smoothing. That means simple exponential smoothing, Holt's method,
Holt-Winters, and the ETS framework that ties them all together.

Third is forecast evaluation, which covers the accuracy metrics and the one rule you must never
break.

And we'll close with a roadmap for the semester.

---

# ▶ SLIDE 3 — Section divider: Why Forecasting Matters

*(This is a divider slide. Read the line, then move on quickly.)*

Let's begin with part one, on why forecasting matters.

Here's the framing I want you to carry all semester. Every business decision made today depends on
some expectation about tomorrow. Forecasting is simply the discipline of making that expectation
explicit, and checkable.

---

# ▶ SLIDE 4 — Forecasting Is Everywhere in Business

Look at this table. It covers four domains, and in each one there's a question somebody is
actually being paid to answer.

The first domain is operations and supply chain. How much inventory should Walmart order for Black
Friday? How many nurses does a hospital need next Tuesday night? Get the first one wrong, and you
either have empty shelves on the busiest day of the year or a warehouse full of unsold goods in
January. Get the second one wrong, and you either burn out your staff or pay for nurses standing
around.

Next is finance and risk. Will credit card defaults rise next quarter? What will the Fed Funds
rate be in six months? Banks hold capital against these answers, and regulators check them.

Then there's marketing and strategy. How many units will this new product sell in its first year?
What's the lifetime value of this customer cohort? Numbers like these decide whether a product
launches at all.

And last, there's macro and public policy. Will GDP grow by two percent or by three percent next
year? What will unemployment be in the fourth quarter? A one-point difference here moves billions
in tax revenue.

**[pause]**

Here's what I want you to notice. Not one of these is a question about statistics. Every single
one is a *decision* waiting on a number. Somebody has to place an order, set a capital reserve,
approve a launch or write a budget, and they're stuck until the number arrives.

That's the line at the bottom of the slide, and it's the sentence I'd most like you to remember
from today: a decision is only as good as the forecast it rests on.

And that cuts both ways. A brilliant forecast attached to no decision is an academic exercise. A
decision resting on a lazy forecast is just a guess wearing a suit.

---

# ▶ SLIDE 5 — The Basic Setup

Now let's get precise. Here's some notation, and we'll use it for the rest of the semester.

A **time series** is an ordered sequence of observations, y-one through y-T, where t indexes time.
The word doing the work in that definition is *ordered*. Order is what makes this different from
every other dataset you've worked with. If I shuffle the rows of a customer database, I've lost
nothing. If I shuffle a time series, I've destroyed it.

There are four pieces of notation on the slide.

The first is y-sub-t, which is simply the observed value at time t.

The second is h, the **forecast horizon**, which is how many periods ahead we're looking.

The third is y-hat, subscript t-plus-h given t. That's our forecast of the value at time
t-plus-h, made using only the information we have at time t. That vertical bar matters
enormously. It says, here is what I knew when I made this call. We'll come back to that bar again
and again.

And the fourth is e-sub-t, the forecast error, which is what actually happened minus what we said
would happen.

To make the horizon concrete, suppose we're forecasting monthly retail sales. Then h equals one is
next month, and h equals twelve is a year out. The problem gets harder as h grows, because
uncertainty compounds. Next month is mostly determined by conditions that already exist. Twelve
months out, the economy could turn, a competitor could enter, or your supply chain could break.

**[pause]**

Now, the line at the bottom of the slide is the theoretical anchor for the whole course.

Under squared-error loss, the best forecast is the **conditional expectation**, which is the
expected value of y at time t-plus-h, given everything we know at time t.

There are two things worth flagging here. The first is the phrase "under squared-error loss."
That qualifier is doing real work. If you care about squared errors, the conditional mean is
optimal. Change the loss function, and the optimal forecast changes. Under absolute error, for
instance, you'd want the conditional *median*. Optimal always means optimal *with respect to
something*.

The second is that script F, which is the information set, everything known at time t. Every
model we build this semester is essentially a different guess about how to turn an information set
into a conditional expectation. Exponential smoothing does it one way, and neural networks do it
another, but that's the target in every lecture.

---

# ▶ SLIDE 6 — Components of a Time Series

Before we model anything, it helps to know what we're looking at. The classical decomposition says
that any series can be written as the sum of four parts. On the slide that's the additive form:
y-sub-t equals T plus S plus C plus I, all at time t.

The first part is the trend, which is the long-run direction. Is the series growing, shrinking, or
flat over the years?

The second is seasonality, which is the regular, calendar-driven pattern. Think of December
retail, summer electricity, or Monday morning traffic. The key word is *regular*: it repeats on a
known cycle.

The third is the cycle, meaning the medium-run swings of the business cycle. These are the ones
people confuse with seasonality, so let me separate the two clearly. Seasonality has a fixed, known
period, and cycles don't. A recession doesn't arrive every forty-eight months on schedule.

And the fourth is the irregular part, the random shocks, which is everything left over.

Which parts matter depends entirely on the series. Retail sales are trend plus seasonality.
Quarterly GDP is trend plus cycle. Daily stock returns are mostly irregular, which is why
predicting them is so hard. And energy demand has all four.

**[pause]**

Now, read the warning box with me, because this is a mistake I see constantly.

Ignoring seasonality causes large, *systematic* errors. They aren't random errors, they're
systematic ones. A non-seasonal model on monthly retail data will consistently under-forecast
December and over-forecast January, every single year.

That word "systematic" is why this matters. Random error averages out, and systematic error
doesn't. If you're wrong by the same amount in the same direction every December, you'll be short
of stock in the same month every year, and no amount of data collection will fix it. The model is
structurally wrong.

---

# ▶ SLIDE 7 — Always Beat a Benchmark First

This slide is about a professional standard, and I want to be blunt about it.

Before you present any forecast model to anyone, check that it beats a naive benchmark. A model
that can't beat the baseline has zero value. I don't mean a little value. I mean zero.

Here are four standard benchmarks, all on the same series, CA_1 FOODS, which is the store you'll
forecast in the homework.

*(Point at the figure.)*

The gray line is the data. The shaded strip on the right is the test year, the fifty-two weeks we
hold back. In each panel, the red line is the forecast, and the blue line is what it's built from.
Every one of these is a shape you could draw with a ruler.

In the bottom left is the **naive** forecast, which says tomorrow equals today. You take the last
training week, the blue dot, and hold it flat for a year. It sounds stupid, but it's extremely hard
to beat for asset prices, because if you could reliably beat it, you'd be rich rather than sitting
in this class.

In the bottom right is the **mean**. That's the average of every training week, the dashed blue
line, held flat. Use it when there's no trend and no season.

In the top right is **drift**. Draw a chord from the very first training week to the last one,
the two blue dots, and keep going at the same slope. That's the average historical change,
extended forward. Use it when there's a trend but no season.

And in the top left is the **seasonal naive** forecast. You take last year, the blue stretch, and
slide it right by fifty-two weeks, so this December equals last December. Use it for any series
with a season.

**[pause]**

Now, read the red numbers. Seasonal naive wins, at sixteen sixty. Then comes drift at seventeen
sixty-four, naive at seventeen ninety-four, and the mean last, at eighteen eighty-one. On a series
with a season, the benchmark that copies the season wins, and for retail it's a genuinely tough
benchmark. Hold on to that sixteen sixty. It's the number every model in this course has to beat on
this series, all the way to Lecture 11, on Bayesian time series models.

**[pause]**

Now look at the line at the bottom of the slide, because it isn't a formality.

In the M4 Competition, one of the big open forecasting competitions, with a hundred thousand
series, many machine learning models *failed to beat* simple exponential smoothing.

Sit with that for a moment. These were sophisticated methods from real researchers, and they lost
to a method from the nineteen fifties that you'll learn in about ten minutes.

There are two lessons here. The first is that complexity is not accuracy. A model isn't better
because it's harder to explain. The second is that the benchmark is how you find out. Without one,
you have no idea whether your fancy model is adding anything at all.

So benchmark first, always. And if your model loses to seasonal naive, that's a finding, not a
failure, so report it.

---

# ▶ SLIDE 8 — Section divider: Exponential Smoothing

That brings us to part two, on exponential smoothing.

The core intuition fits in one sentence: recent observations should carry more weight than old
ones.

---

# ▶ SLIDE 9 — The Problem with Equal Weights

Here's the problem we're solving.

Most classical methods treat every historical observation equally, so every data point gets the
same vote. That's fine if the world is stable, but what if it changes?

Take a concrete case. A retailer renovates a store, and sales shift. If your model weights three
years of pre-renovation data equally, it'll keep forecasting the old store. It'll be wrong for
months, in the same direction, until the new data finally outweighs the old.

We fix this with **exponential weighting**. An observation k weeks old gets a weight of alpha
times one-minus-alpha to the power k. With each step back in time, the weight shrinks by the same
factor. A high alpha means a short memory, and a low alpha means a long one.

*(Point at the left panel first.)*

These are real weights from the store you'll use in the lab, two aisles of Walmart's store TX_1,
with alpha fitted to each. Each bar is the weight on one past week. The vertical axis is on a log
scale, where each gridline is ten times the one below it, and on a log scale, shrinking by the same
factor every week makes a straight staircase.

On the left is household goods, with alpha at zero point six two, so last week alone gets
sixty-two percent of the vote. On the right is food, with alpha at zero point two one, and a much
gentler staircase.

The dashed line is equal weighting, one over two hundred twenty-five, because there are two
hundred twenty-five training weeks. On household goods, every week older than five gets *less*
than an equal share. On food, it's every week older than sixteen.

**[pause]**

Now look at the triangles. Think of the bars as weights sitting on a plank. The triangle is where
the plank balances, which is the average age of the data the forecast is built from. For household
goods, it's zero point six weeks, so essentially last week. For food, it's three point seven
weeks. Equal weighting would balance a hundred and twelve weeks back, more than two years in the
past.

The weights never hit exactly zero, and they add up to one, so this is a proper weighted average.
Alpha is the forgetting rate. With a high alpha, the memory is short: you react fast, and you
forget fast. With a low alpha, the memory is long, so the forecast is stable but slow to notice
change. It's the same store, with two aisles and two very different memories.

Hold on to alpha, because it's about to become the single parameter of our first model.

---

# ▶ SLIDE 10 — Simple Exponential Smoothing (SES)

Here's the model. It's one equation, written two ways.

The first way says the new level is alpha times the current observation, plus one-minus-alpha
times the previous level. The second way is the same equation rearranged, and it's the one to
picture. The new level is the old level plus alpha times the *gap* between what just happened,
y-sub-t, and what the old level said. Every week, the level closes the fraction alpha of that gap,
where alpha is above zero and at most one. So alpha works like a learning rate.

*(Point at the right-hand panel.)*

That's the picture on the right. It shows one real week, the first week of July twenty thirteen,
in food sales at Walmart's store CA_1. Sales jumped four thousand eight hundred and sixty-two units
above the old level. With alpha at zero point three two, the level moved up one thousand five
hundred and thirty-four units. That's about a third of the gap, and the level kept two thirds of
its old belief.

And the forecast is just y-hat equals ell-T, for every horizon h, so it's flat.

**[pause]**

*(Now point at the main panel.)*

Here we have two years of weekly food sales in gray, and then the test year, shaded. Three levels
run through it. The orange one, with alpha at zero point eight, closes eighty percent of every gap.
It chases every wiggle, and it's almost a copy of the data. The blue one, with alpha at zero point
one, closes ten percent. It's smooth, but it's slow. See how long it takes to come down after the
twenty thirteen peak. The red one is the alpha statsmodels picked, zero point three two, by
minimizing next-week squared errors in the training data.

Then the test year starts, and each level runs flat. That flatness is the model's honest admission
of its limits, because SES has no concept of trend or season. We'll fix trend on the next slide.

There's one twist worth knowing. On this test year, the three flat lines land within about thirty
units of RMSE of each other, and the fitted one isn't the best. That's because alpha is fitted to
next-week errors, not to errors a whole year ahead.

**[pause]**

And here's the elegant part. SES *nests both benchmarks* from slide seven. Set alpha to exactly
one, and the level becomes y-sub-t, which is the naive forecast. Let alpha go to zero, and you get
the historical mean.

So SES isn't a competitor to those benchmarks. It's a dial that spans them, and estimating alpha
means letting the data pick the point on that dial.

Use it when there's no trend and no seasonality.

---

# ▶ SLIDE 11 — Holt's Method: Adding Trend

SES gives a flat forecast, so if the series trends, we need more. Holt's insight, from nineteen
fifty-seven, was to track two things instead of one: the level, and the slope.

There are three equations in the box. The **level** equation looks like the one for SES, except
that it compares against the old level *plus* the old trend, because the series was expected to
move. The trend equation applies the same smoothing idea to the slope, controlled by beta-star. And
the forecast is the level plus h times the trend, so it's a sloped line instead of a flat one.

Read ell-t as where the series *is* right now, and b-sub-t as how fast it's moving. They're
position and velocity.

**[pause]**

Now the caution, and this one has cost people real money. That forecast is a straight line,
extended forever, and nothing grows without limit.

*(Point at the chart.)*

Here's Holt fitted to the same food series. Today's level is twenty-one thousand two hundred
seventy-two units, and the fitted slope is seventy units a week. The red line is the undamped
forecast: plus seventy, every week, forever. By the end of a year, that's three thousand six
hundred more units a week than the store sells today. On the real test year, that model's RMSE is
three thousand four hundred and twenty-eight, which is about double what SES got.

The fix is the **damped trend**, in blue. Each week ahead adds phi times the previous week's rise,
where phi sits between zero and one. Here phi is zero point eight five, which is an illustrative
value. The first week adds about sixty units instead of seventy, the next adds about fifty, and the
one after that adds forty-three. It's a staircase whose steps shrink by the same factor each week.

And a staircase like that has a top. Add up all the steps, forever, and you never pass the dashed
line, which is today's level plus five point seven weeks' worth of slope. That's the ceiling, ell-T
plus b-T times phi over one-minus-phi. So instead of adding three thousand six hundred units over a
year, the damped forecast adds at most about four hundred.

**[pause]**

Empirically, the damped trend is one of the strongest performers in forecasting competitions. Use
it for anything beyond about six periods ahead. In the lab you'll fit phi yourself, and I'll ask
you to look hard at where it lands.

---

# ▶ SLIDE 12 — Holt-Winters: Level, Trend, and Seasonality

Holt-Winters adds one more component, seasonality, so now we're tracking three things.

The equations look intimidating, so let me just point at the structure. There's the level, the
trend, and now a **seasonal** equation with a third smoothing parameter, gamma. Notice the
subscript, s-sub-t-minus-m, where m is the seasonal period. That's twelve for monthly data, four
for quarterly data, and seven for daily data with a weekly cycle. We're looking back one full
season.

In the level equation, we subtract the seasonal component before updating. So we de-seasonalize
first, and then we update. Then the forecast adds the season back in. As written, that formula
holds for horizons up to one season. Past that, you reuse the last season's index again.

**[pause]**

Now the comparison, and this is a real decision you'll have to make.

With **additive** seasonality, the swings are constant in size. December adds the same amount
every year, whether the store is big or small.

With **multiplicative** seasonality, the swings grow with the level. December is plus thirty
*percent*, not plus a fixed number of dollars.

How do you tell which one you have? Look at the plot. If the seasonal swings get visibly bigger as
the series grows, it's multiplicative.

US retail sales are the textbook case. The December spike grows every year as overall sales grow,
so it's multiplicative. Retail and tourism usually are.

There's one practical constraint at the bottom of the slide: you need at least two full seasons of
data to initialize. For monthly data, that's twenty-four observations at minimum. With eighteen
months of history, you can't fit a seasonal model, no matter how much you want to.

---

# ▶ SLIDE 13 — The ETS Framework: A Unified View

Now let's tie it together, because it may feel like we've collected three unrelated techniques. We
haven't. They're one family.

Hyndman and Athanasopoulos, the authors of our textbook, unify every exponential smoothing method
as a **state space model**, described by three letters: E for error, T for trend, and S for
seasonal.

The trend can be none, additive, or damped. The season can be none, additive, or multiplicative.
Three times three is nine, and the grid shows all nine as forecast shapes.

*(Point at the figure.)*

These are drawings, not data. The gray is a made-up history, and the red is each method's
forecast.

Read down the first column. With no trend, the forecast is flat, and that's SES. With an additive
trend, it's a straight line up, and that's Holt. With a damped trend, the line bends over and
levels off. That's Holt damped, the ceiling from two slides ago.

Now read across. An additive season repeats the same swing on top of the trend, and that's
Holt-Winters additive. A multiplicative season scales the swing with the level, so as the trend
climbs, the swings grow, and that's Holt-Winters multiplicative. And look at the top row. With no
trend, the level never moves, so additive and multiplicative look the same. A multiplicative season
only shows itself once the level changes.

**[pause]**

So where's the error letter? It isn't in the picture, because it doesn't change the shape. With
the same parameters, additive and multiplicative errors give the same point forecast. What changes
is the prediction interval around it. So each of the nine cells is really two models, which makes
eighteen.

Three of them, the ones with additive errors and a multiplicative season in the right-hand column,
are numerically unstable, because the state equations end up dividing by numbers that can get close
to zero. That's why that column says "error M only." Eighteen minus three is fifteen. It's also why
Holt-Winters multiplicative is ETS M-A-M, with multiplicative errors to go with the multiplicative
season. And every method from the last three slides is one of the fifteen. ETS A-N-N is SES, ETS
A-A-N is Holt, ETS A, A-damped, N is the damped trend, and ETS A-A-A is Holt-Winters additive.

**[pause]**

So these weren't separate tricks. They were points in a structured space, and now you can see the
whole space at once, including combinations we never discussed.

Here's the practical payoff, which is **automatic selection**. You don't have to agonize over which
of the fifteen to use. You fit all fifteen, and you keep the one with the minimum AIC. In Python,
the ETSModel class in statsmodels fits any one of them. The search itself is a short loop over the
fifteen that you write, and the AIC comes back with every fit.

That raises a fair question. If the computer picks the model, why did we walk through the
equations? Because when automatic selection returns something strange, like a multiplicative season
on a series whose swings don't grow, you need to recognize it and know why it's wrong. Automation
is a labor-saving device, not a substitute for judgment.

---

# ▶ SLIDE 14 — Section divider: Forecast Evaluation

That brings us to part three, on forecast evaluation.

The line on this slide is one I mean literally: forecasting without evaluation is not science.
It's storytelling.

---

# ▶ SLIDE 15 — How Do We Measure Forecast Accuracy?

We have our out-of-sample errors, actual minus forecast, one for each week of the test year. So how
do we summarize fifty-two of them in a single number?

There are two classic answers, on the left. MAE is the mean absolute error, where you average the
sizes of the misses. RMSE is the root mean squared error, where you square each miss, average the
squares, and take the square root. Both come out in the same units as your data, which here means
units of food per week. That makes them easy to explain to a manager: "We're off by about fourteen
hundred units a week."

*(Point at the left panel of the figure.)*

Here's the picture that tells them apart. Each test week's miss is drawn as a square, with the size
of the miss as its side, and all fifty-two squares share a corner. The left panel is SES on the
food series, and the right panel is seasonal naive, which predicts the same week last year.

The dashed square has the *average side*, and that's MAE. The filled square has the *average
area*. Its area is the average of the fifty-two squares' areas, and its side is RMSE.

**[pause]**

Now look at what that does. SES and seasonal naive have almost the same dashed square, with an MAE
of thirteen seventy-six against thirteen sixty-five. On MAE, they tie. But SES has a few huge
squares out at the edge. The biggest one is the week of Christmas twenty fifteen, when sales fell
four thousand six hundred units below its flat forecast. Area grows with the square of the side, so
those few big squares dominate the average area. Five weeks carry forty percent of all of SES's
squared error. So the filled square is bigger, with an RMSE of seventeen fifty-four against sixteen
sixty. On RMSE, seasonal naive wins.

Notice, too, that the filled square always contains the dashed one. An average of squares can never
be smaller than the square of the average. So RMSE is always at least as big as MAE, and the gap
between them tells you how lumpy the misses are.

**[pause]**

So which one should you use? It depends on your cost function. If big misses are especially
costly, like a stockout, an understaffed shift or a blown capital reserve, prefer RMSE, because it
reflects that asymmetry. If all errors hurt roughly in proportion to their size, MAE is more
representative.

That's a business judgment, not a statistical one. Ask what the error actually costs.

---

# ▶ SLIDE 16 — Comparing Across Series: MAPE and MASE

Both of those numbers come in units, whether that's units of food, dollars or patients. That means
you can only compare models on the *same* series. An RMSE of four hundred means nothing next to an
RMSE of two, if one series sells forty units a week and the other sells forty thousand.

To compare *across* series, you scale the errors, and there are two ways to do it.

MAPE is the mean absolute *percentage* error, where each miss becomes a percentage of the actual
value. It's easy to explain, but it breaks down when the actual value is near zero. Divide by a tiny
number, and one week can blow up the whole average. Slow-moving products, with weeks of zero sales,
make MAPE useless.

MASE is the mean absolute *scaled* error. The top of the fraction is the MAE you already know. The
bottom is the yardstick: how far off seasonal naive was, on average, on the *training* data, when
it predicted each week from the same week a year earlier.

**[pause]**

So a MASE below one means your errors are smaller than seasonal naive's typical miss in the past.
Above one, you're doing worse than that, and you should say so. It's built on the benchmark idea
from slide seven, and it's one of the two accuracy measures the M4 Competition scored on. It also
has a property the others lack, which is that the number carries its own yardstick. An RMSE of
eight hundred needs context. A MASE of one point two says you missed by more than last year's value
used to.

There's one honest caution, and it comes from our own food series. Both SES *and* seasonal naive
score about zero point five nine there. Why? Because the yardstick was set in the training years,
and in twenty twelve the store's food sales jumped twenty-nine percent. Last year was a terrible
guess that year, so the yardstick is generous. A MASE below one does not mean you beat seasonal
naive on *this* test year. To compare two models on the same series, compare their MASEs with each
other.

---

# ▶ SLIDE 17 — The Golden Rule: Always Evaluate Out-of-Sample

If you take one operational rule from this lecture, take this one: **never** evaluate a forecast
model on the data used to fit it.

In-sample fit measures how well the model memorized the past. It says almost nothing about how well
it predicts the future. The two can even point in opposite directions. A model complex enough to
fit your history perfectly will usually forecast terribly, because it has learned the noise along
with the signal.

The diagram shows the correct setup. The training set runs up to time T. Then comes the forecast
origin, and the test set lies beyond it. The model sees only the training data, and we judge it
only on the test data. The split is *chronological*, because the test set is the future.

**[pause]**

Now the warning box, and this is the single most common technical error I see. Never randomly
shuffle time-series observations.

In scikit-learn, use `TimeSeriesSplit`, never `KFold`. `KFold` is the default in most tutorials,
and shuffled or not, it holds out each block in turn and trains on the others, including the later
ones. On a time series, that puts future observations into the training set and past observations
into validation. The model gets to peek at the answer, and turning shuffling off does not fix it.

This is called data leakage, and the reason it's dangerous is that it doesn't look like an error.
Your cross-validation scores get *better*, and the model looks excellent. Then it goes into
production and fails, and nobody can work out why, because the bug was in the evaluation, not in
the model.

So use `TimeSeriesSplit`, every time. This will come back in every remaining lecture.

---

# ▶ SLIDE 18 — Section divider: Course Roadmap

For the last part, let's look at where the semester goes.

---

# ▶ SLIDE 19 — ECON 8310: Four-Part Structure

There are twelve method lectures, in four parts, and then a synthesis.

Part one is time series models. That's today's lecture, then ARIMA and VAR in Lecture 2, and then
generalized additive models in Lecture 3. This is the classical econometric toolkit.

Part two is tree-based models. It starts with decision trees, then tree ensembles, meaning random
forests and boosted trees together, and then regularization and model selection. This is where we
cross into machine learning, and where feature engineering starts to matter more than the model.

Part three is deep learning: neural networks in PyTorch, then CNN architectures, and then RNNs,
LSTMs and Transformers. We build these from the components up, not just by calling a library.

Part four is Bayesian methods: the foundations, then time series and hierarchical models, and then
Bayesian linear regression. This is where we stop producing a single number and start producing a
full distribution, which is often what the decision actually needed.

Then Lecture 13 closes the course with a synthesis of what all four parts measured, side by side.

**[pause]**

We use Python throughout, with statsmodels, scikit-learn, XGBoost, PyTorch, Prophet, pyGAM, and
PyMC.

One note on sequencing. The parts get harder, but not uniformly. If you come from econometrics,
part one will feel comfortable, and part four will be a genuine shift in thinking. If you come from
data science, it's the reverse. Whichever you are, don't coast through the familiar half. The
methods build on each other, and falling behind in one week makes the next one harder.

---

# ▶ SLIDE 20 — Tools and What You Will Be Able to Do

Quickly, here's the stack. Statsmodels handles the classical time series models, which means ETS,
ARIMA and VAR. Scikit-learn covers machine learning models, pipelines, and cross-validation.
XGBoost is for gradient boosting, and PyTorch is for neural networks. Prophet and pyGAM handle the
additive models, and PyMC handles the Bayesian work.

Get these installed before the next class. There's a test script in the course repository. Run it,
and if something fails, email me now rather than the night before Homework 1.

**[pause]**

And here's what I'm actually promising. By the end of the semester, you'll be able to choose the
right method for a given business problem, which is mostly about knowing what each method assumes.
You'll be able to implement and tune it in Python, and to evaluate it rigorously out-of-sample.
You'll be able to interpret the result for someone non-technical, and to use AI tools effectively to
speed the work up. And you'll know how to avoid the most common mistakes, several of which we've
already named today.

Notice that only one of those six is about writing code. The code is the easy part now, and the
judgment is what you're here for.

---

# ▶ SLIDE 21 — Lecture 1: Key Takeaways

Here are five things to carry out of today.

First, forecasting underpins every forward-looking business decision, and under squared-error
loss, the best point forecast is the conditional expectation.

Second, always beat a benchmark first, whether that's naive, seasonal naive, mean, or drift. If you
can't beat them, you have nothing, and you should say so.

Third, exponential smoothing assigns geometrically declining weights to past observations, and
alpha controls how fast old data is forgotten.

Fourth, SES gives you the level, Holt adds the trend, and Holt-Winters adds seasonality. The ETS
framework unifies all fifteen variants, and AIC selects among them automatically.

And last, evaluate out-of-sample only, and never shuffle time series data. RMSE punishes large
errors, and MASE scales by seasonal naive's misses, so it can compare across series.

**[pause]**

If you remember only two of those, make it the second and the last: benchmark first, and evaluate
honestly. Every method from here on is judged by those two standards.

---

# ▶ SLIDE 22 — References

The readings for this week are in the syllabus. For today, that's chapters one, two, five, and
eight of *Forecasting: Principles and Practice*. Either edition works for this material, the R
edition or the Python one, because the chapter numbers are identical.

Next time, in Lecture 2, we turn to ARIMA and VAR, a regression-based approach to time series with
a rich autocorrelation structure. We'll relax the idea that the past enters only through a smoothed
level, and start modeling the correlation structure of the series directly.

See you then.

---

## Timing guide

Measured from the script (2026-10-16, after the benchmark and ETS figures were added):
**4,985 spoken words**. At 140 wpm — a normal lecture pace — that is **~36 minutes**; 41 if you
read deliberately, 33 if you push.

| Segment | Slides | Words | Runtime |
|---|---|---:|---:|
| Opening & why forecasting matters | 1–7 | 1,583 | ~11 min |
| Exponential smoothing | 8–13 | 1,832 | ~13 min |
| Forecast evaluation | 14–17 | 973 | ~7 min |
| Roadmap & close | 18–22 | 597 | ~4 min |
| **Total** | | **4,985** | **~36 min** |

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
