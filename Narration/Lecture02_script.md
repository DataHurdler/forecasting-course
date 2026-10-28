# Lecture 2 — Recording Script

**ECON 8310: Business Forecasting · ARIMA, VAR & Multivariate Models**

Deck: `Slides/Lecture02_ARIMA_VAR.pdf` (24 pages) · Measured runtime: **~34 minutes** (31–37 depending on pace)

---

## How to use this document

- **`▶ SLIDE n — Title`** marks where to advance. The number is the PDF page.
- Written to be **spoken**. Short sentences. Contractions are fine.
- *Italic parentheticals* are stage directions — do not read them aloud.
- **[pause]** means actually stop for a beat.
- Change anything that doesn't sound like you.

---

# ▶ SLIDE 1 — Title page

Welcome back. This is Lecture 2, on ARIMA, VAR and multivariate models.

Last week, in Lecture 1, we built exponential smoothing. The idea there was that the past enters
your forecast through a smoothed *level*, which is a running summary of where the series is,
updated as each new observation arrives.

Today we do something different. Instead of summarizing the past into a level, we model the
**correlation structure** of the series directly. How does today relate to yesterday, to the day
before, or to the same month last year? ARIMA answers those questions explicitly.

After that, we'll do something exponential smoothing simply can't do, and bring in a *second*
series.

**[pause]**

---

# ▶ SLIDE 2 — Lecture Outline

We'll take this in four parts.

First comes stationarity, which is the entry requirement for everything else today. If you skip
it, nothing downstream is valid.

Second, we'll cover ARIMA and its seasonal cousin, SARIMA.

Third comes VAR, for when two series move each other. That section also covers Granger causality
and ARIMAX.

And we'll close with the takeaways.

---

# ▶ SLIDE 3 — Section divider: Stationarity and Differencing

*(Divider — read and move on.)*

So, part one is stationarity.

Before we fit any model today, the series has to have a stable mean and a stable variance. Let me
explain what that means, and why it isn't optional.

---

# ▶ SLIDE 4 — What Is Stationarity?

A series is **weakly stationary** when three conditions hold. The mean is constant over time. The
variance is constant and finite. And the covariance between two points depends only on how far
apart they are, which is the lag, and not on *when* you are.

Read that third condition carefully, because it's the one people skim. It says the relationship
between January and February should be the same as the relationship between June and July. The
gap matters, and the location doesn't.

**[pause]**

The first two conditions have a picture, and it's on the right. *(Point at the figure.)* This is
TX_1 household goods, the series you forecast in Lab 1, last week. The gray line is the data. The
thick line is a rolling average over a year, and the shaded band runs two standard deviations
either side of it.

The top panel shows the weekly units. Watch the band. It climbs, and the average goes from about
thirty-one hundred units a week to about sixty-five hundred. A constant mean would give you a flat
band, and this one is a ramp.

The bottom panel is the same series as week-to-week changes. Now the band is horizontal and
centered on zero, and it keeps roughly the same width the whole way across. That's what conditions
one and two look like: **the series stays inside a horizontal band of fixed width.**

The panel titles carry a number we'll meet in two slides, from the ADF test. On top it's minus one
point three eight, which is no evidence against a unit root. On the bottom it's minus eight point
five one, which is decisively stationary. So your eye and the test agree.

Now, why does the definition earn its keep?

If the mean drifts, there's no single value to forecast toward. Ask what the long-run level of
this series is, and there's no answer, because it depends on when you ask.

If the variance grows, prediction intervals widen without bound. Your forecast is technically
correct and operationally useless.

Stationarity is what makes the past informative about the future, and every model today rests on
it. When we get to neural networks in Lecture 7, five weeks from now, that assumption is still
hiding in there. It's just less visible.

The usual culprits are trending series, like GDP, prices, sales and almost anything measured in
dollars over time, and variance that grows with the level. First-differencing removes a stochastic
trend, and you just watched it work. A log transform stabilizes the variance. Often you need both.

---

# ▶ SLIDE 5 — Unit Roots and Differencing

Here's the sharpest version of the idea. Take a simple AR one: today equals phi times yesterday,
plus noise. Everything turns on whether phi is below one, or exactly one.

The picture shows what that means. *(Point at the figure.)* Each line starts with one unit shock at
time zero, and tracks how much of it is left, period by period.

The blue line is phi equal to one half. Every period keeps half of what was there, so the shock
halves after one period, then halves again, and by period five it's essentially gone.

The orange line is phi equal to zero point nine. The rule is the same, and every period keeps the
same share, but the share is bigger, so the decline is slower. It takes about six point six periods
to lose half the shock. That's the half-life, and the slide gives the formula for it: the log of
one half, divided by the log of phi. It's the same staircase every time, just with a different
step.

**[pause]**

Now look at the red line. Here phi equals **exactly one**, so every period keeps *all* of the
shock. It never halves, and it never fades. It's absorbed permanently into the level of the series.
That's a random walk, a series with a unit root, and that flat red line is what "permanent" means.

That distinction has real consequences. In a mean-reverting series, a bad quarter is just a bad
quarter. In a random walk, a bad quarter permanently lowers the level you forecast from. It's the
same data, with a completely different business implication.

**[pause]**

The fix is **first differencing**, which means modeling the change rather than the level.
Difference a random walk, and the shock shows up exactly once, in the period it happened, and then
it's gone from the changes. So d equal to zero means the series was already stationary, and d equal
to one means you took one difference. Capital D equal to one means a *seasonal* difference, such as
this December minus last December.

---

# ▶ SLIDE 6 — Testing for a Unit Root

So how do you know which line you're on?

Start with a rule of thumb. A visible trend or a growing variance means the series probably isn't
stationary. But a phi of zero point nine five and a phi of exactly one look almost identical over a
few years of data, so test it formally rather than trusting your eye.

You have two tests, ADF and KPSS, and here's the thing worth remembering: they test **opposite
nulls**. The null hypothesis of the ADF test is that there *is* a unit root. The null of the KPSS
test is that the series *is* stationary. So run both. When they agree, you can be confident. When
they disagree, be suspicious, because you're probably in a borderline case, and you should look at
the plot again.

**[pause]**

*(Point at the warning box.)*

And don't over-difference. If you difference a series that's already stationary, you induce
negative autocorrelation that you then have to model away. You've manufactured the very problem
you're now solving. So test again *after* differencing, and stop at d equal to one once the result
is stationary.

The line at the bottom shows what a clean case looks like, on the series from the picture on slide
four, TX_1 household goods. The ADF statistic on the levels is minus one point three eight, which is
no evidence against a unit root. After one difference it's minus eight point five one, which is
decisively stationary. So d equals one, and you stop there. We'll read that same series' ACF in a
few slides.

---

# ▶ SLIDE 7 — Why the Rule Exists: Spurious Regression

Before we go on, I want to show you *why* that differencing rule exists. Told as a rule, it sounds
like bookkeeping, and it isn't.

Look at these two series. Both of them trend up, and they look related. If you saw this in a
business deck, you'd assume one drives the other.

In fact they have **nothing to do with each other.** Both are random walks that I generated
independently, with different random seeds, no shared shocks and no common cause. There's no
relationship to find.

**[pause]**

Now look at the two numbers on the slide, because they're the whole lesson.

The correlation of their week-to-week *changes* is minus zero point one three. That's essentially
zero, and it's the truth: these series are independent.

The correlation of their *levels* is plus zero point six two. And the levels are what you'd hand to
a regression.

Granger and Newbold demonstrated this in nineteen seventy-four. Regress two independent random
walks on each other, and you reject the null of no relationship roughly **three-quarters of the
time** at the five percent level. That's not the five percent the test promises. It's seventy-five
percent.

Think about what that means. Your t-statistic is large, your R-squared is high, and every
diagnostic you were taught to check looks excellent. And yet the relationship doesn't exist.

The regression isn't detecting a relationship between the series. It's detecting that both of them
wander. Two things that drift will drift *somewhere*, and over any finite sample they'll appear to
drift together or apart, and the regression will happily report that as signal.

**[pause]**

This is why economics has a long history of embarrassing published findings between non-stationary
series, and it's why the rule is to **difference first, then regress.**

Once you difference, you're modeling changes rather than levels, and the spurious correlation goes
away, because the changes really are independent. That's the minus zero point one three.

---

# ▶ SLIDE 8 — ACF and PACF: Model Identification

There are two diagnostic plots here, and together they tell you what model to fit.

The **autocorrelation function**, or ACF, is the correlation between y at time t and y at time t
minus k. It's straightforward: how does today relate to k periods ago?

The **partial** autocorrelation function, the PACF, asks the same question but controls for every
lag in between, and that distinction matters more than it sounds. Suppose today correlates with
three days ago purely *because* today correlates with yesterday, and yesterday correlates with three
days ago. Then the ACF will show that link, and the PACF won't. The PACF isolates the direct link.

**[pause]**

Now the table, which is the practical payoff.

For an AR process of order p, the ACF tails off geometrically, and the PACF **cuts off** sharply
after lag p. So if the PACF has two clear spikes and then nothing, you're looking at an AR two.

For an MA process of order q, it's the mirror image: the ACF cuts off after lag q, and the PACF
tails off.

For an ARMA, both of them tail off. That's less informative, and in practice it's why people reach
for automatic order selection.

And for a random walk, the ACF decays very slowly, and the PACF has one big spike at lag one. That
pattern is your warning sign, and in two slides you'll see it on real data.

---

# ▶ SLIDE 9 — What Those Patterns Actually Look Like

Now let's actually look at them, because "tails off" and "cuts off" mean very little until you've
seen the shapes.

There are four panels, laid out like the table you just saw. The top row is an AR one with a phi of
zero point seven, and the bottom row is an MA one with a theta of zero point six. The left column is
the ACF, and the right column is the PACF. These are exact values, computed from the models
themselves, so they're what you'd see with infinitely long data. Real data adds noise, and that's
the next slide.

**[pause]**

Start at the top left, with the ACF of the AR one. Lag one is zero point seven, which is phi. Lag two
is zero point seven times zero point seven, and lag three multiplies by zero point seven again.
*(Trace the dotted curve.)* Each bar keeps the same share of the one before it, which is the same
staircase you just saw for a shock. It fades out rather than stopping, and that's what "tails off"
means.

At the top right is the PACF of the same process. There's one spike at lag one, and then exactly
nothing. That's what "cuts off" means, and the lag where it cuts off is your p.

**[pause]**

The bottom row is the MA one, and it's the mirror image. The ACF has one spike and then stops.
Notice how short that spike is, at zero point four four. That's not an accident of the drawing. No
MA one, whatever its theta, can have a lag-one autocorrelation above one half, and that's the
dashed line. So if you see a lag-one spike of zero point eight followed by nothing, it isn't a clean
MA one.

The PACF decays, and notice that it alternates sign, flipping above and below the axis. That happens
because theta here is greater than zero. With a negative theta, the PACF decays on one side, with
every bar negative, and simple exponential smoothing, which comes up on the very next slide, has a
negative theta. So the decay is the MA signature, and the sign flips are not.

So the diagnostic is to **find the one that cuts off sharply.**

If the PACF cuts off, you have an AR, and the cut-off lag gives you p. If the ACF cuts off, you have
an MA, and it gives you q. If neither cuts off, and both just fade, you have a mixed ARMA, and that's
when you stop squinting at plots and let `auto_arima` search.

---

# ▶ SLIDE 10 — Difference First, Then Read the ACF

Now for real data. This is the TX_1 household series from Lab 1, the one whose ADF numbers you saw a
few slides ago.

*(Point at the left panel.)* This is the ACF of the levels. It's zero point nine four at lag one, and
still zero point five eight half a year later. It crawls down. And here's the thing: it would crawl
down like that whatever the underlying process was. A slowly decaying ACF tells you nothing about p
or q. It only tells you that you haven't differenced yet.

**[pause]**

So here's one rule, and it's absolute: **always run a unit-root test and difference before you read
either plot.** Students lose a lot of time reading tea leaves in the ACF of an undifferenced series.
Don't.

*(Point at the right panel.)* Now look at the same series after one difference, which is the
week-to-week change. It's a completely different picture. There's one clear negative spike at lag
one, at minus zero point two nine, and nothing else as big as zero point one six.

The shaded band is what pure noise would show. Several later bars poke just past its edge, which is
more than chance alone would give you, so there's a little structure left that a fuller model might
chase. But none of them comes close to the spike at lag one. Don't build a model around a bar that
barely pokes out at lag twenty-two.

So what does one spike at lag one, in the ACF of the change, tell you? Look back at the previous
slide: that's the MA one signature. It's a moving average of order one, on the differenced series,
which makes this an ARIMA zero, one, one.

**[pause]**

And here's why that's satisfying. An ARIMA zero, one, one is simple exponential smoothing, and
you'll see that at the start of the next section. In Lab 1, SES on this very series chose an alpha
of zero point six two. That alpha implies a lag-one autocorrelation of minus zero point three three
in the differences, and the data show minus zero point two nine. So two different methods, from two
different lectures, describe the same series the same way.

---

# ▶ SLIDE 11 — Section divider: ARIMA Models

That brings us to part two, ARIMA, which combines differencing with the AR and MA components in a
single model.

---

# ▶ SLIDE 12 — ARIMA(p, d, q)

Here it is. Take the series and difference it d times, and call the result y-prime. Then fit an
ARMA with p autoregressive lags and q moving-average lags to y-prime, not to the original series.
That's why the y's in the box carry primes.

So that's what the three letters mean: p is the number of AR lags, d is the number of differences,
and q is the number of MA lags.

**[pause]**

What I like about this framework is how much of what you already know turns out to live inside it.

An ARIMA zero, one, zero, with no AR terms, one difference and no MA terms, is a random walk. That's
the naive forecast from Lecture 1, last week.

An ARIMA one, zero, zero is a plain AR one.

And an ARIMA zero, one, one is algebraically equivalent to simple exponential smoothing. The model
we spent twenty minutes on last week is a single point in this space.

So ARIMA isn't a competitor to what we did in Lecture 1. It's a larger room that contains it.

Let's do one by hand, because the mechanics are less mysterious once you've turned the crank.

Take an ARIMA one, one, zero with an estimated phi of one half. We have y at period one hundred
equal to one hundred twenty, and y at period ninety-nine equal to one hundred eighteen. So the
change last period was two.

The model says the next change is half of the last change, and half of two is one. Add that to where
we are, and one hundred twenty plus one gives **one hundred twenty-one**.

That's all there is to it. Differencing means we model the *change*, and then add it back to the
level.

A word on where this comes from, because the name will follow you. Box and Jenkins formalized this
in nineteen seventy as a three-stage cycle. First you identify a candidate order from the ACF and
PACF. Then you estimate the parameters. Then you check the residuals, and if the residuals fail, you
go back to identification. It's a **loop**, not a pipeline. Half a century later that loop is still
the methodology, and it's why every ARIMA workflow you'll see has the same three stages.

**[pause]**

In Python, `auto_arima` from the pmdarima package automates the identify step. It picks d first,
with a unit-root test, which is KPSS by default. Then it searches a grid of p and q and returns
whichever combination minimizes AIC, usually stepwise, so it doesn't try all of them. AIC can't
choose d itself, because models with different differencing are fit to different data, so their
AICs don't compare. Once you've settled on an order, `SARIMAX` in statsmodels fits it.

Use `auto_arima`. But understand its failure mode: it optimizes a number, and it has no idea what
your data means. I've seen it return an ARIMA five, two, four on a series that visibly needs one
difference and an AR one. Its unit-root step chose the second difference, and then AIC over-fitted p
and q. That model had the lowest AIC on that sample, and it was complete nonsense out of sample. AIC
rewards fit with a complexity penalty, and on a short, noisy series the penalty isn't strong enough
to stop it.

So run the search, then look at the ACF and PACF anyway, and ask whether the answer is sensible. If
`auto_arima` returns a second difference and your ADF test said one was enough, trust the test.

---

# ▶ SLIDE 13 — SARIMA: Adding Seasonal Terms

For seasonal data, autocorrelation shows up not just at lags one and two, but at lag m, two m, three
m and so on, which are the seasonal lags. December relates to last December.

SARIMA handles this by adding a second set of AR and MA polynomials that operate at the seasonal
frequency. The notation gets heavy, with p, d and q in lowercase, then capital P, D and Q, and then
m in brackets, but the idea is simple.

Think of it as **two ARIMA models** running at different time scales. The lowercase part handles
the short-run dynamics, and the uppercase part handles the season. They're fitted together.

**[pause]**

Here's what a seasonal lag looks like in real data. *(Point at the left panel.)* This is CA_1 FOODS,
the series from Lecture 1, but daily this time, with its autocorrelation out to five weeks. The red
spikes are at lags seven, fourteen, twenty-one and twenty-eight, and they're the tallest on the
chart, at about zero point seven three to zero point eight. Today looks like the same weekday last
week, and the week before that. That's the weekly season. So here m is seven, not the twelve you'd
use for monthly data. It's the same idea with a different calendar.

Now the right panel. *(Point at it.)* This is the same series after one **seasonal difference**,
which is today minus the same day last week. That's capital D equal to one. The tall weekly peaks
are gone. What's left at lag seven is a single negative spike, at minus zero point four one. One
spike at a single lag is the MA signature you saw on slide nine, and when it sits at lag m, it's a
*seasonal* MA term. So capital Q equals one.

And the bump near lag thirty? That's not noise. California's SNAP benefits land on the first ten
days of every month, and food sales jump when they do, so there's a monthly rhythm hiding inside a
daily series. Take the SNAP days out first, and that bump drops from about zero point three to about
zero point one. A seasonal ARIMA with m equal to seven isn't built for a monthly calendar. That needs
a regressor you supply, like the holiday terms in the Prophet model next week, in Lecture 3.

---

# ▶ SLIDE 14 — SARIMA in Practice: A Four-Step Workflow

In practice, the workflow has four steps.

First, use ADF and KPSS to choose d. Second, use a seasonal unit-root test to choose capital D.
Third, read the ACF and PACF of the differenced series to guide the four order parameters, which is
exactly what we just did by eye on the daily series. And fourth, minimize AIC, then check the
residual ACF with a Ljung-Box test.

**[pause]**

That last step is the one people skip, so let me explain what it's actually doing.

The Ljung-Box test asks whether the residuals are **white noise**. It takes the autocorrelations of
the residuals across the first several lags and tests whether they're *jointly* zero.

The null hypothesis is that they are, meaning nothing is left to model. So here, unusually, you
*want* a large p-value. Failing to reject is the good outcome. That trips people up, because in most
tests you're hunting for significance.

**[pause]**

And think about why it matters. If your residuals still have autocorrelation, there's structure in
the data that your model hasn't captured. Today's error tells you something about tomorrow's error.
That means your forecast is leaving information on the table, and your prediction intervals are too
narrow, because they assume the errors are independent.

So a model can have a great AIC and still fail the Ljung-Box test. AIC compares models to each
other. Ljung-Box asks whether *this* model is adequate on its own terms. They're different
questions, and you want both of them answered.

For monthly retail data with a trend and a December season, a SARIMA with order one, one, one and
seasonal order zero, one, one, at a period of twelve, is a strong starting point. Notice the seasonal
part: one seasonal difference and one seasonal MA term, the same zero, one, one you just read off
the daily chart. It's not there because it's optimal. It's there because it's a good default to
beat before you try anything more elaborate.

---

# ▶ SLIDE 15 — Section divider: VAR Models

Part three is VAR. When two series influence each other, you model them together.

---

# ▶ SLIDE 16 — Why Use Multiple Series?

Everything so far, all of Lecture 1 and everything today, uses only the series' own history. We've
been predicting y from past y.

But business variables are linked, and often the link runs forward in time. Consumer sentiment today
moves retail spending next month. Interest rates today move housing starts within two quarters.
Advertising spend this week moves sales over the following few weeks.

**[pause]**

The principle underneath all of it is in the box: **if x today carries information about y h
periods ahead that isn't already in y's own history, then including x reduces forecast error.**

Notice the qualifier, "not already in y's own history." It's doing real work. If x is just a lagged
reflection of y, it adds nothing. It has to bring genuinely new information.

Granger causality is the formal test of exactly that claim, and we'll get to it right after the VAR.

---

# ▶ SLIDE 17 — VAR or ARIMAX: Which Way Does Influence Run?

That leaves you with a modeling choice, and it turns on whether the influence runs one way or both
ways.

A VAR treats everything **symmetrically**. Every variable predicts every other variable, and no
variable is privileged. Use it when the influence genuinely runs in both directions.

An ARIMAX model runs in one direction. Here x drives y, and not the reverse. Use it when the causal
story is clear. But notice the third point in that column, because it's the catch we'll come back
to: ARIMAX needs a forecast of x itself.

The last line of each column gives the example you'll meet. Retail sales and unemployment plausibly
move each other, so that's a VAR question, and it's the one you test in the lab. Advertising drives
sales and not the other way around, so that's ARIMAX.

---

# ▶ SLIDE 18 — The VAR(p) Model

Here's the structure for two series, written out for one lag, so p equals one. There are two
equations, and each one regresses on the lags of *both* variables. More lags just add more of the
same terms.

Look at where the value is, in the **cross terms**. The coefficient a-one-two is how much y-two last
period predicts y-one this period. That coefficient is precisely what a univariate AR throws away,
because it has no place to put it.

And here's the elegant part. If a-one-two and a-two-one both come back at zero, the system collapses
into two separate AR models. Then you've learned something: these series don't inform each other.
That's a finding, not a failure.

Each equation is estimated by ordinary least squares, so it really is just regression, and the
statsmodels fit chooses the order p by BIC.

**[pause]**

It's worth knowing why this model exists, because it was a reaction to something.

Christopher Sims proposed the VAR in nineteen eighty, in a paper arguing that the large structural
macroeconomic models of the day rested on assumptions nobody could defend. To identify those models,
economists had to declare in advance which variables affected which, and Sims argued those
restrictions were, in his word, incredible. He didn't mean wrong. He meant unbelievable.

His alternative was to stop pretending you know the structure. Let every variable depend on the
lagged values of every variable, symmetrically, and let the data speak.

He shared the Nobel Prize for it in twenty eleven.

I mention this because it frames what a VAR is *for*. It's deliberately agnostic. That's its
strength, because you impose almost nothing, and it's also why it can't answer causal questions on
its own, and why the parameter count explodes. You've traded assumptions for parameters.

**[pause]**

That brings us to the warning, and this one bites people in practice. It's called **parameter
explosion**. A VAR with k variables and p lags has k-squared times p coefficients, so the count is
quadratic in the number of variables.

With five variables and four lags, that's one hundred coefficients.

Sit with that number, one equation at a time. A monthly business series over ten years gives you one
hundred twenty observations per variable, or one hundred sixteen once four are spent on the lags.
Each equation fits twenty slopes plus an intercept. That's twenty-one coefficients from one hundred
sixteen rows, or one coefficient for every five or six observations, in every equation. Add a sixth
variable and it's twenty-five per equation. That model will fit your sample beautifully and forecast
far worse than it fits.

So keep k at four or fewer, unless you're using a regularized LASSO-VAR, which is the kind of thing
Lecture 6, on regularization, gives you the tools for.

---

# ▶ SLIDE 19 — Granger Causality Test

Here's the formal definition. x **Granger-causes** y if past values of x help predict y beyond what
y's own past already explains.

That final clause is the whole test. The question isn't whether x predicts y. It's whether x adds
anything once we've already used y's own history.

The procedure is a straightforward F-test. You fit the y equation with the x lags and without them,
and you test whether all the x-lag coefficients are jointly zero. If you reject, x Granger-causes y.

**[pause]**

Now the caveat, and I want you to take it seriously, because it's the single most abused result in
applied forecasting: **Granger causality is not structural causality.**

It tests *predictive content*. It doesn't establish that x mechanically brings y about.

Here's an example. Umbrella sales Granger-cause rain. Shoppers read the forecast, and they buy
umbrellas when it says rain, so yesterday's umbrella sales carry information that rain's own past
doesn't. But umbrellas don't cause rain.

So report it as "predictive of," and never as "causes." Say it the careful way, and you'll never
have to walk a claim back in front of a room of executives.

---

# ▶ SLIDE 20 — Granger on Real Data: The COVID Window

Here's a concrete question we run in the lab: does unemployment Granger-cause retail sales? The null
is that every unemployment lag is zero in the retail equation.

Notice that we're using macroeconomic series from FRED here, rather than store sales. That's
deliberate.

Two Walmart stores in the same state absolutely *do* move together, and you'll find a significant
Granger test between them in the second part of Homework 1. The problem is that the finding tells
you almost nothing, because both stores are responding to the same regional promotions and the same
weather. The test fires, and you still can't say which one drives the other.

*(Point at the left panels.)* These are the two series the lab tests. Monthly retail sales growth is
on top, and the monthly change in the unemployment rate is below, from nineteen ninety-two to today.
For almost thirty years, both of them wiggle in a narrow band. Then comes the pink strip, from
February twenty twenty to June twenty twenty-one. Retail growth swings sixteen times its pre-COVID
standard deviation. The unemployment rate jumps by more than ten points in a single month, April
twenty twenty, which is sixty-eight times its pre-COVID standard deviation.

**[pause]**

Now, I ran the test before writing this slide, and what came back is better than the tidy answer I
expected. *(Point at the right panel.)* Each line is one direction. The two points on each line are
the F-statistic with COVID in the sample, and with those seventeen months taken out. The dashed line
is the five percent critical value, at about two point four.

On the full sample, both directions are significant, and unemployment looks like the stronger
driver, at eighteen point two against twelve point eight. That's exactly the story everyone expects.
People lose jobs, so people stop spending.

Then drop the COVID window and rerun the test. The intuitive direction collapses to two point four
two, right on the line. It's still significant, but only just. Meanwhile retail predicting
unemployment holds up, slipping only to about ten. So the lines cross. Both directions are
significant in both samples, and what the one shock decides is which of them looks stronger. In the
lab, when you restrict the sample to the years before twenty twenty, you'll see where that decline
ends up.

Sit with that, because there are two lessons in it, and both are worth more than the tidy answer
would have been.

The first lesson is economic, and it's obvious in hindsight. Retail sales are a coincident-to-leading
indicator, and unemployment is a lagging one. Spending falls first, and layoffs follow. That's a
story consistent with the data, and notice that I'm still saying *predicts*, not *causes*. The test
tells us about sequence, not mechanism.

The second lesson is methodological, and it's the one I want you to carry into your own work. On the
full sample, you would have reported a strong, intuitive, publishable finding, and it was being
driven by one enormous shared shock. One episode, seventeen months in a thirty-four-year sample, was
enough to flip the headline result.

You'll run all three windows yourself in the lab.

**[pause]**

Before we leave ARIMA and VAR, there's one more thing these models give you that we've barely
mentioned, and that's **prediction intervals**.

Everything we've produced so far is a point forecast, a single number. But every one of these models
also produces a distribution, and `statsmodels` will hand you an eighty or ninety-five percent
interval alongside the point.

Use them. A forecast of forty thousand units means one thing if the interval runs from thirty-nine
to forty-one thousand, and something completely different if it runs from twenty-five to fifty-five
thousand. It's the same point forecast, with an entirely different inventory decision.

And notice how the intervals behave as the horizon grows. They **widen**, and for a differenced
series they widen without bound. That's the model being honest. It's telling you that a random walk
has no long-run level to forecast toward, so far enough out, it genuinely doesn't know.

A point forecast hides that, and an interval shows it. When we get to Bayesian methods in Lecture 10,
which is Week 11, we'll stop treating the interval as an add-on and make the whole distribution the
output.

---

# ▶ SLIDE 21 — ARIMAX: Exogenous Variables with ARIMA Errors

Here's the last model for today. When one variable clearly *drives* another, rather than the two
influencing each other symmetrically, use ARIMAX.

The structure works like this. You regress y on your exogenous predictors in the usual way, and then
you let the *errors* follow an ARIMA process. That second part is what makes it more than ordinary
regression. Regression on time-series data almost always leaves autocorrelated residuals, and ARIMAX
gives that autocorrelation somewhere legitimate to live, instead of letting it quietly break your
standard errors.

**[pause]**

Use it when three things hold. The causal direction is clear, the way advertising drives sales and
not the other way around. You already have x, or you can forecast it. And adding x actually reduces
cross-validated forecast error. Check that last one rather than assuming it, because adding a
predictor doesn't automatically help.

In code, it's the same `SARIMAX` class from earlier, with your predictors passed in as `exog`.

**[pause]**

Now for the catch that surprises people.

To forecast y at time t plus h, you need x at time t plus h. In other words, you need a forecast of
your own predictor.

Think about what that means operationally. You build a beautiful model of sales driven by
unemployment, and it fits wonderfully. Then someone asks for next quarter's sales, and you realize
you first need next quarter's unemployment. That's its own forecasting problem, with its own error,
and that error now propagates into yours.

If you can't forecast x, ARIMAX buys you nothing at forecast time. Use a VAR instead, because a VAR
generates forecasts for all of its variables at once. It solves the problem by construction.

---

# ▶ SLIDE 22 — Section divider: Key Takeaways and Roadmap

Let's pull it together. Stationarity, ARIMA, VAR and ARIMAX make up the classical time-series
toolkit.

---

# ▶ SLIDE 23 — Lecture 2: Key Takeaways

There are five things to take away from today.

The first is that stationarity comes before you fit anything. Test with ADF and KPSS, which have
opposite nulls, so run both. And if there's a unit root, difference it away, because
first-differencing removes a stochastic trend.

The second is that the ACF identifies the MA order, and the PACF identifies the AR order. Always
inspect them *after* differencing, never before.

The third is that ARIMA combines differencing with AR and MA components, and it contains the naive
forecast and simple exponential smoothing as special cases. SARIMA adds seasonal polynomials. Use
`auto_arima` to search, but keep your eyes open.

The fourth is that a VAR models all of its variables symmetrically, and captures the cross-dynamics
that a univariate model discards. Keep k at four or fewer. And Granger causality tests predictive
content, and predictive content only.

The last one is that ARIMAX adds exogenous predictors. Use it when the causal direction is clear and
x is genuinely forecastable.

**[pause]**

If you take only one thing away, make it **stationarity first.** Almost every ARIMA mistake I see
traces back to someone reading an ACF they should never have looked at.

---

# ▶ SLIDE 24 — References

The readings are in the syllabus: Chapter 9 for ARIMA, sections of Chapter 10 for dynamic
regression, and section twelve point three for VAR. Either edition of the textbook works, because
the chapter numbers are the same.

Next week, in Lecture 3, we turn to generalized additive models, and we'll fit them with Prophet and
pyGAM. We keep the interpretability we've had so far, but we give up the assumption that effects
have to be straight lines.

See you then.

---

## Timing guide

| Segment | Slides | Target |
|---|---|---:|
| Opening & stationarity | 1–10 | ~17 min |
| ARIMA & SARIMA | 11–14 | ~8 min |
| VAR, Granger, ARIMAX | 15–21 | ~16 min |
| Close | 22–24 | ~3 min |

If you need to compress, slide 14's four-step workflow survives trimming. Do **not** compress
slides 5, 10, 19, or 20 — the differencing rule, the "difference before you read the ACF" rule,
and the Granger caveat are the three things students most reliably get wrong.
