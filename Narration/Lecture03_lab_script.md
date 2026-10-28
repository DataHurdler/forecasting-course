# Lab 3 — Recording Script

**ECON 8310: Business Forecasting · GAMs, Prophet, and Several Seasons at Once**

Lab: `Labs/Lecture03_lab.qmd` (6 steps) · Measured runtime: **~25 minutes** of narration
(the in-room version is budgeted at 40)

---

## How to use this document

- **`▶ STEP n — Title`** matches the lab's own headings exactly.
- *Italic parentheticals* are stage directions. **[pause]** is a beat; **[STOP — learner works]**
  is where you tell the viewer to pause the video and do something.
- Numbers are the lab's real output, verified reproducible across repeated runs. Prophet uses
  MAP estimation rather than MCMC here, so it is deterministic — no seed required.
- Nothing here is sacred. If a sentence doesn't sound like you, change it.

### Three corrections that were applied to the lab

Writing this script turned up three claims in the `.qmd` that the measured output did not
support. **All three are now fixed in the lab**, so script and lab agree — this note is here so
you know why the wording changed if you remember the earlier version.

1. **Step 2** said two harmonics cut the error "by roughly a factor of three." It is
   **3,428 → 1,348**, a factor of **2.5**.
2. **The closing** said every method beat the benchmark. True on RMSE; **false on MAE**, where
   the single-harmonic Prophet (3,358) loses to seasonal naive (3,333). The lab now says "on
   RMSE" and points at the disagreement.
3. **The closing** said the simpler model "was competitive." pyGAM **wins** — 3,444 against
   Prophet's 3,935, about 12% better on RMSE and 15% on MAE. Understating your own result
   trains students to do the same.

### In-room language that needs replacing

The closing **Discuss** block. Converted below to a discussion-board prompt.

---

# ▶ OPENING

*(Show the rendered lab, at the top of the document.)*

This is Lab 3, and today we're leaving the Walmart data we used in the first two labs.

This week's lecture, Lecture 3, made a specific claim. It said that a GAM earns its keep when a
series has **several cycles running at once**, because each cycle gets its own smooth term, and
you can still read the model afterward. Retail sales can't demonstrate that. They have essentially
one season, which is Christmas, and one season doesn't need a GAM.

So today we'll use electricity demand. It has three large cycles running at the same time, and an
annual shape that will break the first model we try.

The business framing matters for how you read the results. You're forecasting daily demand for a
grid operator, and being wrong is expensive in **both** directions. If you under-forecast, you're
buying power at spot prices during a peak. If you over-forecast, you've paid to keep generation
spinning for nothing. Hold on to that asymmetry, because it comes back in the closing question.

**[pause]**

You'll need `prophet` and `pygam` installed, along with `electricity_daily.csv` and
`electricity_hourly.csv`, which come from `prep_electricity.py`. You only have to run that script
once.

---

# ▶ SETUP

*(Run the setup cell.)*

```
2041 days — train 1673, test 368
test window: 2017-08-01 to 2018-08-03
```

There are two things to notice here. First, we trim the data to twenty thirteen onward. That's
five and a half years, which is plenty of annual cycles, and it keeps every fit under a second,
which matters because you're going to fit several. Second, we hold out just over a year, from
August twenty seventeen through August twenty eighteen, so the test window contains a full summer
*and* a full winter. That's deliberate. Holding out only six months of an annual cycle tells you
almost nothing.

The `logging` line silences Prophet, which is otherwise extremely chatty on every fit.

---

# ▶ STEP 1 — Three cycles, one series

*(Run the three-panel figure.)*

Before we fit any model, let's look at the three cycles separately.

*(Point at each panel in turn.)*

The first panel is hour of day. Demand bottoms out around four in the morning and peaks in the
early evening, around hour nineteen. People wake up, go to work, come home, and turn things on.

The second panel is day of week. Tuesday is the highest and Sunday is the lowest, and that pattern
is commercial and industrial load.

The third panel is day of year, and it's a strange shape. That shape is the whole point of
Step 2.

*(Run the swing cell.)*

```
hour of day    swing  33.3%   min   4   max  19
day of week    swing  11.6%   min   6   max   1
day of year    swing  44.7%   min 116   max 200
```

**[pause]**

All three cycles are large. The annual swing is forty-five percent of mean demand, the daily swing
is thirty-three percent, and even the weakest, day of week, is nearly twelve percent. None of these
is a rounding error you could ignore and still have a usable forecast.

That's the situation Lecture 3 said GAMs are for. It isn't one dominant season with noise around
it. It's three real cycles, all worth modeling, and all on different time scales.

---

# ▶ STEP 2 — The annual cycle has *two* peaks

*(Point back at the third panel.)*

Look at that annual panel again. It doesn't look like a wave. It looks like the letter M.

*(Run the January, April and July cell.)*

```
January  (days 1-31)     34,403 MW   <- heating
April    (days 95-115)   26,945 MW   <- mild, nothing running
July     (days 180-200)  37,043 MW   <- air conditioning
```

**[pause]**

There it is: high in January, low in April, and high again in July. That's two peaks and two
troughs a year, because heating and cooling are **both** electricity demand, and they happen at
opposite ends of the calendar. April is the month when nobody needs either one.

Now for the consequence, and this is the argument for the whole lecture. We'll fit that annual
shape with a single sine wave, and then with two.

*(Run the harmonics cell.)*

```
one harmonic     in-sample RMSE 3,428 MW
two harmonics    in-sample RMSE 1,348 MW
```

A single sine wave has one peak per cycle. **It can put that peak in summer or in winter, but it
can't do both.** Whichever one it picks, it's badly wrong for half the year. That's three thousand
four hundred megawatts of error, on a series averaging around thirty thousand.

Add a second harmonic, a cycle that repeats *twice* a year, and the error drops to thirteen
forty-eight. That's two and a half times better, from one extra pair of terms, because the second
harmonic is exactly the shape the physics needs.

**[pause]**

That's the mechanism. Prophet's `yearly_seasonality` is this same idea with more terms. By default
it's a Fourier series of order ten, which means ten sine-and-cosine pairs. A pyGAM spline gets
there differently, by bending freely rather than by adding up waves, but it's solving the same
problem: the annual shape isn't a wave, so don't fit it with one.

---

# ▶ STEP 3 — Prophet, and an honest benchmark

*(Run the Prophet cell.)*

Prophet wants two columns named `ds` and `y`, and that's genuinely most of what you need to know
about its interface.

But we benchmark first, always. The benchmark is the same day last year, and notice that it goes
back **three hundred sixty-four** days rather than three hundred sixty-five, which keeps the
weekday aligned. Comparing a Tuesday with a Wednesday would hand the benchmark an error that isn't
really about seasonality.

```
Seasonal naive (364d)    RMSE   4,608   MAE   3,333
Prophet                  RMSE   3,935   MAE   3,001
```

Prophet beats the benchmark, by about fifteen percent on RMSE.

There's one deliberate choice in this cell, which is `daily_seasonality=False`. The data has one
row per day, so there's no variation within the day left to model. If you leave it on, the daily
terms are constant. Every timestamp is midnight, so the daily wave reads the same value every day,
and those terms can't be identified. Prophet would still draw you a daily curve that the data
can't support.

---

# ▶ STEP 4 — The components plot

*(Run `m.plot_components(fc)`.)*

This is the payoff, and it's why Prophet gets used by people who have to defend a forecast in a
meeting.

*(Point at each panel.)*

There are three panels, one for each additive term, and each one is on the scale of the original
series. They show the trend, the weekly cycle and the yearly cycle. Each one is readable on its
own, and each one is a sentence you can say to someone non-technical: "Demand is drifting down
slightly, weekdays run about three thousand megawatts above Sundays, and there are two annual
peaks."

**[pause]**

Now look at the yearly panel. It reproduces the double peak from Step 2. Nobody told Prophet about
heating and cooling, and nobody coded a January term and a July term. The Fourier series simply
had enough flexibility to find the shape that was in the data.

That combination, flexible enough to find the shape and structured enough to show you what it
found, is the entire case for additive models.

---

# ▶ STEP 5 — How much did the extra harmonics buy?

Step 2 made this argument in-sample, on the averaged annual profile. That's suggestive, but it
isn't evidence. So now we test it properly, on the holdout.

`yearly_seasonality` takes an integer, the Fourier order, in place of `True`.

**[STOP — learner works]**

Pause the video here. Fit a Prophet with the yearly order forced to **one**, which is a single
harmonic, and add it to the results table. Everything else stays the same.

*(After the pause, run it.)*

```
single harmonic RMSE 4,269
```

**[pause]**

That's forty-two sixty-nine, against thirty-nine thirty-five for the full-order model. So the
holdout agrees with Step 2. Crippling the annual term costs you real accuracy out of sample, not
just in-sample fit.

But look at where it lands relative to the benchmark. Seasonal naive was forty-six oh-eight, and
the single-harmonic Prophet is forty-two sixty-nine. So it beats the benchmark by about seven
percent, and that's *all* it beats it by.

And here's the part worth stopping on. **On MAE, it loses to the benchmark outright**, thirty-three
fifty-eight against thirty-three thirty-three. It's the same two models and the same holdout, with
the opposite verdict, depending on which error measure you quote.

That isn't a trick, and it isn't a bug. RMSE squares the errors, so it punishes the occasional
large miss, while MAE treats every error in proportion to its size. A model can be better on one
and worse on the other, and when a ranking flips between them, the honest move is to say so,
rather than to quote the metric that flatters you.

---

# ▶ STEP 6 — pyGAM, and reading the smooth directly

*(Run the pyGAM cell.)*

Prophet is a GAM with the time series decisions already made for you. pyGAM makes you state the
terms yourself, which is more work, and gives you more control.

There are only two terms. The first is a cyclic spline on day of year. Setting `basis="cp"` is what
makes December thirty-first join back up with January first, so the smooth doesn't jump at the
year boundary. The second term is day of week, as a categorical factor.

That's all. There's no trend term, and nothing hourly.

```
effective degrees of freedom: 22.5

                          RMSE    MAE
pyGAM (doy + dow)         3444   2550
Prophet                   3935   3001
Prophet (yearly order=1)  4269   3358
Seasonal naive (364d)     4608   3333
```

*(Let the table sit for a moment.)*

**[pause]**

Every model beat the benchmark on RMSE, but on MAE, as you just saw, the single harmonic did not.
The single-harmonic Prophet is the worst of the three models, as Step 2 predicted.

And **pyGAM wins**. It scores thirty-four forty-four against Prophet's thirty-nine thirty-five.
That's about twelve percent better on RMSE, and fifteen percent better on MAE, using two terms
against Prophet's three, with no trend component at all.

That should bother you slightly, and we'll come back to it in the closing question.

*(Run the partial dependence plots.)*

Now for the partial dependence, which is the GAM's version of the components plot.

*(Point at the left panel.)*

Say what the left panel means in words, the way you'd say it to a grid operator. Demand is high in
January, falls through spring to a minimum in late April, climbs to an annual peak in mid-July,
and drops again in the fall. The model learned the M shape directly from the data, by bending,
without being given a single Fourier term.

Read the band too. It's about the same width all year, because every day of the year appears four
or five times in the training data, and a cyclic spline has no ends to fray. That's worth noticing,
because it isn't the usual picture. On a predictor whose extremes are rare, such as unemployment
above ten percent, **the band fans out exactly where the data thin**, and that widening is the
model telling you where it's guessing. A forecast that comes with an honest statement of its own
uncertainty is worth more than a point estimate that doesn't.

---

# ▶ BEFORE YOU LEAVE

*(Show the closing callout.)*

**[STOP — learner works]** — *(replaces the in-room "Discuss")*

Post your answers to these in the Week 3 thread before Lecture 4.

The first question is about the awkward result. pyGAM was given two things, day of year and day
of week. Prophet was given a trend, a weekly cycle, *and* a yearly cycle with ten harmonics.
Prophet has strictly more machinery, and it lost by twelve percent on RMSE and fifteen percent on
MAE.

Give me one reason a more flexible model can lose on a held-out year. Then name the thing you
would check before concluding that pyGAM is genuinely the better choice here, rather than luckier
on this particular holdout.

The second question is the operational one. Your forecast is for a grid operator who has to buy
generation in advance, and the two directions of error cost different amounts. If you
under-forecast, you buy at spot prices during a peak. If you over-forecast, you've paid for
spinning reserve you didn't need.

Which of these models would you actually ship, and what would you need to know before you could
answer that at all?

**[pause]**

Here's the caveat the first question is fishing for. This whole comparison rests on a single
held-out year. What would settle it is walk-forward validation over several forecast origins,
which is the lesson of Lab 2, last week, arriving again one lecture later.

And notice that nothing in the table we just built answers the second question. RMSE and MAE are
both symmetric. They charge you the same for being a thousand megawatts too high as for being a
thousand too low. If the costs aren't symmetric, neither metric is measuring the thing you care
about.

That's not a flaw in this lab. It's the gap between forecast accuracy and forecast *value*, and
it's worth carrying into the rest of the course.

Solutions for the coded parts go up on Canvas after class. See you next week in Lecture 4, where we
start on trees.

---

# Appendix — expected output

Reproducible; verified bit-identical across repeated runs. Prophet uses MAP estimation, not MCMC.

| Quantity | Value |
|---|---|
| Data | 2,041 days (train 1,673, test 368), 2017-08-01 → 2018-08-03 |
| Swing — hour of day | 33.3% (min hour 4, max hour 19) |
| Swing — day of week | 11.6% (min Sunday, max Tuesday) |
| Swing — day of year | 44.7% (min day 116, max day 200) |
| January / April / July | 34,403 / 26,945 / 37,043 MW |
| One harmonic, in-sample | 3,428 MW |
| Two harmonics, in-sample | 1,348 MW (**factor of 2.5**, not 3) |
| pyGAM (doy + dow) | RMSE 3,444 · MAE 2,550 |
| Prophet | RMSE 3,935 · MAE 3,001 |
| Prophet (yearly order = 1) | RMSE 4,269 · MAE 3,358 |
| Seasonal naive (364d) | RMSE 4,608 · MAE 3,333 |
| pyGAM effective d.o.f. | 22.5 |

**Two things to know before recording.**

*The MAE inversion is real and worth keeping.* Seasonal naive has a **better MAE** (3,333) than
the single-harmonic Prophet (3,358) while having a worse RMSE. It is the only metric disagreement
in the table and it makes the point about symmetric loss concretely, so do not smooth over it.

*pyGAM beating Prophet is a genuine result, not a fluke of rounding* — it wins on both metrics,
by about 12% on RMSE and 15% on MAE. The honest caveat, which is what the closing question is
fishing for, is that this is a single held-out year. Walk-forward validation over several
origins is what would settle it, and that is Lab 2's lesson arriving again one lecture later.
