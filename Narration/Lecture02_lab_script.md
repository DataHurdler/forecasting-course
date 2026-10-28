# Lab 2 — Recording Script

**ECON 8310: Business Forecasting · Stationarity, ARIMA, and Granger Causality**

Lab: `Labs/Lecture02_lab.qmd` (6 steps) · Measured runtime: **~26 minutes** of narration
(the in-room version is budgeted at 40)

---

## How to use this document

- **`▶ STEP n — Title`** matches the lab's own headings exactly.
- *Italic parentheticals* are stage directions. **[pause]** is a beat; **[STOP — learner works]**
  is where you tell the viewer to pause the video and do something.
- Numbers in this script are the lab's real output, verified reproducible. **One exception,
  flagged in Step 3:** the ARIMA order is a student choice, so their RMSE will differ from
  yours. The script handles that explicitly — do not read a number there as if it were the
  answer.
- Nothing here is sacred. If a sentence doesn't sound like you, change it.

### In-room language that needs replacing

Three spots: Step 2 (*"say it out loud to the person next to you"*) and the two **Discuss**
blocks in the closing. This script converts all three to a pause plus a discussion-board prompt
and flags each divergence from the `.qmd`.

---

# ▶ OPENING

*(Show the rendered lab, at the top of the document.)*

This is Lab 2. It takes two ideas from Lecture 2, this week's lecture on ARIMA and VAR, and today
you get to watch each one bite.

The first is a rule: **test for stationarity before you read an ACF.** The second is a caveat:
Granger causality is not structural causality. Both sound like the sort of thing you nod at in a
lecture and skip in practice. So that's exactly what we'll do. We'll skip them, and watch it go
wrong.

The business question starts small. You're forecasting weekly household demand at store CA_1, and
you want to know whether watching the food aisle helps you predict it.

Then, in Step 6, we leave the store entirely and run the same test on a macro question: does
unemployment predict retail sales? You'll find that the answer depends on which years you include,
and not slightly, either. It changes completely.

**[pause]**

You need two prepared files this time: `m5_weekly.csv`, from `prep_m5.py`, and `fred_monthly.csv`,
from `prep_fred.py`. Run both once from the repository root before you start. The FRED script needs
no API key, and it caches its data to disk, so the lab itself makes no network call.

---

# ▶ SETUP

*(Run the setup cell.)*

The setup follows the same pattern as Lab 1, last week: imports, a path helper, a series accessor
and a train-test split.

```
277 weeks — train 225, test 52
```

It's the same store dataset as Lab 1, with a different store and the same category: CA_1
household. Fifty-two weeks are held out for testing.

The one new import worth noticing is `redirect_stdout`, from `contextlib`, along with `io`. That
isn't statistics. It's damage control, and I'll show you why in Step 5.

---

# ▶ STEP 1 — Look, then test

*(Run the plot.)*

Always look at the series first.

*(Trace the series.)*

The level wanders. It drifts up, it drifts back, and there's no fixed value it returns to. Your eye
says it's non-stationary. Now make the test agree, because your eye isn't evidence you can put in a
report.

*(Run the two ADF cells.)*

```
levels                       ADF   -1.33   p  0.614   lags 12   UNIT ROOT
first difference             ADF   -5.00   p  0.000   lags 11   stationary
```

**[pause]**

Read those two lines carefully, because this is what a *clean* case looks like, and you want it in
your memory for comparison later.

In levels, the ADF statistic is minus one point three three, with a p-value of zero point six one.
That's nowhere near significance. We fail to reject the null of a unit root, so the series is
non-stationary, exactly as your eye said.

After the first difference, it's minus five point zero zero, with a p-value of effectively zero.
That's decisive, so one difference was enough.

That tells you d equals one, and you now have an evidence trail for that choice rather than a hunch.
You'll be asked to justify a d in Homework 1, and this is what the justification looks like.

---

# ▶ STEP 2 — Now you are allowed to read the ACF

*(Run the four-panel figure.)*

There are four panels. The top row is the raw series, and the bottom row is the differenced series.

*(Point at the top-left panel.)*

Look at the top ACF. It crawls down in a long, slow decay, bar after bar, staying well above the
significance band for more than twenty lags.

Here's the trap. That pattern looks like enormously strong autocorrelation, and a reasonable person
reads it as a sign that there's a lot of structure here to model. **It isn't.** That slow decay is
the signature of an *undifferenced* series, and it appears no matter what the underlying process is.
A random walk produces it, and a random walk is just an AR one with a unit root. White noise plus a
trend produces it too.

It tells you nothing about p or q. It tells you only that you skipped a step.

*(Point at the bottom row.)*

Now look at the bottom row, after differencing. That one is interpretable. The bars drop into the
band quickly, and you can actually read structure off them.

**[STOP — learner works]** — *(replaces "say it out loud to the person next to you")*

Pause the video. Using the bottom row and the four reference patterns from Lecture 2, propose a p, a
d and a q.

Write it down before you continue, and write down *why*, meaning which panel and which lags decided
it. Then post it in the Week 2 thread. I want to see the range of proposals, because there genuinely
is one, and that's part of the lesson.

---

# ▶ STEP 3 — Fit it

*(Show the `order` line.)*

Line one of this cell is `order = (1, 1, 1)`. **Replace it with your proposal.** The lab ships with
one, one, one as a placeholder, so that the file renders for everyone. It isn't the answer.

*(Run the cell.)*

*(Note for the narrator: with the shipped (1,1,1) the 52-week RMSE is **569**. Say your own
number aloud, then say the following.)*

With the placeholder order, one, one, one, the fifty-two-week RMSE comes out at five hundred
sixty-nine.

Your number will differ from mine if you chose a different order, and that's fine, because that's
the exercise. What you should be able to do is say why you chose what you chose.

---

# ▶ STEP 4 — Break it: the horizon trap

*(Run the walk-forward loop. It fits an ARIMA at every origin, so it takes a moment.)*

This is the same model with a different evaluation. Watch what happens to the number.

```
walk-forward 1-step RMSE : 352   (172 origins)
single-split 52-step RMSE: 569
```

**[pause]**

That's three fifty-two against five sixty-nine. The walk-forward number is nearly forty percent
smaller, and it's computed from one hundred seventy-two separate forecasts instead of one, so it
even *looks* more rigorous. More origins and more evaluation give you a smaller error.

It's enormously tempting to quote it. Don't.

They aren't two estimates of the same thing, because they answer different questions. Forecasting
one step ahead, knowing everything up to last week and forecasting next week, is a fundamentally
easier problem than standing at a fixed origin and forecasting fifty-two weeks out. Of course the
error is smaller. It would be alarming if it weren't.

The question is which one matches the decision. If a planner is ordering inventory twelve months
ahead, **the fifty-two-step number is the honest one**, and quoting three fifty-two to that person
isn't optimistic. It's wrong.

**[pause]**

This is the most common way I see forecast accuracy overstated in practice, and almost nobody does
it dishonestly. They compute the number that's easy to compute, and it happens to be the flattering
one.

---

# ▶ STEP 5 — Granger causality, and the trap in it

*(Show the helper function.)*

Does the food aisle help predict the household aisle?

First, a practical note. `grangercausalitytests` prints four test variants for every lag up to
`maxlag`, and for a lag of four, that's a screen and a half of output nobody asked for. The helper
swallows it with `redirect_stdout` and pulls out the single F-test we want. That's what those
imports were for.

*(Run it.)*

```
      food -> household  F =  3.83   p = 0.0048   significant
```

Now the argument order, and please write this down: **the second argument is the candidate cause.**
So `granger("household", "food")` asks whether food helps predict household. Getting this backwards
is the single most common mistake with this function, and it's silent. You get a perfectly
reasonable-looking result to the wrong question.

**[STOP — learner works]**

Now it's your turn. Pause the video, run the other direction, and report both F-statistics. It's
one line: uncomment it, then swap the two arguments.

*(After the pause, run it.)*

```
 household -> food        F =  2.80   p = 0.0263   significant
```

**[pause]**

Both directions are significant. Food predicts household, and household predicts food.

Sit with how strange that is. Household goods don't cause groceries, and groceries don't cause
household goods. Nobody buys paper towels *because* they bought bananas. And yet the test says each
one helps predict the other, at conventional significance, in both directions.

We'll come back to what's generating that, so hold on to the question.

---

# ▶ STEP 6 — The same test on macro data, and why the sample matters

*(Run the FRED cell.)*

Now for the question from the Granger slides in Lecture 2, on real macro data: does unemployment
Granger-cause retail sales?

Both series arrive already stationary. Retail growth is the log-differenced monthly percent change,
and unemployment is the monthly change in the rate. So we can test directly.

```
Full sample: 1992-02 to 2026-06 (412 months)
   unrate_diff -> retail_growth   F =  18.22   p = 0.0000   significant
 retail_growth -> unrate_diff     F =  12.80   p = 0.0000   significant
```

**[pause]**

Both directions are significant, and unemployment is the stronger of the two, with an F of eighteen
point two two against twelve point eight. That matches the story everybody expects: people lose
jobs, so people stop spending. You could write that paragraph without running anything, which is
exactly why you should be suspicious of how satisfying it is.

Now let's drop the COVID window and run it again.

*(Point at the comment in the cell.)* There's one detail in the code to notice before you run it.
The lags are built on the unbroken series first, and only then are the COVID months dropped, along
with the four months after them, whose lags still reach into the window. If you dropped the rows
first, the test would treat January twenty twenty and July twenty twenty-one as neighbors.

*(Run the ex-COVID cell.)*

```
Excluding Feb 2020 - Jun 2021 (395 months)
   unrate_diff -> retail_growth   F =   2.42   p = 0.0480   significant
 retail_growth -> unrate_diff     F =   9.96   p = 0.0000   significant
```

That's seventeen months removed out of four hundred twelve, or four percent of the sample. The
unemployment-to-retail F-statistic falls from **eighteen point two two to two point four two.** It's
still significant, but only just, with a p-value of zero point zero four eight, and it has lost
about seven-eighths of its strength.

The reverse direction went from twelve point eight to nine point nine six. That's only a little
weaker, and still overwhelming.

**[STOP — learner works]**

Now it's your turn. Pause the video, restrict the sample to everything *before* February twenty
twenty, with no COVID period at all and no recovery either, and run both directions.

*(After the pause, run it.)*

```
Pre-COVID, before Feb 2020 (336 months)
   unrate_diff -> retail_growth   F =   1.85   p = 0.1196   NOT significant
 retail_growth -> unrate_diff     F =   9.67   p = 0.0000   significant
```

**[pause]**

Put the three numbers side by side: eighteen point two two, then two point four two, then one point
eight five.

On twenty-eight years of pre-pandemic data, unemployment does **not** Granger-cause retail sales.
The intuitive direction, the one everybody expects, the one that has a mechanism, and the one you'd
put in a slide without checking, is gone.

And the direction nobody leads with, retail predicting unemployment, survives every single cut:
twelve point eight, then nine point nine six, then nine point six seven. Its p-value never rises
above zero point zero zero zero one.

---

# ▶ BEFORE YOU LEAVE

*(Show the closing callouts.)*

There are two questions here, and I want both of them in the Week 2 thread before Lecture 3.

**[STOP — learner works]** — *(replaces "Discuss (Step 5)")*

The first question is about the aisles. You found a two-way relationship between household goods
and groceries, and neither one causes the other. So what's generating it?

And then there's the commercial half, which is the one that matters. Given what you found, is there
anything a planner is *entitled* to do with it? Name one legitimate use, and one thing they must not
conclude.

**[STOP — learner works]** — *(replaces "Discuss (Step 6)")*

The second question is about the macro result. One analyst runs the full sample and reports that
unemployment predicts retail sales. Another runs nineteen ninety-two to twenty nineteen and reports
that it doesn't.

And neither of them has made an **error**. Both ran the correct test, correctly, on a defensible
sample. That's what makes this worth twenty minutes of your attention, rather than being a
cautionary tale about sloppy work.

So what actually happened? And what does it tell you about a Granger result computed on a sample
that contains one enormous shared shock?

Here's one hint worth using. Retail sales are a coincident-to-leading indicator, and unemployment is
a *lagging* one. Firms cut hours and stop hiring after demand falls, not before. Does that help
explain which direction survived every cut?

**[pause]**

Here's where I'd land on that second question. COVID supplied one enormous shared shock. Both
series moved violently in the same few months, and on the full sample those months were enough to
make unemployment look like the stronger predictor. Take them out and that direction weakens, and
before twenty twenty it's gone. So a Granger result on a sample like that can be reporting one
episode rather than how the two series usually move, and it's worth rerunning it without the shock
before you report it. The hint fits, too. Spending falls first and layoffs follow, which is why
retail predicting unemployment survived every cut. And even then, it predicts. It doesn't cause.

Solutions for the coded parts go up on Canvas after class. Next is Lecture 3, on generalized
additive models, where we leave retail data entirely, because it can't show you what we need to
see.

---

# Appendix — expected output

Reproducible; verified across repeated runs. **Step 3 excepted** — that number depends on the
order the student chooses.

| Quantity | Value |
|---|---|
| Series | CA_1 HOUSEHOLD, 277 weeks (train 225, test 52) |
| ADF levels | −1.33, p = 0.614, 12 lags → unit root |
| ADF first difference | −5.00, p < 0.001, 11 lags → stationary |
| ARIMA(1,1,1) 52-step RMSE | 569 *(shipped placeholder order only)* |
| Walk-forward 1-step RMSE | 352, over 172 origins |
| food → household | F = 3.83, p = 0.0048 |
| household → food | F = 2.80, p = 0.0263 |
| **Macro, full sample (412 mo)** | unrate→retail F = 18.22 · retail→unrate F = 12.80 |
| **Ex-COVID (395 mo)** | unrate→retail F = 2.42 (p = 0.048) · retail→unrate F = 9.96 |
| **Pre-COVID (336 mo)** | unrate→retail F = 1.85 (**p = 0.1196, not significant**) · retail→unrate F = 9.67 |

**Note for the instructor.** The FRED full sample ends 2026-06, so it moves as `prep_fred.py` is
re-run. The three F-statistics will drift slightly between terms; the *pattern* — collapse in one
direction, survival in the other — is what's stable, and it's what the discussion questions rest
on. Re-run this appendix before recording if the cache is more than a term old.

**Three things to know before recording.**

*Step 6 is the only place in the course where a Granger verdict flips with the sample window.*
`UNRATE` to `RSXFS` is significant on 1992-2026 and gone before 2020, because COVID supplies one
enormous shared shock. That is the point of leaving M5 for one step — no retail series in this
course has an episode that can teach it — so protect its screen time.

*The clean ADF verdict here is deliberate, and worth naming.* CA_1 HOUSEHOLD gives the same answer
at every lag order. Homework 1 uses the messier series where the verdict flips, so students meet
the unambiguous case first and have something to compare against when it stops being
unambiguous.

*Granger causality is not causality, and the lab's own title says "the trap in it".* Say the
disclaimer in the same breath as the result, not after it — a student who writes "unemployment
causes retail sales" in Homework 1 has taken the wrong sentence away.
