# Lab 1 — Recording Script

**ECON 8310: Business Forecasting · Benchmarks and Exponential Smoothing**

Lab: `Labs/Lecture01_lab.qmd` (6 steps) · Measured runtime: **~24 minutes** of narration
(the in-room version is budgeted at 35, because students type)

---

## How to use this document

- **`▶ STEP n — Title`** marks a section of the lab. The headings match the `.qmd` exactly, so
  a student following along in their own file is never lost.
- *Italic parentheticals* are stage directions — screen actions, not speech.
- **[pause]** means stop for a beat. **[STOP — learner works]** is different: it is where you
  tell the viewer to pause the video and do something. Say it out loud; people do not pause
  unless told.
- Every number in this script is the number the code printed when the lab page was rendered.
  No seeds are involved, so every row should match your screen exactly — except Holt-Winters.
  Its optimizer lands on a parameter boundary, so its numbers depend on the platform: 885 / 814
  on the rendered page, 875 / 804 on Linux with the pinned library versions. If any *other*
  number disagrees with this page, something upstream changed and you should stop and find out
  what.
- Nothing here is sacred. If a sentence doesn't sound like you, change it.

### One thing to fix before recording

The lab has two moments written for a room: Step 4 says *"discuss with the person next to you"*
and the closing says *"Discuss."* Online, that instruction is dead air. This script replaces both
with a **[STOP — learner works]** beat plus a discussion-board prompt. If you also teach this
live, the `.qmd` can stay as it is — the script diverges deliberately, and the divergence is
noted at each spot.

---

# ▶ OPENING — before you open the file

*(Screen: the rendered lab page, top of document.)*

This is Lab 1, and it runs about thirty-five minutes if you type along — which you should.

Lecture 1 made a claim that probably sounded like housekeeping: always beat a benchmark first.
Establish the dumb model, then earn your way past it. It's the kind of thing that sounds like
process rather than substance.

Today you're going to test that claim on a real Walmart series. You'll find it holds. And then,
in the last five minutes, you're going to watch it fail on a different series from the *same
store* — and that failure is the actual lesson.

The business question is concrete. You're forecasting weekly household-goods demand at store
TX_1 to set inventory a year out. Fifty-two weeks. Which method do you ship?

**[pause]**

You need `pandas`, `numpy`, `matplotlib`, `statsmodels`, and the processed M5 file. If
`data/processed/m5_weekly.csv` isn't there, run `python scripts/prep_m5.py` from the repository
root first — it takes a minute or two and you only ever do it once.

---

# ▶ SETUP — nothing to fill in

*(Screen: run the setup cell. Let the output appear.)*

Run this one and move on. It imports, it loads, it splits.

The one piece worth pointing at is `data_path`. It walks up from wherever you are looking for
`data/processed/`, so the lab renders whether you run it from the repository root or from inside
the `Labs` folder. That is not clever — it's defensive. Path errors are the single most common
reason a submission doesn't render, and you'll meet that again in the homework.

The split is the part that matters:

```
277 weeks total — train 225, test 52
```

Two hundred and seventy-seven weeks, a bit over five years. We hold out the last fifty-two —
one full year — and the model never sees it. Not for fitting, not for tuning, not for choosing.
That is the whole discipline of honest evaluation and everything downstream depends on it.

**[pause]**

---

# ▶ STEP 1 — Look at it first

*(Screen: run the plot cell.)*

Never fit anything before you've looked at the series. This takes ten seconds and it will save
you an hour.

*(Point at the plot as you talk.)*

Two things. First, the level roughly doubles across five years — this store's household aisle is
growing, and growing substantially. Second, and this is the one to hold onto: **the level is
still moving at the end of the training window.** It hasn't flattened out. Whatever we fit has
to say something about where the level goes next.

That single observation is the whole lab. Everything that happens in the next twenty minutes is
a consequence of the level moving.

**[pause]**

---

# ▶ STEP 2 — Establish the benchmark

*(Screen: run the seasonal-naive cell.)*

Before any model, the thing to beat.

Seasonal naive says: this week next year looks like this week last year. That's it. One line —
`y.shift(52)`. No parameters, no fitting, no estimation.

```
Seasonal naive  RMSE 973
```

Nine hundred and seventy-three. Remember that number, because everything from here is measured
against it. If a model can't beat 973, it does not deserve to be in production no matter how
sophisticated it is.

**[STOP — learner works]**

Pause here. Before you run anything else, write down a number — an actual number, on paper or in
a comment. What RMSE do you think a proper exponential smoothing model will get on this series?

I'm asking you to commit because in Step 4 you're going to check it, and the gap between what
you expected and what happened is worth more to you than the result itself. Guessing privately
and then quietly adjusting is not the same exercise.

Write it down. Then come back.

---

# ▶ STEP 3 — Fit the three models

*(Screen: run the three-model cell. It takes a few seconds — Holt-Winters is estimating 52
seasonal indices.)*

Three models, three lines, all straight from Lecture 1.

SES — level only. No trend, no seasonality. The simplest thing that could work.

Holt — level plus trend. Undamped for now; we come back to that in Step 5, and the fact that
I'm flagging it should tell you something.

Holt-Winters — level, trend, and seasonality, additive, period 52.

*(Screen: highlight the callout box.)*

One constraint worth respecting. Holt-Winters estimates a seasonal index for every one of the
52 weeks, and it needs two full cycles — 104 observations — just to initialize. You have 225
training weeks, so you're fine. You will not always be fine. When you get to a series with three
years of weekly data and you're wondering why Holt-Winters won't fit, this is why.

**[STOP — learner works]**

Before you scroll to Step 4: predict the ordering. Which of these three wins, which comes last?
Write it down next to your number from Step 2.

Most people rank them by sophistication — Holt-Winters first, because it models the most.
See whether you did.

---

# ▶ STEP 4 — Compare honestly

*(Screen: run the table cell.)*

```
                RMSE   MAE
SES              427   352
Holt-Winters     885   814
Seasonal naive   973   848
Holt            1166  1029
```

**[pause]** — *let this sit on screen. Do not talk over it.*

Read that from the bottom.

**Holt is last.** Eleven sixty-six. Worse than the benchmark — worse than shifting the series
back a year and calling it a day. Holt is the model that adds a trend to SES, and this series
has an obvious trend. It should have helped. It made things twenty percent worse than doing
nothing.

**Holt-Winters is third.** Eight eighty-five here — you may see eight seventy-five, because
this one fit depends on the optimizer and the platform. Either way it beats the benchmark, but
only just, and it loses badly to a model with no trend and no seasonality in it at all. This is
the most sophisticated thing on the table and it's mid.

**SES wins, and it isn't close.** Four twenty-seven against a benchmark of 973 — less than half
the error. The simplest model on the list, the one that only tracks the level, beat everything.

**[pause]**

Now compare that to what you wrote down. If you predicted Holt-Winters would win because it
models the most structure, you are in good company and you have just learned the most useful
thing in this lab: **more structure is not more accuracy.** Every parameter Holt-Winters
estimates is a parameter that can be estimated badly, and 52 seasonal indices from 225
observations is a lot of estimating.

**[STOP — learner works]** — *(replaces the in-room "discuss with the person next to you")*

Pause and answer these in the discussion thread for Week 1:

1. Did any model beat the benchmark, and by how much?
2. Was the winner the model you predicted in Step 3?
3. Holt is worse than doing nothing. Before you watch Step 5 — what's your hypothesis for why?

Post before you continue. The next step gives away the answer to number three, and the guess is
worth more than the answer.

---

# ▶ STEP 5 — Break it, then fix it

*(Screen: run the forecast plot.)*

Here's what went wrong. Look at where Holt's forecast goes.

*(Trace the Holt line with the cursor, left to right, all the way up.)*

That's a straight line, and it does not stop. Holt estimated a positive trend from the training
data — correctly, the series *was* growing — and then extrapolated it linearly for fifty-two
consecutive weeks. By the end of the horizon it's forecasting a level the store has never
reached.

Lecture 1 warned that undamped linear extrapolation stops being credible after about six
periods. You are forecasting fifty-two. The model isn't wrong about the trend; it's wrong about
the trend continuing forever.

Meanwhile SES, the dashed flat line, just says "about here" — and about-here turns out to be a
much better guess than a confident straight line pointed at the ceiling.

**[STOP — learner works]**

Your turn, and it's genuinely one keyword argument. Uncomment the block and fill in the blank:

```python
holt_damped = Holt(train, damped_trend=True).fit()
```

That's the fix. Add it to the table, print the result, and print the estimated phi. Pause here
and do it — it's thirty seconds of typing and you'll remember it because you typed it.

*(Screen: after the pause, run it.)*

```
Holt (damped)    426   351
SES              427   352
```

Phi comes out at **0.800**.

**[pause]**

Three things here, and the last two are the honest ones.

First: damping rescued the model. Eleven sixty-six down to four twenty-six — from worst on the
table to best.

Second: look hard at that phi before you interpret it. Statsmodels only searches phi between
0.8 and 0.995, and 0.800 is the floor of that range, to the third decimal. The optimizer didn't
find eighty percent; it ran into the wall, because it wanted more damping than it was allowed.
So "phi is 0.8" is a fact about the software's default bounds, not about this store. An
estimate sitting exactly on the edge of its allowed range is the first thing to distrust in any
fitted model.

And it isn't the same trend estimate, either. The undamped model carries a slope of about
thirty units a week into the test year. The damped refit set its trend smoothing to zero, so
its starting slope just shrank by a fifth every week for all 225 training weeks — down to
effectively nothing. The damped forecast isn't a bending trend at all. It's a flat line at
6,688 units, within half a unit of SES's flat line.

Third — and don't let this slide past — **damped Holt beat SES by one unit of RMSE.** Four
twenty-six against four twenty-seven. That is not a win, and now you know why: they are the same
forecast. If you report "the damped model is best" you're reporting noise as a finding.

The honest summary is: *on this series, models that keep the forecast near the current level
beat models that extrapolate — and the damped model got there by switching its trend off.*

---

# ▶ STEP 6 — Same store. Different aisle. Opposite answer.

*(Screen: run the FOODS cell.)*

Change one word. `HOUSEHOLD` becomes `FOODS`. Same store, same code, same horizon, same fifty-two
weeks held out.

```
TX_1 FOODS   seasonal naive RMSE 1308
TX_1 FOODS   SES            RMSE 1474
```

**[pause]**

The ranking flips.

On the household aisle, SES beat the benchmark by more than half. On the food aisle of the same
store, the benchmark wins and SES is about thirteen percent worse. Nothing changed except which
shelf we're looking at.

So what's different? It isn't the store. It isn't the method — the code is character-for-character
identical apart from one string.

*(Screen: point at the two percentages in the lab text.)*

It's visible in one number. For household, the test year sits about **fifteen percent above** the
year before it. The level moved, so seasonal naive carried a stale level forward and lost badly —
and a method that tracks the level won.

For foods, the level barely moved — about **one and a half percent down**. When the level doesn't
move, last year is an excellent guess, and there's nothing for a level-tracking model to add.

**It is not the store, and it is not the method. It is whether the level moved.**

**[pause]**

That's the sentence to leave with. And notice what it does to the claim we opened with — "always
beat a benchmark first" isn't process hygiene. It's the only thing that told us the answer was
different on these two series. If you'd fit SES on household, seen it win, and rolled it out
across the store, you'd have shipped a model that's thirteen percent worse than doing nothing on
the food aisle, and nothing in your workflow would have told you.

---

# ▶ BEFORE YOU LEAVE

*(Screen: the closing callout.)*

**[STOP — learner works]** — *(replaces the in-room "Discuss")*

Here's the question, and it's the one that shows up in your job rather than in your homework.

You now have two series where the best method is different. You are responsible for all thirty
series in this dataset, and you have to pick **one** approach to run in production every month.

What do you actually do — and what would you need to measure to decide?

"Fit everything and pick the winner per series" is a legitimate answer. But say what it costs
you: thirty model selections a month, each one a chance to overfit the held-out year; a pipeline
that can silently switch methods between runs; and no single story to tell the person who asks
why the forecast changed.

Post your answer in the Week 1 thread before the next lecture. Two or three sentences. There
isn't a key for this one — I want to see how you reason about it, and we'll pick it up in
Lecture 2 when the methods get more complicated and the temptation gets worse.

**[pause]**

Solutions for the coded parts go up on Canvas after the deadline. See you in Lecture 2.

---

# Appendix — expected output

Everything the lab prints, for checking against your screen. These are deterministic, with
one exception: the Holt-Winters fit sits on a parameter boundary, so its row depends on the
optimizer and the platform. For every other row, if yours differ, the data changed.

| Quantity | Value |
|---|---|
| Series length | 277 weeks (train 225, test 52) |
| Seasonal naive RMSE | 973 |
| SES RMSE / MAE | 427 / 352 |
| Holt (undamped) RMSE / MAE | 1166 / 1029 |
| Holt-Winters RMSE / MAE | 885 / 814 on the rendered page; 875 / 804 on Linux with the pinned versions (optimizer-dependent) |
| Holt (damped) RMSE / MAE | 426 / 351 |
| Estimated phi | 0.800 (statsmodels' lower bound; the fitted trend is ≈ 0) |
| SES estimated alpha | 0.621 |
| TX_1 FOODS seasonal naive RMSE | 1308 |
| TX_1 FOODS SES RMSE | 1474 (12.7% worse) |
| HOUSEHOLD test year vs prior year | +14.5% |
| FOODS test year vs prior year | −1.5% |

**Note for the instructor.** Holt-Winters may emit a `ConvergenceWarning` on this series,
depending on the platform: on Linux with the pinned versions it converges without one, to
875 / 804. The fit lands on a parameter boundary (trend and seasonal smoothing both 0), so the
optimum shifts slightly from machine to machine. The lab suppresses warnings globally in Setup,
so students will not see a warning either way. None of this changes the conclusion (Holt-Winters
loses to SES either way), but it is a second, quieter piece of evidence for the same point: that
model is being asked to estimate more than this sample supports.

**Three things to know before recording.**

*The contrast with Homework 1 is the whole lab, and it is easy to under-sell.* Lab 1 runs on
TX_1 HOUSEHOLD, which shifts about 15% in the test year, so seasonal naive carries a stale level
and loses. Homework 1 runs on CA_1 FOODS, stable inside 5%, where the benchmark wins. Same code,
opposite conclusion — say explicitly that the difference is the series, not the method, or
students will generalize whichever one they saw first.

*Do not let the benchmark's loss here undo Lecture 1's lesson.* A student who watches seasonal
naive lose in the lab and win in the homework should conclude "compute it every time", not
"benchmarks are weak". Land that in the closing.

*The blanks are commented and must stay commented.* Every `___` in this lab sits inside a comment
block, including the dependent lines, because the lab has to render with its blanks unfilled. If
you edit the exercise on camera, keep the comment markers.
