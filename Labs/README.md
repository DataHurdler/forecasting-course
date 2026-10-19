# Labs

The third part of each class session. Your syllabus describes a class as three parts —
review the previous topic, learn the theory, then **try the new technique on the computer**.
These are that third part.

## Naming

One lab per lecture, matching the `Narration/` short-form convention:

```
Labs/Lecture01_lab.qmd
Labs/Lecture02_lab.qmd
...
```

## Format

Quarto `.qmd`, not `.py` or `.ipynb`:

- Students already write `.qmd` for homework — one format, not three
- It renders to HTML for the course site *and* executes, so one file is both the handout
  and the runnable notebook
- It diffs cleanly in git; notebooks rewrite output cells and metadata on every run

## What a lab is, and is not

A lab is **guided and completable in the room** — roughly 30–40 minutes, with the scaffolding
already written and the student filling in the interesting parts. Homework is the opposite:
longer, unscaffolded, and done alone.

The rule of thumb: if a student can get stuck for twenty minutes with no way forward, it
belongs in homework, not a lab.

## Student blanks — the rule that keeps labs renderable

A lab must render with its blanks still blank. Two consequences, learned the hard way when
Labs 1 and 2 shipped with bare `___` on live lines and `quarto render` died on
`NameError: name '___' is not defined`:

**1. A blank is always inside a comment.** Never a live statement.

```python
# ---- your turn: uncomment and fill in the one keyword argument ----
# holt_damped = Holt(train, ___).fit()       # <- damped_trend=True
#
# results["Holt (damped)"] = (rmse(test, holt_damped.forecast(N_TEST)),
#                             mae(test, holt_damped.forecast(N_TEST)))
```

Comment the *dependent lines too*, or the block still breaks.

**2. Blanks go in leaf positions only.** Never blank a value that later steps consume — the
comment would take the rest of the lab down with it. If the exercise is genuinely load-bearing,
either give the line and move the thinking downstream, or supply a clearly-marked starter:

```python
order = (1, 1, 1)     # <- REPLACE with your proposal from Step 2
```

The test is one command, and it is not optional before a lab ships:

```bash
quarto render Labs/LectureNN_lab.qmd
```

## Convention

- Every lab runs end-to-end on the course dataset with no manual steps
- `random_state=42` wherever a result is quoted
- `TimeSeriesSplit`, never `KFold`
- Each lab ends with one question that has no code answer — something to argue about before
  the session closes

Start from `_TEMPLATE_lab.qmd`.

## Overlap with homework

**Revised 2026-08-25.** The original rule was that a lab must never touch the series its homework
asks about. That is no longer a requirement.

Overlap is fine. If a lab hands students the answer to a homework question or two, that is often
a good thing — particularly where the question is scaffolding rather than analysis. In Homework 5
the hard part is not the feedforward network, it is building a windowing `Dataset` and getting
the tensor shapes right. A student who loses three hours there has learned nothing the assignment
was testing.

**But do not force it either way.** Build the lab that teaches the lecture best. Sometimes that
means reusing the homework's setup; sometimes a different series makes the point better, as in
Labs 1 and 2 where the *contrast* between series is the lesson. Both are fine.

The one thing to protect is a result the homework is designed to make surprising — Homework 1's
benchmark winning, Homework 3's planted-noise finding, Homework 5's architecture ranking. Give
away the setup freely; leave the punchline.

The table below records what each lab uses, mostly so collisions are visible when they matter.

## Series map

| | Primary series | Contrast / second series |
|---|---|---|
| **Lab 1** (Week 1) | TX_1 HOUSEHOLD | TX_1 FOODS |
| **HW1 Parts 1–2** (due Wks 2–3) | CA_1 FOODS | CA_3 FOODS |
| **Lab 2** (Week 2) | CA_1 HOUSEHOLD | CA_1 FOODS *(Granger step only)* |
| **Lab 2** Step 6 | FRED `UNRATE` + `RSXFS` | monthly macro, not M5 |
| **HW2** (due Wk 4) | M5 daily, CA_1 FOODS | migrated 2026-08-25 |
| **Lab 3** (Week 3) | PJM electricity, daily | not M5 — see below |
| **Lab 4** (Week 4) | M5 daily, CA_1 FOODS | same series as HW2, *after* HW2 is due |
| **Lab 5** (Week 5) | M5 daily, CA_1 FOODS | continues Lab 4 on the same series |
| **HW3** (due Wk 6) | M5 weekly, all 30 series | pooled panel |
| **Lab 6** (Week 6) | M5 daily, CA_1 FOODS, 62 features | completes the Lab 4–6 arc |
| **HW4** (due Wk 7) | M5 weekly, 46 features | different frequency and panel |

Lab 1 and Lab 2 are deliberately built so the *answer differs* from the homework's:

- **Lab 1**: TX_1 HOUSEHOLD shifts +15% in the test year, so seasonal naive carries a stale
  level and loses. HW1's CA_1 FOODS is stable within 3%, so the benchmark wins. Same code,
  opposite conclusion — that contrast is the lesson.
- **Lab 2**: CA_1 HOUSEHOLD gives a clean ADF verdict at every lag order. HW1 deliberately
  uses the messy series where the verdict flips, so students meet the clean case first.

Lab 3 leaves M5 entirely, for the reason ADR-0001 gives: a GAM's whole selling point is
several seasonal cycles at once, and retail has one. PJM electricity demand has three large
ones (hour-of-day 33%, day-of-week 12%, day-of-year 45%) and a **double-peaked** annual shape —
July air conditioning and January heating — that a single harmonic cannot fit. That failure is
Step 2 of the lab. It also keeps Lab 3 clear of HW2, which stays on M5 daily.

Lab 4 deliberately **reuses HW2's series** — daily CA_1 FOODS. That is safe because HW2 is due
*before* Week 4's class, so nothing is spoiled, and it buys a direct method comparison: students
fit a GAM to that series one week and a tree to it the next, on identical data. It also stays
clear of HW3, which pools all 30 weekly series at a different frequency.

Labs 4 and 5 run as a **pair on one series** — daily CA_1 FOODS. Lab 4 ends with students
averaging 20 bootstrap trees by hand (324 → 266); Lab 5 opens by replacing that with a real forest
(248) and closes on XGBoost (244). One scoreboard, five models, one series.

They do not spoil HW3, which pools all 30 weekly series and asks a different question. Two
deliberate divisions of labour: Lab 5 shows MDI and permutation **disagreeing on real features**
(`snap` moves from 8th to 3rd), while HW3 makes students *plant noise controls* to prove why. And
Lab 5 finds `max_features="sqrt"` losing on 10 features with one dominant predictor; HW3's
27-feature panel is dominated by `units_lag1` too, so `sqrt` loses there as well.

Lab 2 Step 6 leaves M5 on purpose. It delivers the `UNRATE` → `RSXFS` question promised on the
Lecture 2 slide, and it is the only place in the course where a Granger verdict **flips with the
sample window** — significant on 1992–2026, gone before 2020, because COVID supplies one enormous
shared shock. M5 has no episode that can teach that.
