# Figures, round 2: the rest of the ranked lists for Lectures 9–13

**Status:** COMPLETED 2026-10-16 (committed, not pushed; owner to review). APPROVED 2026-10-16. The owner asked what was planned but not built; I listed it and
suggested building in teaching-calendar order, starting with Lecture 9 Part 1 (taught next week)
and leaving L01–L07 until after the semester. Owner: "Okay. Do as you suggested." Then, minutes
later: "I don't want the other updates to happen after the semester." So L01–L07 join this round
(builders G–J below).
**Roadmap:** [visuals-and-geometry audit](2026-10-13_visuals-and-geometry-audit.md), per-deck
sections. **House rules, word diet, report format:** exactly as in the
[round-1 rollout](2026-10-13_figures-rollout.md) ("House rules", "Hands off", "Report").

## Scope

The non-★ candidates that were not built on 2026-10-13, for the decks not yet taught. Items the
audit ranked **Low** are out of scope, and so are two blocked items:
- **L10 Bayes mosaic** — needs the owner to define a "strong week".
- **L13 skill bars** — needs the HW01 key reconciled first.

| Builder | Deck (week) | Figures |
|---|---|---|
| A | L09 Part 1 (wk 9) | Check 2: the 30 series on one axis (91% between-series variance, as Pythagoras: total² = between² + within²) plus the mean ACF |
| A | L13 (wk 16) | Reuse A's 30-series figure on the frame that cites the 91%; a KFold row added to the train/test timeline on the pipeline-errors frame; the L08 March-2020 aisle photo (`Figures/Lecture08/source/aisle-march-2020.jpg`, see its SOURCE.md) for structural breaks |
| B | L09 Part 2 (wk 10) | Recurrence = 52 hops vs attention = one arc (path length); attention's cost as the area of a T×T square |
| C | L10 (wk 11) | Four Metropolis chains from scattered starts, warm-up shaded, plus their histogram against the exact Beta(35,215); priors by support (N(0,1) on σ puts half its mass on impossible values; the HalfNormal fold); prior-predictive counts out of 200 for flat / Beta(5,45) / Beta(500,4500) |
| D | L11 Part 1 (wk 12) | Structural decomposition: observed / level / seasonal / residual stacked on a shared scale (top = vertical sum of the rest); centred vs non-centred traces (the funnel itself stays in Part 2). The level random-walk fans only if they add something `band_width` does not already show |
| E | L11 Part 2 (wk 13) | The question: a strip plot of the raw SNAP effect by category across 30 stores (no MCMC) |
| F | L12 (wk 14–15) | Posterior as an area: b_price density with its HDI, the area left of −0.5 (P = 0.46) and a line at −1 marking revenue break-even; the DAG the "Directed Acyclic Graphs" frame describes but does not draw; the ROPE on b_event |

**Added 2026-10-16 — the taught decks (L01–L07)**, same rules, non-Low candidates only:

| Builder | Deck | Figures |
|---|---|---|
| G | L01 | The four benchmarks across the test year with RMSEs (seasonal naive 1,660 / drift 1,764 / naive 1,794 / mean 1,881), each as its geometric shape; ETS 3×3 grid of forecast shapes |
| G | L02 | Stationarity as a fixed-width band (levels vs differences, ADF in titles); Granger with COVID shaded; SARIMA daily ACF before/after the seasonal difference |
| H | L03 | Prophet component panels; additivity forces parallel curves vs the real weekday/weekend gap; Fourier order (one wave vs adding the 182.5-day wave) |
| H | L04 | The real tree's partition of the lag4 × lag52 plane beside it; train/test RMSE vs `max_depth`; 20 stumps vs 20 deep trees for one test point |
| I | L05 | Bagging variance ρ + (1−ρ)/B vs B; boosting stages 1/2/5/50; the OOB grid sorted by date shows the leak |
| I | L06 | LASSO path coloured by weekly lag; test RMSE vs n/k (OLS vs ridge vs benchmark); elastic-net ball on the same ellipses; CV curve with the one-SE rule unless it gives away a lab "Your turn" |
| J | L07 | Gradient descent on the real lag7/lag14 bowl (smooth / zig-zag / diverge at 2/λ_max); train vs validation by epoch, 3 seeds; weight matrices to scale; activation slopes |

## After the builders

Coordinator: verify every deck (Beamer 0/0, slide-qa, parity, narration join, gates), publish with
`sync_to_docs.sh slides`, re-render any lab whose prose changed, update the audit plan's build
table and its stale ○ marks, commit. **Push only after the owner says so** (these are student-facing).

## Outcome (2026-10-16, coordinator)

All ten builders reported; the coordinator re-verified every deck afterwards.
- **36 new figures** across all 14 decks except L08. Pages grew: L02 22→24, L03 19→20, L04 20→21,
  L05 25→27, L06 19→21, L07 21→23, L10 20→21, L11P1 19→20, L12 16→17, L13 21→22.
- **Beamer:** all 15 decks 0 errors, 0 overfull, 0 undefined. **HTML:** published with
  `sync_to_docs.sh slides`; slide-qa 0 overflow. **Parity** 15/15; every narration's `Deck:` line
  and last slide match its PDF; `check-book-claims` 10/10; `check_links` 0 broken; assignment sync ok.
- **Words:** every frame in every deck is now ≤ 90 (the coordinator trimmed L13 "Lesson 1" and
  "Closing", L09P1 BPTT and takeaways, L08 takeaways).
- **Labs re-rendered and published:** Lab 4 (Step 2 sentence now true) and Lab 11 Part 1 (Step 2
  "same model" corrected).

**Corrections found while building:**
- L04: "full tree: zero bias" → zero *training* error; "depths 5–10 usually useful" → test error stops
  improving at depth 3 on this data.
- L10: the deck printed ESS 3,686, which is the **lab's** Beta(2,2) SNAP model, not the deck's
  Beta(5,45) one, and the deck's code had no seed. The code now passes `random_seed=42` and prints
  3,290 (reproduced by the coordinator).
- L11P1: the centred and non-centred fits are **not** the same model. `GaussianRandomWalk`'s default
  start is Normal(0, 100); the hand-written walk starts near 0. σ_level is 449 vs 517. Prose corrected
  to "almost the same model" in deck, lab and both narrations; the models are unchanged.
- L01 narration: auto-ETS cannot return a "multiplicative trend" (not among the 15).

**Numbers that differed from the audit's estimates (figures print the measured values):** L07
2/λmax = 0.57 (audit 0.56); L07 best held-out epochs 118 / 143 / 176 (audit 80–132) — superseded 2026-10-18: the figure now chooses on a validation slice (see below).

**Owner calls (reported, nothing edited):**
1. L11P1: make the two lab models identical? Either change moves σ_level, the interval widths,
   coverage, RMSE, the 2,376 ESS, HW06 Part 1 and CLAUDE.md. The instructor key
   (`solutions/HW06_Part1_solutions.qmd:113`) repeats "Same model, same posterior".
2. L12: the ROPE width (±5%, taken from HW07 Q5, stated on the slide as a business choice), and the
   lecture now demonstrates HW07 Q5's method on one predictor (b_event).
3. L07: Lab 7 trains 60 epochs; held-out error keeps flattening to ~118–176. Train longer?
4. L09P2: the quadratic-cost warning moved from the heatmap slide to the new cost-square slide.
5. L10 MCMC shown as two figures on two frames (walk, then histogram), and a new prior-predictive frame.
6. Lab 6 Step 3 asks students to find the lag pattern "before reading on"; the L06 path slide now shows it.

## Owner decisions, 2026-10-18

The owner reviewed the round, said "Push." (ac5a386 pushed and confirmed live), then decided:
1. L11P1 models: **keep as they are** (Option A). Done: the HW06 Part 1 key's "Same model, same
   posterior" became a grader's note (instructor repo + the gitignored local copy).
2. L12 ROPE: **keep** (Option A); the overlap with HW07 Q5/Q1 is accepted.
3. Lab 7: **stays at 60 epochs**.
4. L09P2 cost warning: **keep** on the cost-square slide.
5. L10 two-slide sampler and prior-predictive split: **keep**.
6. Lab 6 Step 3: **reword**. Done: the prompt now asks which survivor is not a multiple of 7 and
   why the LASSO keeps it; the callout answers (`lag1`, the freshest reading of the level); the lab
   narration matches.

The two blockers, "take actions as you recommend":
- **L10 Bayes mosaic**: a "strong week" is one **above its own calendar year's median**. Measured on
  CA_1 FOODS (277 weeks, high-SNAP = `snap_days >= 5` as in Lab 10): P(SNAP) 0.26, P(strong | SNAP)
  0.76, P(strong | no SNAP) 0.40, P(SNAP | strong) 0.40.
- **HW01 reconciliation, then the L13 skill bars**: keys and narration scripts reconcile to the
  walkthrough (students' own statsmodels calls); damped Holt uses `ExponentialSmoothing`; the two
  grading notes that would mark a correct student wrong are fixed first. The skill bars then take
  their HW01 rows from the same walkthrough code.

### Outcome of the 2026-10-18 follow-ups (coordinator)
- **L10:** new frame 8 "Bayes' Theorem, Drawn" (`bayes_mosaic`); 22 pages. All five probabilities
  reproduced and asserted; the pattern holds in each of the six years.
- **HW01 reconciled** (forecasting-instructor): both keys and both narration scripts now quote the
  walkthrough output; the two grading notes are fixed; PDFs rebuilt with `build_narration_pdfs.py`;
  local gitignored copies synced. Assignment wording reported, not edited (P1 Q2 does not name the
  Holt-Winters trend; P2 Q3(d) "mean RMSE across all steps" reads as MAE; P2 Q3 invites the 1-step
  vs 52-step comparison the key calls a trap).
- **L13 skill bars** (`skill_bars`) on "Lesson 1": eight bars, each against its own benchmark, all
  reproduced (HW1 −8 / −6 / −3%; L11 +4%; HW2 +9 / +12%; HW3 +58%; HW4 +65%). The slide's HW1 claim
  is narrowed to "every … model the assignment asked for". The one-week-ahead caveat is on the figure.
- **L07 early stopping redrawn** at the owner's request: the epoch is chosen on a validation slice (the
  last 52 training weeks), never on test weeks. Validation picks 107 / 132 / 80, test there 949 / 948 /
  1,003 vs 1,159 / 1,411 / 1,053 at epoch 60 in the same runs; the test optimum is shown but not used.
- Verified: L07 / L10 / L13 Beamer 0/0/0, slide-qa 0 overflow, parity, every frame ≤ 90 words,
  check-book-claims 10/10; slides republished; Lab 6 re-rendered.
