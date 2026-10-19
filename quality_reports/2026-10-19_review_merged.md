# All three reviews, merged (2026-10-19)

Sources:
- [Full-course correctness review](2026-10-19_full_course_review.md) — cloud session, 78 findings
  (C = critical, M = major, m = minor). Reviewed `main` at `4de1172`, i.e. before the L09P1 fixes.
- [Proofreading](2026-10-18_proofread_all_decks.md) — 15 decks, about 36 items.
- [Pedagogy](2026-10-18_pedagogy_all_decks.md) — 15 decks, about 60 items.

**Verification by the coordinator:** C1, C2, C3, C4, C5, C9, C10, C12, C13, C14, C15 and M1 were
re-checked against the source, the cached numbers or the data. C6, C7, C8 and C11 rest on the cloud
session's re-runs plus sound reasoning, and were not re-run locally. The rest are taken from the reports
pending the fix pass, where each edit is checked against its source before it is made.

**Calendar (week 9, 2026-10-19).** Taught: L01–L08. This week: L09 Part 1, HW05 Part 1 due.
Not yet due: HW05 Part 2 (wk 11), Part 3 (wk 12), HW06 Part 1 (wk 13), Part 2 (wk 14), HW07 (wk 15).
Rule (owner, 2026-10-13): fix only assignments not yet due; graded ones wait for next term.

## A. Errors students will see — fix now (no decision needed)

**Upcoming lectures and labs (L09 Part 2 → L13)**
- C3 L09P2 + Lab: "could not tell last week from a year ago" — the readout is the last position; it
  loses the order of the *other* 51 weeks. (Fix together with the known "shuffling leaves every output
  unchanged" item.)
- C10 L11P2: pooling's loss was CA_3, not WI_2 (WI_2 was a win, pulled past its truth).
- C11 Lab 12: the series intercepts remove the confounding; centering only helps the sampler.
- C12 L13: "our first two Bayesian fits failed" — L10 passed; L11P1 failed R-hat, L11P2 only divergences.
- M5 Lab 10: the wrong prior moves P(θ > 0.30) from 0.08 to 0.61 — "barely changed" is false.
- M6 L11P1: the "unchecked ARIMA formula" contrast; the real difference is parameter uncertainty.
- M22 L12 narration: b_snap 3.7% is complete pooling, not "consistent with the hierarchy" (FOODS ≈ 8%).
- M24 L11P1 lab narration: 127 divergences → 43. M25 L11P1 teaser: "thirty series" → ten.
- m29–m35 (L09–L13 minor numeric/claim items), m33 ("six/four model fits" → five/three).

**Taught lectures and their labs (L01–L08) — the site is wrong today**
- C2 L08 + Lab 8 (+ L13, narrations): the flatten head sees the whole window; a longer window adds
  context *and* parameters. In the 26-week table, the window — not the receptive field — hides lag 52.
- C6 Lab 4 Step 4: the extremes effect is regression to the mean, not the training-range ceiling.
- C7 Lab 5: HW03 is lag1-dominated; `sqrt` loses there too — drop the prediction.
- C8 Lab 6: the 62-feature design has rank 58; the OLS coefficient column is the solver's pick.
- C9 L06 + Lab 6: the model *did* have day-of-week dummies; under heavy penalty it kept the weekly lags.
- C14 Lab 3: 12% on RMSE, 15% on MAE. C15 Lab 7: 7,530 windows are *fewer* than 8,310 rows.
- M4 L05: the forest's win is one-step-ahead vs year-ahead ETS/ARIMA; most of it is last week's actuals.
- M7 L06: Bergmeir et al. (2018) shows K-fold is valid for pure AR models — cite it for what it says.
- M9 L08: He et al. say the degradation is *not* vanishing gradients — an optimisation difficulty.
- M10 L01: `KFold` does not shuffle by default, and leaks anyway. M11 L03: GAMs fit interactions you specify.
- M12 L03: correlated predictors are a GAM's weak case (concurvity), not its use case.
- M3 Lab 1 narration: Holt-Winters is optimiser/platform-dependent — drop "if yours differ, the data changed".
- M17–M21, M26 (lab/lecture narration numbers), m1–m28 (minor items in L01–L08).
- Proofreading-flagged fact: L08 two stacked width-3 filters see 5 weeks, not 6. L05 63% vs 64%.

**Assignments not yet due**
- C4 HW05 P2 + P3: "shuffling destroys the ordering" → "use shuffle=False so runs are reproducible".
- C5 HW05 P3: 5,900 → 5,160 windows. C3 HW05 P2 Q3: the ablation premise (see above).
- M27 HW05 P1–P3 prompts: five architectures, not four.
- M14 HW07: no promotions/competition in M5. m34 HW07: intercept prior N(ȳ, 1); break-even is b > −1.
- M1 all not-yet-due assignments: the initial prompt says "stop and reply PROMPT_LIMIT_REACHED", but the
  syllabus, AI policy, AGENTS.md and the validator all say the budget is a target. Align the prompts.

**Course documents (published)**
- C13 datasets guide: electricity is PJM East hourly load, no temperature covariate.
- M15 no promo/store-type columns in M5. M16 30 weekly series, not "thousands".
- M2 synthesis is Week 15, not 16 (narrations, datasets guide, CLAUDE.md). m36 assorted document facts.

**Proofreading** — all ~36 items are mechanical (citations moved in the mirror, outline titles, deck
titles, typos, "practised", "modelled", subject–verb). Plus the PDF code-indentation bug in 7 decks.

## B. Pedagogy — split by size
- **Small (one line, no reordering):** missing definitions (softmax, Q/K/V projections, HDI, Beta, GCV,
  edof, global pooling), unlabeled figure encodings, dataset tags in captions, takeaway lines the
  narration calls "the point", notation warnings. Recommend: apply with the fixes in A.
- **Large (reorder frames, new slides, new figures):** e.g. L02 define AR/MA before the ACF; L11P2 show
  the centred code before the trap; L12 move the DAG before the coefficient table; L13 rescope the
  scoreboard; a walk-forward figure in L01; an LSTM cell diagram. Recommend: owner picks, deck by deck,
  after A is done.

## C. Next-term list (graded assignments) — add to `plans/2026-10-18_next-term-changes.md`
- C1 HW04: sort `train` by week before `TimeSeriesSplit` (folds currently cut across series).
- M8 HW04: the "k large relative to n" rationale does not hold (n/k = 112).
- M13 HW02: drop the "common holdout with HW1" objective and the HW1 row in the RMSE column.
- m4 HW01: lag features are 1–4 and 52 only. M1 for the graded assignments.
- M3 HW01 keys: Holt-Winters numbers are platform-dependent (1,791 here, 1,768.5 on Linux).

## D. Owner decisions
1. **HW05 Part 1 Q4 (due this week)** asks students to explain how "two convolution layers see about
   seven weeks regardless of window length" limits a longer window — a false premise (C2).
   Recommendation: correct the wording now, have the key accept either reasoning, and post a Canvas note.
2. **Pedagogy, large items:** pick deck by deck after A, or skip.
