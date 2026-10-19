# Changes held for next term

**Status:** APPROVED 2026-10-18, deliberately NOT applied this term. The owner's rule for Fall 2026
is to fix only assignments that are not yet due; these are already graded, and changing a graded
assignment's text mid-term could confuse students who look back at it. Apply after the last
grades are in, then copy each assignment to the student repo (`python3 scripts/check-assignment-sync.py`).

Found while reconciling the HW01 keys to the walkthroughs (2026-10-18). The keys already credit
students who were misled by the current wording, so nothing here affects this term's grading.

## HW01 Part 1 (`Homework/HW01_Part1_ETS.qmd`)

1. **Q2 never names the Holt-Winters trend.** It lists "Holt-Winters (additive seasonality,
   period = 52)". The walkthrough fits an additive trend (1,791); a literal reading with no trend
   gives 1,547, which beats seasonal naive (1,660).
   **Change to:** "Holt-Winters (additive trend and additive seasonality, period = 52)". Optional
   extension: "Refit with no trend, then with a damped trend. What changes, and why?"

## HW01 Part 2 (`Homework/HW01_Part2_ARIMA_VAR.qmd`)

2. **Q3(d) asks for "the mean RMSE across all steps".** Each step is one 1-week-ahead forecast, so
   its RMSE is the absolute error and the mean of those is the MAE (1,200), not the pooled RMSE of
   all 173 forecasts (1,496).
   **Change to:** "Report the RMSE of all 173 one-step forecasts together: square every error,
   average, then take the square root."
3. **Q3's response prompt invites an unfair comparison.** It asks students to compare the
   walk-forward ARIMA RMSE (1 week ahead) with Part 1's best smoothing model (52 weeks ahead, one
   origin). The key calls it a trap but only docks the wrong conclusion.
   **Change to:** make the trap the question — "Why can't these two numbers be compared directly?
   Which ARIMA number *can* be compared with Part 1?" — and update both HW01 Part 2 keys
   (`forecasting-instructor/solutions/` and the walkthrough) to match.

## Added 2026-10-19 from the full-course review (graded assignments)

Source: [merged review list](../2026-10-19_review_merged.md), bucket C.

4. **HW04 (C1): the "time-aware" CV is not walk-forward.** The setup sorts by store, category, then
   week, and `TimeSeriesSplit` cuts by row position, so folds validate on other series over the same
   weeks. **Change:** `train = train.sort_values("week_start")` before building `X_train`, or pass the
   Q5 date-cutoff splits as `cv=`. Update the HW04 key and walkthrough to match.
5. **HW04 (M8): the reason for 46 features does not hold.** n/k = 112; even an oracle penalty beats OLS
   by 0.3 units. **Change:** present the 46 features as a test of selection and interpretability.
6. **HW02 (M13): the "common holdout with HW1" objective contradicts its own callout.** **Change:**
   compare against a seasonal-naive benchmark on the daily holdout; drop the HW1 row from the RMSE column.
7. **HW01 Part 1 (m4): the AI prompt lists "units_lag1 through units_lag52".** The CSV has lags 1–4 and
   52 only. **Change:** "(units_lag1 through units_lag4, units_lag52)".
8. **All graded assignments (M1): the initial prompt says to stop at the budget** ("reply
   PROMPT_LIMIT_REACHED"), while the syllabus, AI policy, AGENTS.md and the validator treat it as a
   target. **Change:** "log it as over budget and continue"; checklist "every prompt logged".
   (Not-yet-due assignments are fixed this term.)
9. **HW01 keys (M3): Holt-Winters numbers are optimiser/platform-dependent** (1,791 on the rendering
   Mac, 1,768.5 on Linux with the pinned versions). **Change:** quote a range, and note it in the key.
