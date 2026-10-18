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
