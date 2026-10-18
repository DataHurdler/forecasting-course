# Visuals and geometry audit, all decks except Lecture 8

**Status:** APPROVED 2026-10-13; ★ figures and the five geometry pictures BUILT in every deck the same day. Owner decisions:
- **The 52-week window is canonical** for the L9 Part 2 and L13 scoreboards.
- **Build in teaching-calendar order** ("Your recommendation is sound. Follow the teaching calendar.")
- L9 Part 1 is done: [its plan](2026-10-13_lecture09-part1-visuals.md). L9 Part 2 is next.
**Date:** 2026-10-13
**Ask (owner, verbatim):** "Identify places where other lectures can benefit from adding
images/graphic examples. Also, instead of equations (algebra), consider expressing examples and
formulas using geometry."

**Method:** six read-only reviewers, one per group of decks, read every frame of the Beamer
source (and the lab where a number was in question). They checked numbers in memory against
`data/processed/` and the labs' own code. Nothing was written to the repo. Content errors marked
✔ were re-checked by hand against the source. Errors marked ○ were reported by a reviewer but
not yet re-checked.

**Update 2026-10-16:** every ○ row below was verified and fixed in the
[round-1 rollout](2026-10-13_figures-rollout.md) on 2026-10-13, so all rows now read ✔. The
L04/L05 weekly RMSEs did not reproduce and were replaced by the figure script's 1,456 / 1,115.

**Correction to my own brief:** I told the reviewers that Lecture 10 had 16 display equations and
Lecture 13 had 30. Both counts were wrong because `\\[3pt]` table-row spacing matched the
pattern. The real counts are 0–5 per deck, and Lecture 13 has none. The algebra in this course
lives in sentences and tables more than in display math. Each pictured fact has to be stated
exactly, so the "geometry" swaps below are mostly *a picture of a sentence*.

**House style (from Lecture 8):** one matplotlib figure per concept, PDF for Beamer and SVG for
Quarto, drawn at display size (SCALE = 0.55). Use the course's own data, and put the slide's
measured numbers in the figure. One generator script per deck: `scripts/figures/lectureNN_figures.py`.

---

## Build status (update this table as each deck lands)

This file is the standing roadmap for figures and geometry in every deck. Each deck gets its own
build plan when its turn comes (as Lecture 9 Part 1 did); this table says where each one stands.

| Deck | Taught (Fall 2026) | Status | Build plan |
|---|---|---|---|
| L01–L07 | weeks 1–6 | **Built 2026-10-13** (rollout): first figures + geometry pictures + word diet + verified error fixes | [rollout](2026-10-13_figures-rollout.md) |
| L08 CNNs | week 8 (Oct 15) | **Built**; live 2026-10-13; word diet 2026-10-13 ("Why Convolution?" split, new wiring figure) | [L08 plan](2026-10-13_lecture08-visuals-and-theme-fix.md) |
| L09 Part 1 | week 9 | **Built**; live 2026-10-13; word diet (Check 2 split) | [L09P1 plan](2026-10-13_lecture09-part1-visuals.md) |
| L09 Part 2 | week 10 | **Built** 2026-10-13 | [L09P2 plan](2026-10-13_lecture09-part2-visuals.md) |
| L10–L13 | weeks 11–16 | **Built 2026-10-13** (rollout) | [rollout](2026-10-13_figures-rollout.md) |

**Round 2, 2026-10-16** ([plan](2026-10-16_figures-round2.md)): every remaining non-Low candidate in
every deck was built, 36 new figures in all, with these exceptions:
- **Unblocked and built 2026-10-18:** L10 Bayes mosaic (strong week = above its own year's median); L13 skill bars (after the HW01 reconciliation).
- **Skipped on purpose:** L06 CV curve (it is Lab 6 Step 4's "Your turn"); L11P1 level fans (would
  redraw `band_width` at another scale); L07 early-stopping-vs-ridge panel (no room).
- **Still open:** the items ranked Low (L02 #6, L03 #6, L04 #6, L05 #6, L06 #6, L07 #6, L10 #6,
  L11P2 #4, L12 #5).

## Recurring geometric ideas (a visual vocabulary for the whole course)

Five pictures carry most of the swaps. If each is drawn the same way every time it appears,
students learn it once:

1. **A weighted average is a balance point between points.** Applies to the forest prediction
   (L05), the attention output (L09P2), the posterior mean (L10) and hierarchical shrinkage
   (L11P2). SES in L01 is the first one: the level closes the fraction α of the gap each week.
   Forests and attention cannot extrapolate for the same reason: they stay inside the hull.
2. **A squared error is an area.** RMSE is the side of the average square. Applies to the CART
   split (L04), the smoothing penalty (L03, area under the squared bending) and the Christmas
   closures (L13).
3. **A penalty is a shape the solution has to touch.** Smoothing (L03), cost-complexity pruning
   (L04, a line of slope −α touching the hull), and the diamond and disc (L06).
4. **Geometric decay: the same staircase, k steps of ×φ.** SES weights α(1−α)^k and the
   damped-trend ceiling (L01); the AR(1) ACF φ^k, shock half-lives and the ARIMA forecast
   asymptote (L02); the vanishing gradient 0.7^k (L09P1). On a log axis each one is a straight
   line. Drawn the same way three times, L09's "why an RNN forgets" becomes L02's AR(1) again.
   The √h fan (L11P1) is its additive cousin.
5. **A rotation or change of coordinates.** Positional encoding as clock hands (L09P2), Fourier
   pairs as arrows (L11P1) and non-centred parameterization (L11P2).

---

## Content errors found along the way (fix regardless of figures)

| | Deck:line | Problem | Fix |
|---|---|---|---|
| ✔ | L01:271–274 | Lists ETS(A,A,M) among the "15 valid combinations" | FPP excludes (A,N,M), (A,A,M) and (A,A_d,M) as unstable, which is how 18 becomes 15. The Holt-Winters multiplicative model inside the 15 is ETS(M,A,M) |
| ✔ | L02:259 | The ARIMA box writes the ARMA on y_t | It should be on the differenced series, y′_t, as the sentence says |
| ✔ | L02:200 (TikZ) | MA(1) lag-1 spike drawn at 0.94, which is ρ ≈ 0.67 on the AR panel's own scale (0.98 = φ 0.7) | No MA(1) has \|ρ₁\| > 0.5. With θ = 0.6, ρ₁ = 0.44. Regenerate from ArmaProcess with a labelled axis |
| ✔ | L09P1:151 | 0.7^26 printed as 0.000 002 | 0.000 094 (the lab already prints this); 0.7^52 ≈ 9×10⁻⁹ |
| ✔ | L09P1:178 | Forget gate credited to Hochreiter 1997 | The forget gate is Gers, Schmidhuber & Cummins (2000); bib entry needed |
| ✔ | L09P1:404 (and lab prose) | "snap_days ≈ 0 at every lead" | The lab prints −0.128 at t+2 and +0.116 at t+4: the SNAP calendar repeating, not announcement |
| ✔ | L09P2:225, 258, 304 | Says 26-week windows and 5,940 examples | The lab runs **52-week** windows (6,750 windows, train 5,160). The table pairs Transformer 990 (52-wk) with RNN 842 / LSTM 987 (26-wk); the matching 52-wk numbers are 805 / 998. The conclusion survives. **Owner decided 2026-10-13: 52 weeks** for the L09P2 and L13 scoreboards |
| ✔ | L12:101, 135–136, 238–239 | Treats −0.5 as the elastic/inelastic boundary | The boundary is −1. The whole HDI [−0.596, −0.387] is inelastic, which is *why* revenue rises in every draw |
| ✔ | L12:265 | "XGBoost: best tabular accuracy" | The course scoreboard has LASSO 744 vs XGBoost 781 |
| ✔ | L11P2:267 vs :359 | 8.2× vs 8.1× | Pick one |
| ✔ | L05:111–112 | "OOB noisy below ~1,000 rows because each observation is out-of-bag for too few trees" | Each observation is OOB for ≈0.37·B trees whatever n is. Small n is noisy because there are few rows to average. On a time series the real problem is that OOB trees saw later weeks |
| ✔ | L05:328–329 | "`gamma` penalizes complexity the way ℓ1 does" | `gamma` is a per-leaf count penalty (ℓ0-like, like `ccp_alpha`); XGBoost's ℓ1 is `reg_alpha` |
| ✔ | L03:281 (TikZ) | The PDP "Rising" band runs below the x-axis (−0.37) and above the axis tip (2.02); the keybox describes a falling curve no panel shows | Replace with a real partial dependence plot, or fix the coordinates |
| ✔ | L06:153 | "Likely to hit a corner first" | True only for small budgets. On lag7/lag14 (ρ = 0.78) the corner wins only for t ≲ 0.18, about ⅕ of the OLS ℓ1 size (checked by hand on the LARS path) |
| ✔ | L01 Holt / Lab 1 Step 5 | Fitted φ = 0.80 on all three series | That is statsmodels' lower bound (0.8–0.995), so the lab's "what does φ < 1 mean?" is answered by the bound |
| ✔ | L07:46 | NN output "smooth, continuous" | With ReLU it is continuous and piecewise linear |
| ✔ | Labs/Lecture07_lab.qmd:246 | Gradients accumulate "in that epoch" | Without `zero_grad`, they accumulate across all 60 epochs |
| ✔ | L05:131 | "Last year's value always dominates" | The deck's own importance tables have lag4 (weekly) or `dow` (daily) on top |
| ✔ | L05:315–317 | Newton step "converges in fewer trees" | Only for non-squared losses; for squared error the Hessian is constant |
| ✔ | L09P1:126 | Singular value > 1 → explodes | It only makes explosion *possible* |
| ✔ | L12:118 | b_event "at most about half a percent either way" | The lower HDI end is about −1.1% |
| ✔ | L12:188–190 | Confounder named as series only | Time is a second confounder: series means alone give +1.70, the wrong sign |
| ✔ | L12:59–60 | Ridge/lasso ↔ prior | Holds at the posterior *mode*; a Laplace posterior mean is not sparse |
| ✔ | L13:102 | Year-over-year growth figures | 2011→2012 was +28.6%; say "since 2012" |
| ✔ | L04:101 | Flowchart numbers invented (≤38,000 → 61,300) | CA_1 FOODS weekly runs 12k–27k; draw the real tree |
| ✔ | L04/L05 numbers | No script found that generates the weekly tree and forest RMSEs (L04:300–304, L05:195–200, 227–229) | The figure script would become their source |

---

## Per-deck candidates (ranked; ★ = the one to make first)

**Across the course:** outside Lecture 8 no slide plots a real series. The five TikZ figures
are schematics (L01 timeline, L02 simulated walks and ACF, L03 partial dependence, L04 flowchart).

### L01 ETS and evaluation (taught)
1. ★ **SES (:165)**: CA_1 FOODS. Levels at α = 0.8, 0.1 and the fitted 0.32, each flat through the test year. Inset: one week's gap y_t − ℓ_{t−1}, with the level closing α of it. Replaces the six α bullets.
2. **Benchmarks (:109)**: the four benchmarks across the test year with RMSEs: seasonal naive 1,660 (the number L11P1 beats), drift 1,764, naive 1,794, mean 1,881. Geometry: naive and mean are horizontal lines; seasonal naive is last year shifted right; drift is the chord from first to last point, extended.
3. **Equal weights (:140)**: weight bars for α = 0.62 (TX_1 HOUSEHOLD) and 0.21 (TX_1 FOODS) against the 1/225 line. The centre of mass sits at age (1−α)/α: 0.6 vs 3.7 weeks, same store.
4. **Holt (:199)**: the undamped line climbs 70 units/week (RMSE 3,428 vs SES 1,754). Damped at φ = 0.85, it bends under a ceiling at ℓ_T + b_T·φ/(1−φ) = ℓ_T + 5.7 b_T, which replaces the Σφ^j.
5. **Accuracy (:289)**: each test-week error drawn as a square. MAE is the average side; RMSE is the side of the average square, so RMSE ≥ MAE. SES vs seasonal naive tie on MAE (1,376 vs 1,365) but not on RMSE (1,754 vs 1,660). Five weeks carry 40% of SES's squared error.
6. ETS 3×3 grid of forecast shapes (:255).

### L02 ARIMA and VAR (taught)
1. ★ **ACF/PACF (:116, :145)**: the real ACF of TX_1 HOUSEHOLD levels (0.94 → 0.76) beside its first difference (one spike, −0.29). Regenerate the 2×2 from ArmaProcess on a labelled axis. Geometry: the AR(1) ACF is the staircase φ^k. The −0.29 is close to the −0.33 implied by Lab 1's α = 0.62, so the SES ↔ ARIMA(0,1,1) link shows in the data.
2. **Stationarity (:38)**: levels above differences, each with a rolling mean and a ±2 sd band; ADF −1.38 vs −8.51 in the titles. Conditions 1–2 = "stays in a horizontal band of fixed width".
3. **Unit roots (:63)**: the response to one shock for φ = 0.5, 0.9, 1.0, with half-lives (1 and 6.6 periods). A flat line at φ = 1 is what "permanent" means.
4. **Granger (:368)**: retail growth and unemployment changes 1992–2026 with COVID shaded (moves 16× and 68× their pre-2020 sd). Shows why the test result depends on that window.
5. **SARIMA (:278)**: the daily ACF with peaks at 7/14/21/28, then after the seasonal difference. Note m = 7 here vs 12 on the slide.
6. The by-hand forecast 121 → 122 asymptote (:256), the same shape as L01's damped trend; Low.

### L03 GAMs (taught)
1. ★ **Penalty term (:159)**: electricity_daily, mw_mean vs day of year. Three fits (λ≈0, the GCV choice, huge λ), with edof and holdout RMSE. A strip under the first two shades (f″)². Geometry: the penalty is the area under squared bending; it is zero only for a straight line, so λ→∞ is the OLS *line*. Use `basis="ps"`; the lab's cyclic `cp` basis forces a flat line.
2. **Prophet workflow (:213)**: the components panels that :223–225 describe but don't show.
3. **Additivity (:77)**: weekday vs weekend monthly load. The additive fit forces parallel curves (same slope everywhere), but the real gap is 2,006 MW in December vs 4,063 in July. Replaces both derivative expressions.
4. **Prophet Fourier order (:190)**: one sinusoid has one peak a year. Heating plus cooling needs the 182.5-day wave too.
5. **PDP (:264)**: replace the TikZ with the real s(doy) and f(dow) plots (see errors).
6. **"Generalized" (:126)**: identity / exp / logistic link with the forbidden regions shaded. Low priority.

### L04 Decision trees (taught)
1. ★ **CART split (:138)**: CA_1 FOODS weekly, units vs lag4. Top: the cut, two leaf means and residual sticks. Bottom: SSE_L+SSE_R against the threshold, with the minimum marked (20,604). Geometry: slide a cut and total the squared vertical distances.
2. **What is a tree (:101)**: the real depth-2 tree next to its partition of the lag4 × lag52 plane (rectangles shaded by leaf mean). Replaces the invented numbers.
3. **Depth vs bias–variance (:192)**: train/test RMSE vs `max_depth` on weekly data. The lab claims the deck already draws this; it doesn't.
4. **Bias–variance (:60)**: 20 stumps vs 20 deep trees for one test point. Label the gap "miss of the average", not bias, because noise is included.
5. **Pruning (:220)**: training SSE vs leaves. The α-chosen tree is where a line of slope −α touches the lower hull.
6. Greedy XOR (:161): Low.

### L05 Tree ensembles (taught)
1. ★ **Correlation problem (:125)**: the root-split feature in 300 trees. Bagging picks `dow` 298/300; `sqrt` spreads the roots. Show each ensemble's test RMSE honestly (`sqrt` loses here).
2. **Bagging (:67)**: Var/σ² = ρ + (1−ρ)/B against B for ρ = 0.9, 0.5, 0.1, with floors at ρ. More trees moves along a curve; decorrelation drops to a lower curve. Schematic: ρ can't be measured from one forest.
3. **Which one when (:391)**: a forest trained to 2019 on RSXFS, pinned under the training maximum while actuals climb to 665k. Geometry: every prediction is a weighted average of training targets, so it stays inside [min, max].
4. Boosting stages 1/2/5/50 (:281). 5. The OOB grid sorted by date shows the leak (:94). 6. XGBoost Newton step as a parabola vertex −G/(H+λ) on a Poisson loss (:313); Low.

### L06 Regularization (taught)
1. ★ **Why LASSO zeros out (:137)**: real RSS ellipses for standardized lag7/lag14 (OLS 0.52, 0.34; ρ = 0.78), with the disc and the diamond at a budget where the corner wins. Caption: "two coefficients, constrained form; each λ is some budget t; the corner wins only for small t".
2. **Regularization path (:195)**: LASSO coefficients vs log α. Multiples of 7 share a colour, so the drop-out order recovers the weekly cycle.
3. **Why regularize (:39)**: test RMSE vs n/k. OLS goes 280 → 3.6M; Ridge stays 278 → 770; benchmark line at 625.
4. **Elastic net (:166)**: the α = 0.5 ball beside the diamond, on the same ellipses. Pointed only on the axes, curved elsewhere, so it keeps correlated lags together.
5. CV curve with the one-SE rule (:211). This duplicates a lab "Your turn"; keep one or the other.
6. Unstandardized penalty as a stretched ellipse (:87); Low.

### L07 Neural nets (taught; reviews plan section C)
1. ★ **Nonlinearity (:95)**: the W-shaped year of electricity load. Best line (RMSE 4,513) labelled "any stack of linear layers", against four ReLU hinges summed. Geometry: H ReLU units = a piecewise-linear curve with ≤H kinks, a spline with learned knots (back to L03). Draw it at fixed knots as "what 4 units can represent", because a seed-42 fit was non-monotone in width.
2. **Training (:189)**: gradient descent on the real lag7/lag14 bowl. Smooth for η < 0.28, zig-zag for 0.28–0.56, diverges above 0.56 (= 2/λ_max). Second panel: early stopping tracks the ridge path. Caption: an exact bowl only because the model is linear.
3. **Overfitting (:213)**: train vs validation RMSE by epoch, 3 seeds. Validation bottoms at epochs 80–132; the lab stops at 60.
4. **Model size (:149)**: the weight matrices drawn to scale. The 128×128 block is 73% of 22,657 parameters. This supplies the "diagram" :281 refers to.
5. **Activations (:62)**: ReLU/sigmoid/tanh with maximum slopes (¼, 1, 0/1); sets up L09's vanishing gradient. Drop the box-and-arrow neuron (it only redraws σ(wᵀx+b)).
6. `zero_grad` on the bowl (:295); Low.

### L09 Part 1 RNN/LSTM — BUILT 2026-10-13 (#1, #2, #3, #4 done; #5 deferred)
1. ★ **Why an RNN forgets (:137)**: log-scale gradient survival over k = 0–52. Measured ‖∂h_T/∂h_{T−k}‖ for the trained RNN, plus the at-init curve and the 0.7 example. Geometry: repeated multiplication is a straight line on a log axis, with slope = log spectral radius (0.6 at init → 0.83 trained). Replaces the table, including its wrong row.
2. **Cell state (:239)**: two backward lanes in the L08 residual_block style. RNN hops pass through ×W·tanh′; LSTM hops are ⊙f and +. Inset: measured forget gates, median 0.54. **Don't draw an undiminished highway:** after 4 weeks the cell lane keeps 0.08.
3. **Check 1 (:356)**: RMSE vs epochs at all 7 checkpoints; the lab already plots it. Slopes can't be read from a table.
4. Unrolled RNN over the real 26-week strip (:88), reused on BPTT (:114) with thinning backward arrows.
5. Check 2 (:387): 30 series on one axis (the 91% between-series share) plus mean ACF. Fix :404 first.

### L09 Part 2 Transformers (week 10) — BUILT 2026-10-13 (#1, #2, #3, #5 done; #4 path figure not built)
**Settle the window question first**, since every figure prints these numbers.
1. ★ **Attention heatmap (:91, :116)**: 4 heads × 52 lags from the trained model, plus December 2015. Measured: three heads sit on lags 2–7, one is uniform, and lag 52 gets ≤ uniform. There is no annual-cycle head, so :91 and :116 should say what it *did*. Caveat: attention weights are not importance.
2. **Positional encoding (:178)**: dials for 4 frequency pairs plus the 52×52 PE·PE′ matrix, constant along its diagonals. Geometry: each pair is a clock hand, and a shift rotates every hand by its own ω_i·Δ, so similarity depends only on Δ. State the limits: not monotone, W_Q/W_K in between, no pair has a 52-week period.
3. **Dot-product attention (:69)**: q·k = |q| × the shadow of k on q (keys of unequal length, so "angle only" is not implied). Softmax bars, then the output inside the values' convex hull: a head blends, it can't extrapolate.
4. Recurrence path of 52 hops vs one arc; O(T²) as the area of a T×T square (:39, :109).
5. Encoder block diagram reusing residual_block (:150).

### L10 Bayesian foundations (week 11)
1. ★ **"The Update, in Pictures" (:124), which has no picture**: Beta(5,45) prior, normalized likelihood, Beta(35,215) posterior. Below: a lever with 0.10 weighted 50 and 0.15 weighted 200, balancing at 0.14 (exact). Inset: 3/20 → 0.114 and 300/2,000 → 0.149. Tick the prior *mean*, not its peak (0.083).
2. **MCMC (:226)**: four Metropolis traces from scattered starts with warm-up shaded, plus a histogram against the exact Beta(35,215). Geometry: each step compares two heights, so p(D) cancels.
3. **Bayes' theorem (:98)**: a unit-square mosaic on CA_1 FOODS weeks (high-SNAP × strong week). The owner must define "strong".
4. Priors by support (:161), including N(0,1) on a σ with half its area on impossible values, and the HalfNormal fold.
5. Prior predictive (:300): flat / Beta(5,45) / Beta(500,4500) predictive counts out of 200.
6. Repeated-sample CIs vs one posterior (:65); Low.

### L11 Part 1 Bayesian TS (week 12)
1. ★ **The interval is the product (:270)**: 94% band over 52 held-out weeks, 52/52 inside, ±30% *on average*. Geometry: variance = σ² + h·σ_level², so the band widens like √h (the lab shows it roughly doubling from weeks 1–13 to 40–52).
2. **Structural decomposition (:94)**: stacked observed / level / seasonal / residual panels on a shared scale. The top panel is the vertical sum of the rest.
3. **Fourier (:151)**: waves k = 1, 2, 3 and their sum over the detrended weeks. Each (a_k, b_k) pair is an arrow: length = amplitude, angle = timing (amplitudes ≈ 970, 360, 110).
4. Level random walk: small vs large σ_level fans (:120); overlaps #1.
5. Centred vs non-centred traces (:203); save the funnel for Part 2.

### L11 Part 2 Hierarchical (week 13)
1. ★ **Shrinkage (:251, :282)**: one arrow per series from the unpooled to the hierarchical estimate; thin series crimson, μ_b dashed, held-out truth as a ring. 8.2× = the ratio of mean arrow lengths. WI_2 shows over-shrinkage honestly.
2. **The trap (:210)**: the funnel in (b_j, log σ_b) with 127 divergences, against z_j with 1. Geometry: non-centring is a change of coordinates that removes the σ_b-dependent width. Exact under the prior; plot a thin series.
3. **The question (:36)**: a strip plot of the raw SNAP effect by category, 30 stores (no MCMC). Reusable in L13.
4. Posterior predictive for a new store (:315); Low.

### L12 Bayesian regression (week 14–15)
1. ★ **Confounding (:187)**: log price vs log units, pooled slope −2.8 (between categories), against the within-series residual cloud with slope −0.49. Must remove trend as well as series means.
2. **Posterior as area (:122, :220)**: the b_price density with HDI, the area left of −0.5 (0.446), and a line at −1 "revenue break-even". Fix the elasticity boundary first.
3. **DAG (:155)**: the slide calls a DAG "a picture" and has none. TikZ.
4. ROPE on b_event (:93). 5. Revenue fan for 0–20% price rises (:220); Low.

### L13 Synthesis (week 16)
1. ★ **Scoreboard (:64)**: a dot plot of RMSE from 0, the seasonal-naive line, arrows for the RNN/LSTM moving from 30 to 100 epochs, and a seed-spread ruler. Depends on the window decision.
2. **Christmas (:284)**: the closures against the seasonal-naive forecast, 2012–15. Squares: side 537 (average) vs 2,045–2,485 (closures); each closure square is about 18× the average area (re-checked by the reviewer: 4.61% of squared error).
3. Skill bars, 1 − RMSE/naive (:102). Needs the HW1 key reconciled first.
4. 30 series on one axis: Pythagoras between/within, 91% (:126).
5. A KFold row in L01's train/test TikZ (:228). 6. Reuse L08's March-2020 photo for structural breaks (:378).

---

## Practical notes

- **Sampling and training inside figure scripts.** L09 figures need a trained network (≈1–2 min,
  seeded); L10 #2 can use a hand-coded Metropolis sampler; L11P1, L11P2 and L12 need PyMC draws.
  The choice is to sample inside each figure script (minutes) or have the labs save their draws.
- **Each reviewer measured with one seed.** Re-run at the deck's settings before printing any
  number.
- **Lab overlap.** L06 #5 and L04 #3 draw what a lab exercise asks students to draw; pick one
  home for each.
