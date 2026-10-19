# Full-course correctness review — ECON 8310, 2026-10-19

**Revision reviewed:** `main` at `4de1172` (all line numbers are at this revision).
**Scope:** correctness only (numbers, method claims, cross-document consistency, logistics). Wording,
typos, notation, layout and pedagogy were out of scope. No course file was changed.
**Method:** ten parallel reviewers, one per lecture group, plus one for Lecture 13, the course documents
and logistics. Each finding was then checked again before it went into this report: the critical ones
by re-running the reviewer's script or reading the code and data directly, the rest against the quoted
lines. Duplicates found by two groups are merged, and every place a finding appears is listed.

---

## 1. Summary

### What was checked

| Material | Count | How |
|---|---|---|
| Beamer decks (`Slides/*.tex`) | 15 of 15 | read in full; Quarto mirrors spot-checked for numbers that differ from the `.tex` (none found) |
| Narration scripts (lecture + lab) | 29 of 29 | read in full; spoken numbers compared with the deck, the lab printout and `measured.json` |
| Labs (`Labs/*.qmd`) | 14 of 14 | read in full; printed outputs compared with `docs/labs/*.html`; cheap steps re-run |
| Figure generators + `measured.json` | 15 scripts, 14 caches | numbers compared; L04/L05 caches re-derived with zero differences |
| Assignments (`Homework/*.qmd`) | 11 of 11 | read in full; setup code re-run where cheap (HW01–HW04, HW05 window counts, HW07 OLS) |
| Documents | 4 `ECON8310_*.md`, `scripts/site.yml`, `Labs/README.md`, CLAUDE.md tables, syllabus | cross-checked against each other and against `data/processed/*.csv` |

**Re-computed and matching (selection):** seasonal naive 1,660 / 2,152 / 4,608; L01 SES/Holt rows; all
L02 ADF/ARIMA/VAR/Granger numbers; L03 swings, Fourier RMSEs, additivity gaps; every L04/L05 figure
number and the sklearn parts of Labs 4–5; L06 LASSO path and exit order (as a set); HW03 pipeline (RF
899, noise feature 17th of 29); HW04 OLS/LASSO 744; window counts 7,530 / 6,750 / 5,160 / 5,940;
RNN/LSTM/Transformer parameter counts 4,545 / 17,985 / 67,329; the 91% between-series variance share;
Beta-Binomial posteriors and HDIs; the L10 mosaic probabilities (0.26 / 0.76 / 0.40); L11P1 RMSE 1,589
vs 1,660, 52/52 coverage and the ±30/16/39% widths; L12 naive and controlled OLS elasticities; every
Christmas-closure number in L13 (0.25% of days, 4.6% of squared error, 158 event days of which 5 are
closures); all due weeks and budgets between `site.yml` and the assignment headers.

### What could not be checked, and why

- **Neural networks, MCMC, Prophet, pyGAM, XGBoost.** None of these were retrained, re-sampled or re-run,
  as the brief required (torch, pymc, prophet, pygam and xgboost were not installed). Their numbers were
  compared with `measured.json` and the rendered lab HTML and are marked "checked against cache".
  Numbers with no cache at all: L10 ESS 3,290, L12 "ESS 184 → 3,097" (only 3,097 is cached), the
  single-harmonic Prophet 4,269 / 3,358 (a commented-out "your turn" cell), and the L02 ARIMA(5,2,4)
  anecdote.
- **The student repo (`forecasting-env`)** is not in this session. Not checked: `policy/homework_limits.json`,
  the AI policy, the online-store counts, and the student copies of the assignments. This matters for
  finding M1.
- **Instructor keys** (`forecasting-instructor`) are not in this repo.
- **Run times** ("~95 s", "four minutes") depend on hardware and were not checked.

### Totals

| Severity | Count |
|---|---|
| Critical — wrong number or false claim in a deck, lab, assignment or student document | 15 |
| Major — inconsistency between two places, or an overstated claim | 27 |
| Minor — other | 36 |
| **Total** | **78** |

One reviewer finding was dropped as rounding: L12 narration's HDI widths "0.100 against 0.207" against
a printed 0.099 / 0.208, which differs only by which HDI routine is used.

**Severity rule.** A wrong number or false claim in primary student-facing material (a deck, lab prose,
an assignment, or a course document) is **critical**. When the narration contradicts what the lab or
deck correctly prints, that is an inconsistency between two places, so it is **major**. Errors that live
only in internal files (CLAUDE.md) are **minor**.

**Cross-cutting patterns worth fixing as a class:**
1. **Platform-dependent optimiser output.** Holt-Winters fits sit on a parameter boundary, so the same
   code with the pinned library versions gives different RMSEs on Linux than on the machine that
   rendered the pages (M3). The narration tells students that a mismatch means their data changed.
2. **Leftovers from the Rossmann/Monash era.** Several documents still describe data the course no
   longer uses: C13, M14, M15, M16, and the minor CPI finding.
3. **The course contradicts itself across lectures.** L07 states correctly that shuffling training
   windows does not leak; HW05 P2/P3 say the opposite (C4). L01's KFold explanation conflicts with L13's
   (M10). The L11P2 deck names a different series from its own narration (C10).

---

## 2. Findings

Each entry gives the ID, severity, every place the text appears, the quote, what is actually true with
evidence, one fix, the confidence, and how it was verified. Scripts referred to as `work/Gxx/*.py` are
the reviewers' scratch scripts; each one runs the course's own code unchanged unless the entry says
otherwise.

### Critical

#### C1 — HW04's "time-aware" cross-validation is not walk-forward
- **Where:** `Homework/HW04_XGBoost_Regularization.qmd:102-103` (sort), `:134-139` (train/X_train), used at `:266` (Q2 GridSearchCV `TimeSeriesSplit(n_splits=3)`) and `:354-355, :363` (Q4 `ElasticNetCV`/`RidgeCV` with `TimeSeriesSplit(n_splits=5)`); rendered in `docs/homework/HW04_XGBoost_Regularization.html`.
- **Says:** setup `.sort_values(["store_id", "cat_id", "week_start"])`; Q2 "Use `TimeSeriesSplit(n_splits=3)` as the CV strategy … Why must you use `TimeSeriesSplit` rather than standard k-fold CV for this problem?"
- **True:** `TimeSeriesSplit` cuts by row position, and `X_train` is ordered series by series, each running 2012→2015. So every fold trains on some series over the whole period and validates on *other* series over the *same* weeks: validation weeks come before training weeks, and the validation stores are unseen (their one-hot columns are constant in training). This is the "future trains the model" failure the course warns about, applied to hyperparameter selection in Q2 and Q4. Q5 uses date cutoffs and is unaffected.
- **Evidence:** read the setup cell directly (sorting at :102-103; `train = df[df["week_start"] < cutoff]` keeps that order). `work/G06/hw2.py` on the setup verbatim: `train is time-sorted? False`; `fold 1: train weeks 2012-01-30..2015-05-11 series 8 | val weeks 2012-01-30..2015-05-11 series 8`. The choice changes the answer: RidgeCV alpha 0.106 (HW order) vs 44.2 (time-sorted); ElasticNetCV(l1=1) 0.69 vs 82.9.
- **Fix:** add `train = train.sort_values("week_start")` before building `X_train`/`y_train`, or pass the Q5 date-cutoff splits as `cv=`.
- **Confidence:** high · **Verified:** code reading + re-computation.

#### C2 — Lab 8 and L08: "a longer window adds no context because the receptive field is 7 weeks" is false for the network the lab trains
- **Where:** `Labs/Lecture08_lab.qmd:231-233`; `Slides/Lecture08_CNNs.tex:296-297, :350-351` (`Quarto/Lecture08_CNNs.qmd:226, :259`); `Slides/Lecture13_Synthesis.tex:140`; `Narration/Lecture08_lab_script.md:132-134, :265-266, :320-322`; `Narration/Lecture08_script.md:561-562`; `Homework/HW05_Part1_FFN_CNN.qmd:310-313` (Q4 is built on the same premise).
- **Says:** Lab: "The receptive field is stuck at seven weeks no matter how much history you supply. What a longer window *does* add is parameters … so you get more capacity to overfit and no additional context to justify it." Deck: "a seven-week receptive field cannot see the annual lag LASSO was handed directly."
- **True:** seven weeks is the receptive field of one conv-layer-2 unit. Lab 8's model ends in `Flatten() → Linear(64·(W−6), 1)` (`Labs/Lecture08_lab.qmd:146`), which gives a separate weight to every conv position. Together the positions cover the whole window, so with W = 52 the output depends on week t−52. A longer window adds context as well as parameters. In the 26-week comparison table, lag 52 is out of reach because it is outside the window, not because of the receptive field.
- **Evidence:** `work/G07/rf.py` (numpy forward pass of the Lab 8 architecture, perturbing one input week at a time): `52 flatten output depends on 52 of 52 input weeks; week 0 (= lag 52) True`. Closed form: ∪ⱼ [j, j+6] over j = 0…W−7 = [0, W−1].
- **Fix:** "each conv feature sees 7 weeks, but the flatten head sees all of them, so a longer window adds context *and* parameters"; in the 26-week table, blame the window length for the missing annual lag.
- **Confidence:** high · **Verified:** code reading + re-computation.

#### C3 — L09 Part 2: the model without positional encoding "could not tell last week from a year ago", yet it reads its forecast from the last position
- **Where:** `Labs/Lecture09_Part2_lab.qmd:179-181`; `Slides/Lecture09_Part2_Transformers.tex:258-259` (`Quarto/...:209`); `Narration/Lecture09_Part2_script.md:294-295, :405-407, :495-496`; `Narration/Lecture09_Part2_lab_script.md:86-88, :164-165`; `Homework/HW05_Part2_Sequence.qmd:273-275` (Q3 asks students to explain the ablation with this argument).
- **Says:** "It was simply working from an **unordered bag of 52 weeks** — it could not tell last week from a year ago." Lab script: "'Sales were 3,000 last week and 9,000 a year ago' and '… 9,000 last week and 3,000 a year ago' are the same bag of numbers."
- **True:** without position, the encoder is permutation-*equivariant*: the output at position j is F(x_j, multiset of all weeks). The model returns `self.fc(self.enc(h)[:, -1, :])` (`Labs/Lecture09_Part2_lab.qmd:134`), which is F(x_{t−1}, bag). The last week's values reach the readout through the query and the residual path, so the model knows last week exactly. What it loses is the order of the other 51 weeks. The lab script's own example is a pair this model *can* tell apart. This is separate from the already-known "shuffling leaves every output unchanged" item, though the two share a root cause and can be fixed together.
- **Evidence:** `Labs/Lecture09_Part2_lab.qmd:130-134` (read directly); every operation except attention is per-position (post-norm layer rebuilt in `scripts/figures/lecture09_part2_figures.py:166-170`).
- **Fix:** "Without it the model still knows which week is most recent, because it reads out there, but sees the other 51 as an unordered bag." Change the example to two *earlier* weeks and adjust HW05 P2 Q3.
- **Confidence:** high · **Verified:** code reading.

#### C4 — HW05 Parts 2 and 3: "shuffling destroys the ordering the sequence models exist to exploit"
- **Where:** `Homework/HW05_Part2_Sequence.qmd:104-105`; `Homework/HW05_Part3_Comparison.qmd:113-114` (both rendered in `docs/homework/`). The same idea is in `Narration/Lecture09_Part2_script.md:414-418` and `Narration/Lecture09_Part1_lab_script.md:97-98` (failure lists naming `shuffle=True`).
- **Says:** "**Do not shuffle.** `DataLoader(..., shuffle=False)` for every model here. Shuffling destroys the ordering the sequence models exist to exploit, and it will not warn you."
- **True:** `shuffle=True` changes which windows share a minibatch, not the order of weeks inside a window (`X.append(arr[i:i+W])`, `Labs/Lecture09_Part1_lab.qmd:71`). Every model re-initialises its hidden state for each window, so nothing depends on the order of windows. The course's own L07 slide says exactly this.
- **Evidence:** `Slides/Lecture07_NeuralNets.tex:251-252`: "`shuffle=False` keeps runs identical. Shuffling these training windows would not leak the future --- each carries its own past; shuffling *before* the time split would." HW05 Part 1 (`:124-126`) already uses the reproducibility reason, and its Q1 (`:241`) asks whether shuffling leaks.
- **Fix:** copy HW05 Part 1's wording ("use `shuffle=False` so runs are reproducible"); in the two narration lists, say "shuffling before the split".
- **Confidence:** high · **Verified:** cross-reference (both sides read).

#### C5 — HW05 Part 3: "your 52-week training set holds roughly 5,900 windows"; it holds 5,160
- **Where:** `Homework/HW05_Part3_Comparison.qmd:276`; `Narration/Lecture09_Part2_lab_script.md:251` ("5,900 overlapping windows").
- **Says:** "Your 52-week training set holds roughly 5,900 windows across 30 series."
- **True:** under the assignment's rule (target week before 2015-05-17), 52-week windows give 5,160 training windows; 5,940 is the 26-week count. The deck (`Slides/Lecture09_Part2_Transformers.tex:179, :251`), the lab printout, `measured.json` and `Narration/Lecture13_script.md:224` all say 5,160.
- **Evidence:** two reviewers counted independently: `13 train 6330 | 26 train 5940 | 52 train 5160`. The lab prints "6,750 windows of 52 weeks — train 5,160, test 1,590" (`docs/labs/Lecture09_Part2_lab.html`, checked).
- **Fix:** "roughly 5,200 windows (5,160)" in both places.
- **Confidence:** high · **Verified:** re-computation.

#### C6 — Lab 4 Step 4 blames trees for regression to the mean
- **Where:** `Labs/Lecture04_lab.qmd:223, :236-237, :249, :266-270`; `Narration/Lecture04_lab_script.md:311-320, :329-331, :343-345`.
- **Says:** "The predictions cannot leave the training range … Now watch what that does at the extremes." / "the tree systematically under-predicts the busiest days and over-predicts the quietest ones" / Discuss: "Give the mechanical reason — think about what a leaf prediction actually is."
- **True:** choosing test days by their *actual* value (top or bottom decile of `y_test`) guarantees under-prediction at the top and over-prediction at the bottom for any imperfect forecaster: this is regression to the mean. The training-range ceiling never binds, because no test day is outside the training range. A linear regression on the same features, which *can* extrapolate, shows a larger pull toward the middle. Sorting by the tree's *prediction* reverses the sign.
- **Evidence:** re-ran `work/G04/lab4.py` (the lab's split and features): `range 0 5016 … test max 4717 n test > train max 0`; `fulltree hi under 186 | lo over 207`; OLS with dow dummies `hi under 255 | lo over 269`; selected on prediction: top decile actual 3,820 vs prediction 4,137.
- **Fix:** reframe the step as regression to the mean, or demonstrate the ceiling on a series that actually leaves its range (L05's RSXFS example).
- **Confidence:** high · **Verified:** re-computation (re-run during verification).

#### C7 — Lab 5 promises HW03 has "no single dominant" feature and that `sqrt` will win there; neither holds
- **Where:** `Labs/Lecture05_lab.qmd:159-160`; `Narration/Lecture05_lab_script.md:195-197, :451-454`; `Narration/Lecture05_script.md:273-274`.
- **Says:** "In Homework 3 you will run a forest on 27 features with no single dominant one. Check whether the ranking flips there." Narration: "twenty-seven features should restore the textbook answer."
- **True:** `units_lag1` dominates HW03: MDI 0.788 in an all-features forest. HW03 Q6 itself refers to "the dominance of `units_lag1`". On HW03's own split, `sqrt` loses by about 70 units, a wider gap than in Lab 5, and it loses at every seed tried.
- **Evidence:** re-ran `work/G04/hw3.py` (HW03 code verbatim): `mf None 828.4 | mf 0.6 800.8 | mf sqrt 898.5`; `all-feature MDI {'units_lag1': 0.788, 'units_lag4': 0.151, …}`. `hw3seed.py`: None vs sqrt at seeds 0/1/2/7 = 833/896, 826/891, 826/893, 828/875.
- **Fix:** drop the prediction, or say HW03 is lag1-dominated, so `sqrt` should be expected to lose there too.
- **Confidence:** high · **Verified:** re-computation (re-run during verification).

#### C8 — Lab 6's 62-feature design has rank 58; the lab narration says OLS has a unique solution
- **Where:** `Narration/Lecture06_lab_script.md:121-126`; `Labs/Lecture06_lab.qmd:100` ("OLS is fine"), `:115, :122-123` (the `OLS_max_coef` column read as a growth story).
- **Says:** "there are still more observations than parameters, so OLS has a unique solution the whole way down this table … The matrix does not have to be singular, only close to it."
- **True:** the design is exactly singular at every training size. `roll7_mean`, `roll14_mean` and `roll28_mean` are exact averages of lags that are already features (lag1…lag28), and all seven `dow_` dummies are included. That makes four exact dependencies, so rank 58 of 62. `LinearRegression` silently returns the minimum-norm least-squares solution, so the `OLS_max_coef` column reflects that solver's choice, not a property of OLS. The RMSE column is still well defined, because the same identities hold in the test rows, so the lesson that error explodes survives.
- **Evidence:** feature construction read directly (`Labs/Lecture06_lab.qmd:65-79`). `work/G06/lab6b.py`: `rank of standardized X_train 58 of 62`; singular values `… 6.72 5.55 5.6e-14 2.0e-14 1.4e-14 1.3e-14`; `roll7 identity max diff 0.0`; rank 58 at n = 1,210 / 400 / 200 / 120 / 80.
- **Fix:** drop one dow dummy and the three duplicating rolling means, or say plainly that the design is exactly collinear and the coefficient column is the solver's pick.
- **Confidence:** high · **Verified:** re-computation + code reading.

#### C9 — L06: "nobody told the model about the week", yet the features include seven day-of-week dummies
- **Where:** `Slides/Lecture06_Regularization.tex:206-207`; `Quarto/Lecture06_Regularization.qmd:178`; `Labs/Lecture06_lab.qmd:196-197`; `Narration/Lecture06_script.md:467-469`; `Narration/Lecture06_lab_script.md:236-239`.
- **Says:** "Seven of the last eight are multiples of 7: nobody told the model about the week."
- **True:** the 62 features include `dow_0…dow_6` and `roll7_mean/std` (`Labs/Lecture06_lab.qmd:68, :76-79`). At small penalty the largest coefficients *are* dow dummies. The defensible finding is that, under heavy penalty, LASSO keeps the weekly lags and drops the dummies. The lab script contradicts itself two sentences later ("The day-of-week dummies were there as an alternative route").
- **Evidence:** `work/G06/lab6c.py`: `biggest at smallest alpha: dow_5 174.4, lag1 161.4, dow_6 148.0`; every dow dummy exits by log10 α ≈ 1.7, before any of the last eight lags (2.25–2.73).
- **Fix:** "The model also had day-of-week dummies, yet under heavy penalty it kept the weekly lags and dropped the dummies."
- **Confidence:** high · **Verified:** code reading + re-computation.

#### C10 — L11 Part 2 deck names WI_2 as the cost of pooling; the series pooling lost on is CA_3
- **Where:** `Slides/Lecture11_Part2_Hierarchical.tex:266-267`; `Quarto/Lecture11_Part2_Hierarchical.qmd:201`.
- **Says:** "Pooling won on four series of five; WI\_2 shows the cost, pulled past its truth."
- **True:** from `Figures/Lecture11_Part2/measured.json`: CA_3 unpooled 0.105, pooled 0.082, truth 0.102, so the unpooled estimate is closer and this is the one series pooling lost. WI_2 unpooled 0.338, pooled 0.103, truth 0.207: pooling *won* (error 0.104 vs 0.132). The lecture narration (`Narration/Lecture11_Part2_script.md:526, :536`) gets this right, so the deck contradicts it.
- **Evidence:** `b_unpooled`, `b_hier` and `heldout_truth` arrays in `measured.json` (read directly).
- **Fix:** "Pooling won on four series of five; CA_3 lost narrowly. WI_2 was pulled past its truth, yet still ended closer than its own estimate."
- **Confidence:** high · **Verified:** against cache.

#### C11 — Lab 12 says centering removed the confounding; the series intercepts do that
- **Where:** `Labs/Lecture12_lab.qmd:128-130`; `Narration/Lecture12_lab_script.md:140-141, :209, :382`.
- **Says:** "Center log price within each series, so the model uses only price variation *inside* a series over time." Narration: "Centering within series is what removed the confounding."
- **True:** the model already has per-series intercepts `a[idx]`, and by Frisch–Waugh–Lovell those confine the price coefficient to within-series variation. Centering is a reparameterisation that leaves the coefficient unchanged; what it improves is sampling, which is what the deck ("ESS 184 → 3,097") and HW07 say.
- **Evidence:** statsmodels on the lab's data: `ly ~ C(series) + lpc + snap + event + yr` → b_price −0.4960; with uncentred `lp` → −0.4960, identical CI [−0.598, −0.394]. With N(ȳ, 1) priors on the intercepts and 277 weeks per series, the Bayesian fit differs only through that prior, which is negligible here.
- **Fix:** "The series intercepts remove the between-series confounding; centering within series leaves the coefficient unchanged and makes the sampler's job easier."
- **Confidence:** high · **Verified:** re-computation.

#### C12 — L13: "our first two Bayesian fits … failing convergence outright"
- **Where:** `Slides/Lecture13_Synthesis.tex:283-284`; `Quarto/Lecture13_Synthesis.qmd:198`; `Narration/Lecture13_script.md:440-441`.
- **Says:** "Ignoring $\hat{R}$ and divergences & Our first two Bayesian fits returned reasonable-looking forecasts while failing convergence outright."
- **True:** the course's first Bayesian fit passed every check (L10: R-hat 1.00, ESS 3,290, 0 divergences; Lab 10 prints R-hat 1.001, 0 divergences). The fits that failed are L11P1's centred fit (R-hat 1.022) and L11P2's centred fit. The L11P2 fit passes R-hat (1.003) and ESS (966); it fails only on its 43 divergences, and it estimates SNAP effects, not a forecast.
- **Evidence:** `docs/labs/Lecture10_lab.html`, `Lecture11_Part1_lab.html` ("R-hat 1.022"), `Lecture11_Part2_lab.html` ("centered r_hat 1.003 ess 966 divergences 43").
- **Fix:** "Two of our fits, the centred structural model (L11P1) and the centred hierarchy (L11P2), looked fine while failing R-hat or divergences."
- **Confidence:** high · **Verified:** against cache.

#### C13 — The datasets guide describes the wrong electricity dataset
- **Where:** `ECON8310_Datasets.md:24, :113-115, :125-128` (published as `docs/files/datasets.html`).
- **Says:** "Half-hourly electricity demand, with temperature as a covariate. We take it from the Monash Time Series Forecasting Repository" … "temperature drives demand non-linearly … That U-shape is precisely what a smooth term in a GAM captures."
- **True:** the course uses PJM East (PJME) *hourly* metered load from a GitHub mirror (`scripts/prep_electricity.py:1-12`). Neither file has a temperature column, so students cannot fit the temperature U-shape the guide promises. `Labs/README.md` correctly says PJM.
- **Evidence:** `head -1`: `electricity_daily.csv` → `date,mw_mean,mw_peak,mw_trough,dow,is_weekend,doy,month,year`; `electricity_hourly.csv` → `datetime,mw,hour,dow,month`.
- **Fix:** describe PJM East hourly load (2002–2018 daily, 2016–2018 hourly) with no covariate, and remove the temperature paragraph.
- **Confidence:** high · **Verified:** data inspection + cross-reference.

#### C14 — Lab 3: pyGAM beat Prophet "by roughly 12% on both RMSE and MAE"; on MAE the gap is 15%
- **Where:** `Labs/Lecture03_lab.qmd:271`; `Narration/Lecture03_lab_script.md:30-31, :281-282, :370-371`.
- **Says:** "Yet on this holdout the simpler model won, by roughly 12% on both RMSE and MAE."
- **True:** the lab's printed table reads `pyGAM (doy + dow) 3444 2550 / Prophet 3935 3001`. RMSE: 1 − 3,444/3,935 = 12.5%. MAE: 1 − 2,550/3,001 = 15.0%.
- **Evidence:** rendered `docs/labs/Lecture03_lab.html` (checked during verification).
- **Fix:** "by about 12% on RMSE and 15% on MAE".
- **Confidence:** high · **Verified:** against cache + arithmetic.

#### C15 — Lab 7: "the window count is far larger than the row count"; 7,530 windows < 8,310 rows
- **Where:** `Labs/Lecture07_lab.qmd:95-96` (rendered in `docs/labs/Lecture07_lab.html`).
- **Says:** "**Notice the window count is far larger than the row count.** Every week is the end of one window and the interior of twenty-five others, so 8,310 rows become ~7,500 overlapping examples."
- **True:** 30 × (277 − 26) = 7,530 windows, which is 780 *fewer* than the 8,310 rows. The point about overlap is right, but the first sentence contradicts the lab's own printout. HW05 P1 Q1 (`:239-241`) asks students to explain why the count is *fewer*.
- **Evidence:** lab prints `30 series, 8,310 rows` and `X shape (7530, 26, 4)`.
- **Fix:** "Notice the window count is close to the row count, yet the windows overlap heavily: every week sits inside 26 of them."
- **Confidence:** high · **Verified:** re-computation + against cache.

### Major

#### M1 — Prompt budget: every assignment treats it as a hard stop; the syllabus, troubleshooting guide and home page call it "a target, not a limit"
- **Where:** all 11 `Homework/HW*.qmd` initial prompts (e.g. `HW01_Part1_ETS.qmd:135-136`, `HW04_XGBoost_Regularization.qmd:205-206`); "N or fewer entries" checklists in 10 of 11 (e.g. `HW01_Part1:228`, `HW01_Part2:251`, `HW03:362`, `HW04:454`). Against: `syllabi/ECON8310_2026Fall.md:164`, `ECON8310_Troubleshooting.md:159-162`, `docs/index.html:447`.
- **Says:** HW: "If a prompt would take me past my budget of 4, do not do the work. Reply exactly PROMPT_LIMIT_REACHED, and still log that prompt." Syllabus: "**The budget is a target, not a limit.** … **You may go over it** — log the extra prompts … That costs you nothing."
- **True:** the two cannot both hold. A student who follows the syllabus has the assistant refuse prompt N+1 and then fails the checklist.
- **Evidence:** `grep -l PROMPT_LIMIT_REACHED Homework/*.qmd | wc -l` → 11; the syllabus paragraph (read directly).
- **Fix:** choose one policy. If it is a target, replace the refusal with "log it as over budget and continue", and change the checklist to "every prompt logged; over-budget prompts noted". Also check the student repo's `policy/homework_limits.json`, which this session could not see.
- **Confidence:** high · **Verified:** cross-reference.

#### M2 — Synthesis week: Week 15 in the schedule, Week 16 in five other places
- **Where:** `scripts/site.yml:225-240` and `syllabi/ECON8310_2026Fall.md:393, :398` (Week 15 = Lecture 13; Week 16 = presentations). Against: `Narration/Lecture12_lab_script.md:8, :354`; `Narration/Lecture13_script.md:13`; `ECON8310_Datasets.md:59, :177`; `CLAUDE.md:236, :249`.
- **Says:** "See you in Week 16 for the synthesis." / "Week 16 on the syllabus." / "We use this in Week 16."
- **True:** the published schedule puts Lecture 13 in Week 15, and Week 16 is the final project presentations.
- **Evidence:** `site.yml` (read directly) and the syllabus lines above.
- **Fix:** change those references to Week 15.
- **Confidence:** high · **Verified:** cross-reference.

#### M3 — Holt-Winters results depend on the platform; the narration says a mismatch means the data changed
- **Where:** `Narration/Lecture01_lab_script.md:18-20, :172, :186, :357-358, :366, :375`; `docs/labs/Lecture01_lab.html` (prints 885 / 814); `Figures/Lecture13_Synthesis/measured.json` and the L13 `skill_bars` figure (`Slides/Lecture13_Synthesis.tex:118`; HW1 Holt-Winters −8%); `Narration/Lecture13_script.md:162, :185`.
- **Says:** Lab 1 Holt-Winters "885 / 814 … These are deterministic; if yours differ, the data changed." / "Holt-Winters emits a `ConvergenceWarning` on this series". L13: "Holt-Winters, minus eight percent"; backups "about six and seven percent".
- **True:** with the pinned numpy 2.4.6 / scipy 1.16.3 / statsmodels 0.14.6 on Linux and the unchanged data (`m5_weekly.csv` last committed 2026-09-01), Lab 1's cells print Holt-Winters **875 / 804**, with no ConvergenceWarning. HW01's walkthrough model gives **1,768.5 (−6.6%)**, not 1,791 (−8%); the damped-trend backup gives +7.2%. Every other row (SES, Holt, seasonal naive, all ARIMA rows) matches to the unit. The fit lands on a boundary (trend and seasonal smoothing both 0), so the optimum depends on the platform. The qualitative claims all survive. The trouble is that students on Linux or Windows see different numbers and are told their data is wrong. The HW01 keys, which quote 1,791, are affected the same way. HW01 is already graded, so this matters next term.
- **Evidence:** re-ran `work/G01/labcells.py` (all Lab 1 `{python}` cells unchanged) → `Holt-Winters 875 804`, and `work/G13/hw1.py` → `Holt-Winters (additive) 1768.52 skill -6.6% … damped trend 1539.59 skill 7.2% … no trend 1547.38 skill 6.8%`, both during verification.
- **Fix:** mark Holt-Winters numbers as optimiser-dependent ("≈ 875–885"; "−7 to −8%"), drop "if yours differ, the data changed" for those rows, and make the ConvergenceWarning note conditional.
- **Confidence:** high (that the numbers differ); medium (that the cause is the platform) · **Verified:** re-computation.

#### M4 — L05: "the first to beat seasonal naive" compares a one-week-ahead forest with year-ahead ETS/ARIMA
- **Where:** `Slides/Lecture05_TreeEnsembles.tex:221-225`; `Quarto/Lecture05_TreeEnsembles.qmd:183, :185`; `Narration/Lecture05_script.md:376-390, :691-693`.
- **Says:** "the first in this course to do so on this series … lags, price, and SNAP days are information ETS and ARIMA never had."
- **True:** in the test year the forest is fed `units_lag1..4`, the actual sales of the previous weeks, which makes it a one-step-ahead forecast. The course's test-year ETS/ARIMA numbers are 52-step. Lags are the series' own history, which ETS/ARIMA do use. At one step, naive, SES and ARIMA all beat 1,660, and a forest on own-history lags alone gets 1,231, about 430 of the 545-unit gain before any price or SNAP column.
- **Evidence:** `work/G04/horizon.py` (L05 figure split): `naive lag1 (1-step) 1344.8 | SES 1-step 1303.7 (52-step 1755.9) | ARIMA(1,1,1) 1-step 1260.6 (52-step 1735.2) | RF all 8 features 1115.0 | RF own lags only 1230.7`.
- **Fix:** state the horizon, and say most of the gain comes from seeing last week's actuals; drop "information ETS and ARIMA never had".
- **Confidence:** high · **Verified:** re-computation.

#### M5 — Lab 10: "with enough data the prior stops mattering"; the deliberately wrong prior moves the headline probability from 0.08 to 0.61
- **Where:** `Labs/Lecture10_lab.qmd:189` (callout title), `:191-192`, `:218-219` (Discuss); `Narration/Lecture10_lab_script.md:302`.
- **Says:** "All three posteriors land close together, because 277 observations dominate." / "a deliberately wrong prior barely changed the answer."
- **True:** the Beta(20,2) prior is worth about 22 observations against 277. Posterior mean 0.308 vs 0.263; 94% HDIs [0.258, 0.358] vs [0.214, 0.313] barely overlap; P(θ > 0.30) is 0.608 vs 0.084. The lab script's own appendix (`:355-356`) concedes "a little generous".
- **Evidence:** closed form (scipy), re-run during verification: `(2,2) mean 0.263 P>.30 0.084`; `(20,2) mean 0.308 P>.30 0.608`.
- **Fix:** "with enough data the prior is dragged most of the way", and state the residual effect.
- **Confidence:** high · **Verified:** re-computation.

#### M6 — L11P1 contrasts "unchecked" ARIMA intervals with the Bayesian band; the contrast is false
- **Where:** `Slides/Lecture11_Part1_BayesianTS.tex:72-74`; `Quarto/Lecture11_Part1_BayesianTS.qmd:65`; `Narration/Lecture11_Part1_script.md` (~113-128).
- **Says:** "A prediction interval from ARIMA widens too, but according to a formula derived under assumptions you did not check." Narration: "probably could not check … An ARIMA interval you can only accept or reject."
- **True:** an ARIMA interval also follows from the model you wrote down, and L02 teaches how to check it (`Slides/Lecture02_ARIMA_VAR.tex:223, :226`, Ljung-Box). The local-level model is ARIMA(0,1,1)-equivalent, and the deck's own slide gives its widening as a closed-form formula (`:291-293`). The real difference is narrower: the Bayesian band also carries parameter uncertainty.
- **Fix:** "ARIMA's interval plugs in point estimates of its parameters; this one also carries their uncertainty, and each source of widening is a named component."
- **Confidence:** medium · **Verified:** cross-reference.

#### M7 — L06's KFold warning cites Bergmeir, Hyndman & Koo (2018), which shows K-fold is valid for autoregressive models
- **Where:** `Slides/Lecture06_Regularization.tex:224-226`; `Quarto/Lecture06_Regularization.qmd:191`; `Narration/Lecture06_script.md:525-532`; related `Narration/Lecture06_lab_script.md:340-344`.
- **Says:** "Never use KFold here. Random folds train on the future and validate on the past: CV looks excellent and the forecast fails \parencite{Bergmeir2018}."
- **True:** the cited paper (`Bibliography_base.bib:294-302`) shows that standard K-fold is valid for purely autoregressive models with uncorrelated errors, and in its experiments it compares favourably with out-of-sample evaluation. On Lab 6's lag-regression, shuffled KFold did *not* "look excellent and fail": its CV estimate was closer to test error, and the alpha it chose forecast better.
- **Evidence:** `work/G06/kfold.py` (Lab 6 Step 4 pipeline and grid): `TSS alpha 19.64 CV 323.4 test 286.3 | KFold-shuffle alpha 3.71 CV 287.9 test 278.3`.
- **Fix:** cite the paper for what it says (K-fold is fine for pure AR with white-noise errors; prefer time-ordered splits under trend or serial correlation), and drop the unconditional "CV looks excellent and the forecast fails".
- **Confidence:** high (citation); medium (how strongly the lab-script passage should change) · **Verified:** re-computation + cross-reference.

#### M8 — HW04's reason for 46 features (to reach the regime where regularisation "earns its keep") does not hold on its own data
- **Where:** `Homework/HW04_XGBoost_Regularization.qmd:66-73`.
- **Says:** "With k that much smaller than n, there is very little for a shrinkage penalty to do … Regularization earns its keep when k is *large relative to* n … So we build the feature set …"
- **True:** HW04 has k = 46 and n = 5,160, so n/k = 112 (HW03: 191). Lab 6 found regularisation "buys essentially nothing" already at n/k = 19.5. On HW04's own features, even the best penalty chosen on the test set improves on OLS by 0.3 units.
- **Evidence:** `work/G06/hw.py`: `n/k 112.17 | OLS test 744.44 | ridge oracle best 744.45 | lasso oracle best 744.16`.
- **Fix:** present the 46 features as a test of selection and interpretability, not of a large k relative to n, or shrink the training window as Lab 6 does.
- **Confidence:** high · **Verified:** re-computation.

#### M9 — L08 attributes ResNet's degradation problem to fading gradients; the cited paper rules that out
- **Where:** `Slides/Lecture08_CNNs.tex:175-176, :210-211`; `Quarto/Lecture08_CNNs.qmd:140, :165`; `Narration/Lecture08_script.md:356-357, :434-435`.
- **Says:** "deeper networks got *worse* — not from overfitting, but because gradients faded on the way back."
- **True:** He et al. (2016, §4.1), which the deck cites, write that the degradation "is unlikely to be caused by vanishing gradients … we also verify that the backward propagated gradients exhibit healthy norms with BN", and call it an optimisation difficulty. The "not overfitting" part and the additive gradient path (`:182`) are fine.
- **Fix:** "deeper plain networks trained worse — an optimisation difficulty, not overfitting; the skip connection makes the identity easy to learn and gives the gradient an additive path."
- **Confidence:** high · **Verified:** cross-reference (external source).

#### M10 — L01: "`KFold` … shuffles"; by default it does not, and it leaks the future anyway
- **Where:** `Narration/Lecture01_script.md:586-588`; deck `Slides/Lecture01_ETS_Eval.tex:311-314` (`Quarto/...:261`).
- **Says:** narration "`KFold` is the default in most tutorials, and it shuffles." Deck: "Use TimeSeriesSplit in scikit-learn, never KFold. Shuffling lets future data train the model".
- **True:** `KFold(n_splits=5, *, shuffle=False)` cuts contiguous blocks; every fold except the last still trains on later blocks. A student who sets `shuffle=False` would wrongly conclude the problem is fixed. L13's narration (`:387`) describes KFold correctly.
- **Evidence:** `KFold(3).split(range(9))` → test [0,1,2] with train [3..8].
- **Fix:** "KFold, shuffled or not, holds out each block and trains on the others, including later ones."
- **Confidence:** high · **Verified:** re-computation.

#### M11 — L03: a GAM "cannot represent" an interaction
- **Where:** `Slides/Lecture03_GAMs.tex:298` (table "GAM & Yes & No"), `:307-309`; `Quarto/Lecture03_GAMs.qmd:238`; `Narration/Lecture03_script.md:585, :593-596`.
- **Says:** "if the effect of price depends on whether it is a holiday week, a GAM cannot represent that and a tree can."
- **True:** GAMs fit interaction terms you specify: by-factor smooths (`s(price, by=holiday)`) and tensor products (`te()` in pyGAM and mgcv; Wood 2017 ch. 5, the deck's text). The real limitation is the one the same table gives linear regression: "Manual".
- **Fix:** table "Manual" for GAM; "a GAM represents it only if you add the term (`te()` or a by-factor smooth); a tree finds it automatically."
- **Confidence:** high · **Verified:** cross-reference.

#### M12 — L03: "reach for pyGAM when you have many correlated external predictors"
- **Where:** `Slides/Lecture03_GAMs.tex:229-230`; `Quarto/Lecture03_GAMs.qmd:177`; `Narration/Lecture03_script.md:452`.
- **True:** correlated predictors cause concurvity, the GAM's known weak case. The same deck's table recommends GAMs for "Few predictors, explain effects" (`:298`), and the narration (`:502-516`) warns that "under concurvity, the individual curves become unreliable".
- **Fix:** "reach for pyGAM when the drivers are a few non-calendar predictors whose shape you want to see."
- **Confidence:** high · **Verified:** cross-reference.

#### M13 — HW02 promises a "common holdout" RMSE comparison with HW1 that its own callout forbids
- **Where:** `Homework/HW02_GAMs.qmd:28` (objective), `:21`, `:302` (table row "Best HW1 model … *from HW1*" in the Test RMSE column).
- **Says:** "Compare GAM-family models against ARIMA/ETS on a common holdout set using RMSE".
- **True:** HW1 is weekly (52-week holdout) and HW02 is daily (365-day holdout); the assignment's own callout (`:320-322`) says the two RMSEs "must not be put side by side".
- **Fix:** change the objective to "…against a seasonal-naive benchmark on a common daily holdout, and explain why the HW1 RMSEs are not comparable", and take the HW1 row out of the RMSE column.
- **Confidence:** high · **Verified:** cross-reference.

#### M14 — HW07's overview describes a model of "promotions, competition, and store characteristics"
- **Where:** `Homework/HW07_Bayesian_Regression.qmd:46-47`.
- **True:** the assignment regresses log units on log price, SNAP, events, a trend and series effects, using M5. M5 has no promotion or competition variables, and no question uses any.
- **Evidence:** `m5_weekly.csv` header; the HW07 setup and initial prompt.
- **Fix:** "…a Bayesian regression of weekly log units on log price, SNAP and event weeks, a trend and series effects."
- **Confidence:** high · **Verified:** cross-reference.

#### M15 — The datasets guide credits M5 with "promo" and "store type" features
- **Where:** `ECON8310_Datasets.md:70-72`.
- **Says:** "A wide tabular feature set: price, promo, SNAP, calendar effects, store type, category, and lags."
- **True:** M5 has neither. The panel carries `avg_price, snap_days, event_days`, store/category/state IDs, and lags. HW03/HW04 feature lists match the panel.
- **Fix:** "price, SNAP, calendar events, store/category/state identifiers and lags."
- **Confidence:** high · **Verified:** cross-reference.

#### M16 — The datasets guide says the deep-learning lectures have "thousands of parallel series"
- **Where:** `ECON8310_Datasets.md:73-74`.
- **Says:** "Thousands of parallel series and nearly 2,000 time steps. Neural networks need volume, and this has it."
- **True:** Lectures 7–9, their labs and HW05 use 30 weekly series × 277 weeks (5,160 training windows at W = 52). L13 Lesson 2 (`Slides/Lecture13_Synthesis.tex:138-139, :161`) and L09P2 (`:251`) argue this is *not* enough volume.
- **Fix:** "a 30-series weekly aggregate; Lesson 2 shows that is not enough for the largest models."
- **Confidence:** high · **Verified:** cross-reference + data count.

#### M17 — Lab 3 narration: "Monday highest", "Mondays run about a thousand megawatts above Sundays"
- **Where:** `Narration/Lecture03_lab_script.md:96, :200-201, :353`.
- **True:** the lab prints `day of week swing 11.6% min 6 max 1` with 0 = Monday, so the peak is **Tuesday**. Monday is about 2,950 MW above Sunday in the training data (3,013 in Prophet's weekly component). `scripts/prep_electricity.py:12` also says "Tuesday peak".
- **Evidence:** lab HTML (checked); daily means by day of week on the full file, re-run during verification: Mon 32,672, Tue 33,272, Sun 29,406.
- **Fix:** "Tuesday highest, Sunday lowest"; "weekdays run about three thousand megawatts above Sundays".
- **Confidence:** high · **Verified:** re-computation + against cache.

#### M18 — Lab 3 narration: "Every model beat the benchmark"
- **Where:** `Narration/Lecture03_lab_script.md:278`.
- **True:** the same script (`:242`) says the single-harmonic Prophet loses on MAE (3,358 vs 3,333), and the lab itself (`Labs/Lecture03_lab.qmd:261`) was corrected to "on RMSE".
- **Fix:** "Every model beat the benchmark on RMSE."
- **Confidence:** high · **Verified:** cross-reference (the 3,358 itself is not cached; see §1).

#### M19 — Lab 7 narration: "the setting with the worse spread has the marginally better mean"
- **Where:** `Narration/Lecture07_lab_script.md:241-243`.
- **True:** in the printed table, weight_decay = 1e-4 has the worse spread (sd 134 vs 86) *and* the worse mean (1,088 vs 1,084); no weight decay is better on all four columns.
- **Evidence:** `docs/labs/Lecture07_lab.html`: `0 1084 86 946 1176` / `0.0001 1088 134 991 1343` (checked).
- **Fix:** "The setting with the tighter spread also has the marginally better mean."
- **Confidence:** high · **Verified:** against cache.

#### M20 — Lab 5 narration: OOB error is worse because each OOB prediction uses a "smaller forest"
- **Where:** `Narration/Lecture05_lab_script.md:222-226`.
- **True:** Step 2 of the same lab shows 100 trees cost about 1 unit (249 vs 248), and a 110-tree forest scores 247.7. That cannot explain a 49-unit gap (297 vs 248). The gap comes from which rows are scored: OOB covers 2012–2015 (per-year OOB 311 / 319 / 283 / 239), including three Christmas zero days; dropping those gives 275.
- **Evidence:** `work/G04/oob.py`.
- **Fix:** drop or downplay reason one; attribute the gap to the period scored.
- **Confidence:** high · **Verified:** re-computation.

#### M21 — L04 narration's extrapolation example (29% growth, train four years, forecast the fifth) does not happen on this series
- **Where:** `Narration/Lecture04_script.md:581-585`.
- **True:** almost all of the 29% is 2011→2012 (weekly means by year 15,631 / 20,101 / 20,852 / 21,129 / 20,205). Training on four years leaves 0 of 52 year-five weeks above the training maximum (26,584 vs 23,619). The Lab 4 narration (`:315-316, :412`) says the ceiling "did not bite".
- **Evidence:** `work/G04/l4narr.py`.
- **Fix:** use a series that actually leaves its range (RSXFS), or say CA_1 plateaued.
- **Confidence:** high · **Verified:** re-computation.

#### M22 — L12 narration: b_snap = 3.7% is "consistent with what the hierarchy told us last week"
- **Where:** `Narration/Lecture12_script.md:182-183`.
- **True:** last week's FOODS hierarchy gave μ_b = 0.078 (about 8%; `measured.json`); the 30-series hierarchy gave 0.011. L12's single b_snap pools all 30 non-exchangeable series. It matches the cross-category average of raw effects ((0.089 + 0.006 + 0.011)/3 = 0.035), which is complete pooling, the answer L11P2 taught students to reject.
- **Fix:** drop the clause, or "a panel-wide average; last week's FOODS hierarchy put the FOODS lift near 8%."
- **Confidence:** medium · **Verified:** re-computation + against cache.

#### M23 — Lab 11P2 narration's model answer to "name the other five" differs from the deck and HW06
- **Where:** `Narration/Lecture11_Part2_lab_script.md:430-432` (answering `Labs/Lecture11_Part2_lab.qmd:425-427`).
- **Says:** "ridge, LASSO, the bagged ensemble, the random forest, boosting's learning rate".
- **True:** the deck (`Slides/Lecture11_Part2_Hierarchical.tex:153-155`), HW06 Q6 (`Homework/HW06_Part2_Hierarchical.qmd:334-335`) and the lecture narration list GAM smoothing penalty, pruning, `reg_lambda`, Ridge/LASSO, weight decay. Bagging and random forests average models; they do not shrink an estimate toward a common value.
- **Fix:** use the deck's list.
- **Confidence:** high · **Verified:** cross-reference.

#### M24 — Lab 11P1 narration gives the retracted 127 divergences
- **Where:** `Narration/Lecture11_Part1_lab_script.md:367` (vs `:123`, "forty-three").
- **True:** the L11P2 lab prints 43 at seed 42; CLAUDE.md records that 127 did not reproduce.
- **Fix:** 127 → 43.
- **Confidence:** high · **Verified:** against cache.

#### M25 — L11P1 previews "hierarchical models, for thirty series at once"; L11P2 fits ten and argues the thirty cannot be pooled
- **Where:** `Slides/Lecture11_Part1_BayesianTS.tex:328`; `Quarto/Lecture11_Part1_BayesianTS.qmd:252`; `Narration/Lecture11_Part1_script.md:634`.
- **True:** L11P2's model uses the ten FOODS series (`Slides/Lecture11_Part2_Hierarchical.tex:65`), and its opening argues "Pooling all thirty would tell the model they are interchangeable" (`:44-45`). A 30-series fit exists, but only as the counter-example. The L11P1 lab narration says "ten series".
- **Fix:** "hierarchical models — ten series at once, some with barely any data."
- **Confidence:** medium · **Verified:** cross-reference.

#### M26 — L05 narration: the forest sees "three of the eight features" per split; `sqrt` of 8 gives 2
- **Where:** `Narration/Lecture05_script.md:333-334`.
- **True:** `max_features='sqrt'` with 8 features → `int(sqrt(8))` = 2 (the daily figure's "3 of 10" is right).
- **Evidence:** `RandomForestRegressor(max_features='sqrt').fit(<8 cols>).estimators_[0].max_features_` → 2.
- **Fix:** "two of the eight".
- **Confidence:** high · **Verified:** re-computation.

#### M27 — HW05: "four architectures … FFN, 1D CNN, LSTM, and Transformer"; there are five, and Part 1 builds two
- **Where:** `Homework/HW05_Part1_FFN_CNN.qmd:45-55, :157-158`; `Homework/HW05_Part2_Sequence.qmd:136-137`; `Homework/HW05_Part3_Comparison.qmd:145-146` (all in the initial prompt pasted to the AI).
- **True:** HW05 has five architectures. Part 2 Q1 is the vanilla RNN, and Part 3 says "Compare five architectures — FFN, CNN, RNN, LSTM and Transformer" (`:49`). Part 1 covers only FFN and CNN.
- **Fix:** "five … FFN, 1D CNN, vanilla RNN, LSTM, Transformer", and limit the Part 1 prompt to FFN and CNN.
- **Confidence:** high · **Verified:** cross-reference.

### Minor

| ID | Where | Says | True (evidence) | Fix | Conf. | Verified |
|---|---|---|---|---|---|---|
| m1 | `Narration/Lecture06_script.md:534`; `CLAUDE.md:222` | "exactly as in Lecture 1's walk-forward evaluation"; CLAUDE.md lists "walk-forward CV, DM test" for L01 | L01 uses one fixed split; walk-forward first appears in Lab 2 (`Labs/Lecture02_lab.qmd:171`); "Diebold"/"DM test" appears nowhere in Slides, Quarto, Narration, Labs or Homework (grep) | "Lab 2's walk-forward evaluation"; drop the two items from the CLAUDE.md row | high | cross-ref |
| m2 | `Slides/Lecture01_ETS_Eval.tex:326-352`; `Narration/Lecture01_script.md:25, :149, :607, :680`; `Slides/Lecture13_Synthesis.tex:33` | "Twelve lectures, four parts"; L13 "Twelve weeks" | The syllabus has a Lecture 13; Lectures 1–12 span Weeks 1–14 in `site.yml` | "twelve method lectures and a synthesis"; "fourteen weeks" | high | cross-ref |
| m3 | `Slides/Lecture01_ETS_Eval.tex:194` | ŷ_{T+h\|T} = ℓ_T + h b_T + s_{T+h−m} | Valid only for h ≤ m; general form s_{T+h−m(k+1)}, k = ⌊(h−1)/m⌋ (FPP3 §8.3); the slide names m = 12, where h > m is routine | add "(for h ≤ m)" or the general index | high | cross-ref |
| m4 | `Homework/HW01_Part1_ETS.qmd:112` (AI prompt) | "lag features (units_lag1 through units_lag52)" | The CSV has units_lag1–4 and units_lag52 only; HW03:132 states it correctly; an assistant may generate `units_lag5` and fail | "(units_lag1 through units_lag4, units_lag52)" | high | data header |
| m5 | `Slides/Lecture01_ETS_Eval.tex:282-283`; Quarto `:245`; `Narration/Lecture01_script.md:552` | MASE "is the standard in the M-Competitions" | M4 scores OWA (sMAPE + MASE); M1–M3 predate MASE; M5 used WRMSSE | "one of the two accuracy measures in M4" | medium | external |
| m6 | `Slides/Lecture02_ARIMA_VAR.tex:312-313`; `Narration/Lecture02_script.md:597` | "The rooster Granger-causes the sunrise." | Under the deck's own definition (`:305-306`, "beyond what y's own past already explains"), sunrise's own past predicts it perfectly, so crowing adds nothing; the example shows plain correlation (a common textbook illustration, but inconsistent with the slide's definition) | an example with incremental content (e.g. umbrella sales and rain) | medium | cross-ref |
| m7 | `Narration/Lecture02_script.md:388-395` | auto_arima picks the ARIMA(5,2,4) because it "minimizes AIC" | pmdarima picks d with a unit-root test (KPSS by default) before the AIC search; AIC cannot compare different d | "its unit-root step chose d = 2; AIC then over-fitted p and q" | medium | docs (not run) |
| m8 | `Labs/Lecture02_lab.qmd:253-257`; `scripts/figures/lecture02_figures.py:183-211`; deck figure `:319-327`; both L02 narrations | ex-COVID Granger F = 2.91 / 6.74 | Dropping 17 rows and then lagging by position splices Jan 2020 onto Jul 2021; built on the contiguous series: u→r F = 2.42 (p = 0.048), r→u F = 9.96. Conclusions unchanged | build lags first, then drop rows whose target or lags fall in the window | high | re-run in verification |
| m9 | `Narration/Lecture02_script.md:284-286` | alternating PACF "is characteristic of an MA process" | Alternation happens only for θ > 0; the course's own MA(1) (SES ≡ ARIMA(0,1,1)) has θ = −0.38, whose PACF decays on one side (`ArmaProcess(ma=[1,-0.38]).pacf`: −0.33, −0.12, −0.05 …) | "with θ > 0 the PACF alternates; with θ < 0, as in SES, it does not" | high | re-computation |
| m10 | `Narration/Lecture02_script.md:567-569` (deck `:294-296`) | "a hundred parameters from a hundred and twenty data points" | A 5-variable, 120-month VAR has 600 observations; each equation fits 21 coefficients on 116 rows | "21 coefficients per equation" framing | medium | arithmetic |
| m11 | `Homework/HW01_Part2_ARIMA_VAR.qmd:231` | CA_1 and CA_3 are "40 miles apart" | M5 publishes no store locations; no source in the repo | "two stores in the same state" | medium | data header |
| m12 | `CLAUDE.md:268` | HW 1b uses "Store 1 + Store 2" | The assignment uses CA_1 and CA_3 FOODS (`:65, :204, :215`); CA_2 would reverse the Granger result (p = 0.23 vs 4e-5) | "CA_1 + CA_3 FOODS" | high | re-computation |
| m13 | `Slides/Lecture03_GAMs.tex:272`; Quarto `:216` | "Every day of the year is equally covered" | Days 1–212 appear 5×, days 213–365 4× in training; the lab says "four or five times" | "four or five times" | high | re-computation |
| m14 | `Narration/Lecture03_lab_script.md:294` | trough "around mid-April" | Day 116 = 26 April; the deck says "late April" | "late April" | high | re-computation |
| m15 | `Slides/Lecture03_GAMs.tex:259`; Quarto `:206`; `Narration/Lecture03_script.md:483, :487-488` | `gridsearch` tunes λ "by cross-validation" / "held-out error" | pyGAM's gridsearch minimises GCV (in-sample approximation; nothing is held out); the deck's own figure says "λ chosen by GCV" | "by GCV" | high | cross-ref |
| m16 | `Labs/Lecture03_lab.qmd:171-172`; `Narration/Lecture03_lab_script.md:183-185` | leaving daily seasonality on, "Prophet will happily fit noise" | On midnight timestamps the daily Fourier terms are constant (sin = 0, cos = 1): unidentified, not fitting noise | "the daily terms are constant on daily data and cannot be identified" | medium | feature construction only |
| m17 | `Narration/Lecture04_lab_script.md:214-215` | the full tree is "the *worst* model in the table" | The stump is worse (413 vs 406) | "worse than every tree from depth 2 to 12" | high | against cache |
| m18 | `Slides/Lecture05_TreeEnsembles.tex:42`; Quarto `:53`; `Narration/Lecture05_script.md:53` | "At a 1907 county fair" | Galton's ox contest was at Plymouth in 1906; 1907 is when *Vox Populi* appeared in *Nature* | "1906" | medium | external |
| m19 | `CLAUDE.md:225` | L04 covers "entropy/information gain" | L04 is regression-only; entropy appears once in a footnote (`:169`) | "regression CART, cost-complexity pruning" | high | grep |
| m20 | `Slides/Lecture06_Regularization.tex:254` (+ penalty `:159-163`); Quarto `:222`; `Narration/Lecture06_script.md:578, :583-584`; `HW04:333` | "`l1_ratio` is α" | sklearn halves the ℓ2 term (α·ρ‖w‖₁ + ½α(1−ρ)‖w‖²), so the mapping holds only at 0 and 1 | write the penalty as λ[α‖β‖₁ + (1−α)/2 ‖β‖²] | high | sklearn docstring |
| m21 | `Labs/Lecture06_lab.qmd:175-184`; `Narration/Lecture06_lab_script.md:15, :212-220, :404` | elimination order printed "in reverse order"; "Deterministic" | Tied features (lag7/lag28; lag1/lag21) print in Python set order, which varies with the hash seed; the narration and the rendered lab disagree | `sorted(survivors - nz, key=FEATURES.index)` | high | re-computation |
| m22 | `Narration/Lecture06_lab_script.md:87, :115-117` | "Four years"; n = 120 "is the best-fitting model on this slide" | 1,210 days = 3.3 years; n = 80 has a lower training RMSE (84 vs 145) | "about three years"; drop "best-fitting" | high | re-computation |
| m23 | `Slides/Lecture08_CNNs.tex:147` vs Quarto `:118`, `Narration/Lecture08_script.md:43, :301` | "Thirty years of architectures" vs "Twenty years" | The table runs 1998–2015 | one figure ("Two decades") in both | high | cross-ref |
| m24 | `Narration/Lecture08_script.md:315-317` | two stacked 3×3 filters are "strictly more expressive" than a 5×5 | Neither family contains the other; VGG (§2.3) calls the stack a regularisation; the deck's own wording (`:156-157`) is fine | "adds a nonlinearity while constraining the filter" | medium | arithmetic + source |
| m25 | `Slides/Lecture07_NeuralNets.tex:90-91`; Quarto `:81`; `Narration/Lecture07_script.md:193-195` | one hidden layer approximates "*any* continuous function arbitrarily well" | The theorem holds on compact (bounded) domains; a finite ReLU net is linear outside its last kink, which matters in a lecture about extrapolation | add "on a bounded input range" | high | cross-ref |
| m26 | `Slides/Lecture07_NeuralNets.tex:320`; Quarto `:265`; `Narration/Lecture07_script.md:639-640` | RNN/LSTM "Extrapolates: yes" | With a linear head on tanh-bounded states, \|ŷ\| ≤ ‖W_out‖₁ + \|b\|: they can exceed the training maximum only up to a fixed ceiling | "bounded", with a footnote | medium | closed form |
| m27 | `Narration/Lecture08_lab_script.md:51-52` | flattened positions 1 and 5 are "the same quantity four weeks apart" | Time-major flatten of (26, 4): indices 1 and 5 are the same channel *one* week apart (Lab 7 narration says so) | "one week apart" | high | code reading |
| m28 | `Narration/Lecture07_lab_script.md:154`; `Narration/Lecture07_script.md:591-592`; `Narration/Lecture08_lab_script.md:342` | "by about 45%" after quoting 1,109 vs 744; "see what the loss curve looks like"; "no better or worse forecast" | 1,109/744 = 1.49 (the 45% uses a five-seed mean not yet shown); Lab 7 plots no loss curve; Lab 8's 52-week window is 123 RMSE *worse* | "about half"; "what it does to the forecast"; "worse forecast" | high | against cache |
| m29 | `Slides/Lecture09_Part1_RNN_LSTM.tex:116-118`; `Narration/Lecture09_Part1_script.md:244-248` | "about 0.79 a step: 0.4% … 26 weeks back, 0.001% last year" | The percentages are read correctly off the curve, but 0.79²⁶ = 0.2% and 0.79⁵¹ = 0.0006%; the 0.79 slope is fitted from step 5 onward | "past the first few steps, about 0.79 a step; measured, 0.4% …" | medium | arithmetic |
| m30 | `Narration/Lecture09_Part1_script.md:698-699`; `Narration/Lecture09_Part1_lab_script.md:144-145, :380` | "tied once both reached convergence"; LSTM/RNN ratio "**Exactly** four" | The deck says the LSTM is still improving at 100 epochs; 17,985 / 4,545 = 3.957 (only the recurrent layer is exactly 4×) | "tied by 100 epochs"; "almost exactly four" | high | re-computation |
| m31 | `Narration/Lecture09_Part2_script.md:301-302`; `Slides/Lecture09_Part2_Transformers.tex:307-308` (+ Quarto `:235`, narration `:539-542`) | "the original paper adds thirty-two pairs"; M5 "won by ensembles of exactly the methods in Lectures 5 and 6" | Vaswani et al. used d_model = 512 (256 pairs; 32 is this model's); the M5 winner was LightGBM (L05), with no penalised-linear (L06) entry among the top solutions | "our model adds 32 pairs"; "gradient-boosted trees — Lecture 5's method" | high / medium | external |
| m32 | `Narration/Lecture11_Part1_lab_script.md:307`; `Narration/Lecture11_Part1_script.md:597-599` | trade "worth 5%"; no earlier model "from ETS through the Transformer" had a checkable interval | 1 − 1,589/1,660 = 4.3%, and every other place says 4%; ETS intervals are taught in L01 (`:231`) and HW02 plots Prophet bands | "4%"; "we never checked their intervals; here we did" | high / medium | re-computation |
| m33 | `Labs/Lecture11_Part2_lab.qmd:32`; `Labs/Lecture12_lab.qmd:28`; lab narrations `L11P2:15, :442`, `L12:18, :360` | "Six model fits" / "Four model fits" | `fit()` is called 5 and 3 times; the HTML prints 5 and 3 diagnostic blocks | "Five" / "Three" | high | grep + cache |
| m34 | `Homework/HW07_Bayesian_Regression.qmd:263-266`, `:366` | "weakly informative … `alpha ~ Normal(0, 1)`"; "Revenue rises if the percentage volume loss is smaller than the 5% price gain" | ly averages 8.60 (series means 7.17–10.21), so N(0,1) sits 7–10 SDs away; the lab uses N(ȳ, 1). The break-even loss is 1 − 1/1.05 = 4.76% (b > −1), not 5% | `Normal(ly.mean(), 1)`; "rises if 1.05 × (1 + change) > 1, i.e. b > −1" | high | re-computation |
| m35 | `Slides/Lecture12_BayesianRegression.tex:133` (+ Quarto `:121`); `Labs/Lecture12_lab.qmd:253-255`; `Narration/Lecture12_script.md:258`; `Slides/Lecture13_Synthesis.tex:417` (+ narration `:679-680`) | "A significance test would have announced the direction and said nothing about either size question"; L13 "Held-out data caught our worst two" | The OLS 94% CI [−0.598, −0.394] already excludes −1, so a test against −1 settles "inelastic"; what only the posterior gives is a probability per threshold. L13's own slide 15 (`:291-293`) credits held-out data with one catch, not two | "the usual p-value (against zero) gives only direction"; "caught one; a DAG caught another" | medium | re-computation / cross-ref |
| m36 | `Labs/README.md:133-135`; `ECON8310_Datasets.md:56-57, :86-87`; `ECON8310_About.md:156`; `ECON8310_Troubleshooting.md:50`; `CLAUDE.md:248-249`; L13 scoreboard `Slides/Lecture13_Synthesis.tex:82` (+ L08/L09/L12 decks); assignment suggested prompts | Lab 5 "closes on XGBoost (230)"; Christmas forecast "wrong by 100% of the true value"; FRED "consumer price index"; "One executable lab per lecture"; "five .csv files"; "there is no `prep_favorita.py`"; XGBoost 781; per-question suggested prompts | Lab 5 prints 244; the true value is 0 (it is 100% of the *forecast*); `prep_fred.py` pulls UNRATE, UMCSENT and RSXFS only; L13 has no lab; the guide and rubric add `onlinestore_weekly.csv` (six files; unverified in the student repo); `scripts/prep_favorita.py` exists (6cd617c); `measured.json` records 792 under the pinned xgboost 3.4.1 (781 is kept deliberately in `lecture13_figures.py:150-151` but not labelled; rank unchanged); suggested prompts sum to 3/6/8/9/10/13 against budgets 4/8/10/12/12/16 in HW01 P1, HW01 P2, HW02, HW03, HW04, HW07 | 244; "100% of its own forecast"; drop CPI; "except the synthesis"; "six"; drop the clause; 792 or a footnote; note the spare prompts | high / medium | cache, grep, data headers |

---

## 3. Notes for the owner

- **Known items not re-reported:** everything listed in the brief, `quality_reports/plans/2026-10-18_next-term-changes.md`, and the Owner calls / Owner decisions sections of `2026-10-16_figures-round2.md`. Two findings sit next to known items but are distinct. C3 (the readout position) is a different claim from "shuffling leaves every output unchanged". M3 (Holt-Winters platform dependence) does not change the next-term HW01 Q2 item: the no-trend fit gives 1,547 here too.
- **Extra evidence on a known item (not re-reported):** adding lag 52 to a lags 1–4 regression on the L09 panel lowers residual RMSE from 834 to 823 (t = 13.5). This supports the known L09P1 Check-2 item ("the recent past is a sufficient statistic" is overstated).
- **Rendered copies:** most findings in lab prose and assignments also appear in `docs/` (labs, homework, book and workbook chapters). Re-render after fixing, and copy assignment fixes to the student repo (`python scripts/check-assignment-sync.py`).
- **Which findings touch work students are doing now:** C1 (HW04), C2, C3, C4 and C5 (HW05), M1 (all assignments), M14 and m34 (HW07), M13 (HW02), m4 (HW01). Check them against the due dates before deciding which wait until next term.
