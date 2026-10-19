# Fix pass log — 2026-10-19 review findings

**Branch:** `fix/review-2026-10-19`, cut from `main` at `9e7e257`. One commit per deck or document; no pull request.
**Sources:** `quality_reports/2026-10-19_review_merged.md` (Bucket A), `2026-10-19_full_course_review.md` (C/M/m IDs),
`2026-10-18_proofread_all_decks.md` (PR-), `2026-10-18_pedagogy_all_decks.md` (PED-). Seven fixers, one per deck group,
edited disjoint files; the coordinator reviewed each group's diff against its sources before committing.

## Not done, and why

- **Nothing was compiled or rendered.** This container has no `xelatex` and no `quarto`. Fixers estimated vertical fit
  from the last-built PDFs, and a few figures were shrunk slightly to make room (L04 two figures, L09P2 two figures).
  **Before publishing:** compile all 15 decks, render the mirrors and the 11 edited labs, rebuild book/workbook, and
  check the tightest pages each group names below (L04 p.5/11, L05 p.8/14, L06 path frame, L07 comparison and neuron
  frames, L10 p.9/17, L11P1 p.13/15, L12 p.5, the L09P2 frames that gained a line).
- **m8 (Granger lag construction): skipped — done locally.** It changes F statistics printed on a slide, in a figure
  and in the lab.
- **Bucket C/D, owner calls/decisions, C2, the six L09P1 fixes, ✔ items:** not touched, per the brief. HW01–HW04 and
  HW05 Part 1 not edited; their parts of M1, M27 (P1 still says "four" at lines 45 and 157) and m20 (HW04:333) wait.
- **`.venv` (untracked, not committed):** a fixer's pip run left mixed numpy/scipy/scikit-learn builds; `import scipy`
  fails. Rebuild with `pip install -r requirements.txt` before rendering. All gates below ran with the system Python.

## Coordinator items

| ID | files changed | what changed (one line) — or "skipped: <reason>" |
|---|---|---|
| PR-L11P1 / PR-L11P2 takeaway titles | L11P1 and L11P2 .tex, .qmd, narration header | "Lecture 11 Part N: Key Takeaways" → "Lecture 11, Part N: Key Takeaways", matching the decks' subtitles and Lecture 9 (one commit per deck). |
| Week 16 sweep | — | Remaining "Week 16" references (rubric, CLAUDE.md) are the project presentations, which are Week 16; left as is. |

## Group FA — Lectures 1–3

| ID | files changed | what changed (one line) — or "skipped: <reason>" |
|---|---|---|
| M3 | Narration/Lecture01_lab_script.md | "How to use" bullet, Step 4 ("Eight eighty-five here — you may see eight seventy-five…"), appendix lead and HW row now say Holt-Winters is optimizer/platform-dependent (885/814 rendered page; 875/804 Linux + pinned versions); "if yours differ, the data changed" kept only for the other rows; ConvergenceWarning note now conditional ("may emit… on Linux… converges without one"). Labs/Lecture01_lab.qmd makes no determinism claim — unchanged. |
| M10 | L01 .tex, .qmd, Narration/Lecture01_script.md | Warningbox: "never `KFold`: shuffled or not, `KFold` lets future data train the model"; narration: KFold "shuffled or not, holds out each block in turn and trains on the others — including the later ones… Turning shuffling off does not fix it." |
| m2 (L01 part) | L01 .tex, .qmd, Narration/Lecture01_script.md | Four-part slide adds "Lecture 13 closes the course with a synthesis of all four parts."; narration "twelve lectures" → "twelve method lectures and a closing synthesis" / "in every lecture" / "Twelve method lectures in four parts, then a synthesis" / "Every method from here on", plus one line introducing Lecture 13 on slide 19. |
| m3 | L01 .tex, .qmd, Narration/Lecture01_script.md | HW forecast line: "(m = seasonal period; h ≤ m)"; narration adds "as written, for horizons up to one season". |
| m5 | L01 .tex, .qmd, Narration/Lecture01_script.md | "the standard in the M-Competitions" → "one of the M4 Competition's two accuracy measures" (citation kept). |
| m6 | L02 .tex, .qmd, Narration/Lecture02_script.md | Rooster/sunrise → "umbrella sales Granger-cause rain, because shoppers read the forecast" (narration: umbrellas carry information rain's own past does not; umbrellas do not cause rain). |
| m7 | Narration/Lecture02_script.md | auto_arima now "picks d first, with a unit-root test — KPSS by default — then searches p and q by AIC"; AIC cannot compare different d; ARIMA(5,2,4) anecdote: "its unit-root step chose the second difference, and AIC then over-fitted p and q". |
| m8 | — | skipped: done locally (changes printed F statistics in slide, figure and lab). |
| m9 | Narration/Lecture02_script.md | PACF sign flips attributed to θ > 0; with negative θ (SES) the PACF decays on one side; "the decay is the MA signature; the sign flips are not". |
| m10 | L02 .tex, .qmd, Narration/Lecture02_script.md | Warningbox: "**100**: 21 per equation, from about 116 months in ten years"; narration reframed per equation (21 coefficients from 116 rows, one per five or six observations; 25 per equation with a sixth variable); "forecast like a coin flip" → "forecast far worse than it fits". |
| m13 | L03 .tex, .qmd | PDP caption: "Every day of the year appears four or five times" (narration already said so). |
| m14 | Narration/Lecture03_lab_script.md | "minimum around mid-April" → "minimum in late April". |
| m15 | L03 .tex, .qmd, Narration/Lecture03_script.md | pyGAM slide: gridsearch tunes λ "by GCV (generalized cross-validation), an in-sample shortcut to leave-one-out error"; narration: GCV described as an estimate with nothing held out (penalty slide), "by GCV" (pyGAM slide), "optimizes an estimate of held-out error", knots paragraph and takeaway "by GCV or cross-validation". |
| m16 | Labs/Lecture03_lab.qmd, Narration/Lecture03_lab_script.md | "Prophet will happily fit noise" → daily terms are constant on midnight timestamps, cannot be identified; Prophet would draw a daily curve the data cannot support. |
| C14 | Labs/Lecture03_lab.qmd, Narration/Lecture03_lab_script.md | "12% on both RMSE and MAE" → "about 12% on RMSE and 15% on MAE" (lab Discuss block; narration header note, Step 6, closing, appendix). |
| M11 | L03 .tex, .qmd, Narration/Lecture03_script.md | Table GAM/Interactions "No" → "Manual"; text: "you must add that interaction yourself (`te()`); a tree finds it unprompted"; narration: "a real limit — though not an absolute one… `te()` in pyGAM, or a smooth of price fitted separately by holiday… A tree finds it on its own"; "Interactions, only the ones you specify". |
| M12 | L03 .tex, .qmd, Narration/Lecture03_script.md | "reach for pyGAM when you have many correlated external predictors" → "reach for pyGAM to see the shape of a few non-calendar drivers"; narration adds that many correlated predictors are where GAM curves stop being trustworthy. |
| M17 | Narration/Lecture03_lab_script.md | "Monday highest" → "Tuesday highest"; "Mondays run about a thousand megawatts above Sundays" → "weekdays run about three thousand megawatts above Sundays"; appendix "max Monday" → "max Tuesday". |
| M18 | Narration/Lecture03_lab_script.md | "Every model beat the benchmark." → "…on RMSE — on MAE, as you just saw, the single harmonic did not." |
| PR-L01: takeaway "richer" | Quarto/Lecture01_ETS_Eval.qmd | "richer" → "rich" (the .tex wins). |
| PR-L01: package capitalization | L01 .tex, .qmd, Narration/Lecture01_script.md | Four-part slide and narration: statsmodels, scikit-learn, XGBoost, PyTorch, Prophet, pyGAM, PyMC. |
| PR-L01: "time-series" modifier | L01 .tex, .qmd, narration quote | "Never randomly shuffle time-series observations." |
| PR-L02: VAR(p) shows lag 1 only | L02 .tex, .qmd, Narration/Lecture02_script.md | "For two stationary series and $p = 1$, …"; narration "written out for one lag… More lags just add more of the same terms." |
| PR-L02: "time-series" modifier | L02 .tex, .qmd | "classical time-series toolkit". |
| PR-L02: outline "and Roadmap" | Quarto/Lecture02_ARIMA_VAR.qmd | "4. Key Takeaways and Roadmap". |
| PR-L02: Key Takeaways title | L02 .tex, .qmd, Narration/Lecture02_script.md | "Key Takeaways" → "Lecture 2: Key Takeaways" (frame, `## `, `# ▶ SLIDE 23` header). |
| PR-L02: British spelling | Narration/Lecture02_script.md | "centred" → "centered". |
| PR-L03: "modelled" | L03 .tex, .qmd, Narration/Lecture03_script.md | → "modeled". |
| PR-L03: "General Additive Modeling" | L03 .tex (\section + \sectionslide), .qmd (outline + divider), narration header | → "pyGAM for Generalized Additive Modeling". |
| PR-L03: Facebook vs Meta | L03 .tex, .qmd, Narration/Lecture03_script.md | "at Meta" → "at Facebook (now Meta)"; subtitle "Facebook's Prophet" kept. |
| PR-L03: GCV never spelled out; gridsearch two ways | L03 .tex, .qmd | Spelled out on the pyGAM slide; both slides now say GCV (see m15). The penalty frame keeps "GCV" — it is full (1.8 pt free, both caption lines at the margin), so the spell-out could not go at first use without overflow. |
| PR-L03: outline "and Roadmap" | Quarto/Lecture03_GAMs.qmd | "5. Key Takeaways and Roadmap". |
| PR-L03: "time-series" modifier | L03 .tex, .qmd, narration header | Section "Prophet for Time-Series GAMs" (matches "time-series clothing" on the next slide). |
| PR-L03: narration divider headers | Narration/Lecture03_script.md | "Section divider: Motivation" / "…: pyGAM" were abbreviated; now the full section titles. |
| PR-L01/02/03: citations in mirror | — | Checked: every \parencite/\textcite key appears in the .qmd at the same place; no uncited abbreviations (FPP/BMCP) in these decks. No change. |
| INDENT-L01/L02/L03 | — | No `\texttt{\ ` lines in these decks (0 matches before and after). No change. |
| PED-L01-6: MASE<1 caveat | L01 .tex, .qmd | Muted line: "MASE < 1 is no win over seasonal naïve on the test weeks: on CA_1 FOODS both score 0.59." |
| PED-L01: dataset tag | L01 .tex, .qmd | SES caption: "CA_1 FOODS. No trend, no season." |
| PED-L02-5: "Lecture 1's TX_1 HOUSEHOLD" | L02 .tex (3), .qmd (3), Narration/Lecture02_script.md (3) | → "Lab 1's" (Lecture 1 shows the series only in the weights figure; Lab 1 forecasts it, SES α 0.621). |
| PED-L03-1: GCV / edof | L03 .tex, .qmd, Narration/Lecture03_script.md | GCV defined in one line on the pyGAM slide; edof defined in narration only ("the 'edof' above each panel, the number of parameters the curve really uses") — skipped on the slide: the penalty frame has no vertical room. |
| PED-L03-3: dataset never introduced | L03 .tex, .qmd | Additivity frame: "Lab 3's daily electricity load (PJM East), by month" (source: scripts/prep_electricity.py). |
| PED-L03 (m1-adjacent) | Narration/Lecture03_lab_script.md | Appendix "Lecture 1's lesson arriving again" → "Lab 2's lesson" (walk-forward first appears in Lab 2). |
| PED skipped | — | L01 Holt damped-trend dataset tag, L01 ETS AIC gloss (frame full, 1.8 pt), L02 θ = α − 1 formula for ARIMA(0,1,1) = SES (frame 7.9 pt free), L03 "interaction" named on the additivity frame (3.2 pt free): no room without a layout change I could not compile-check. |

**FA — arithmetic and sources for changed numbers**

- **MASE 0.59 (L01):** denominator = mean |y_t − y_{t−52}| over CA_1 FOODS training weeks 53–225 = 2,332.9
  (from `Figures/Lecture01_ETS_Eval/measured.json` `ca1_all_units`); SES MAE 1,376.2 / 2,332.9 = 0.590;
  seasonal-naive MAE 1,365.4 / 2,332.9 = 0.585. Both 0.59 (matches the narration and G01).
- **VAR (L02, m10):** k²p = 5²·4 = 100; per equation kp + 1 = 5·4 + 1 = 21; ten years monthly T = 120, usable
  T − p = 116; 116 / 21 = 5.5 ("one coefficient for every five or six observations"); k = 6: 6·4 + 1 = 25.
- **C14 (Lab 3):** printed table pyGAM 3,444 / 2,550, Prophet 3,935 / 3,001 (`docs/labs/Lecture03_lab.html`):
  1 − 3,444/3,935 = 12.5%; 1 − 2,550/3,001 = 15.0%.
- **M17:** lab prints `day of week swing 11.6% min 6 max 1`, panel titled "0=Mon" → max is Tuesday.
  Prophet weekly component Mon +554.5, Sun −2,458.3 (measured.json) → 3,013; data Mon − Sun 2,954 → "about three thousand".
- **M3:** 885 / 814 is what `docs/labs/Lecture01_lab.html` prints; 875 / 804 is the finding's Linux re-run with
  numpy 2.4.6 / scipy 1.16.3 / statsmodels 0.14.6; boundary fit (trend and seasonal smoothing both 0) from the finding.
- **m13:** 212 days × 5 and 153 days × 4 in training (finding evidence; measured.json [5.0, 3.98]).
- **m14:** day 116 = 26 April.

## Group FB — Lectures 4–5

| ID | files changed | what changed (one line) — or "skipped: <reason>" |
|---|---|---|
| C6 | Labs/Lecture04_lab.qmd; Narration/Lecture04_lab_script.md | Step 4 reframed. The ceiling is still stated as an absolute but "never binds here". The extremes are now regression to the mean: days picked by actual sales, so any imperfect forecaster errs this way, and OLS pulls harder. Discuss prompt now asks why any forecaster does this and what selecting on the prediction would show. Narration: rewritten extremes paragraph (OLS 255/269; selected-on-prediction 3,820 vs 4,137), scatter-shape line, discussion prompt, appendix note, and a dated "corrections" note. **Code unchanged** (all 11 chunks still match docs/labs/Lecture04_lab.html). |
| C7 | Labs/Lecture05_lab.qmd; Narration/Lecture05_lab_script.md; Narration/Lecture05_script.md | Dropped the prediction that HW03 flips the ranking. All three now say HW03's 27 features are dominated by `units_lag1` (as `dow` dominates here) and `sqrt` loses there too. Covers lab callout, lab narration Step 3 + appendix, lecture narration slide 10. |
| M4 | Slides/Lecture05_TreeEnsembles.tex; Quarto/Lecture05_TreeEnsembles.qmd; Narration/Lecture05_script.md | "Random Forest in Python": dropped "the first in this course" and "information ETS and ARIMA never had". Slide now says both trees forecast only one week ahead while Lectures 1–2 forecast a whole year. Muted line: SES and ARIMA beat 1,660 one week ahead too; a forest on the series' lags alone scores 1,231; most of the gain is last week's actual sales. Narration slide 14 rewritten (≈430 of the 545-unit gain from lags; features ≈116) and the takeaway close fixed. |
| M20 | Narration/Lecture05_lab_script.md | Dropped the "smaller forest" reason (100 trees = 249 vs 248, so one unit cannot explain 49). Gap attributed to the period scored: per-year OOB 311/319/283/239; three Christmas zero days; 275 without them. "Wrong model on the wrong period" → "wrong period". |
| M21 | Narration/Lecture04_script.md | Slide 19: CA_1's 29% growth was mostly year 1→2 and the store then plateaued, so a four-year tree never meets a fifth-year week above its training max. The ceiling is shown on L05's RSXFS series instead ($450bn ceiling vs $665bn). |
| M26 | Narration/Lecture05_script.md | Slide 13: "three of the eight features" → "two of the eight". |
| m17 | Narration/Lecture04_lab_script.md | "the *worst* model in the table" → "worse on data it has not seen than every tree from depth 2 to depth 12". |
| m18 | Slides/Lecture05_TreeEnsembles.tex; Quarto/Lecture05_TreeEnsembles.qmd; Narration/Lecture05_script.md | Galton's ox: 1907 → 1906. |
| PR-L05: 63% vs 64% | Slides/Lecture05_TreeEnsembles.tex; Quarto/Lecture05_TreeEnsembles.qmd; Narration/Lecture05_script.md | Warningbox now reads "still trained on about 63% of the weeks after it", matching the first line. Narration says "sixty-three percent — the same share as any bootstrap sample". Arithmetic is below. |
| PR-L04: citations in .qmd | Quarto/Lecture04_DecisionTrees.qmd | [@James2021], [@Breiman1984], [@Molnar2022] moved into the `## ` headings, where the .tex has them, as L05's mirror already does. Removed the in-body [@Breiman1984]. |
| PR-L04: "Next:" line | Slides/Lecture04_DecisionTrees.tex | .tex now uses the .qmd wording, which is the factual one: "Next: Tree Ensembles — variance reduction through bootstrap aggregation, and sequential error correction with boosting." |
| PR-L04: TimeSeriesSplit import only in .qmd | Quarto/Lecture04_DecisionTrees.qmd | Removed the unused import line from the mirror's code template (the .tex wins). |
| PR-L04: Asymmetry bullet | Slides/…04….tex; Quarto/…04….qmd; Narration/Lecture04_script.md | "extreme values distort coefficients equally in both directions" → "a rise and a fall of the same size move the forecast equally, even when the real response is lopsided". Narration matched, with a price example. |
| PR-L04: "trade" → "tradeoff" | Slides/…04….tex; Quarto/…04….qmd; Narration/Lecture04_script.md | Takeaway 2 changed, plus the narration's "Two." |
| PR-L04: MDI bias described differently | Slides/…04….tex; Quarto/…04….qmd; Narration/Lecture04_script.md; Narration/Lecture05_lab_script.md | L04 now uses L05's wording, "biased toward features with many split points". L04 narration keeps "high-cardinality" as the jargon gloss. Lab 5 narration's "Lecture 4's footnote named it" now quotes the new wording. |
| PR-L04/L05: importance "sum" vs "average" | Slides/…05….tex; Quarto/…05….qmd; Narration/Lecture05_script.md | L05's MDI definition: sum the weighted impurity reduction over the splits on j, average over trees, normalize. L04's single-tree "sum" was already right. |
| PR-L04: SHAP promise | Slides/…04….tex; Quarto/…04….qmd | "Permutation importance and SHAP give more reliable attributions" → "Permutation importance gives a more reliable ranking; we return to it in Lecture 5". L05 covers permutation importance only. |
| PR-L05: outline "and Roadmap" | Quarto/Lecture05_TreeEnsembles.qmd | Outline item 7 → "Key Takeaways and Roadmap". |
| PR-L05: XGBoost table text dropped in 5 .qmd rows | Quarto/Lecture05_TreeEnsembles.qmd | Restored "(use early stopping)", "; needs more trees", "; depth >6 risks overfit", "(like RF)", "; stabilizes". |
| PR-L05: "naïve" vs "naive" | Slides/…05….tex; Quarto/…05….qmd | Table row "Seasonal naïve" → "Seasonal naive", matching the slide's text and later decks. |
| PR-L05: defaults column `\texttt` | Slides/…05….tex; Quarto/…05….qmd | `None` and `False` set as code, like `1.0` and `True`. |
| PR-L05: American spelling (narration) | Narration/Lecture05_lab_script.md | "favours" → "favors"; "apologise" → "apologize" (in the rewritten appendix line). |
| INDENT-L05 | Slides/Lecture05_TreeEnsembles.tex | 3 code lines `\texttt{\ ` → `\texttt{\mbox{}\ ` (RF template line 2, XGBoost template lines 2–3). LaTeX-only; the .qmd has no such bug. `grep -c '\\texttt{\\ '` → 0 in both decks. |
| PED-X4: dataset tags | Slides/…04….tex; Quarto/…04….qmd; Slides/…05….tex; Quarto/…05….qmd | Captions tagged: L04 bias–variance ("Weekly CA_1 FOODS, 20 bootstrap refits per model"), L04 depth ("Weekly CA_1 FOODS, one week ahead"), L05 OOB ("Weekly CA_1 FOODS forest"), L05 root splits ("Lab 5 (daily CA_1 FOODS)"). |
| PED-X4 (rest of L04) | — | skipped: the CART, rectangles and pruning frames follow the frame that already names "weekly CA_1 FOODS". The pruning frame is also full: in the last PDF its content starts right under the title (y=29 pt). |
| PED-X5-L04: "the point" | Slides/…04….tex; Quarto/…04….qmd; Narration/Lecture04_script.md | Takeaway 6 → "A tree is a step function that cannot extrapolate; its high variance needs ensembles (Lecture 5)". Narration "Six." matched ("ensembles do not solve the extrapolation"). |
| PED-X5-L05: "the point" | Slides/…05….tex; Quarto/…05….qmd; Narration/Lecture05_script.md | New takeaway 8: "The forest beat naive on its **inputs**, not its algorithm." To fit the 90-word cap, the muted "Next:" line was shortened to "Next: Regularization & Model Selection." (a roadmap, not a caveat). Narration "Seven things" → "Eight", and the old "one thing that isn't on the slide" becomes "Eight." with the M4 horizon point. |
| PED-L04-2: depth-3 result | Slides/…04….tex; Quarto/…04….qmd; Narration/Lecture04_script.md | The depth figure already prints "lowest: depth 3, 1,213". The caption adds the comparison "(seasonal naive: 1,660)" from measured.json and the horizon "one week ahead". Narration slide 11 says the same. Figure height 4.1→3.9 cm to absorb the extra caption line. |
| PED-L05-3: OOB figure encoding | Slides/…05….tex; Quarto/…05….qmd | New muted line under the figure: "Weekly CA_1 FOODS forest. Gray: in that tree's bootstrap sample; white: out of bag." The red/boxed encodings were already labeled in the figure. |
| m19 | — | skipped: CLAUDE.md belongs to another group (per task). |

**FB — arithmetic and sources for changed numbers**

- **63%.** 1 − (1 − 1/173)^173 = 1 − 0.3668 = 0.633 (`measured.json` weekly.oob_share_expected 0.3668; oob.oob_share 0.3667). The scorers' share of later weeks is measured at 0.6358 (`oob.scorers_share_of_later_weeks_seen`). Its expectation is 1 − (1 − 1/172)^173 = 1 − e^(173·ln(171/172)) = 1 − 0.3647 = 0.635. So both are about 63% (0.633 and 0.636). The point now stated is that the scorers see the same share as any bootstrap sample. No figure prints 64% (checked oob_grid.svg labels).
- **M4.** All from the review's evidence (`work/G04/horizon.py`): one-step naive 1,344.8, SES 1,303.7, ARIMA(1,1,1) 1,260.6, RF all features 1,115.0, RF own lags 1,230.7. Gain over seasonal naive: 1,660 − 1,115 = 545. Before any price/SNAP column: 1,660 − 1,231 = 429 ("about 430"). Features: 1,231 − 1,115 = 116 ("the last hundred or so"). Forest-own-lags vs one-step ARIMA: (1,261 − 1,231)/1,261 = 2.4% ("within a few percent").
- **C6** (review evidence `work/G04/lab4.py`). Train max 5,016; test max 4,717. OLS with dow dummies: 255 under / 269 over, against the tree's 186 / 207 (lab printout). Top decile selected on the prediction: actual 3,820 vs predicted 4,137.
- **M20.** Lab printout: n=100 → 249, n=300 → 248, OOB 297 vs test 248, so 297 − 248 = 49. Per-year OOB 311/319/283/239 and 275 without the three Christmas zero days come from the review's evidence (`work/G04/oob.py`).
- **M21** (review evidence `work/G04/l4narr.py`). Weekly means by year 15,631 / 20,101 / 20,852 / 21,129 / 20,205; 20,205 / 15,631 = 1.29. Zero of 52 year-five weeks exceed the 4-year training max. RSXFS: train_max 449,643 ($450bn) and last actual 665,054 ($665bn), from `measured.json` and as printed on L05's slide.
- **M26.** int(sqrt(8)) = 2.
- **PED-L04-2.** `measured.json`: depth_sweep depth 3 test_rmse 1,212.98 (→ 1,213, already on the figure) and seasonal_naive_test_rmse 1,659.53 (→ 1,660).
- **m17.** Lab 4 printout: full tree 406 vs depths 2–12 = 369, 337, 303, 313, 385 (all lower); the stump is 413.

## Group FC — Lectures 6–7

| ID | files changed | what changed (one line) — or "skipped: <reason>" |
|---|---|---|
| C8 | Labs/Lecture06_lab.qmd; Narration/Lecture06_lab_script.md | Lab: "OLS is fine" → "OLS forecasts well". The table reading is now about RMSE only, plus a new paragraph: the design is exactly collinear (roll7/14/28 means = lag averages, dow dummies sum to 1), rank 58 of 62, so the `OLS_max_coef` column is `LinearRegression`'s minimum-norm pick. RMSE is still well defined. Narration: "unique solution / not singular" replaced with the same explanation; the coefficient-reading paragraph now leads with RMSE; "where OLS was fine" → "forecast well"; appendix note now says the design is exactly singular. Features and code unchanged. |
| C9 | Slides/L06 .tex + .qmd (path frame); Labs/Lecture06_lab.qmd; Narration/Lecture06_script.md; Narration/Lecture06_lab_script.md | "nobody told the model about the week" → "the model also had day-of-week dummies, yet under heavy penalty it kept the weekly lags and dropped the dummies" (deck). Same finding in the lab callout and in both narrations; the lab-script contradiction ("There is no 'week' feature … dummies were there as an alternative route") is removed. |
| C15 | Labs/Lecture07_lab.qmd; Narration/Lecture07_lab_script.md | "window count is far larger than the row count … ~7,500" → "close to the row count, yet the windows overlap heavily … 8,310 rows become 7,530 windows, and most weeks sit inside 26 of them as input". I left out the reason for the lost windows on purpose, because HW05 P1 Q1 asks students to work it out. Narration: "twenty-six windows, once as a target, twenty-five as interior" → "as input in twenty-six windows, and once more as a target" (the arithmetic is below). |
| M7 | Slides/L06 .tex + .qmd (CV frame); Narration/Lecture06_script.md; Narration/Lecture06_lab_script.md | Warning box: "Never use KFold … CV looks excellent and the forecast fails [Bergmeir2018]" → "Default to TimeSeriesSplit, as in Lecture 1. Random KFold trains on the future to validate the past: valid for a pure autoregression with uncorrelated errors [Bergmeir2018], not under trend or serial correlation." To stay at 90 words I trimmed "and record" → "recording" and dropped "more stable" from the one-SE line (the pedagogy review called it untested). Narration states the paper's result and that shuffled K-fold does no harm on Lab 6's lag regression (evidence: kfold.py, below). Lab narration Step 5 softened the same way. Kept the takeaway "never random K-fold" as the course's practice rule. |
| M19 | Narration/Lecture07_lab_script.md | "the worse spread has the marginally better mean" → "the tighter spread also has the marginally better mean". |
| m1 | Narration/Lecture06_script.md | "Lecture 1's walk-forward evaluation" → "Lab 2's walk-forward evaluation". The CLAUDE.md half is not my file, and it is already done: by the end of my run the L01 row no longer contains "walk-forward CV, DM test" (grep). |
| m20 | Slides/L06 .tex + .qmd (Python frame); Narration/Lecture06_script.md | The elastic-net **formula is kept**, because the figure (`elastic_net.pdf`, titled "α = 0.5") is drawn with weights α : (1−α) (`lecture06_figures.py:431`). Under the halved form the same picture would be α = 1/3, and the "0.02" caption number would move. Instead, "`l1_ratio` is α" → "`l1_ratio` sets the Ridge/LASSO mix, like α, but matches α only at 0 and 1: sklearn halves the ℓ2 term." Narration: the same small print, plus "l1_ratio of one half is our alpha of two thirds"; "is the textbook's alpha" → "is, near enough, the textbook's alpha". HW04 not touched. |
| m21 | Labs/Lecture06_lab.qmd (code, Step 3); Narration/Lecture06_lab_script.md | `for gone in survivors - nz:` → `for gone in sorted(survivors - nz, key=FEATURES.index):   # ties leave together; fix their order`. I ran the lab's setup and Step 3 (scratch venv, pinned versions). The sorted code prints lag28, lag7, lag35, lag364, lag14, lag21, lag1, lag56, which is exactly the narration's existing block and appendix row, so "Deterministic" now holds. The narration adds one line: two pairs (lag28/lag7, lag21/lag1) leave at the same step and the lab sorts each tie. **The rendered docs/labs/Lecture06_lab.html still prints lag7 first until Lab 6 is re-rendered.** |
| m22 | Narration/Lecture06_lab_script.md | "Four years" → "About three years"; dropped "It is the best-fitting model on this slide". |
| m25 | Slides/L07 .tex + .qmd (universal approximation); Narration/Lecture07_script.md | Added "on a bounded input range". To stay ≤ 90: "With enough of them" → "With enough" and "that you have the data to identify it" → "that your data can identify it". Narration adds "over a bounded range of inputs" and one line on why it matters (a ReLU net is a straight line past its last kink). |
| m26 | Slides/L07 .tex + .qmd (comparison frame); Narration/Lecture07_script.md | RNN/LSTM "Extrapolates: yes" → "bounded*", with the footnote "*tanh-squashed states cap the forecast at a fixed ceiling." To stay ≤ 90: "the sequence column above starts saying yes, and it is the reason to learn this machinery now" → "the sequence column says yes, and why we learn this machinery now". The narration explains the bounded row. |
| m28 (L07 parts) | Narration/Lecture07_lab_script.md; Narration/Lecture07_script.md | "by about 45%" (after 1,109 vs 744) → "by about half"; "see what the loss curve looks like" → "see what it does to the forecast" (Lab 7 prints RMSE, not a loss curve). The L08 part is left to its group. |
| PR-L06: outline | Quarto/L06 .qmd | Outline item 5 "Key Takeaways" → "Key Takeaways and Roadmap", matching the .tex section. |
| PR-L06: Ridge | Slides/L06 .tex + .qmd | "so ridge keeps both" → "so Ridge keeps both". The `^{\text{ridge}}` math label and the spoken narration were left as they are. |
| PR-L06: X_tr | Quarto/L06 .qmd | Code template `.fit(X_train, y_train)` → `.fit(X_tr, y_tr)` (.tex wins). |
| PR-L06: FPP §7.5 | Slides/L06 .tex + .qmd (subset selection) | Uncited "FPP §7.5" removed from the muted line. "AIC, BIC or CV picks the winner \parencite[\S7.5]{Hyndman2021}" / `[@Hyndman2021, §7.5]` (the key already exists; FPP3 §7.5 is "Selecting predictors"). |
| PR-L07: James2021 | Quarto/L07 .qmd | `[@James2021]` moved from the "evidence of Homework 4" sentence to the `##` heading, where the .tex has it. This is the L05 mirror convention; narration headers omit citations, as in L05. |
| PR-L07: Homeworks | Slides/L07 .tex + .qmd | "Homeworks 3 and 4" → "Homework 3 and Homework 4". |
| PR-L07: Early Stopping fragment | Slides/L07 .tex + .qmd | "Lab 7's network for 200 epochs, with … held out …:" → "We train Lab 7's network for 200 epochs, holding out the last 52 training weeks as a validation slice:". |
| INDENT-L06 | Slides/Lecture06_Regularization.tex | 2 lines `\texttt{\ ` → `\texttt{\mbox{}\ ` (LaTeX-only; the .qmd has no such bug). grep afterwards: 0. |
| INDENT-L07 | Slides/Lecture07_NeuralNets.tex | 15 lines, same fix (LaTeX-only). grep afterwards: 0. |
| PED-L06-1: takeaway "the point" | Slides/L06 .tex + .qmd (takeaways) | The muted Next line now carries the cross-lecture skeleton: "Next: neural networks, where the fit-plus-penalty skeleton of Lectures 3–6 returns as weight decay." "and dropout" was dropped, because dropout is not a penalty term. Frame at 89 words. |
| PED-L07-2: activations | Slides/L07 .tex + .qmd (neuron frame) | One inline definition: "a nonlinear activation σ(·), such as ReLU's max(0, z), wrapped around it". The paragraph stays on two lines. Sigmoid and tanh ranges are already drawn and labeled in the figure. |
| PED-L06-7: path frame sklearn naming | — | skipped: the path figure's x-axis label already reads "penalty strength, log₁₀ λ (sklearn calls it alpha)". |
| PED-L07-5: λ_max definition | — | skipped: the gd_bowl figure's own caption line defines it ("λmax = 3.51, the bowl's steepest curvature"). Another caption line on a 4.6 cm figure frame risks vertical overflow, and I had no LaTeX here to check. |
| PR cross-deck: notation reuse (t, n, σ, α) | — | skipped: renaming, not mechanical. Pedagogy owner call. |

**FC — arithmetic and sources for changed numbers**

- **C15:** 8,310 rows / 30 series = 277 weeks per series. Windows = 30 × (277 − 26) = 30 × 251 = 7,530 (the lab
  prints `X shape (7530, 26, 4)`), so 780 = 30 × 26 fewer than the rows. Week j is an input in windows
  i ∈ [j−25, j], i.e. up to 26 windows, and it is the target of window j−26: "26 as input, once more as a target".
- **m22:** 1,210 days / 365 = 3.3 years. Training RMSE n=120 is 145 and n=80 is 84 (review G06-8, `work/G06/lab6e.py`).
- **m28:** 1,109 / 744 = 1.49, so "about half". The appendix's "45%" is 1,084 / 744 = 1.457 (five-seed mean), so it was left alone.
- **M19:** docs/labs/Lecture07_lab.html: wd=0 mean 1,084, sd 86; wd=1e-4 mean 1,088, sd 134. Tighter spread and better mean.
- **m20:** sklearn: ½n⁻¹‖y−Xw‖² + αρ‖w‖₁ + ½α(1−ρ)‖w‖², so ℓ1 : ℓ2 = ρ : (1−ρ)/2; the slide has α : (1−α). Equal ratios at
  ρ = ½ give (1−α)/α = ½, so α = 2/3 ("l1_ratio of one half is our alpha of two thirds"). The figure is drawn with a = 0.5,
  b = 1 − 0.5 (`lecture06_figures.py:431`; measured.json `elastic_net.level` 0.055 = 0.5·0.1 + 0.5·0.01). Under the halved
  form, that ball is (1−α)/(2α) = 1, i.e. α = 1/3, not the 0.5 in its title. That is why the formula was kept.
- **M7:** review evidence `work/G06/kfold.py`: TimeSeriesSplit alpha 19.64, CV 323.4, test 286.3; shuffled KFold alpha 3.71,
  CV 287.9, test 278.3. That is the basis for "shuffled K-fold does no harm on Lab 6's lag regression".
- **C8:** review evidence `work/G06/lab6b.py`: rank 58 of 62 at n = 1,210/400/200/120/80; roll7 identity max diff 0.0.
- **C9:** review evidence `work/G06/lab6c.py`: largest coefficients at the smallest alpha are dow_5 174, lag1 161, dow_6 148;
  every dow dummy exits by log10 α ≈ 1.71, before the last eight (2.25–2.73).
- **m21:** re-run of the Lab 6 setup and Step 3 in a scratch venv with the requirements.txt pins (sklearn 1.9.0, scipy 1.16.3,
  numpy 2.4.6, pandas 3.0.5). Script: `scratchpad/fc_env/scripts/lab6_order.py`. Exit groups: 179.57 [lag56] alone;
  209.91 [lag1, lag21]; 245.38 [lag14]; 391.94 [lag364]; 458.16 [lag35]; 535.57 [lag7, lag28], the same as measured.json.
  With `key=FEATURES.index` the cell prints lag28, lag7, lag35, lag364, lag14, lag21, lag1, lag56.
- **m26:** with a linear head on tanh-bounded states, |ŷ| ≤ ‖W_out‖₁ + |b| (review m26).

## Group FD — Lectures 8, 9 Part 1, 9 Part 2; HW05 Parts 2–3

| ID | files changed | what changed (one line) — or "skipped: <reason>" |
|---|---|---|
| C3 | Slides/L09P2 .tex, Quarto/L09P2 .qmd, Labs/Lecture09_Part2_lab.qmd, Narration/Lecture09_Part2_script.md, Narration/Lecture09_Part2_lab_script.md, Homework/HW05_Part2_Sequence.qmd | Without positional encoding the model still knows last week, because it reads out at the last position, and sees the other 51 as an unordered bag. "Last week vs a year ago" became "two weeks ago vs a year ago". The lab-script example now uses two earlier weeks. HW05 P2 Q3 now gives the correct premise. The ablation slide reads "what the order of those 51 weeks is worth" in place of "the price of permutation invariance". |
| KNOWN-L09P2 "shuffling leaves every output unchanged" | same deck, mirror, lab and both narrations | "Attention is permutation-invariant: every output is unchanged" became "Attention ignores order: each week's output moves with it". Narration and lab script name it permutation-*equivariant*. Takeaway 5 now reads "Attention ignores order", in the deck, the mirror and the narration. |
| C4 | HW05_Part2_Sequence.qmd, HW05_Part3_Comparison.qmd, Narration/Lecture09_Part2_script.md, Narration/Lecture09_Part1_lab_script.md | The HW05 P2/P3 "Do not shuffle … destroys the ordering" bullet is replaced by HW05 Part 1's wording, copied verbatim ("Split by time before anything else … `shuffle=False` … so your runs are reproducible"). In both narration failure lists, `shuffle=True` became "shuffling (rows) before the time split". |
| C5 | HW05_Part3_Comparison.qmd, Narration/Lecture09_Part2_lab_script.md | "roughly 5,900 windows" became "5,160 windows" (HW05 P3 Q3.2), and "5,900 overlapping windows" became "5,160 overlapping windows" (lab script). |
| M1 | HW05_Part2_Sequence.qmd, HW05_Part3_Comparison.qmd | The PROMPT_LIMIT_REACHED rule now reads "If this prompt takes me past my budget of N, say so, log it as over budget, and continue." (N = 10, 3). The checklist line now reads "`PROMPT_LOG.md` complete for Part N: every prompt logged; over-budget prompts noted", the same wording HW06 uses. |
| M27 | HW05_Part2_Sequence.qmd, HW05_Part3_Comparison.qmd | The initial prompt now reads "Homework 5 covers five … FFN, 1D CNN, vanilla RNN, LSTM, and Transformer encoder", plus "This part covers the last three" (P2) or "This part compares all five" (P3). **HW05 Part 1 skipped** (off-limits per brief); its prompt still says "four" at lines 45 and 157. |
| M9 | Slides/Lecture08_CNNs.tex, Quarto/Lecture08_CNNs.qmd, Narration/Lecture08_script.md, Narration/Lecture09_Part1_script.md | ResNet: "deeper plain networks *trained* worse — not overfitting, but an optimization difficulty", and "Each block learns only a change to x". The "What Carries Over" skip row now reads "a deep stack that is hard to train". Narration: "the authors checked, and the gradients were healthy". The L08 closing callback says that for the LSTM, unlike ResNet, fading gradients really are the problem. In the L09P1 narration, "Same problem: gradients dying …" became "A related problem: a signal that has to pass through many layers". |
| KNOWN-L08 "two stacked width-3 filters see 5 weeks, not 6" (+ "far fewer weights") | Slides/L08 .tex, Quarto/L08 .qmd, Narration/Lecture08_script.md | Not fixed by 9e7e257. The row now reads "Two stacked width-3 filters see five weeks, with a nonlinearity between them. VGG's weight saving was a 2D effect: in 1D it is 6 weights against 5." The narration says the same. |
| m23 | Slides/L08 .tex, Quarto/L08 .qmd, Narration/Lecture08_script.md | "Thirty years" (.tex) and "Twenty years" (.qmd and narration :43, :301) both became "Two decades". |
| m24 | Narration/Lecture08_script.md | "strictly more expressive" became "not strictly more expressive: the VGG authors describe the stack as a regularization, because it constrains what the filter can be while adding that nonlinearity" (with 18 vs 25 weights). "Cheaper and more expressive" became "Fewer weights, more nonlinearity". |
| m27 | Narration/Lecture08_lab_script.md | "the same quantity four weeks apart" became "one week apart". |
| m28 (L08 lab narration part) | Narration/Lecture08_lab_script.md | "longer window, no better or worse forecast" became "longer window, worse forecast". The L07 parts of m28 were skipped: not my files. |
| m29 | Slides/L09P1 .tex, Quarto/L09P1 .qmd, Narration/Lecture09_Part1_script.md | Not covered by ✔2, which fixed only the 52-week label. The slide now reads "Past the first few steps, ours keeps about 0.79 a step; measured, 0.4% … 0.001% …". The narration says the percentages are measured, not multiplied out from the slope. |
| m30 | Narration/Lecture09_Part1_script.md, Narration/Lecture09_Part1_lab_script.md, Labs/Lecture09_Part1_lab.qmd, HW05_Part3_Comparison.qmd | "tied once both reached convergence" became "tied by a hundred epochs, with the LSTM still improving". "**Exactly** four" became "almost exactly … 3.96, which the printout rounds to 4.0". In the appendix, "ratio exactly 4.0×" became "4.0× as printed (3.96; the recurrent layer alone is exactly 4×)". The same convergence claim was also fixed in Lab 9P1 prose ("at convergence they return a tie" became "by 100 epochs") and in the HW05 P3 callout ("trained to convergence" became "by 100 epochs, with the LSTM still improving"). |
| m31 | Slides/L09P2 .tex, Quarto/L09P2 .qmd, Narration/Lecture09_Part2_script.md | "The original paper adds thirty-two pairs" became "Our model adds thirty-two pairs". "M5 … won by ensembles of exactly the methods in Lectures 5 and 6" became "won by gradient-boosted trees, Lecture 5's method", in the deck, the mirror and the narration. |
| M2 (L08 instance; not in the review's Where list) | Slides/L08 .tex, Quarto/L08 .qmd, Narration/Lecture08_script.md | The March-2020 aisle note "more in Week 16" became "Week 15". Lecture 13 (Synthesis) is Week 15 in scripts/site.yml, and the aisle photo is shown at Slides/Lecture13_Synthesis.tex:400. |
| C2-residual (C2 itself already applied in 9e7e257) | Narration/Lecture08_script.md, Narration/Lecture08_lab_script.md, Narration/Lecture09_Part1_script.md, Labs/Lecture08_lab.qmd | These are spots 9e7e257 missed that contradict the corrected slides or callout. L08 narration takeaway four now voices the corrected slide. "Remember that number — seven weeks. It explains the results" became "seven weeks per filter, twenty-six in the window; the window is the one that explains the results". The Lab 8 narration's bold "More data … unless the architecture can reach it" now matches the lab callout. L09P1 narration :25 "it only sees about seven weeks back" became "its filters only see". Lab 8 Step 1 "each output neuron still sees only seven" became "each convolution unit". The coordinator can revert these if C2 is meant to stay frozen. |
| PR-L08: section subtitles ×3 | Quarto/L08 .qmd, Narration/Lecture08_script.md | The .qmd now matches the .tex: "Two decades of architectures, and three ideas that survived." and "The same machinery, one dimension, and an honest measurement." The missing Key Takeaways subtitle was added ("What convolution is for, and what it is not for."). Narration divider text for slides 11 and 16 was set to the same words, so build_book drops the duplicate. |
| PR-L08: "don't" on takeaways | .tex, .qmd | "usually don't" became "usually do not". |
| PR-L08: "Pooling and Dropout" vs content | .tex (\section + \sectionslide), .qmd (outline + `#`), narration SLIDE 9 header | The section is renamed "Pooling". The section has no dropout content. Parity script and header check both pass. |
| PR-L08: filter "length" vs "width" | .tex, .qmd, narration | "a filter of length K" and "Filter length K" became "width", matching the deck's width-3 and width-5 usage. |
| PR-L08: "well engineered" | .tex, .qmd, narration | Now "well-engineered". |
| PR-L08: architecture years vs citation years | .tex, .qmd | A muted note was added under the summary table: "VGG, Inception and ResNet appeared a year before the papers cited for them." The bib entries are Simonyan2015, Szegedy2015 and He2016, against table years of 2014/2014/2015. LeNet (1998) matches, so it is excluded. |
| PR-L09P1: "Lecture 9, Part 1" comma | .tex, .qmd frame title, narration header + doc title | The takeaways frame is now "Lecture 9, Part 1: Key Takeaways", matching the subtitle. This was also applied to L09P2 for consistency. build_book's norm_title ignores punctuation, so book matching is unaffected. |
| PR-L09P1: Bengio1994 placement | Quarto/L09P1 .qmd | Moved from "our own RNN … measured [@Bengio1994]" to the heading `## Why a Vanilla RNN Forgets [@Bengio1994]`, where the .tex has it (the L05 mirror uses the same convention). |
| PR-L09P2: deck title | Quarto/L09P2 .qmd, Narration/Lecture09_Part2_script.md | The .qmd title "Transformers" became "Attention and Transformers", as in the .tex. The narration subtitle line and opening line were updated to match. |
| PR-L09P2: Vaswani2017 placement | Quarto/L09P2 .qmd | Moved from the body to `## Scaled Dot-Product Attention [@Vaswani2017]`, as in the .tex. |
| PR-L09P2: "training examples" vs "training windows" | .tex, .qmd, narration (2 places) | The 5,160 is now "training windows" everywhere. |
| PR-L09P2: narration setext bug (found while editing) | Narration/Lecture09_Part2_script.md | Three paragraphs ran straight into `---` with no blank line, which Markdown renders as an H2 heading. A blank line was added before each (after the slides 6, 10 and 11 text). |
| INDENT-L08 | Slides/Lecture08_CNNs.tex | 5 lines: `\texttt{\ ` became `\texttt{\mbox{}\ `. LaTeX-only (the .qmd code block is unaffected). grep now finds 0. |
| INDENT-L09P2 | Slides/Lecture09_Part2_Transformers.tex | Same fix on 3 lines; LaTeX-only; grep now finds 0. |
| PED-L08: global pooling / flatten head | .tex, .qmd, narration | The muted note on the PyTorch pipeline slide now reads "`AdaptiveAvgPool1d(1)` is **global average pooling**: one mean per filter over the whole time axis. A **flatten head** skips it and gives every position its own weight." The narration says the same. |
| PED-L09P2-3: softmax, Q/K/V, d_k | .tex, .qmd, narration | Added "Q, K and V are three learned projections of the same weeks; d_k is the length of each key." and "Softmax exponentiates the scores and divides by their total". The narration adds the projections line and "sixteen, per head". |
| PED-L09P2-5: cost-square caption | .tex, .qmd | The caption now reads "Each cell is one query–key score: O(T²) time and memory, fine at T = 52, painful at T = 10,000." It was trimmed to stay at or under 90 words. |
| PED-L09P2-6: RNN/LSTM 26→52-week numbers | .tex, .qmd, narration | The intro line on "Does It Work?" adds "(Part 1's headline 842 and 987 were at 26 weeks)". The narration says the same. |
| PED-L09P1: LSTM-era notation line | .tex, .qmd, narration | Added "⊙ multiplies element by element" to the gates frame. The narration adds that the circled dot is element-wise and that "f-t here is the forget gate — not last week's filter". |
| HW05 P1 part of M27 | — | skipped: HW05 Part 1 is off-limits (brief). |
| m28 L07 parts | — | skipped: L07 files are not in group FD. |

**FD — arithmetic and sources for changed numbers**

- **m29:** 0.79^26 = e^(26·ln 0.79) = e^(−6.13) = 0.0022 (0.22%), and 0.79^51 = e^(−12.02) = 6.0e-6 (0.0006%). The deck's 0.4% and 0.001% are the measured `gradient.trained[26]` = 0.00364 and `trained[51]` = 1.06e-5 in `Figures/Lecture09_Part1/measured.json`. `per_step_trained` = 0.7919 is fitted for k ≥ 5 (`scripts/figures/lecture09_part1_figures.py:217`, "past the first few steps").
- **m30:** RNN recurrent layer = 4·64 + 64·64 + 64 + 64 = 4,480; plus the head (65) gives 4,545. LSTM recurrent = 4 × 4,480 = 17,920; plus 65 gives 17,985. 17,985 / 4,545 = 3.957. The lab prints `parameter ratio: 4.0x the RNN` (docs/labs/Lecture09_Part1_lab.html, checked).
- **Stacked filters:** receptive field = 3 + (3 − 1) = 5 weeks. Weights per in/out channel pair: 3 + 3 = 6 against 5 for one width-5 filter in 1D. VGG's 2D case: 9 + 9 = 18 against 25.
- **C5:** `Figures/Lecture09_Part2/measured.json` gives `train_windows` 5160. The lab prints "6,750 windows of 52 weeks — train 5,160, test 1,590" (docs/labs/Lecture09_Part2_lab.html, checked).
- **M2 (L08):** scripts/site.yml has Week 15 = Lecture 13 Synthesis and Week 16 = presentations.
- **Year note:** the bib years are LeCun1998 1998, Simonyan2015 2015, Szegedy2015 2015 and He2016 2016. The table years are 1998/2014/2014/2015.
- No other number was added or changed.

## Group FE — Lectures 10, 11 Part 1; HW06 Part 1

| ID | files changed | what changed (one line) — or "skipped: <reason>" |
|---|---|---|
| M5 | `Labs/Lecture10_lab.qmd`, `Narration/Lecture10_lab_script.md` | Callout retitled "With enough data the prior is outweighed, not erased". The text now quotes the printed means 0.265 / 0.259 / 0.309 and "a prior worth 22 observations … dragged most of the way", and adds the residual effect: P(θ > 0.30) goes from 0.080 (Step 4) to about 0.61. Discuss premise rewritten: the mean moved about 0.04, yet P(θ > 0.30) went from 0.08 to about 0.61. The narration now says this aloud after "edge of the right answer". The narration's Discuss is rewritten the same way. The appendix gets a P(θ > 0.30) row (0.084 vs 0.608 exact), and its "Do not oversell" note now opens with a dated correction naming the old wording. |
| M6 | `Slides/Lecture11_Part1_BayesianTS.tex`, `Quarto/…qmd`, `Narration/Lecture11_Part1_script.md` | Slide 5 now reads "A prediction interval from ARIMA widens too, but it plugs in point estimates of the parameters. This one also carries their uncertainty, and each source of widening is a named part of the model." The narration drops "you did not check and probably could not check" and "only accept or reject". It now says the ARIMA interval also follows from a model, Lecture 2 taught the residual check, and the real difference is plug-in point estimates vs. carried parameter uncertainty. |
| M24 | `Narration/Lecture11_Part1_lab_script.md` | Appendix: "127 divergences" → "43 divergences" (`docs/labs/Lecture11_Part2_lab.html` prints `divergences   43`; matches `:123` "forty-three"). |
| M25 | `Slides/Lecture11_Part1_BayesianTS.tex`, `Quarto/…qmd`, `Narration/Lecture11_Part1_script.md` | "Next: hierarchical models, for thirty series at once." → "…for ten series at once." (deck + mirror). Narration: "when you have thirty series" → "ten series". The finding's longer wording ("— ten series at once, some with barely any data") would have put the takeaways frame at 93 words, so I kept the 8-word form (89). The narration already says "some of them barely have any data". |
| m32 (a) | `Narration/Lecture11_Part1_lab_script.md` | "Whether that trade is worth 5%" → "worth 4%". |
| m32 (b) | `Narration/Lecture11_Part1_script.md` | Slide 17: "every single one, from ETS through the Transformer — that question had no answer at all…" → "For the models before this one, we never asked. ETS had intervals, and Prophet drew a band in Homework 2, but we scored every model on its point forecast and never checked a band against held-out weeks. Here we did." Verified: L01 `:231` (error type sets the intervals), HW02 `:192` (Prophet uncertainty bands), and no `coverage` / `conf_int` / `yhat_lower` anywhere in Labs 1–9 or HW01–HW05. The deck's own "No point-forecast model so far let us." was left alone: the finding names only the narration, and a model that returns a point forecast really does not let you ask. |
| M1 (HW06 P1) | `Homework/HW06_Part1_Foundations_TS.qmd` | Initial prompt: the PROMPT_LIMIT_REACHED bullet → "If this prompt takes me past my budget of 10, say so, log it as over budget, and continue." Checklist: "complete with **10 or fewer** entries" → "complete for Part 1: every prompt logged; over-budget prompts noted". |
| PR-L10: deck title | `Quarto/Lecture10_BayesianFoundations.qmd` | YAML title "Bayesian Statistics I --- Foundations" → "Bayesian Statistics I: Foundations" (the .tex wins). |
| PR-L10: citations in .qmd | `Quarto/Lecture10_BayesianFoundations.qmd` | `[@Gelman2013]`, `[@MartinKumarLao2021]` and `[@McElreath2020]` moved from body sentences into the `## ` headings, where the .tex has them in the frame titles (same convention as L05's mirror). |
| PR-L10: "Even a few means" | `Slides/…tex`, `Quarto/…qmd`, `Narration/Lecture10_script.md` | Deck + mirror: "Even a few divergences mean the posterior is explored poorly." Narration: "explored wrongly" → "explored poorly". |
| PR-L10: dash run-on | `Slides/…tex`, `Quarto/…qmd` | "assumptions too --- functional form, variable selection, sampling scheme --- they are just not…" → "assumptions too: functional form, variable selection, sampling scheme. They are just not…" |
| PR-L10: Beta math vs plain | `Slides/…tex`, `Quarto/…qmd` | Slide 9 `$\text{Beta}(5, 45)$` → plain "Beta(5, 45)", to match slide 19 and "Normal(0, 1)". |
| PR-L10: takeaway colon | `Slides/…tex`, `Quarto/…qmd` | `\textbf{Priors:}` / `**Priors:**` → `\textbf{Priors}:` / `**Priors**:`, matching items 1, 2 and 5. |
| PR-L11P1: deck title | `Quarto/Lecture11_Part1_BayesianTS.qmd`, `Narration/Lecture11_Part1_script.md` | YAML title "Bayesian Statistics II --- Time Series" → "Bayesian Structural Time Series" (the .tex wins). The narration's spoken title-page line changes to match. The document header line, which names the series, is unchanged. |
| PR-L11P1: "Don't" in title | `Slides/…tex`, `Quarto/…qmd`, `Narration/Lecture11_Part1_script.md` | "The Fix: Reparameterize, Don't Sample More" → "…, Do Not Sample More" in the frame title, the `## ` heading and the `# ▶ SLIDE 13` header. |
| PR-L11P1: "With … On" | `Slides/…tex`, `Quarto/…qmd`, `Narration/Lecture11_Part1_script.md` | "A Random Walk With a Prior On It" → "A Random Walk with a Prior on It". This matches the course's lowercase-preposition titles (L01 "The Problem with Equal Weights", L12 "Scenario Analysis with…"). All three places changed. |
| PR-L11P1: BMCP §4.6.1 | `Slides/…tex`, `Quarto/…qmd`, `Narration/Lecture11_Part1_script.md` | "BMCP §4.6.1 covers exactly this" → "It is the standard fix \parencite[§4.6.1]{MartinKumarLao2021}" / `[@MartinKumarLao2021, §4.6.1]`. The key exists (`Bibliography_base.bib:481`), and the deck's References frame, empty until now, gets an entry. Narration: "Martin, Kumar and Lao work it in section four-point-six-point-one of their book." |
| PR-L11P1: "Next … thirty series" | — | see M25. |
| INDENT-L11P1 | `Slides/Lecture11_Part1_BayesianTS.tex` | 9 lines `\texttt{\ ` → `\texttt{\mbox{}\ ` (model code box, 8 lines; centered-form table cell, 1 line). `grep -c '\\texttt{\\ '` → 0. LaTeX-only: the .qmd code blocks keep their indentation. L10 already had 0. |
| PED-L10-4: θ, 𝒟 undefined | `Slides/…tex`, `Quarto/…qmd`, `Narration/Lecture10_script.md` | Definition box title "The update rule" → "The update rule, for an unknown $\theta$ and data $\mathcal{D}$". It goes in the title bar because the body was full (last line 6 pt above the footer). "Unknown" rather than "parameter" also covers slide 8's yes-or-no θ. The narration adds one sentence. |
| PED-L10-3: Beta before defined / "worth 50" | `Slides/…tex`, `Quarto/…qmd`, `Narration/Lecture10_script.md` | Slide 9 opening → "a Beta(5, 45) prior, like 5 successes in 50 trials, then 30 in 200." That is still one line, and it explains "worth 50 observations" and the 0.10 mean through the slide's own lever arithmetic. The narration adds "Read a Beta prior as a count you already have…". |
| PED-L10-8: HDI undefined; 94% vs 95% | `Slides/…tex`, `Quarto/…qmd`, `Narration/Lecture10_script.md` | Slide 17 adds "HDI: the narrowest interval holding 94% of the draws. 94 or 95 is a choice; say which." To make room, the 2-line opener became 1 line: "MCMC can fail quietly: it always returns samples, from the posterior or not." The narration adds the definition and the slide-5 95% vs 94% note. |
| PED-L11P1-9: 1,660 vs 2,152 | `Slides/…tex`, `Quarto/…qmd`, `Narration/Lecture11_Part1_script.md` | Slide 15 opener → "CA_1 FOODS, 52 weeks held out (Lecture 9's 2,152 was on the 30-series panel)." The narration adds that 2,152 is seasonal naive over the 30-series panel and that this is one series, Homework 1's benchmark. Verified against `Figures/Lecture13_Synthesis/measured.json`: `board/naive 2152.12` with `test_rows 1590` (30 series × 53 weeks, cutoff 2015-05-17), and `skill/hw01/naive` = `skill/l11p1/naive` = 1659.53 (CA_1 FOODS, n_test 52, first test week 2015-05-25). The two windows differ by about one week, so the note says "different series", not "different window". |
| Lab 11P1 prose | — | skipped: no finding in scope touches `Labs/Lecture11_Part1_lab.qmd` (checked with grep for 127, thirty, 5%, BMCP and the old titles). |
| L10 "We'll meet this for real in Lecture 11, twice" (`Narration/Lecture10_script.md`, divergences paragraph) | — | not changed (no finding names it). Flag for the coordinator: L11P1's failed fit had **0** divergences, so divergences appear only in L11P2. "Twice" is true only if "this" means sampler failure in general. |

**FE — arithmetic and sources for changed numbers**

- Lab 10 printed (`docs/labs/Lecture10_lab.html`): Step 4 `P(theta > 0.30) = 0.080`; Step 5 posterior means
  0.265 / 0.259 / 0.309. Shift quoted as "about 0.04": 0.309 − 0.265 = 0.044.
- Closed form (pure-Python integration; the venv's scipy import is broken), posterior Beta(a+72, b+205):
  Beta(2,2) → Beta(74,207): mean 0.2633, P(θ>0.30) 0.0841. Beta(5,15) → mean 0.2593, P 0.0578.
  Beta(20,2) → Beta(92,207): mean 0.3077, **P(θ>0.30) 0.6077** → "about 0.61".
- "A prior worth 22 observations": Beta(20,2), a + b = 22. "Most of the way": data share 277/(277+22) = 0.926;
  from the printed values (0.909 − 0.309)/(0.909 − 0.260) = 0.92.
- m32: 1 − 1,589/1,660 = 0.0428 → 4% (`measured.json`: 1588.77 / 1659.53).
- Slide 9 "like 5 successes in 50 trials": Beta(5,45) → 5/50 = 0.10 = the prior mean the slide already
  states, and its lever (50×0.10 + 200×0.15)/250 = 0.14.

## Group FF — Lectures 11 Part 2, 12; HW06 Part 2, HW07

| ID | files changed | what changed (one line) — or "skipped: <reason>" |
|---|---|---|
| C10 | Slides/Lecture11_Part2_Hierarchical.tex, Quarto/Lecture11_Part2_Hierarchical.qmd | "WI_2 shows the cost, pulled past its truth" → "CA_3 lost narrowly. WI_2 was pulled past its truth, yet still ended closer than its own estimate." (narration already correct) |
| C11 | Labs/Lecture12_lab.qmd (Step 2 prose), Narration/Lecture12_lab_script.md (Step 2 intro, Step 3 "irony" paragraph, appendix note) | The series intercepts remove the between-series confounding; centering leaves the coefficient unchanged and helps the sampler. The Step 3 irony now says that centering, a sound choice, is what made the divisor small (0.0455) |
| M22 | Narration/Lecture12_script.md | b_snap 3.7%: dropped "consistent with what the hierarchy told us last week" → "one average across all thirty series. Last week's FOODS hierarchy put the FOODS lift near eight percent." |
| M23 | Narration/Lecture11_Part2_lab_script.md | Model answer "ridge, LASSO, the bagged ensemble, the random forest, boosting's learning rate" → the deck/HW06 list: GAM smoothing penalty, tree pruning, `reg_lambda`, Ridge and LASSO, weight decay |
| m33 | Labs/Lecture11_Part2_lab.qmd, Narration/Lecture11_Part2_lab_script.md (2 places), Labs/Lecture12_lab.qmd, Narration/Lecture12_lab_script.md (2 places) | "Six model fits / six fits" → five; "Four model fits / four fits" → three |
| m34 | Homework/HW07_Bayesian_Regression.qmd | Q3 intercept prior `Normal(0, 1)` → `Normal(df.ly.mean(), 1)` with a one-line reason; Q6(b) break-even → "1.05 × (1 + volume change) > 1, that is, b > −1 (a volume loss under 1 − 1/1.05 ≈ 4.76%, not 5%)" |
| m35 (L12 part) | Slides/Lecture12_BayesianRegression.tex, Quarto/Lecture12_BayesianRegression.qmd, Labs/Lecture12_lab.qmd, Narration/Lecture12_script.md, Narration/Lecture12_lab_script.md | Deck/mirror: "A p-value gives only the direction." → "The usual p-value, against zero, gives only the direction." Lab and both narrations: "a significance test … said nothing about either size question" / "No significance test would ever have given you that map" → the same narrowed wording. L13 part not in scope (other group) |
| M14 | Homework/HW07_Bayesian_Regression.qmd | Overview: "store sales as a function of promotions, competition, and store characteristics" → "weekly log units on log price, SNAP and event weeks, a trend and series effects" |
| M2 | Narration/Lecture12_lab_script.md | "Week 16 is synthesis" and "See you in Week 16 for the synthesis" → Week 15 (site.yml: Week 15 = synthesis, Week 16 = presentations) |
| M1 | Homework/HW06_Part2_Hierarchical.qmd, Homework/HW07_Bayesian_Regression.qmd | PROMPT_LIMIT_REACHED rule → "If this prompt takes me past my budget of N, say so, log it as over budget, and continue." (N = 9 / 16); checklist "N or fewer entries" → "every prompt logged; over-budget prompts noted" |
| m36 (HW07 prompts) | Homework/HW07_Bayesian_Regression.qmd | Added under "## Questions": "The suggested prompts below add up to 13 of your budget of 16; the remaining 3 are spare." HW06 P2 checked: sums to 9 = budget 9, so no line was added |
| m36 (XGBoost 781 in L12 Method Map) | — | skipped: my brief limits m36 to HW07's suggested prompts; the 781 row is left for the coordinator/L13 owner (needs a decision across L08/L09/L12/L13) |
| PR-L11P2: "even the machine" | L11P2 .tex + .qmd | "Counts move with the seed, even the machine." → "…, and even with the machine." |
| PR-L11P2: stray newline in \section | L11P2 .tex | `\section{Fitting It, and the Trap⏎}` → one line (LaTeX-only) |
| PR-L11P2: `shape=10` vs `shape=J` | L11P2 .tex + .qmd; Narration/Lecture11_Part2_script.md | Defined J on "The Model in PyMC": "`J` is the number of series (10)." `shape=10` in the earlier table stays and matches J = 10. The narration says "J is the number of series, ten here" |
| PR-L11P2: "BMCP §4.6.1" (+ empty References) | L11P2 .tex + .qmd; Narration/Lecture11_Part2_script.md | Funnel footnote → "Posterior geometry matters \parencite[§4.6.1]{MartinKumarLao2021}." / `[@MartinKumarLao2021, §4.6.1]` (key exists). This fills the References frame. The narration's spoken "BMCP section…" → "section four-point-six-point-one of Martin, Kumar and Lao, the book cited on the slide" |
| PR-L12: deck title .tex vs .qmd | Quarto/Lecture12_BayesianRegression.qmd | qmd title "Bayesian Statistics III --- Bayesian Linear Regression" → "Bayesian Statistics III: Bayesian Linear Regression" (.tex wins) |
| PR-L12: warningbox lead bold only in .qmd | L12 .tex | `\textbf{Confidently, precisely wrong}` added in the .tex. The .qmd already bolds it; its `.key-result` accent left as an HTML-only decoration |
| PR-L12: empty References frame | L12 .tex + .qmd | Cited the collider claim the deck already makes ("Controlling for it creates a spurious association") with \parencite{McElreath2020} / [@McElreath2020] (Statistical Rethinking 2e, ch. 6, collider bias). No frame added |
| INDENT-L11P2 | L11P2 .tex | 7 code lines `\texttt{\ ` → `\texttt{\mbox{}\ ` (LaTeX-only; .qmd has no such bug). `grep -c '\\texttt{\\ '` → 0 in both L11P2 and L12 |
| PED-L11P2-6: name the five thinned series | L11P2 .tex + .qmd | Held-out frame: "The weeks we thinned away still exist" → "The weeks we thinned from CA_1, CA_3, TX_1, TX_3 and WI_2 still exist" (from measured.json `thin`), before "four of five" |
| PED-L11P2-4: J undefined | L11P2 .tex + .qmd | see PR-L11P2 `shape=J` row |
| PED-L12-4: `idx`, `J`, `lpc`, `yr` undefined | L12 .tex + .qmd | PyMC frame: added "`J` is the number of series (30), `idx` maps each row to its series, and `yr` counts years since the first week." (`lpc` was already defined on the frame) |
| PED-HDI | — | skipped: L12 already defines HDI (frame "Reading a Posterior Coefficient"). In L11P2 the HDI appears only on "What the Model Actually Concluded", which is at 90 words and was not flagged for this deck. The missing definition was flagged in L10 (another group) |
| EXTRA-L12lab: interval widths | Narration/Lecture12_lab_script.md | Appendix "0.100 against 0.207" → "0.099 against 0.208", matching the lab's printed output and the script's own Step 2 section |
| EXTRA: American spelling | Narration/Lecture11_Part2_lab_script.md, Narration/Lecture12_script.md | "modeller" → "modeler"; "labelled" → "labeled" |

**FF — arithmetic and sources for changed numbers**

- **C10** (`Figures/Lecture11_Part2/measured.json`). CA_3: unpooled 0.1045, pooled 0.0819, truth 0.1018, so the errors are 0.0027 unpooled vs 0.0199 pooled and pooling lost. WI_2: unpooled 0.3385, pooled 0.1029, truth 0.2067, so the errors are 0.132 unpooled vs 0.104 pooled and pooling won. The pooled value sits below the truth while the unpooled one sits above it, hence "pulled past". `thin` = CA_1, CA_3, TX_1, TX_3, WI_2.
- **m33**: Lab 11P2 calls `fit()` 5 times (u30, h30, uf, hf, cent). Lab 12 calls it 3 times (naive, ctrl, std). `docs/labs/Lecture12_lab.html` prints three fit outputs (Naive, Controlled, standardized).
- **m34**: on `data/processed/m5_weekly.csv` (HW07 filtering), `ly.mean()` = 8.602 and the series means run 7.17–10.21, so N(0, 1) sits 7.2–10.2 SDs away. Break-even: 1.05^(1+b) > 1 ⇔ b > −1, and the volume loss must be under 1 − 1/1.05 = 0.0476.
- **m36 (HW07)**: suggested prompts 1+2+3+2+2+3+0 = 13 against a budget of 16, so 3 are spare. HW06 P2: 1+3+2+2+1+0 = 9 against a budget of 9.
- **L12 lab widths**: the lab prints "width of 94% HDI 0.099" (naive) and "0.208" (controlled).
- **J**: L11P2's hierarchy is over j = 1…10 FOODS series (deck equation). L12's lab prints "30 series".
- **M22**: μ_b = 0.078 (L11P2 measured.json `mu_b` 0.0778), spoken as "near eight percent".

## Group FG — Lecture 13; course documents; CLAUDE.md

| ID | files changed | what changed (one line) — or "skipped: <reason>" |
|---|---|---|
| C12 | .tex, .qmd, narration | Silent Failures II row 4: "Our first two Bayesian fits … failing convergence outright" → "Our centered fits in Lecture 11 Parts 1 and 2 looked fine while failing R̂ or divergences." Narration: P1 centered fit R-hat 1.022; P2 passed R-hat/ESS, failed only on 43 divergences. |
| C13 | ECON8310_Datasets.md | Table row "Half-hourly" → "Hourly". §3 now says PJM East hourly load, daily 2002–2018 and hourly 2016–2018, demand only with no temperature or other covariate, from a public mirror of PJM data. The Monash link and the temperature U-shape paragraph are gone. In their place: the double-peaked yearly cycle (July max, smaller January peak, spring/autumn lows) that a GAM smooth captures. |
| M15 | ECON8310_Datasets.md | "price, promo, SNAP, calendar effects, store type, category, and lags" → "price, SNAP, calendar events, store/category/state identifiers, and lags". |
| M16 | ECON8310_Datasets.md | "Thousands of parallel series … this has it" → a weekly aggregate of 30 series × 277 weeks. The synthesis lecture shows this is not enough for the largest models. |
| M2 | narration, Datasets ×2, CLAUDE.md ×3 | Week 16 → Week 15 at Lecture13_script.md:13, Datasets :59 and :177, CLAUDE.md L13 row and the "shock cameo" line. CLAUDE.md "all 16 weeks plus finals week" → "all 16 weeks (Week 15 is the synthesis, Week 16 the project presentations)". site.yml has 16 entries and no finals week, and the syllabus calls Week 16 itself final-exam week. |
| m2 | .tex, .qmd, narration | "Twelve weeks" → "Fourteen weeks": .tex section slide :33 and the .qmd divider. Header comment now matches the deck title. Narration "twelve weeks" → "fourteen weeks" ×3 (slide 1, slide 7, slide 15). "Twelve lectures" kept where it means Lectures 1–12 (narration slides 1 and 3). |
| m35 | .tex, .qmd, narration | Closing item 5: "Held-out data caught our worst two" → "Held-out data and a DAG caught our worst two." Narration: "Held-out data caught one of them; a DAG drawn before the model caught the other." |
| m36 Lab 5 | Labs/README.md | "closes on XGBoost (230)" → "(244)", as `docs/labs/Lecture05_lab.html` prints. |
| m36 Christmas % | ECON8310_Datasets.md | "100% of the true value" → "100% of its own forecast". |
| m36 CPI | ECON8310_Datasets.md | Dropped "the consumer price index"; "a handful of monthly US series" → "three monthly US series" (prep_fred.py: UNRATE, UMCSENT, RSXFS). |
| m36 labs | ECON8310_About.md | "One executable lab per lecture" → "One executable lab per lecture except the synthesis". |
| m36 csv count | ECON8310_Troubleshooting.md | "you should see five .csv files" → "you should see m5_weekly.csv and the other course .csv files". The datasets guide, rubric and CLAUDE.md say the clone also has onlinestore_weekly.csv (six files). I could not open the student repo (it is not attached to this session), so I worded it to be true either way. |
| m36 prep_favorita | CLAUDE.md | Dropped "there is no `prep_favorita.py`" (scripts/prep_favorita.py exists). The line now reads "Favorita as of 2026-08-31 — Week 15 does not use it." |
| m36 XGBoost 781 | .tex, .qmd, narration | 781 is NOT changed. Added a scriptsize gray note under the scoreboard table (right column): "XGBoost's 781 is from an earlier xgboost; the pinned 3.4.1 gives 792." The .qmd has the same note in `.neutral` style. Narration adds one sentence: still second place (792 < 805). Frame now 83 words. |
| m1 | CLAUDE.md | L01 row: dropped "walk-forward CV, DM test". |
| m12 | CLAUDE.md | HW 1b row "Weekly, Store 1 + Store 2" → "Weekly, CA_1 + CA_3 FOODS" (HW01_Part2:110). |
| m19 | CLAUDE.md | L04 row "CART, entropy/information gain" → "regression CART, cost-complexity pruning". |
| C7 (echo) | Labs/README.md | Outside the named scope but the same fact as C7. The README said HW3's 27-feature panel "gives the textbook answer"; it now says HW3 is also dominated by `units_lag1`, so `sqrt` loses there as well (C7 evidence: MDI 0.788; None 828.4 vs sqrt 898.5). Without this, the README contradicts the fixed Lab 5. The coordinator may revert. |
| PR-L13: practised | .tex, .qmd, narration | "practised" → "practiced". |
| PR-L13: does not → do not | .tex, .qmd, narration | "Three variations on one family does not count/satisfy" → "do not". |
| PR-L13: fragment | .tex, .qmd | "Which does not count toward the three, and is not optional." → "It does not count toward the three, and it is not optional." |
| PR-L13: RMSE vs Test RMSE | — | skipped: `scripts/check-book-claims.py:82` parses the .qmd header literally as `\| Model \| Test RMSE \| Where \|`. Changing it broke the gate (3 MISMATCH), so I reverted it. Widening the .tex header to "Test RMSE" does not fit: the table already spans 156.5 pt of a 161.5 pt column in the current PDF. |
| PR-L13: Classical vs Time series | .tex, .qmd, narration | Project frame "Classical, trees, deep learning, Bayesian" → "Time series, trees, …", to match the deck's own Arc table. The Arc table cell is 2.3 cm, so "Classical time series" would wrap there. |
| PR-L13: start here | .tex, .qmd | Table header "start here" → "Start here". |
| PR-L13: section subtitles ×5 | .qmd | All five `#` dividers now carry the .tex subtitles. Two were reworded and one was changed by m2; "Choosing a Method" and "Where to Go Next" had no subtitle and now have one. |
| PR-L13: deck title | .qmd, narration, .tex comment | .qmd title "Synthesis --- What the Semester Measured" → "Synthesis: What the Results Taught Us" (the .tex wins). Narration header and the spoken slide-1 title match. |
| PR-L13: FPP citations | .tex, .qmd | "FPP Ch.~11" → `\parencite[ch.~11]{Hyndman2021}` / `[@Hyndman2021, ch. 11]` (FPP3 ch. 11 is hierarchical/grouped series). The References frame now prints an entry. "FPP-Py Ch. 15" is left as text: no bib key exists for the Python edition, and Hyndman2021 is the R edition (13 chapters), so a ch. 15 cite would be wrong. |
| PED-L13-3 (naive's numbers) | .tex, .qmd | "carry 4.6% of its squared error" → "of the seasonal naive's squared error". |
| PED-L13-3 (squares) | — | skipped: the figure already decodes the squares ("area = squared error", "error on 25 Dec: the side of a square", "average day / RMSE 537"). The PDF page is vertically full: content y = 39–218 plus the keybox, and other full pages end at 229–238. One more line risks the footline. |
| PED-L13-4 | .tex, .qmd | Lesson 1: added "Skill $= 1 - \text{RMSE} \div \text{benchmark RMSE}$." It fits the free space on the paragraph's third line, which ends at x = 205 of 439 pt. Frame 60 words. |
| PED-L13-6 | .tex, .qmd, narration | Row 4 is now consistent with C12 and with the L11P2 lab (R-hat and ESS pass; the divergences fail). |
| INDENT-L13 | — | not affected: `grep -c '\\texttt{\\ '` = 0 (L13 is not in the list). |

**FG — arithmetic and sources for changed numbers**

- Weeks (scripts/site.yml): Lecture 12 = Week 14, Lecture 13 = Week 15, Week 16 = "Final project presentations". There are 16 entries and no finals week; the syllabus (:398) calls Week 16 final-exam week. So Lectures 1–12 span Weeks 1–14, which gives "fourteen weeks".
- XGBoost: `Figures/Lecture13_Synthesis/measured.json` board.xgboost = 792.43 (xgboost_version 3.4.1). round → 792. 792 < 805 (RNN), so the rank is unchanged (second).
- Christmas: actual 0 on each 25 Dec; forecasts 2,230/2,045/2,485/2,365. The error equals the forecast, so it is 100% of the forecast and undefined as a % of the true value.
- Electricity (computed from `data/processed/electricity_*.csv`): daily 2002-01-01..2018-08-03 (6,059 rows); hourly 2016-01-01..2018-08-03 (22,681 rows); no temperature column. Monthly mean MW: Jul 37,882 (max), Jan 34,343 (second peak, smaller), Apr 27,861 and Oct 28,117 (lows). The prep_electricity.py docstring agrees: "July maximum … secondary January maximum".
- M5 weekly: 30 series × 277 weeks = 8,310 rows (G13 re-count).
- Lab 5: the scoreboard prints `XGBoost (lr=0.1, 100 trees) 244`.
- L11 fits (lab HTML): L11P1 centered R-hat 1.022, ESS 274, 0 divergences. L11P2 centered r_hat 1.003, ess 966, divergences 43. L12 naive fit R-hat 1.0018, ESS 2,321 (passes, so "two worst answers passed diagnostics" holds).
- Layout (from the current PDF, page height 255 pt, footline at y ≈ 248, text width 425 pt): Closing page content ends at y = 201, so the item 5 wrap fits. Silent Failures II row 4 is about 92 characters against a line 1 capacity of about 61, so it stays 2 lines (the page is full at y = 231). Scoreboard: the note goes in the right column, which is shorter than the figure. References page ends at y = 216, so one extra table line fits.

## Final checks (all 15 decks, at the branch head)

### Frame parity (`python3 scripts/check_mirror_parity.py`)

```
deck                               frames  slides  status
Lecture01_ETS_Eval                     17      17  ok
Lecture02_ARIMA_VAR                    19      19  ok
Lecture03_GAMs                         14      14  ok
Lecture04_DecisionTrees                16      16  ok
Lecture05_TreeEnsembles                19      19  ok
Lecture06_Regularization               15      15  ok
Lecture07_NeuralNets                   17      17  ok
Lecture08_CNNs                         17      17  ok
Lecture09_Part1_RNN_LSTM               18      18  ok
Lecture09_Part2_Transformers           14      14  ok
Lecture10_BayesianFoundations          16      16  ok
Lecture11_Part1_BayesianTS             14      14  ok
Lecture11_Part2_Hierarchical           14      14  ok
Lecture12_BayesianRegression           12      12  ok
Lecture13_Synthesis                    16      16  ok
```

### Words of running text per frame (`python3 scripts/slide_word_counts.py <Deck>`)

No frame is above 90 (baseline maximum 90; total running text 14,520 → 14,882 words across all frames).

```
== Lecture01_ETS_Eval
  25  Lecture01_ETS_Eval              1. Forecasting Is Everywhere in Business
  88  Lecture01_ETS_Eval              2. The Basic Setup
  84  Lecture01_ETS_Eval              3. Components of a Time Series
  56  Lecture01_ETS_Eval              4. Always Beat a Benchmark First
  63  Lecture01_ETS_Eval              5. The Problem with Equal Weights
  62  Lecture01_ETS_Eval              6. Simple Exponential Smoothing (SES)
  80  Lecture01_ETS_Eval              7. Holt's Method: Adding Trend
  62  Lecture01_ETS_Eval              8. Holt-Winters: Level, Trend, and Seasonality
  63  Lecture01_ETS_Eval              9. The ETS Framework: A Unified View
  70  Lecture01_ETS_Eval             10. How Do We Measure Forecast Accuracy?
  89  Lecture01_ETS_Eval             11. Comparing Across Series: MAPE and MASE
  64  Lecture01_ETS_Eval             12. The Golden Rule: Always Evaluate Out-of-Sample
  23  Lecture01_ETS_Eval             13. ECON 8310: Four-Part Structure
  78  Lecture01_ETS_Eval             14. Tools and What You Will Be Able to Do
  75  Lecture01_ETS_Eval             15. Lecture 1: Key Takeaways
== Lecture02_ARIMA_VAR
  88  Lecture02_ARIMA_VAR             1. What Is Stationarity?
  69  Lecture02_ARIMA_VAR             2. Unit Roots and Differencing
  90  Lecture02_ARIMA_VAR             3. Testing for a Unit Root
  79  Lecture02_ARIMA_VAR             4. Why the Rule Exists: Spurious Regression
  43  Lecture02_ARIMA_VAR             5. ACF and PACF: Model Identification
  38  Lecture02_ARIMA_VAR             6. What Those Patterns Actually Look Like
  59  Lecture02_ARIMA_VAR             7. Difference First, Then Read the ACF
  87  Lecture02_ARIMA_VAR             8. ARIMA($p$,$d$,$q$)
  72  Lecture02_ARIMA_VAR             9. SARIMA: Adding Seasonal Terms
  73  Lecture02_ARIMA_VAR            10. SARIMA in Practice: A Four-Step Workflow
  72  Lecture02_ARIMA_VAR            11. Why Use Multiple Series?
  77  Lecture02_ARIMA_VAR            12. VAR or ARIMAX: Which Way Does Influence Run?
  90  Lecture02_ARIMA_VAR            13. The VAR($p$) Model
  67  Lecture02_ARIMA_VAR            14. Granger Causality Test
  35  Lecture02_ARIMA_VAR            15. Granger on Real Data: The COVID Window
  80  Lecture02_ARIMA_VAR            16. ARIMAX: Exogenous Variables with ARIMA Errors
  83  Lecture02_ARIMA_VAR            17. Lecture 2: Key Takeaways
== Lecture03_GAMs
  90  Lecture03_GAMs                  1. The Flexibility--Interpretability Tradeoff
  73  Lecture03_GAMs                  2. Why Additivity Matters for Interpretation
  86  Lecture03_GAMs                  3. What Is a GAM?
  90  Lecture03_GAMs                  4. What the ``Generalized'' Actually Means
  46  Lecture03_GAMs                  5. Controlling Smoothness: The Penalty Term
  84  Lecture03_GAMs                  6. Prophet: Forecasting at Scale
  67  Lecture03_GAMs                  7. Prophet's Seasonality: How Many Waves?
  34  Lecture03_GAMs                  8. Prophet: Code and Workflow
  82  Lecture03_GAMs                  9. pyGAM: Flexible Additive Modeling
  90  Lecture03_GAMs                 10. Interpreting GAM Results: Partial Dependence Plots
  89  Lecture03_GAMs                 11. GAMs vs.\ Other Methods: When to Choose Each
  75  Lecture03_GAMs                 12. Lecture 3: Key Takeaways
== Lecture04_DecisionTrees
  68  Lecture04_DecisionTrees         1. The Limits of Linear Models
  53  Lecture04_DecisionTrees         2. The Bias--Variance Tradeoff
  67  Lecture04_DecisionTrees         3. What Is a Decision Tree?
  53  Lecture04_DecisionTrees         4. The Same Tree, Drawn as Rectangles
  70  Lecture04_DecisionTrees         5. The CART Algorithm (Regression)
  90  Lecture04_DecisionTrees         6. Greedy Is Not Optimal --- and That Matters
  87  Lecture04_DecisionTrees         7. Depth Controls the Bias--Variance Tradeoff
  27  Lecture04_DecisionTrees         8. Two Ways to Stop a Tree Overfitting
  87  Lecture04_DecisionTrees         9. Cost-Complexity Pruning: A Line Touching the Curve
  16  Lecture04_DecisionTrees        10. Decision Tree Regression in Python
  88  Lecture04_DecisionTrees        11. Feature Importance
  47  Lecture04_DecisionTrees        12. Interpreting and Communicating Tree Predictions
  90  Lecture04_DecisionTrees        13. When a Single Tree Falls Short
  90  Lecture04_DecisionTrees        14. Lecture 4: Key Takeaways
== Lecture05_TreeEnsembles
  74  Lecture05_TreeEnsembles         1. The Wisdom of Crowds
  77  Lecture05_TreeEnsembles         2. Bagging: The Big Idea
  49  Lecture05_TreeEnsembles         3. More Trees, or Less Alike Trees
  82  Lecture05_TreeEnsembles         4. Out-of-Bag (OOB) Error: Free Cross-Validation
  77  Lecture05_TreeEnsembles         5. The Problem Bagging Does Not Fully Solve
  63  Lecture05_TreeEnsembles         6. Random Forest Hyperparameters
  78  Lecture05_TreeEnsembles         7. Random Forest Feature Importance
  62  Lecture05_TreeEnsembles         8. Random Forest in Python
  86  Lecture05_TreeEnsembles         9. Boosting vs.\ Bagging: The Core Difference
  90  Lecture05_TreeEnsembles        10. The Gradient Boosting Algorithm
  77  Lecture05_TreeEnsembles        11. Boosting, Tree by Tree
  84  Lecture05_TreeEnsembles        12. XGBoost: Three Advances
  26  Lecture05_TreeEnsembles        13. XGBoost: Key Hyperparameters
  68  Lecture05_TreeEnsembles        14. XGBoost in Python
  46  Lecture05_TreeEnsembles        15. Tree Methods: Which One When
  66  Lecture05_TreeEnsembles        16. Why No Tree Can Extrapolate
  89  Lecture05_TreeEnsembles        17. Lecture 5: Key Takeaways
== Lecture06_Regularization
  49  Lecture06_Regularization        1. Why Regularize?
  58  Lecture06_Regularization        2. Regularization: Fit Plus a Penalty
  86  Lecture06_Regularization        3. Subset Selection: The Classical Answer
  78  Lecture06_Regularization        4. Ridge Regression ($\ell_2$)
  85  Lecture06_Regularization        5. LASSO ($\ell_1$)
  86  Lecture06_Regularization        6. Why LASSO Zeros Out and Ridge Does Not
  85  Lecture06_Regularization        7. Elastic Net: Both Penalties at Once
  67  Lecture06_Regularization        8. The Grouping Effect, Drawn
  80  Lecture06_Regularization        9. The Regularization Path
  90  Lecture06_Regularization       10. Choosing $\lambda$ by Cross-Validation
  66  Lecture06_Regularization       11. Regularization in Python
  55  Lecture06_Regularization       12. Linear Methods: Which One When
  89  Lecture06_Regularization       13. Lecture 6: Key Takeaways
== Lecture07_NeuralNets
  84  Lecture07_NeuralNets            1. Why Go Beyond Trees?
  66  Lecture07_NeuralNets            2. From Linear Regression to a Single Neuron
  76  Lecture07_NeuralNets            3. Why the Nonlinearity Is the Whole Trick
  89  Lecture07_NeuralNets            4. Universal Approximation: Existence, Not Estimation
  87  Lecture07_NeuralNets            5. Feedforward Network: Layers and Forward Pass
  80  Lecture07_NeuralNets            6. How Big Is This Model, Actually?
  89  Lecture07_NeuralNets            7. Training: Loss, Gradients, and the Update Step
  65  Lecture07_NeuralNets            8. The Learning Rate, on a Real Bowl
  79  Lecture07_NeuralNets            9. Preventing Overfitting --- Where Lecture 6 Returns
  71  Lecture07_NeuralNets           10. Early Stopping, on Lab 7's Network
  30  Lecture07_NeuralNets           11. PyTorch: Dataset and DataLoader
  82  Lecture07_NeuralNets           12. PyTorch: Defining the Model
  37  Lecture07_NeuralNets           13. PyTorch: The Training Loop
  90  Lecture07_NeuralNets           14. Neural Networks vs.\ Tree Methods
  89  Lecture07_NeuralNets           15. Lecture 7: Key Takeaways
== Lecture08_CNNs
  81  Lecture08_CNNs                  1. Why Convolution? A Dense Layer Wastes Its Weights
  63  Lecture08_CNNs                  2. Why Convolution? Two Ideas
  53  Lecture08_CNNs                  3. How a Convolutional Layer Works
  54  Lecture08_CNNs                  4. Convolution, Step by Step
  44  Lecture08_CNNs                  5. What a Filter Sees
  84  Lecture08_CNNs                  6. Pooling: Downsampling Feature Maps
  90  Lecture08_CNNs                  7. LeNet to VGG: Depth and Small Filters
  90  Lecture08_CNNs                  8. Inception and ResNet: Depth That Trains
  60  Lecture08_CNNs                  9. What a Trained CNN Learns
  58  Lecture08_CNNs                 10. What Carries Over to Time Series
  88  Lecture08_CNNs                 11. Why Use a 1D CNN for Forecasting?
  83  Lecture08_CNNs                 12. 1D CNN Forecasting Pipeline in PyTorch
  69  Lecture08_CNNs                 13. Does It Actually Work?
  87  Lecture08_CNNs                 14. CNN Architecture Summary
  90  Lecture08_CNNs                 15. Lecture 8: Key Takeaways
== Lecture09_Part1_RNN_LSTM
  77  Lecture09_Part1_RNN_LSTM        1. Two Architectures That Cannot See Time
  90  Lecture09_Part1_RNN_LSTM        2. The Recurrent Idea: Carry State Forward
  48  Lecture09_Part1_RNN_LSTM        3. Vanilla RNN: One Equation, Repeated
  90  Lecture09_Part1_RNN_LSTM        4. Training It: Backpropagation Through Time
  76  Lecture09_Part1_RNN_LSTM        5. Why a Vanilla RNN Forgets
  79  Lecture09_Part1_RNN_LSTM        6. LSTM: A Cell State and Three Gates
  72  Lecture09_Part1_RNN_LSTM        7. What the Gates Are Doing, in Plain Terms
  83  Lecture09_Part1_RNN_LSTM        8. Why the Cell State Fixes the Gradient
  31  Lecture09_Part1_RNN_LSTM        9. LSTM for Forecasting in PyTorch
  83  Lecture09_Part1_RNN_LSTM       10. Does It Work? The Scoreboard So Far
  89  Lecture09_Part1_RNN_LSTM       11. Why the Gated Model Lost at 30 Epochs
  67  Lecture09_Part1_RNN_LSTM       12. Check 1: Was It a Fair Fight?
  70  Lecture09_Part1_RNN_LSTM       13. Check 2: What in the Data Explains It
  83  Lecture09_Part1_RNN_LSTM       14. Check 2, Continued: Nothing Arrives Early
  67  Lecture09_Part1_RNN_LSTM       15. RNN and LSTM in Context
  88  Lecture09_Part1_RNN_LSTM       16. Lecture 9, Part 1: Key Takeaways
== Lecture09_Part2_Transformers
  72  Lecture09_Part2_Transformers    1. Two Problems Built Into Recurrence
  89  Lecture09_Part2_Transformers    2. Scaled Dot-Product Attention
  67  Lecture09_Part2_Transformers    3. What That Buys a Forecaster
  88  Lecture09_Part2_Transformers    4. Multi-Head Attention
  76  Lecture09_Part2_Transformers    5. The Encoder Block
  90  Lecture09_Part2_Transformers    6. Positional Encoding: Attention Cannot Tell Time
  88  Lecture09_Part2_Transformers    7. Why Transformers Took Over
  28  Lecture09_Part2_Transformers    8. Transformer Encoder in PyTorch
  66  Lecture09_Part2_Transformers    9. Does It Work?
  87  Lecture09_Part2_Transformers   10. Reading That Result Honestly
  79  Lecture09_Part2_Transformers   11. Sequence Architecture Comparison
  89  Lecture09_Part2_Transformers   12. Lecture 9, Part 2: Key Takeaways
== Lecture10_BayesianFoundations
  59  Lecture10_BayesianFoundations   1. Frequentist vs.\ Bayesian: What Is Random?
  87  Lecture10_BayesianFoundations   2. What This Buys You in a Meeting
  87  Lecture10_BayesianFoundations   3. Bayes' Theorem
  67  Lecture10_BayesianFoundations   4. Bayes' Theorem, Drawn
  71  Lecture10_BayesianFoundations   5. The Update, in Pictures
  63  Lecture10_BayesianFoundations   6. Priors You Will Actually Use
  61  Lecture10_BayesianFoundations   7. Choosing a Prior You Can Defend
  66  Lecture10_BayesianFoundations   8. MCMC: The Big Idea
  68  Lecture10_BayesianFoundations   9. The Samples Are the Answer
  59  Lecture10_BayesianFoundations  10. PyMC: A First Model
  71  Lecture10_BayesianFoundations  11. Did the Sampler Actually Work?
  55  Lecture10_BayesianFoundations  12. Prior Predictive Checks
  58  Lecture10_BayesianFoundations  13. Three Priors, Three Predictions
  78  Lecture10_BayesianFoundations  14. Lecture 10: Key Takeaways
== Lecture11_Part1_BayesianTS
  90  Lecture11_Part1_BayesianTS      1. What Every Model So Far Gave You
  72  Lecture11_Part1_BayesianTS      2. Three Things That Become Possible
  52  Lecture11_Part1_BayesianTS      3. Structural Time Series: One Equation
  86  Lecture11_Part1_BayesianTS      4. The Level: A Random Walk with a Prior on It
  75  Lecture11_Part1_BayesianTS      5. Seasonality as Fourier Terms
  37  Lecture11_Part1_BayesianTS      6. The Model in PyMC
  60  Lecture11_Part1_BayesianTS      7. The First Fit Failed --- Here Is What That Looks Like
  65  Lecture11_Part1_BayesianTS      8. The Fix: Reparameterize, Do Not Sample More
  89  Lecture11_Part1_BayesianTS      9. Does It Work? The Point Forecast
  67  Lecture11_Part1_BayesianTS     10. The Interval Is the Product
  90  Lecture11_Part1_BayesianTS     11. Honest, and Too Wide to Order Against
  89  Lecture11_Part1_BayesianTS     12. Lecture 11, Part 1: Key Takeaways
== Lecture11_Part2_Hierarchical
  83  Lecture11_Part2_Hierarchical    1. The Question
  87  Lecture11_Part2_Hierarchical    2. Exchangeability: the Assumption Under Every Hierarchy
  77  Lecture11_Part2_Hierarchical    3. Complete Pooling and No Pooling
  74  Lecture11_Part2_Hierarchical    4. The Hierarchy Is One Extra Line
  90  Lecture11_Part2_Hierarchical    5. Why This Is the Right Answer, Not a Compromise
  63  Lecture11_Part2_Hierarchical    6. The Model in PyMC
  90  Lecture11_Part2_Hierarchical    7. The Trap: Write It the Obvious Way and It Breaks
  87  Lecture11_Part2_Hierarchical    8. Why It Breaks: a Funnel
  77  Lecture11_Part2_Hierarchical    9. Shrinkage, Measured
  85  Lecture11_Part2_Hierarchical   10. Checked Against Data the Model Never Saw
  90  Lecture11_Part2_Hierarchical   11. What the Model Actually Concluded
  90  Lecture11_Part2_Hierarchical   12. Lecture 11, Part 2: Key Takeaways
== Lecture12_BayesianRegression
  77  Lecture12_BayesianRegression    1. OLS vs.\ Bayesian Regression: What Changes?
  58  Lecture12_BayesianRegression    2. Bayesian Linear Regression in PyMC
  73  Lecture12_BayesianRegression    3. Reading a Posterior Coefficient
  31  Lecture12_BayesianRegression    4. Asking the Posterior a Direct Question
  78  Lecture12_BayesianRegression    5. Too Small to Matter? A ROPE
  89  Lecture12_BayesianRegression    6. Directed Acyclic Graphs
  85  Lecture12_BayesianRegression    7. Confounding, Measured
  86  Lecture12_BayesianRegression    8. Scenario Analysis with Posterior Samples
  48  Lecture12_BayesianRegression    9. Course Method Map
  70  Lecture12_BayesianRegression   10. Lecture 12: Key Takeaways
== Lecture13_Synthesis
  66  Lecture13_Synthesis             1. The Arc of the Course
  83  Lecture13_Synthesis             2. The Scoreboard
  60  Lecture13_Synthesis             3. Lesson 1: The Benchmark Is a Real Competitor
  45  Lecture13_Synthesis             4. Lesson 2: More Sophisticated Did Not Mean Better
  60  Lecture13_Synthesis             5. Lesson 2, Continued: A Problem This Data Does Not Have
  75  Lecture13_Synthesis             6. Lesson 3: One Idea Appeared Six Times
  37  Lecture13_Synthesis             7. Which Method, When
  44  Lecture13_Synthesis             8. Errors That Fail Silently I: the Pipeline
  69  Lecture13_Synthesis             9. Errors That Fail Silently II: the Code Was Fine
  67  Lecture13_Synthesis            10. The Day Every Model in This Course Fails
  54  Lecture13_Synthesis            11. What a Defensible Forecast Report Contains
  52  Lecture13_Synthesis            12. Where You Do This Yourself
  59  Lecture13_Synthesis            13. What This Course Did Not Cover
  90  Lecture13_Synthesis            14. Closing
```

### Other gates

- `\texttt{\ ` at the start of a code line: 0 in every deck.
- Narration `# ▶ SLIDE n — Title` headers: every header matches its frame title. The four the scripted check flagged
  (L02 `ARIMA(p, d, q)`; L06 `Ridge Regression (ℓ₂)`, `LASSO (ℓ₁)`, `Choosing λ by Cross-Validation`) differ only
  in how the narration writes the deck's math, and this branch did not change those titles.
- Lab code: every `{python}` chunk is unchanged except Lab 6 chunk 5 (m21's tie sort).
- `check-book-claims.py` 10/10 ok · `check-repo-hygiene.py` clean · `check-site-index.py` ok ·
  `check-assignment-sync.py`: student repo not present here, so it compared nothing.
- No file under `docs/`, `Figures/`, `scripts/`, no `.pdf`/`.html`, no `measured.json` and no `.bib` was changed.

## Assignments edited — copy to the student repo (`forecasting-env/assignments/`), then run `check-assignment-sync.py`

- `Homework/HW05_Part2_Sequence.qmd` (C3, C4, M1, M27)
- `Homework/HW05_Part3_Comparison.qmd` (C4, C5, M1, M27, m30)
- `Homework/HW06_Part1_Foundations_TS.qmd` (M1)
- `Homework/HW06_Part2_Hierarchical.qmd` (M1)
- `Homework/HW07_Bayesian_Regression.qmd` (M1, M14, m34, spare-prompts line)
