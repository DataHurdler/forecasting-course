# Pedagogy review, all 15 decks (2026-10-18) — RAW, NOT YET VERIFIED

Five `pedagogy-reviewer` agents (read-only, run locally in worktrees), three decks each. Findings
are proposals; nothing is applied until the coordinator verifies them and the owner approves.
Status key: ☐ unverified · ✔ verified · ✗ refuted · ⊘ conflicts with an owner decision

## L04–L06 (reviewer 2)
**Cross-deck**
- ☐ X1 α means four things (L01 SES, L03 GAM intercept, L04 pruning price, L06 elastic-net mix); penalty strength is λ in L03/L06 but α in L04 → write L04's price as λ (sklearn `ccp_alpha` = λ/n).
- ☐ X2 σ² = Var(y) (L02), irreducible noise (L04), one tree's variance (L05) → `σ²_tree` in L05.
- ☐ X3 L03→L04 bridge ("GAMs bend but stay additive; trees add interactions") only in narration.
- ☐ X4 four unlabelled datasets across L04/L05/L06 → dataset tag in each caption.
- ☐ X5 takeaways omit the narration's "the point": L04 (step function cannot extrapolate), L05 (gain came from features), L06 (the cross-lecture penalty skeleton).
- ☐ X6 no frame poses a question to students.
- ☐ X7 forecast setup (one-step-ahead, true lags) never stated in L04/L05.

**L04** (verdict: strongest of the three)
- ☐ 1 Bias–variance frame sits early, uses "stumps/deep trees/bootstrap" before trees are defined, six frames from its payoff → move or cross-reference; define bootstrap once.
- ☐ 2 No forecast result → "depth 3: RMSE 1,213 vs seasonal naive 1,660" in the depth caption.
- ☐ 3 Python template uses `max_depth=5` though the deck finds depth 3 best.
- ☐ 4 "Greedy Is Not Optimal" uses invented numbers without saying so.
- ☐ 5 Instability claim in the depth caption has no evidence → point to the bootstrap dot-plot.
- ☐ 6 "Cannot extrapolate" missing from takeaways; "typically 20–40% lower RMSE" unsupported.
- ☐ 7 "Asymmetry … equally in both directions" bullet contradicts itself (same as proofreader L4-7).
- ☐ 8 `y_{t-4}` vs `units_lag4` mapping never stated.
- ☐ 9 "do the job better" (pruning) asserted (low).

**L05** (verdict: good arc; ordering and one contradiction)
- ☐ 1 Bagging opener misses the callback to L04's bootstrap dot-plot (its diamond *is* a bagged forecast).
- ☐ 2 OOB contradiction: the OOB frame warns it is invalid for time series, but the hyperparameter table ("free CV estimate"), the Python template (`oob_score_`) and "Which One When" ("OOB for free") sell it.
- ☐ 3 OOB figure's encoding never explained on the slide.
- ☐ 4 "The Problem Bagging Does Not Fully Solve" overloaded and ends with RF losing, then the template uses `max_features='sqrt'`.
- ☐ 5 "More Trees, or Less Alike Trees" reads as a fragment; schematic status only in a code comment.
- ☐ 6 Variance formula before its figure.
- ☐ 7 "Gradient" never explained; slope/curvature/log-loss/Poisson used without prerequisites.
- ☐ 8 RF "cuts RMSE 23%" is against an overfit deep tree; vs a depth-3 tree (~1,213) it is ~8%.
- ☐ 9 Boosting algorithm before its picture; η = 0.3 in the figure vs "above 0.3 risks overshooting".
- ☐ 10 Deep (bagging) vs shallow (boosting) trees never contrasted on a slide.
- ☐ 11 "Best" accuracy for XGBoost unsupported; early stopping used before defined.
- ☐ 12 Extrapolation frame comes after the decision table.
- ☐ 13 `reg_lambda`/`reg_alpha` = Ridge/LASSO a week early; "leaf size" ambiguous (means leaf weights).
- ☐ 14 Importance "sum" (L04) vs "average" (L05) (same as proofreader L4-10); MDI frame before its forest.
- ☐ 15 Pacing: seven dividers for 17 content frames.

**L06** (verdict: most honest; thesis withheld)
- ☐ 1 Takeaways omit the lecture's stated purpose (penalty skeleton across L03/L04/L06).
- ⊘ 2 "Choosing λ by CV" has no figure → **owner-approved skip**: the CV curve is Lab 6 Step 4's "Your turn". A vertical line at the chosen λ on the path figure may still be fine.
- ☐ 3 Geometry frame teaches the exception (corner wins only for t < 0.16) more than the rule; "budget t" not linked to λ.
- ☐ 4 "Elastic Net usually the right default" unsupported; "reach" ≠ "budget".
- ☐ 5 Ridge used before defined; "the benchmark" and the 62 predictors only in narration.
- ☐ 6 No worked number (+800/−740 → +30/+28 only in narration).
- ☐ 7 Path frame: sklearn `alpha` naming warning three frames later; "more stable" claim untested.
- ☐ 8 `l1_ratio` = α and sklearn `alpha` = λ buried in a 9 pt footnote.
- ☐ 9 AIC/BIC unexplained (low). ☐ 10 Pacing: four no-new-evidence frames at the end.


## L01–L03 (reviewer 1)
**Cross-deck**
- ☐ Walk-forward CV is taught on no slide (L01 shows one split + "use TimeSeriesSplit"), yet HW01 P2(d) and L06 ("as in Lecture 1") assume it. CLAUDE.md lists "walk-forward CV, DM test" for L01; neither is in the deck → add a walk-forward figure to L01, and correct CLAUDE.md.
- ☐ The "beat the benchmark (1,660)" promise is not paid off in L02/L03: no ARIMA/VAR/Prophet/pyGAM score against a benchmark on a slide; L02 shows no forecast at all.
- ☐ Notation collisions: α, φ (damping vs AR), h (horizon vs Prophet holiday), g (link vs Prophet trend), s, f (smooth vs pyGAM factor `f()`), T (sample size vs trend T_t), e_t vs ε_t.
- ☐ AIC/BIC/GCV/edof used unexplained. ☐ Data thread breaks (L02 five series; L03 electricity with no bridge).
**L01** ☐1 TimeSeriesSplit prescribed, multi-origin never shown · ☐2 benchmarks frame uses RMSE/test weeks before defined; benchmark formulas not on slide · ☐3 weights α(1−α)^k and the SES recursion never connected · ☐4 ETS frame: "9×2−3" unexplained, ETS codes never written, AIC unglossed · ☐5 Holt-Winters frame thin after split; HW never fitted to data · ☐6 MASE<1 caveat only spoken · ☐7 T clash; T/S/C/I decomposition never used again · ☐8 roadmap after content, says 12 lectures.
**L02** ☐1 ACF/PACF identification before AR/MA are defined; table before figure · ☐2 no forecast, interval or benchmark; prediction-interval narration has no slide · ☐3 "VAR or ARIMAX" compares before ARIMAX is defined; Granger H₀ not linked to a₁₂ · ☐4 ARIMA(0,1,1)=SES payoff unverifiable (formula missing) · ☐5 "Lecture 1's TX_1 HOUSEHOLD" — L01 never forecast it (only in the weights figure) · ☐6 COVID Granger slide omits the variable transforms; takeaways omit spurious regression and sub-sample lesson · ☐7 unit-root frame overloaded (d, D, Δ, half-life) · ☐8 no PACF on real data; workflow jumps m=7 → m=12 · ☐9 no opening hook.
**L03** ☐1 splines/knots/GCV/edof first appear in one caption · ☐2 notation clashes (g, α, h, f vs `f()`, s) · ☐3 dataset never introduced ("Lab 3's" before the lab) · ☐4 Prophet-accuracy and concurvity caveats only in narration; no benchmark RMSE · ☐5 "Generalized" frame interrupts the arc · ☐6 "interaction" not named on the additivity frame · ☐7 pyGAM code ≠ the model plotted · ☐8 Prophet formula before picture; holiday term has no example · ☐9 PDP band lesson on an uninformative example · ☐10 five dividers for 15 frames.

## L07–L09P1 (reviewer 3)
**Cross-deck**
- ☐ 1 "Backprop multiplies local derivatives" is on no slide (L07 says only "chain rule, run backward"; L08's multiply-vs-add sentence is narration-only), yet L09 relies on it.
- ☐ 2 L07's "each sigmoid scales a gradient by at most ¼ … matters in Lecture 9" is never paid off and misleads (vanilla RNN uses tanh; LSTM cell path scales by f, not σ′).
- ☐ 3 L08 blames the CNN's loss on a 7-week receptive field; L09's 26-week RNN can't see lag 52 either yet wins → hedge L09's "reading a sequence in order" causal claim.
- ☐ 4 σ generic (L07) vs sigmoid (L09); f filter (L08) vs forget gate (L09).
**L07** ☐1 no worked backprop example · ☐2 activations named only in the figure; "¼" forward reference · ☐3 "Early Stopping, on Lab 7's Network" comes before the network is defined; markers explained only in narration · ☐4 FFN frame equation-first (weight_blocks one frame later) · ☐5 learning-rate frame never states the three regimes; λ_max undefined · ☐6 2,864 bound unexplained (calendar-day mean, in-sample) · ☐7 "training windows" used before L08 defines windows.
**L08** ☐1 **FACT: two stacked width-3 filters see 5 weeks, not 6** (deck's own rule; narration repeats it); RF = 1 + Σ(K−1) never stated · ☐2 code shows the losing pooled head; "Flatten head" undefined; global pooling undefined · ☐3 the ~500 RMSE cost is global pooling, not the local 2×2 pooling taught · ☐4 channels / multi-channel conv / 1×1 / dilation untaught; parameter counts underivable · ☐5 formula is true convolution, figure and Conv1d are cross-correlation · ☐6 "Pooling and Dropout" has no dropout · ☐7 eleven frames before any forecasting data · ☐8 pooling patch = shelf edge only in narration.
**L09P1** (taught next week; **six fixes applied 2026-10-19**, owner-approved: 1 c̃_t defined + σ/stacking line [diagram not added], 2, 3, 4, 6, and "next four slides") ✔1 **c̃_t used, never defined; no LSTM cell diagram**; [h,x] unexplained · ✔2 **gradient figure is the 52-week RNN** but the slide implies the 26-week model; "0.001% a year back" needs a 52-week window; 0.79^26 ≠ quoted 0.4% (fitted past step 5) · ✔3 **Check 2 vs window table**: "nothing needs carrying across a gap / sufficient statistic" vs RNN 878→842→805 improving with window; ACF 0.43 at lag 52 · ✔4 BPTT: "roughly W_h" → "largest singular value × tanh′ ≤ 1" · ☐5 vanilla RNN frame: hidden size 64, last-state-only output, 4 channels, h₀ — all narration-only · ✔6 PyTorch frame shows only LSTM; `nn.RNN` unpacks `out, h` · ☐7 Check 1: LSTM ahead at 10 epochs, two seeds, still falling at 100 — only in figure/narration · ☐8 Check 2 Continued: fragment, no figure, three diagnostic questions missing from takeaways, forget-gate 0.52 thread not closed · ☐9 Thanksgiving story conflicts with Check 2 · ☐10–13 causal claim; "solves a problem this data does not have" predicts a tie not a loss; "next three slides" → four; narration "comes second".

## L09P2–L11P1 (reviewer 4)
**L09P2** ☐1 "it needs scale" asserted with no budget/seed spread — the exact mistake Part 1 warned against · ☐2 heatmap before heads/layers are defined; rows/panels unexplained on slide · ☐3 Q,K,V as projections of the same weeks never stated; d_k, W^O, softmax undefined · ☐4 "shuffling leaves every output unchanged" — self-attention is permutation-*equivariant*; readout is the last position · ☐5 cost square uncaptioned on slide · ☐6 RNN/LSTM numbers changed from Part 1 (26→52 weeks) without a footnote; "since Homework 3" vs "as Homework 4" · ☐7 opening implies attention fixes recurrence; Part 1 found no long-memory need · ☐8 `self.pe` never built; `[:, -1, :]` unexplained · ☐9 no questions.
**L10** ☐1 no bridge model (Normal likelihood, unknown σ, multi-parameter) before L11's PyMC mechanics · ☐2 Ridge = Normal prior / LASSO = Laplace bridge missing (L11 later relies on it) · ☐3 Beta(5,45) used before Beta is introduced; "worth 50 observations" unexplained · ☐4 θ, 𝒟 undefined on the formula slide; the mosaic's θ is an event (a SNAP week), then θ becomes a rate; formula before picture; gray-0.40 footnote raises a puzzle with no payoff · ☐5 divergences required in the diagnostics table, but only NUTS has them (the MCMC slide teaches Metropolis) · ☐6 prior predictive placed after fitting; code before figure · ☐7 opens on an abstract table · ☐8 94% vs 95% intervals and HDI undefined · ☐9 p(𝒟) integral never written · ☐10 one question only.
**L11P1** ☐1 opening stock-out question never answered (no 95th-percentile order quantity) · ☐2 code slide shows the non-centred fix before the failure; trace rows not labelled in text; heights explanation in an 8-pt footnote · ☐3 forecast mechanics (extending the walk h steps) absent · ☐4 promises "trend" and component error bars; model is local level, figure shows means only · ☐5 priors unjustified; no prior predictive check despite L10's "every time" · ☐6 "ARIMA's formula under unchecked assumptions" vs this model's own σ²+hσ²_level (= SES's) · ☐7 "calibrated" vs 52/52 at 94% (conservative) · ☐8 random-walk frame has no picture · ☐9 RMSE 1,660 here vs 2,152 in L09 unexplained; "won't beat XGBoost" unsupported · ☐10 minor: "thirty series" teaser, BMCP, warningbox used for a good result, decomposition features only in narration.

## L11P2–L13 (reviewer 5, resumed 2026-10-19)
**Cross-deck** ☐ no audience questions on slides (L13's "ask which a diagnostic would catch" is answered on the slide) · ☐ headline numbers recapped in L13 live only in labs/homework/narration · ☐ heavy dividers for short sections.
**L11P2** (verdict: strongest of the three) ☐1 non-centred code shown before the trap; no callback to L11P1's identical fix · ☐2 funnel frame (~95 words) never says why a funnel defeats the sampler, nor the consequence (σ_b biased up → under-pooling) → split in two · ☐3 "thinned to 8 weeks" purpose (mimic new stores, keep an answer key) only in narration · ☐4 full model never written (likelihood, a_j, σ); `s` vs `sigma`; J undefined; HalfNormal(0.25) unexplained · ☐5 precision-weighted shrinkage rule only in a muted footnote · ☐6 the five thinned series never named before "four of five / WI_2" · ☐7 "strength estimated, not tuned" claimed as new, but L11P1 estimated σ_level · ☐8 `b`/`mu_b` vs L12's `b`/`mu`.
**L12** ☐1 **one pooled b_price and b_snap across all 30 series contradicts L11P2's lesson** (b_snap 0.036 blend vs FOODS 0.078); L13 lists exactly this as a silent failure · ☐2 model and coefficient table before the DAG/confounding frames that justify the controls; pricing question only at the scenario slide · ☐3 collider asserted, no example on the slide (revenue = price × units only in narration) · ☐4 "do not standardize" consequence (22×) only in the lab; `idx`, `J`, `lpc`, `yr` undefined · ☐5 Ridge/LASSO-as-prior asserted without example; likelihood never written; "OLS since Lecture 2" (first slide use is L6) · ☐6 Course Method Map duplicates L13, cites "our board" before L13 shows it, lists Ridge/LASSO "Uncertainty: No" one slide after calling them posterior modes · ☐7 `b` reused; ROPE width for price not discussed (low).
**L13** ☐1 **scoreboard**: "every model in this course" = 7 models on one panel; CNN 26-wk vs others 52-wk; RF on 27 features vs LASSO/XGBoost on 46; top three inside the deck's own 20–90 seed spread → retitle, label windows/features, "top three within seed noise" · ☐2 opening question ("where does knowledge come from") never returns; Bayesian third has no payoff slide (L11P1 interval, L12 scenario); L06 placed under "Trees" · ☐3 Christmas frame: numbers are the seasonal naive's, slide says "every method failed"; squares never decoded on slide · ☐4 Lesson 1 skill bars: one-week groups (+58/+65%) read as contradicting "the benchmark beat every model"; skill not defined in body · ☐5 numbers recapped from homework/labs only (4.3 cycles, 17th of 29, nine fits, +0.011 vs +0.089, 22×, unshifted rolling features) → cite sources · ☐6 Silent Failures II row 4: L11P2's centred fit passed R̂/ESS and failed only on divergences; row is a sampler failure on a "code was fine" frame; last line answers its own question · ☐7 "Which Method, When" gives conflicting advice for the M5 panel (LASSO won) · ☐8 Lesson 3 lists dropout as a weight penalty · ☐9 pacing: four text/table frames in a row; project frame inside "Doing It Honestly"; aisle photo uncaptioned; "where the course is actually assessed" overstates the rubric · ☐10 "Lesson 2, Continued" reads as a fragment.

## Still pending
- The full-course review (numbers and claims): moved to a Claude Code on the web session; prompt in `quality_reports/plans/2026-10-19_cloud-full-course-review-prompt.md`.
