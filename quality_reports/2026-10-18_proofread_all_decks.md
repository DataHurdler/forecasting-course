# Proofreading, all 15 decks (2026-10-18) — RAW, NOT YET VERIFIED

Five `proofreader` agents (read-only, run locally in worktrees, not in the cloud as first
reported), three decks each, `.tex` authoritative vs `.qmd` mirror. Per
`.claude/rules/proofreading-protocol.md`: Phase 1 (propose) only. Nothing is applied until the
coordinator verifies each item against the files and the owner approves.

Status key: ☐ unverified · ✔ verified · ✗ refuted

## Cross-deck patterns (fix once, everywhere)
- ☐ **Citation in frame title (.tex) moved into the body (.qmd)**, sometimes onto a different claim:
  L04 (James2021, Molnar2022 missing from .qmd; Breiman1984 placement), L07 (James2021 attached to
  "evidence of Homework 4"), L09P1 (Bengio1994 attached to "our own trained RNN"), L09P2 (Vaswani2017),
  L10 (Gelman2013, MartinKumarLao2021, McElreath2020).
- ☐ **Outline omits "and Roadmap"** in .qmd: L02, L03, L05, L06.
- ☐ **Section-divider subtitles differ or are missing** in .qmd: L08 (×3), L13 (×5).
- ☐ **Deck titles differ** .tex vs .qmd: L09P2, L10, L11P1, L12, L13; Bayesian series uses ":" vs "---" inconsistently.
- ☐ **Uncited abbreviations**: "BMCP §4.6.1" (L11P1, L11P2 → MartinKumarLao2021), "FPP §7.5" (L06),
  "FPP Ch. 11 / FPP-Py Ch. 15" (L13 → Hyndman2021). L11P2, L12, L13 print empty References frames.
- ☐ Bib key `James2021` has `year = {2023}` (low; invisible to students).
- ☐ Notation reuse: `t` (threshold L04 / time / budget L06), `n` (layer width vs sample size L07),
  `T_t` trend vs `T` sample size (L01), `g` link vs Prophet trend (L03), `σ` activation (L07) vs sigmoid (L09),
  `p` vs `k` features (L05/L06), `α` (pruning / elastic-net mix / sklearn alpha).
- ☐ "Key Takeaways" frame title pattern: L02 lacks "Lecture 2:".

- ☐ **Code boxes lose their indentation in the PDF** (a leading `\ ` after `\texttt{` is dropped): L05 (3 lines), L06 (2), L07 (15), L08 (5), L09P2 (3), L11P1 (9), L11P2 (7). Fixed in L09P1 on 2026-10-19 with `\texttt{\mbox{}\ …`, as L10 already does. Found by the coordinator, not a reviewer.

## Per deck
**L01** ☐ takeaway "richer" (.qmd) vs "rich" (.tex) · ☐ package-name capitalization mixed on the four-part slide.
**L02** ☐ VAR(p) frame shows only lag 1 — add "(shown for p = 1)" · ☐ "time-series" hyphenation as modifier.
**L03** ☐ "modelled" → "modeled" · ☐ "General Additive Modeling" → "Generalized" (section title + outline) ·
☐ "Facebook's Prophet" vs "at Meta" · ☐ "GCV" never spelled out; gridsearch described two ways.
**L04** ☐ "Next:" line differs .tex/.qmd (qmd better describes L05) · ☐ TimeSeriesSplit import only in .qmd ·
☐ "Asymmetry — … distort coefficients equally in both directions" contradicts its label (owner wording) ·
☐ "trade" → "tradeoff" · ☐ MDI bias described differently L04 vs L05 · ☐ importance "sum" (L04) vs "average" (L05) ·
☐ L04 promises SHAP in L05; L05 never mentions it.
**L05** ☐ XGBoost table: .qmd dropped text in 5 rows · ☐ "naïve" vs "naive" on one slide · ☐ defaults column `\texttt` inconsistent.
**L06** ☐ "ridge" → "Ridge" · ☐ code template `X_tr` (.tex) vs `X_train` (.qmd).
**L07** ☐ "Homeworks 3 and 4" → "Homework 3 and Homework 4" · ☐ fragment on "Early Stopping" frame ("Lab 7's network for 200 epochs, …" has no verb).
**L08** ☐ "don't" on takeaways · ☐ section "Pooling and Dropout" has no dropout content · ☐ filter "length" vs "width" ·
☐ "well engineered" → "well-engineered" · ☐ VGG/Inception/ResNet years (arXiv) vs citation years.
**L09P1** ✔ "the next three slides" → four (fixed 2026-10-19) · ☐ "Lecture 9, Part 1" comma inconsistency.
**L09P2** ☐ "training examples" vs "training windows" (5,160).
**L10** ☐ "Even a few means … explored wrongly" → "Even a few divergences mean … explored poorly" ·
☐ dash run-on on "Choosing a Prior" · ☐ Beta(·) math vs plain · ☐ takeaway colon inside/outside bold.
**L11P1** ☐ "Don't" in a frame title · ☐ "With … On" title capitalization · ☐ "Next: … thirty series" vs Part 2's ten FOODS series.
**L11P2** ☐ "Counts move with the seed, even the machine" → "and even with the machine" · ☐ stray newline in `\section{…}` ·
☐ `shape=10` vs `shape=J` (J never defined).
**L12** ☐ warningbox lead phrase bold only in .qmd.
**L13** ☐ "practised" → "practiced" · ☐ "Three variations … does not" → "do not" · ☐ fragment "Which does not count…" ·
☐ scoreboard header "RMSE" vs "Test RMSE" · ☐ part names "Classical" vs "Time series" · ☐ "start here" capitalization ·
☐ "Twelve weeks"/"twelve lectures" (course is 16 weeks).

## Flagged for the numeric/domain pass (not proofreading)
- ☐ L08 "What Carries Over": two stacked width-3 filters see 5 weeks, not 6, and use 6 weights per channel, not "far fewer" than one width-6 filter.
- ☐ L09P1 scoreboard: "Recurrence beats convolution decisively" also attached to the LSTM (987 vs CNN 1,032, a 45 gap).
- ☐ L05: "63%" (line ~99) vs "about 64%" (line ~107).
