# CLAUDE.MD -- Academic Project Development with Claude Code

**Project:** ECON 8310: Business Forecasting
**Institution:** University of Nebraska at Omaha
**Branch:** main

---

## Core Principles

- **Plan first** -- enter plan mode before non-trivial tasks; save plans to `quality_reports/plans/`
- **Verify after** -- compile/render and confirm output at the end of every task
- **Single source of truth** -- Beamer `.tex` is authoritative; Quarto `.qmd` derives from it
- **Quality gates** -- nothing ships below 80/100
- **[LEARN] tags** -- when corrected, save `[LEARN:category] wrong → right` to MEMORY.md
- **Python-first** -- labs and analysis use Python (statsmodels, scikit-learn, matplotlib); `random_state=42`

---

## Folder Structure

```
bsad8310-forecasting/
├── CLAUDE.MD                    # This file
├── .claude/                     # Rules, skills, agents, hooks
├── Bibliography_base.bib        # Centralized bibliography
├── Figures/                     # Figures and images
├── Preambles/header.tex         # LaTeX headers
├── Slides/                      # Beamer .tex files (LectureNN_Title.tex)
├── Labs/                        # In-class hands-on exercises (LectureNN_lab.qmd)
├── Narration/                   # Recording scripts (LectureNN_script.md)
├── Quarto/                      # RevealJS .qmd files + theme
├── docs/                        # GitHub Pages (auto-generated)
├── scripts/                     # Utility scripts + Python notebooks
├── quality_reports/             # Plans, session logs, merge reports
├── explorations/                # Research sandbox (see rules)
├── templates/                   # Session log, quality report templates
└── master_supporting_docs/      # Papers and existing slides
```

---

## Commands

```bash
# LaTeX (3-pass, XeLaTeX only)
cd Slides && TEXINPUTS=../Preambles:$TEXINPUTS xelatex -interaction=nonstopmode file.tex
BIBINPUTS=..:$BIBINPUTS bibtex file
TEXINPUTS=../Preambles:$TEXINPUTS xelatex -interaction=nonstopmode file.tex
TEXINPUTS=../Preambles:$TEXINPUTS xelatex -interaction=nonstopmode file.tex

# The published course URL lives in ONE place: scripts/course_url.txt
python scripts/set_course_url.py                       # show it + where it appears
python scripts/set_course_url.py https://new.host/path/ # rewrite all of them, then re-render
# It spans BOTH repos: 5 documents here + 3 in the student repo (README, assignments/README,
# STUDENT_QUICKSTART). The student repo is found via $FORECASTING_ENV_REPO or a sibling
# Forecasting-Env directory; if neither exists the script says so loudly rather than skipping.

# Publish to GitHub Pages (docs/). Targets: all | slides | labs | homework | docs | books | LectureNN
# 'all' includes the book and workbook -- they are assembled from the decks, labs and
# narration, so any source edit leaves them stale until they are rebuilt.
./scripts/sync_to_docs.sh            # everything -- slow, labs fit real models
./scripts/sync_to_docs.sh slides     # RevealJS mirrors only (PDFs are not published)
./scripts/sync_to_docs.sh Lecture07  # one lecture

# The landing page is GENERATED, never hand-edited. Add a lecture, lab, assignment or
# document to scripts/site.yml; sync_to_docs.sh rebuilds docs/index.html from it and then
# refuses to finish if anything published under docs/ is not linked from the page.
python scripts/build_index.py                          # rebuild it now
python scripts/check-site-index.py                     # current? links resolve? nothing orphaned?

# The book and workbook are generated; their PREFACES are the only hand-written pages in
# them. Every claim a preface makes is bound to its source, so editing one means the claim
# and its source have to still agree.
python scripts/check-book-claims.py

# Assignments are authored in Homework/ and COPIED to the student repo; the setup guide,
# AI policy and quickstart are authored THERE and rendered here. Both directions of
# cross-repo drift are invisible to every other gate, which runs inside one repo.
python scripts/check-assignment-sync.py

# Student-visible external links (skips commented-out blocks)
python scripts/check_links.py

# Beamer <-> Quarto parity: every deck's frame count must equal its mirror's '## ' count
python scripts/check_mirror_parity.py                 # all decks
python scripts/check_mirror_parity.py LectureNN_Name  # + list that deck's frame titles

# Quality score
python scripts/quality_score.py Quarto/LectureNN_Title.qmd

# Render an in-class lab (must run end-to-end without interaction)
quarto render Labs/LectureNN_lab.qmd

# Environment: Python 3.12 venv at .venv (see requirements.txt)
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
# macOS also needs: brew install libomp   (for xgboost)
```

---

## Quality Thresholds

| Score | Gate | Meaning |
|-------|------|---------|
| 80 | Commit | Good enough to save |
| 90 | PR | Ready for deployment |
| 95 | Excellence | Aspirational |

---

## Skills Quick Reference

The full workflow ships in `.claude/` (60 skills, 18 agents, 8 hooks, 30 rules), inherited
from the [claude-code-my-workflow](https://github.com/pedrohcgs/claude-code-my-workflow)
template at v2.5.1. The ones that matter most for this course:

**Slides and lectures**

| Command | What It Does |
|---------|-------------|
| `/compile-latex [file]` | 3-pass XeLaTeX + bibtex |
| `/deploy [LectureNN_Title]` | Render Quarto + sync to docs/ |
| `/create-lecture` | Full lecture creation workflow |
| `/translate-to-quarto [file]` | Beamer → Quarto translation |
| `/qa-quarto [LectureN]` | Adversarial Quarto vs Beamer QA |
| `/extract-tikz [LectureN]` | TikZ → PDF → SVG |
| `/new-diagram` | Author a new TikZ diagram |
| `/slide-excellence [file]` | Combined multi-agent review |
| `/visual-audit [file]` | Slide layout audit |
| `/pedagogy-review [file]` | Narrative, notation, pacing review |
| `/proofread [file]` | Grammar/typo/overflow review |
| `/devils-advocate` | Challenge slide design |
| `/validate-bib` | Cross-reference citations |

**Teaching**

| Command | What It Does |
|---------|-------------|
| `/syllabus` | Draft or revise the syllabus |
| `/scaffold-exercises` | Generate problem sets / homework scaffolding |
| `/teach-from-paper [file]` | Turn a paper into teaching material |

**Analysis and code**

| Command | What It Does |
|---------|-------------|
| `/data-analysis [dataset]` | End-to-end Python analysis (statsmodels/sklearn) |
| `/review-r [file]` | R code quality review (R scripts only) |
| `/capture-environment` | Record the runtime environment |
| `/diagnose` | Debug a failing build or script |

**Research**

| Command | What It Does |
|---------|-------------|
| `/lit-review [topic]` | Literature search + synthesis |
| `/research-ideation [topic]` | Research questions + strategies |
| `/interview-me [topic]` | Interactive research interview |
| `/review-paper [file]` | Manuscript review |

**Session and repo**

| Command | What It Does |
|---------|-------------|
| `/commit [msg]` | Stage, commit, PR, merge |
| `/checkpoint` | Save a session checkpoint |
| `/context-status` | Report context usage |
| `/compress-session` | Compress the session log |
| `/learn` | Record a [LEARN] entry to MEMORY.md |
| `/verify-claims` | Check stated claims against reality |
| `/deep-audit` | Full repo audit |

Run `ls .claude/skills` for the complete list. Gate scripts live in `scripts/`
(`check-*.py`); `scripts/backtest.sh` runs them all, though several are
template-maintenance gates that this course repo does not satisfy.

---

## Beamer Custom Environments

| Environment         | Effect                      | Use Case                              |
|---------------------|-----------------------------|---------------------------------------|
| `keybox`            | UNO-blue highlighted box    | Key formulas, forecast accuracy rules |
| `definitionbox[T]`  | Blue-bordered titled box    | Formal definitions (stationarity etc) |
| `warningbox`        | Red-accent warning box      | Common pitfalls, assumption violations|
| `examplebox[T]`     | Green-accent titled box     | Worked examples, business applications|

## Quarto CSS Classes

| Class              | Effect                    | Use Case                           |
|--------------------|---------------------------|------------------------------------|
| `.keybox`          | Gold-bordered box         | Key formulas, rules (mirrors Beamer keybox) |
| `.definitionbox`   | Blue left-rule + title    | Formal definitions (use `.definitionbox-title` inner div for title) |
| `.warningbox`      | Red left-rule box         | Common pitfalls, assumption violations |
| `.examplebox`      | Green left-rule + title   | Worked examples (use `.examplebox-title` inner div for title) |
| `.key-result`      | Bold UNO-blue accent      | Key takeaways per slide            |
| `.interpretation`  | Indented italic           | Model interpretation callouts      |
| `.smaller`         | 85% font size             | Dense content / long equations     |
| `.python-output`   | Monospace gray box        | Code output / model results        |
| `.hi`              | Bold blue accent text     | Key terms inline                   |
| `.neutral`         | Gray muted text           | Asides, caveats, footnotes         |
| `.positive`        | Green bold text           | Pros, recommended use cases        |
| `.negative`        | Red bold text             | Cons, contraindicated use cases    |

---

## Current Project State

**Beamer status:** All 15 decks written and compiling clean (0 errors, 0 overfull).
**Quarto status:** All 15 mirrors written and at frame parity — verify with
`python scripts/check_mirror_parity.py`.
**Site status:** `docs/` carries all 16 weeks plus finals week — slides (HTML only -- PDFs are deliberately not published), 14 labs,
11 homework submissions, and the syllabus, datasets guide and project rubric as HTML.
**Note:** The pre-redesign lecture set (Lecture01_Intro through Lecture12_Capstone) was removed in the Fall 2026 cleanup; it remains recoverable from git history. Its QA reports were kept in `quality_reports/`. Its lab notebooks (`scripts/Lecture*.ipynb`) were **deleted 2026-08-31**: they had been kept as "the only worked Python outside Homework/", which stopped being true once the 14 executable labs landed, and their filenames collided with the current numbering (`Lecture10_NeuralNets.ipynb` beside a Lecture 10 that is Bayesian Foundations). They loaded `statsmodels` toy data, never the course datasets. Recoverable from git history.

| Lecture | Beamer | Quarto | Key Content |
|---------|--------|--------|-------------|
| 1: Intro, ETS & Forecast Evaluation | `Lecture01_ETS_Eval.tex` ✓ | `Lecture01_ETS_Eval.qmd` ✓ | SES, Holt, Holt-Winters, ETS framework, RMSE/MAE/MAPE/MASE, walk-forward CV, DM test. |
| 2: ARIMA, VAR & Multivariate Models | `Lecture02_ARIMA_VAR.tex` ✓ | `Lecture02_ARIMA_VAR.qmd` ✓ | Stationarity, ARIMA/SARIMA, auto-ARIMA, VAR, Granger causality, ARIMAX. |
| 3: Generalized Additive Models | `Lecture03_GAMs.tex` ✓ | `Lecture03_GAMs.qmd` ✓ | GAM structure, smoothing penalty, splines, Prophet, pyGAM, partial dependence plots. |
| 4: Decision Trees | `Lecture04_DecisionTrees.tex` ✓ | `Lecture04_DecisionTrees.qmd` ✓ | Bias-variance tradeoff, CART, entropy/information gain, sklearn, feature importance. |
| 5: Tree Ensembles — RF & Boosted Trees | `Lecture05_TreeEnsembles.tex` ✓ | `Lecture05_TreeEnsembles.qmd` ✓ | Bagging, bootstrap aggregation, feature subsampling, OOB error, MDI vs permutation importance, gradient boosting, XGBoost (Newton step, col subsampling). |
| 6: Regularization & Model Selection | `Lecture06_Regularization.tex` ✓ | `Lecture06_Regularization.qmd` ✓ | Subset selection vs shrinkage, Ridge, LASSO, Elastic Net, l1/l2 geometry, regularization paths, choosing λ by walk-forward CV. |
| 7: Introduction to Neural Networks | `Lecture07_NeuralNets.tex` ✓ | `Lecture07_NeuralNets.qmd` ✓ | Neurons, activations, FFN layers, MSE loss, backprop, Adam, dropout, PyTorch Dataset/DataLoader/training loop. |
| 8: CNN Architectures | `Lecture08_CNNs.tex` ✓ | `Lecture08_CNNs.qmd` ✓ | Convolution, pooling, LeNet→VGG→Inception→ResNet, residual connections, 1D CNN for time series. |
| 9 Part 1: RNNs & LSTMs (Wk 9) | `Lecture09_Part1_RNN_LSTM.tex` ✓ | `Lecture09_Part1_RNN_LSTM.qmd` ✓ | Recurrence, BPTT, vanishing gradients, LSTM gates, cell state as additive path. Measured: at a fixed 30-epoch budget the vanilla RNN beats the LSTM at all three windows (842 vs 987 at 26 weeks) — but the two tie by 100 epochs (830 vs 821), and 91% of the panel's variance is between-series. Two added frames (Checks 1 and 2) teach the budget dependence and the three data diagnostics. |
| 9 Part 2: Transformers (Wk 10) | `Lecture09_Part2_Transformers.tex` ✓ | `Lecture09_Part2_Transformers.qmd` ✓ | Scaled dot-product attention, multi-head, encoder block, positional encoding (measured: 990 with, 1,282 without — a 292 penalty). Ties the LSTM; loses to the vanilla RNN. |
| 10: Bayesian Statistics I — Foundations | `Lecture10_BayesianFoundations.tex` ✓ | `Lecture10_BayesianFoundations.qmd` ✓ | Frequentist vs Bayesian, Bayes' theorem, priors (Beta/Normal/Exponential), MCMC/NUTS, PyMC, prior predictive checks. |
| 11 Part 1: Bayesian TS (Wk 12) | `Lecture11_Part1_BayesianTS.tex` ✓ | `Lecture11_Part1_BayesianTS.qmd` ✓ | Structural decomposition, local level, Fourier seasonality. Measured: RMSE 1,589 vs naive 1,660; 94% interval 100% covered but ±30% wide. |
| 11 Part 2: Hierarchical (Wk 13) | `Lecture11_Part2_Hierarchical.tex` ✓ | `Lecture11_Part2_Hierarchical.qmd` ✓ | **Exchangeability first** (the ten FOODS series), then complete/no/partial pooling, non-centred parameterization. Measured: thin series shrink 8.2× more; held-out RMSE 0.126 → 0.049; 127 divergences → 1 (R-hat 1.044 → 1.002). |
| 12: Bayesian Statistics III — Linear Regression | `Lecture12_BayesianRegression.tex` ✓ | `Lecture12_BayesianRegression.qmd` ✓ | Bayesian linear regression, posterior coefficient distributions, HDI, DAGs, scenario analysis, course method map. |
| 13: Synthesis *(Wk 16, no lab)* | `Lecture13_Synthesis.tex` ✓ | `Lecture13_Synthesis.qmd` ✓ | Course-wide scoreboard, the three lessons the measurements taught, method-choice guide, a two-part silent-failure catalogue (pipeline bugs vs. correct code / wrong answer), the Christmas-closure limit (0.25% of days, 4.6% of squared error), and the final-project frame. |

---

## Homework Assignments

**Dataset architecture:** see
[`quality_reports/decisions/ADR-0001-course-dataset-architecture.md`](quality_reports/decisions/ADR-0001-course-dataset-architecture.md).
**M5/Walmart** is the spine, supplemented by FRED (L02 VAR/Granger) and electricity demand
(L03 multiple seasonality). **Two datasets are project-only:** the **online store** (`onlinestore_weekly.csv`, 454 weeks
2018–2026, revenue indexed to Nov 2025 = 100, built by `forecasting-instructor/prep_ico_weekly.py`
from a gitignored dump; shipped in the student repo, documented as §5 of the datasets guide) and
**Favorita** as of 2026-08-31 — there is no
`prep_favorita.py` and Week 16 does not use it. The Week 16 shock cameo is built on M5's own
Christmas closures (units = 0 on 25 Dec, five years; `is_event` fires on 158 days of which 5
are closures), which needs no second dataset. The migration off
Rossmann is **complete** — no Rossmann reference remains in `Homework/` or `Labs/`.

**Status:** All 11 submissions written and rendering. Each part's checklist states its own prompt
budget, and the folder stem is also the commit-message prefix and the key in the student repo's
`policy/homework_limits.json` (e.g. `hw06_part1`).
**Data prep:** Run each once from the repo root:
`python scripts/prep_m5.py` → `data/processed/m5_weekly.csv`, `m5_daily.csv`
`python scripts/prep_fred.py` → `fred_monthly.csv` (L02 Granger; no API key)
`python scripts/prep_electricity.py` → `electricity_daily.csv`, `electricity_hourly.csv` (L03 GAMs)
`python scripts/prep_favorita.py` → `favorita_weekly.csv`, `favorita_daily.csv` (**final projects only**; needs a Kaggle account and a 4.6 GB download)
**Format:** Quarto `.qmd` (students render to HTML). Each assignment includes an initial Codex prompt and per-question prompt budgets.
**Student repo:** https://github.com/DataHurdler/forecasting-env (Codex workflows, submission structure, validation script). Assignment `.qmd` files are authored in `Homework/` here and **copied** to that repo's `assignments/` — keep them in sync.

| HW | File | Lectures | Dataset | Prompt Budget | Key Tasks |
|----|------|----------|---------|---------------|-----------|
| 1a | `HW01_Part1_ETS.qmd` ✓ | L01 | Weekly, Store 1 | 4 | ETS (SES/Holt/HW), held-out RMSE/MAE |
| 1b | `HW01_Part2_ARIMA_VAR.qmd` ✓ | L02 | Weekly, Store 1 + Store 2 | 8 | ARIMA walk-forward CV, VAR, Granger causality, method comparison |
| 2 | `HW02_GAMs.qmd` ✓ | L03 | **M5 daily, CA_1 FOODS** | 10 | Prophet components + SNAP/event regressors, pyGAM cyclic spline, seasonal-naive benchmark |
| 3 | `HW03_Trees_RF.qmd` ✓ | L04 + L05 | **M5 weekly, 30 series** | 12 | Tree visualization, RF + OOB, MDI vs permutation with planted noise controls, walk-forward CV |
| 4 | `HW04_XGBoost_Regularization.qmd` ✓ | L06 | **M5 weekly, 46 engineered features** | 12 | XGBoost tuning incl. `reg_lambda`, LASSO path, per-mix alpha tuning, cross-model CV |
| 5a | `HW05_Part1_FFN_CNN.qmd` ✓ | L07 + L08 | M5 weekly, window 26 | 9 | Windowing Dataset, FFN, 1D CNN, receptive-field experiment |
| 5b | `HW05_Part2_Sequence.qmd` ✓ | L09 | M5 weekly, window 52 | 10 | Vanilla RNN, LSTM, Transformer + positional-encoding ablation |
| 5c | `HW05_Part3_Comparison.qmd` ✓ | L07–L09 | M5 weekly | 3 | Five-architecture comparison, full-course table, reflection |
| 6a | `HW06_Part1_Foundations_TS.qmd` ✓ | L10 + L11 Pt 1 | M5 weekly, CA_1 FOODS | 10 | Beta-Binomial + prior predictive, structural TS, calibration vs usefulness |
| 6b | `HW06_Part2_Hierarchical.qmd` ✓ | L11 Pt 2 | M5 weekly, 10 FOODS series | 9 | Exchangeability check, pooled/unpooled/hierarchical, divergence reproduction, shrinkage, held-out validation |
| 7 | `HW07_Bayesian_Regression.qmd` ✓ | L12 | **M5 weekly, 30 series** | 16 | DAG, naive vs controlled OLS, Bayesian regression, ROPE, price-scenario simulation |
