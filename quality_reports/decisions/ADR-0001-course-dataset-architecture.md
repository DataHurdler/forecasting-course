# ADR-0001: Course Dataset Architecture

**Status:** ACCEPTED
**Date:** 2026-08-24
**Context:** Course-wide data strategy — triggered by a Holt-Winters fit in HW01 that cannot run on the current dataset

## Problem

The course needs a dataset spine used **both for in-class worked examples and for homework**.
Using the same data in lecture and in assignments is the point: students see a method
demonstrated on a series in class, then apply the next method to the same series themselves,
so differences across methods are attributable to the method rather than to the data.

**Today there is no continuity at all.** Lecture examples quote results on **RSXFS** — a
monthly FRED retail-sales series, see L02, L05, L08 — while homework uses **Rossmann**, a
weekly store panel. A student cannot carry one example across the two, and the numbers on the
slides come from data they never touch.

The current choice for homework (Rossmann Store Sales, 30 stores × ~134 weekly observations,
Jan 2013 – Jul 2015) is also too short: Holt-Winters with a 52-week season needs 2m = 104 observations to initialize, which
leaves almost no test set and caused HW01 Q2 to specify a fit that cannot execute.

The four method families in this course want incompatible things from data:

| Need | Driven by |
|---|---|
| >= 2 full seasonal cycles, ideally many | L01 ETS |
| Multiple series that genuinely influence each other | L02 VAR / Granger |
| Exogenous regressors | L02 ARIMAX, L06 regularization |
| Daily granularity + holidays + several seasonalities | L03 GAM / Prophet |
| Wide tabular feature set | L04–L06 trees |
| Large observation count, many parallel series | L07–L09 deep learning |
| Explicit group structure for partial pooling | L11 hierarchical Bayes |
| Interpretable covariates for a DAG | L12 Bayesian regression |

No single public dataset satisfies all of these well.

## Options considered

### Option A: Keep Rossmann

Stay with the current dataset and work around its length.

**Pro:** Zero migration cost; `prep_rossmann.py` and 53 references across slides and homework already exist.
**Con:** ~134 weeks cannot support a 52-week season with a usable test set; no real hierarchy;
weekly only, so Prophet's multiple-seasonality story cannot be told; too small for deep learning.

### Option B: M5 / Walmart as a single spine

Use the M5 Forecasting–Accuracy dataset for everything.

**Pro:** 1,941 days (~5.3 years); 42,840 series across 12 aggregation levels on two crossed
hierarchies (state→store, category→dept→item); calendar with events and SNAP; weekly
`sell_prices`; free on Zenodo and via Nixtla `datasetsforecast`, so **no Kaggle login**;
Walmart is immediately legible to MBA students.
**Con:** Large raw download; single-seasonality retail understates what Prophet is for.

*(Corrected 2026-08-24 after testing the data: the original con claimed retail stores do not
Granger-cause one another. That is empirically false for M5. Two same-state stores share
regional shocks and the test is significant — see the correction note below.)*

### Option C: M5 spine + targeted supplements

M5 as the spine, with additional datasets introduced only where M5 demonstrably cannot show
something.

**Pro:** Keeps one dataset as the through-line for cross-method comparison, while each
supplement earns its place by covering a specific gap — and the *reason* for each supplement
is itself a teachable point about matching data to method.
**Con:** Students meet more than one data format; slightly more prep code.

### Option D: Favorita as the spine

Corporación Favorita Grocery Sales (Ecuador, 2013–2017).

**Pro:** Daily; hierarchy via store clusters; holidays; a daily **oil price** giving a genuine
macro→retail transmission channel; a documented magnitude-7.8 earthquake on 2016-04-16 that
visibly distorts sales for weeks.
**Con:** 125M rows raw; Kaggle-only; shorter than M5; overlaps M5 almost entirely in shape.

## Decision

**Chose:** Option C — M5 spine with targeted supplements.

**Rationale:** A single spine is what makes cross-method comparison meaningful — students see
ETS, ARIMA, XGBoost, an LSTM, and a Bayesian model on the *same* series and can attribute
differences to method rather than to data. M5 is the only candidate long enough to make the
52-week season work with room to spare, and the only one with a real hierarchy for L11. Where
M5 genuinely cannot demonstrate something, a supplement is introduced deliberately and the
reason is stated to students.

## The architecture

| Dataset | Role | Why not the spine |
|---|---|---|
| **M5 / Walmart** | Spine, L01–L12 | — |
| **FRED** macro series | L02 VAR & Granger causality | Gives a case where the causal *direction* is defensible — unemployment moves retail spending, not the reverse. M5 gives the complementary case where the test fires but the direction is not identified. Free API, no login, `pandas_datareader` support. |
| **Electricity demand** (half-hourly) | L03 multiple seasonality | Daily + weekly + yearly at once is what Prophet and GAMs are for; retail shows only one. Monash Forecasting Repository, free, no login. |
| **Favorita** | Week 16 cameo + final-project seed | Provides a **dated exogenous shock** (2016-04-16 earthquake) that M5 has no equivalent of. Used to close the course: every method fails together, because the future stopped resembling the past. Also a good project dataset — different enough from the spine that students must transfer rather than copy lecture code. |

## Consequences

- **These datasets are used in lecture as well as in homework.** Worked examples on slides,
  live-coded demonstrations, and assignment questions all draw on the same series, so a
  student can carry one running example through the whole semester.
- **The datasets are introduced in a standalone document**, `ECON8310_Datasets.md`, not in a
  Lecture 1 slide. A slide was drafted and then reverted: the material is reference content
  students return to all semester, and it needs more room than a slide allows — where to
  download each set, what is in each file, and why each supplement exists. The *why there is
  more than one* framing is the point of the document, and it is a teaching point rather than
  an apology.
- `scripts/prep_rossmann.py` is replaced by a new `prep_m5.py` that subsets M5. A comparable
  subset (10 stores × 3 categories, daily) is ~58,000 rows against the current ~4,000.
- Existing slide examples that quote Rossmann numbers (L02, L05, L08) must be re-run against
  M5, not merely renamed — the RMSE figures currently on those slides would otherwise be wrong.
- The Holt-Winters constraint in HW01 Part 1 becomes non-binding: ~277 weekly observations
  against the 104 required. The runtime assertion stays as a guard.
- ~53 Rossmann references must be updated: 3 lecture decks (L02, L05, L08), 8 homework files,
  and the prep script.
- **No lecture currently teaches structural breaks or intervention analysis.** The Week 16
  Favorita cameo would be its only appearance, and pairs with the already-assigned FPP Ch. 13
  (§13.9, outliers and missing values).
- HW02–HW07 should not be restructured until the migration happens, or that work is done twice.
- The same applies to `Labs/` (in-class hands-on exercises, one per lecture, `LectureNN_lab.qmd`).
  The folder, README, and template are in place and are dataset-agnostic; the individual labs
  should be authored against M5 rather than written twice.

## Rejected alternatives — why not

- **A (keep Rossmann):** the dataset is structurally too short for a 52-week season; the HW01
  bug was a symptom, not an isolated mistake.
- **B (M5 alone):** would understate what Prophet is for. *(The original reason given here ---
  that M5 has no real causal relationships to test --- was wrong; see the correction below.)*
- **D (Favorita spine):** duplicates M5's shape while being shorter, larger to download, and
  Kaggle-gated. Its one unique asset — the earthquake — is retained as a cameo under Option C.

---

## Correction — 2026-08-24, after testing against the data

The original version of this ADR justified bringing in FRED by asserting that **two Walmart
stores do not meaningfully Granger-cause one another**, so a VAR fitted to M5 would teach the
mechanics on a case with no relationship to find. That assertion was made from intuition and
is **empirically false**.

Measured on the prepared weekly panel, CA_1 and CA_3 FOODS:

- contemporaneous correlation of weekly **changes**: **+0.85**
- Granger test, CA_3 lags added to the CA_1 equation, 4 lags: **F(4, 263) = 8.51** — clearly
  significant

Two stores 40 miles apart share regional promotions, weather, holidays, and local economic
conditions. Of course they co-move.

**This makes the teaching case better, not worse.** M5 now supplies the *harder* and more
realistic lesson: the test fires, and a student still cannot claim CA_3 drives CA_1, because
the relationship is confounded by shared shocks. That is exactly the "Granger causality is not
structural causality" point, delivered on data where the temptation to over-claim is real
rather than hypothetical.

FRED still earns its place, for the complementary reason: it supplies a case where the causal
*direction* is defensible — unemployment plausibly moves retail spending and not the reverse.
Between them the two datasets cover both situations a student will meet.

HW01 Part 2 Q4 is built on the M5 case and asks students to reason about precisely this.

---

## Correction 2 — 2026-08-25, after testing the FRED pairing

The architecture table justifies FRED as the dataset that **"gives a case where the causal
*direction* is defensible — unemployment moves retail spending, not the reverse."**

That assertion was, like the M5 one corrected above, made from intuition. Measured on the
cached series (`data/processed/fred_monthly.csv`, 1992-02 to 2026-06, 412 months, 4 lags,
retail log-differenced and unemployment differenced):

| Sample | unemployment → retail | retail → unemployment |
|---|---|---|
| Full, 1992–2026 | F = 18.22, p < 0.0001 | F = 12.80, p < 0.0001 |
| Excluding Feb 2020 – Jun 2021 | F = 2.91, p = 0.021 | F = 6.74, p < 0.0001 |
| Pre-COVID, 1992–2020 | **F = 1.85, p = 0.12 — not significant** | F = 9.67, p < 0.0001 |

**The direction is not defensible in the way claimed.** The reverse direction — retail
predicting unemployment — is the one that survives every sample cut, and the intuitive
direction disappears entirely once COVID is removed. This is economically sensible in
hindsight: retail sales are a coincident-to-leading indicator and unemployment is a lagging
one. Spending falls first; layoffs follow.

**Consumer sentiment was tested too and is worse for the purpose.** `UMCSENT` → `RSXFS` is
insignificant at every lag and in every subsample (p = 0.14–0.66). The Lecture 2 slide named
sentiment, so the slide was also promising an example that returns a null.

### What changed as a result

- FRED **keeps its place in L02**, but for a different and better reason: it is the only
  pairing in the course where a Granger verdict **flips with the sample window**. That is a
  lesson M5 cannot teach, because M5 contains no comparable shared shock.
- The L02 slide now asks the `UNRATE` → `RSXFS` question (not sentiment) and states plainly
  that the test fires both ways and that which direction looks stronger depends on whether
  COVID is in the sample.
- `scripts/prep_fred.py` caches `UNRATE`, `UMCSENT`, `RSXFS` to
  `data/processed/fred_monthly.csv` from FRED's public CSV endpoint — no API key, and no
  network needed during class.
- Lab 2 gained **Step 6**, which runs the test on all three windows and makes the flip the
  point of the exercise.

**The pattern is now twice-observed and worth naming:** both times this ADR asserted what a
causality test *would* show, it was wrong, and in both cases the measured answer made better
teaching material than the assumed one. No further claim about what a test will show belongs
in this document unless it has been run.

---

## Amendment — 2026-08-25: electricity source substituted

The architecture table names the **Monash Forecasting Repository** as the source for the
electricity supplement. Monash's Zenodo endpoints return **HTTP 403** to a scripted download,
so the dataset cannot be fetched by a prep script the way FRED and M5 can. The `tsibbledata`
`vic_elec` set (the FPP textbook's own) is reachable, but only as an R `.rda` binary, which
would add `pyreadr` as a student dependency to read.

**Substituted:** PJM Interconnection hourly metered load (PJME zone), 2002-01-01 to
2018-08-03, via a public mirror of the PJM releases — plain CSV, no login, no extra
dependency. Cached by `scripts/prep_electricity.py`.

It satisfies the requirement the ADR actually stated, and exceeds it. Measured on the
prepared file:

| Cycle | Swing | Trough → peak |
|---|---|---|
| hour of day | 33.3% | 4am → 7pm |
| day of week | 11.6% | Sunday → Tuesday |
| day of year | 44.7% | late April → mid July |

The annual cycle is **double-peaked** — ~37,000 MW in July (air conditioning), ~34,400 MW in
January (heating), ~26,900 MW in the April trough. That shape is the pedagogical asset: a
single harmonic must place its peak in summer *or* winter, and on the Lab 3 holdout
(Aug 2017 – Aug 2018) forcing `yearly_seasonality=1` costs about 300 MW of RMSE against
Prophet's default order of 10. Retail has nothing comparable, which is the original reason
this supplement exists.

**Sourcing caveat, recorded deliberately:** this is a community mirror, not a primary
publisher, so the URL is less durable than FRED's. The prep script caches to
`data/processed/`, so a broken upstream affects only a fresh setup, not a working checkout.
If the mirror disappears, EIA-930 (`eia.gov`, official, reachable) is the fallback and covers
2015 onward.

---

## Addendum, 2026-09-21 — a fifth dataset, for projects only

A sixth option was added to the project menu and nothing above is retracted: the spine, the two
supplements and Favorita all stand.

`onlinestore_weekly.csv` — 454 weeks (2018-01-01 to 2026-09-07) of orders, units and indexed
revenue from a small online retailer still trading today. It ships in the student repository;
there is no download and no account. Revenue is indexed to November 2025 = 100 because the
series is published and the store's real takings are not ours to publish; no forecasting property
survives rescaling any worse for it.

It answers something none of the four could: **an unmodelled series**. Every other dataset here
has a literature a student can consult. This one has none, which is the condition the course
claims to prepare them for. The cost is honest and stated in the guide — it is short, thin and
lumpy, and much of the work is deciding what not to model.

The raw source is a CubeCart database dump carrying customer names, addresses, phone numbers,
emails and IP addresses. It lives in the private instructor repository, gitignored, and is
processed by `prep_ico_weekly.py`; nothing but weekly totals leaves it.
