ECON 8310 — The Data We Use
===========================
### A guide to the five datasets in this course

[← Course website](https://www.luozijun.com/forecasting-course/) · [About this course](https://www.luozijun.com/forecasting-course/files/about.html)

---

## Why there is a spine, and why there are supplements

Almost everything in this course — lecture examples, lab demonstrations, and homework —
runs on **one** dataset. That is deliberate. When exponential smoothing and an LSTM disagree
in Week 9, you should be able to say the difference came from the **method**. If we switched
datasets every lecture, you could never make that claim: the data would be a confound.

Four other datasets appear during the semester. Each one is here for exactly **one** reason —
something the spine genuinely cannot demonstrate. Noticing *which* dataset a question needs is
itself part of the skill this course is teaching.

| Dataset | Where it appears | Why it is in the course |
|---|---|---|
| **M5 / Walmart** | Throughout — the spine | Five years of daily store-item sales with prices, holidays, and a real store/category hierarchy |
| **FRED** macro series | Lecture 2 | Gives a case where the causal *direction* is defensible — unemployment moves retail spending, not the reverse |
| **Electricity demand** | Lecture 3 | Half-hourly data has daily, weekly *and* yearly cycles at once. Retail has one |
| **Favorita** (Ecuador) | Optional, for final projects | A rich second panel with promotions, holidays and an oil price — and a documented natural experiment |
| **Online store** | Optional, for final projects | A small shop still trading today: no published answer to copy, and real business messiness at a scale you can hold in your head |

---

## 1. M5 / Walmart — the spine

**What it is.** Daily unit sales for 3,049 products across 10 Walmart stores in three US
states, covering **1,941 days** — roughly five and a half years. It was the dataset for the M5
forecasting competition, one of the most studied benchmarks in the field, so when you read a
paper claiming a method works, there is a good chance it was tested on exactly this data.

**Where to get it.** Free, and **no Kaggle account is required**:

- Zenodo mirror: [https://zenodo.org/records/10203108](https://zenodo.org/records/10203108)
- Python package: [`datasetsforecast`](https://nixtlaverse.nixtla.io/datasetsforecast/m5.html) from Nixtla
- Original competition (needs a Kaggle login): [M5 Forecasting — Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy)

We use a prepared subset, not the raw download. Run the prep script once before your first
assignment; it writes the processed files the homework expects.

**What is in it.**

| File | Contents |
|---|---|
| Sales | Daily units sold per item, per store |
| `calendar` | Dates, weekday, month, year, plus event/holiday flags and a SNAP benefits indicator |
| `sell_prices` | Weekly price per item per store |

**The shock it already contains.** On **25 December**, every year, the stores are shut and
recorded sales are **exactly zero** — five occurrences in the sample, against a daily average of
about 2,800 units. The seasonal-naive benchmark predicts roughly 2,300 on those days, so it is
wrong by 100% of the true value on a date known years in advance. The `is_event` flag fires on
**158** days, of which only **5** are closures, so the feature every model in this course relies
on cannot distinguish "the doors are locked" from an ordinary Super Bowl. We use this in Week 16.

**Why it can carry the whole course.**

- **Lecture 1 (ETS).** Aggregated to weekly it gives ~277 observations per series. Holt-Winters
  needs two full 52-week cycles — 104 — to estimate its seasonal indices, so this clears the
  requirement with room to spare.
- **Lecture 2 (ARIMA, ARIMAX).** Long enough to identify an order from ACF and PACF, and
  `sell_prices` gives you a genuine exogenous regressor: price drives demand.
- **Lecture 3 (GAMs, Prophet).** Daily data with an explicit holiday and event calendar — the
  input Prophet was designed around.
- **Lectures 4–6 (trees, boosting, regularization).** A wide tabular feature set: price, promo,
  SNAP, calendar effects, store type, category, and lags. Plenty of correlated predictors,
  which is what makes regularization worth doing.
- **Lectures 7–9 (deep learning).** Thousands of parallel series and nearly 2,000 time steps.
  Neural networks need volume, and this has it.
- **Lecture 11 (hierarchical Bayes).** The hierarchy is real and explicit — state → store, and
  category → department → item. This is what partial pooling is *for*: a store with a short
  history borrows strength from the stores around it.
- **Lecture 12 (Bayesian regression).** Price → demand is a clean causal question with obvious
  confounders, which is exactly what a DAG is for.

---

## 2. FRED macro series — Lecture 2

**What it is.** The Federal Reserve Bank of St. Louis publishes hundreds of thousands of
economic time series, free and without a login. We use a handful of monthly US series: retail
sales, unemployment, the consumer price index, and consumer sentiment.

**Where to get it.** [https://fred.stlouisfed.org](https://fred.stlouisfed.org), or directly
from Python with `pandas_datareader`, which pulls the series by code.

**Why we need it and M5 cannot substitute.** Lecture 2 covers **Vector Autoregression** and
**Granger causality** — methods for several series that influence one another.

M5 can demonstrate the mechanics perfectly well: two Walmart stores in the same state share
regional promotions, weather, and holidays, and their weekly changes correlate at about 0.85.
A Granger test between them comes back strongly significant. You will run exactly that in
Homework 1.

But notice what that case *cannot* settle. Neither store causes the other. Both respond to
something else, and the test cannot tell you so. That is the honest limit of the method, and
it is why we also bring in macroeconomic data — where the causal *direction* is defensible.

Macroeconomic series do genuinely move each other. Unemployment affects retail spending;
retail spending feeds back into sentiment. There is a real transmission mechanism, so a
significant Granger test means something you can reason about — and an insignificant one is
informative rather than inevitable.

---

## 3. Electricity demand — Lecture 3

**What it is.** Half-hourly electricity demand, with temperature as a covariate. We take it
from the [Monash Time Series Forecasting Repository](https://forecastingdata.org/), a free
curated archive of 30 forecasting datasets that requires no login.

**Why we need it and M5 cannot substitute.** Lecture 3 covers GAMs and Prophet, whose real
strength is handling **several seasonal cycles at the same time**. Retail sales have one
meaningful season: the yearly cycle. Electricity demand has three simultaneously —

- a **daily** cycle (demand peaks morning and evening),
- a **weekly** cycle (weekdays differ from weekends),
- a **yearly** cycle (heating and cooling).

On top of that, temperature drives demand non-linearly — demand rises when it is very cold
*and* when it is very hot. That U-shape is precisely what a smooth term in a GAM captures and
a linear model cannot. Retail data would let you fit the method; electricity lets you see why
it exists.

---

## 4. Favorita — optional, for final projects

**What it is.** Daily sales for Corporación Favorita, a large Ecuadorian grocery chain,
covering 2013 to 2017. It includes promotions, store metadata, national holidays, and — because
Ecuador's economy is oil-dependent — a daily **oil price**.

**Where to get it.** [Kaggle: Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/c/favorita-grocery-sales-forecasting).
This one does require a Kaggle account, and you must accept the competition rules on the site
before the download will work.

The raw download is about 4.6 GB and `train.csv` alone is roughly 125 million rows, which is not
a useful starting point. Run the prep script once and work from what it writes:

```bash
kaggle competitions download -c favorita-grocery-sales-forecasting -p data/raw/
unzip 'data/raw/*.zip' -d data/raw/
python scripts/prep_favorita.py
```

It produces two files:

| File | |
|---|---|
| `data/processed/favorita_weekly.csv` | store × product family, weekly, with promotions, holidays, the oil price and store metadata |
| `data/processed/favorita_daily.csv` | one store × family, daily — if you would rather work one series than a panel |

**Store × family, not item level.** Item level is four thousand series, most of them mostly zeros,
which teaches sparsity rather than forecasting. Store × family is the same shape as the M5 panel
you already know, so your methods transfer while the data does not — which is the point of using
it.

**The earthquake is left unflagged.** There is no column marking 16 April 2016. If your project
studies that break, find it in the series.

**What makes it interesting.** On **16 April 2016**, a magnitude 7.8 earthquake struck Ecuador.
Sales spiked for weeks afterward as people bought water and relief supplies, and the competition
organizers documented it in the data description, so it is not something you have to
reverse-engineer.

A forecast made on 15 April 2016 — by exponential smoothing, by ARIMA, by XGBoost, by an LSTM,
by anything in this course — is catastrophically wrong for the following month. Not because the
model was badly specified, but because the future stopped resembling the past, and every method
here assumes it does not. If your project wants to study a genuine structural break rather than
simulate one, this is the cleanest one available.

*(We make the related point in Week 16 without leaving the spine dataset — see the Christmas
closures in §1, where a **known, dated** event defeats every model in the course. Favorita's
earthquake is the harder case: an event nobody could have known about at all.)*

**For final projects.** Favorita is also a good project dataset if your group wants one
ready-made: it is rich enough for serious work and different enough from the spine that you
cannot reuse lecture code unchanged — which is rather the point.

---

## 5. An online store — optional, for final projects

**What it is.** Weekly sales from a single small online retailer, **454 weeks from 1 January
2018 to 7 September 2026**. The shop is real, independently owned, and still trading. Its order
history was shared for this course.

The store is not named here, and nothing in the file identifies it, its products, or anyone who
bought from it. The file is weekly totals and nothing else.

**Where to get it.** Nowhere — it is already in your repository, at
`data/processed/onlinestore_weekly.csv`. There is no download and no prep script.

| Column | |
|---|---|
| `week_start` | the Monday the week begins |
| `orders` | orders placed that week |
| `orders_intl` | how many of those came from outside the US |
| `units` | items across those orders |
| `revenue_index` | revenue, indexed (see below) |
| `revenue_index_intl` | the part of that revenue from outside the US |
| `merch_index`, `shipping_index` | the same revenue split into goods and shipping |

**The international columns are parts, not separate series.** Domestic is the subtraction:
`revenue_index - revenue_index_intl`, and likewise for orders. Everything is on one index base,
so the two halves add back to the total exactly. The US is 63% of orders and international is
37% of revenue — the export orders are worth more each — spread across **55 countries** led by
Canada, Australia and the UK. That is a grab-bag, not one export market, so treat it as two
series at most rather than a per-country panel.

**Revenue is an index, not dollars.** November 2025 = 100, and every other week is relative to
it. A week at 50 did half that month's average business; a week at 200, twice. The store's actual
takings are its owner's business, and a forecast does not need them: growth, volatility,
seasonality and every error metric behave identically under rescaling. Your RMSE will simply be
in index points.

**Why it is worth your time, and what it will cost you.** Every other dataset in this course is
famous. Hundreds of people have published models of M5; you can look up what works. Nobody has
ever modelled this series. There is no answer to check yourself against, which is the situation
you will actually be in at work.

It is also **small and awkward in ways the big public datasets are not**: a few orders in a
typical week, weeks with none at all, and a level that does not sit still across eight years.
Methods that shine on Walmart's thousands of units a day have less to work with here. Deciding
what is signal, what is noise, and what should not be modelled at all is most of the work — and
it is the part the famous datasets let you skip.

**Nothing in this series is flagged for you.** As with Favorita's earthquake, if something in
the history is not ordinary demand, the data will not say so. Finding it, deciding what it is,
and defending how you handled it is the project.

**One thing you have that you will not usually have:** the owner is reachable through me. Ask me
the business questions the data cannot answer — what changed in a given year, how the store
sells, what a quiet week looks like from the inside. Treat me as the client; that is what a
client is for, and it is how the work is actually done.

---

## Getting set up

1. Install the Python packages listed in the syllabus.
2. Run the data preparation script once, before your first assignment. It downloads and
   subsets M5 and writes the processed files that the homework expects. (The online-store
   file in §5 is committed to the repository already — nothing to run.)
3. Confirm the output files exist before starting Homework 1.

If any step fails, email me *before* the assignment is due rather than the night it is due.

---

*This document accompanies the course syllabus. If a dataset detail here and the syllabus ever
disagree, the syllabus governs on deadlines and grading; this document governs on data.*
