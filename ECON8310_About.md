# About This Course

**ECON 8310 · Business Forecasting**

This page holds what the course *is* — what it teaches, what it assumes you already know, and
which books it reads from. It is deliberately free of anything specific to one university, one
term or one section: no dates, no grading scale, no room, no policies. Those live on Canvas for
enrolled students.

Everything else on this site — the slides, the labs, the assignments, the book and the workbook —
is the course material itself, and is the same in every offering.

---

## What this course covers

The course develops practical forecasting skills for graduate students in business, economics and
data science. It works through a broad suite of methods — from classical time series models to
machine learning to Bayesian inference — applied to real business forecasting problems. Three
things are emphasized throughout: implementation in Python, interpretation of model output for
business decisions, and honest comparison of methods against each other and against a benchmark.

The first half covers traditional econometric forecasting; the second half turns to predictive
analytics and machine learning.

**On AI.** This course is designed to fully integrate modern AI tools, in both teaching and
learning. The instructor uses AI a lot, says so when he has, and asks you to do the same. The
[AI use policy](https://www.luozijun.com/forecasting-course/files/ai-policy.html) sets out what that means in practice: any assistant is fine, every
assignment carries a prompt budget and a prompt log, and the written interpretation is yours.

---

## What you will be able to do

By the end of the course you will be able to:

1. Apply exponential smoothing, ARIMA and VAR models to univariate and multivariate business time
   series, and evaluate their forecast accuracy using walk-forward cross-validation.
2. Build Generalized Additive Models and use Prophet to decompose trend, seasonality and external
   regressors.
3. Engineer features from time series data and apply decision trees, random forests and gradient
   boosting to tabular forecasting problems.
4. Design, train and evaluate feedforward networks, CNNs, LSTMs and Transformer encoders for
   sequence forecasting using PyTorch.
5. Apply Bayesian inference to specify priors, compute posteriors and generate calibrated
   probabilistic forecasts using PyMC.
6. Communicate model results and business recommendations to both technical and non-technical
   audiences.

*(These outcomes were drafted with help from Claude.)*

---

## What it assumes you know

Familiarity with **basic statistics** — regression and hypothesis testing — and **introductory
programming**.

**No prior knowledge of machine learning or time series analysis is required.** The course starts
from the beginning on both.

---

## The books

All are free online. Nothing needs to be purchased.

### The main text, in two editions

The primary textbook exists in both an R and a Python version. The Python version is newer and has
two extra chapters. You are *highly encouraged* to use Python — many basic forecasting tasks can be
done in R if that is what you already know, but when the course reaches the more advanced
techniques, Python is the only way through.

- ***Forecasting: Principles and Practice, the Pythonic Way*** (**FPP-Py**) —
  [otexts.com/fpppy](https://otexts.com/fpppy/)
- ***Forecasting: Principles and Practice (3rd ed.)*** (**FPP3**, the R edition) —
  [otexts.com/fpp3](https://otexts.com/fpp3/)

**How the two editions line up.** Chapters 1 through 13 carry the same numbers and titles in both,
so for the first two-thirds of the course either edition works. Two differences matter:

- **Chapter 12 is numbered differently.** In FPP-Py, §12.4 is *Bootstrapping and bagging*. In FPP3,
  §12.4 is *Neural network models* and bootstrapping/bagging is §12.5. Every Chapter 12 assignment
  in the schedule gives both numbers.
- **Chapters 14 and 15 exist only in the Python edition** — *Neural networks* and *Foundation
  forecasting models*. These are exactly the chapters the second half of the course reads. There is
  no R equivalent.

That is the practical reason for the Python recommendation. Through Lecture 6 the choice of
language is yours; **from Lecture 7 onward the readings exist only in Python**, and so does all of
the course code.

### Two gaps FPP cannot fill

FPP contains **no tree-based methods and no regularization**. Lectures 4 through 6 use:

- ***An Introduction to Statistical Learning with Applications in Python*** (**ISLP**) —
  [statlearning.com](https://www.statlearning.com/)

ISLP is by four of the same statisticians behind the standard graduate reference in this area, it
is written at this level, and every chapter ends with a Python lab. Chapter 8 covers decision
trees, bagging, random forests and boosting together; Chapter 6 covers regularization. That is
Lectures 4 through 6 almost exactly.

Neither book covers **Bayesian methods**. Lectures 10 through 12 use:

- ***Bayesian Modeling and Computation in Python*** (**BMCP**), by Martin, Kumar and Lao —
  [bayesiancomputationbook.com](https://bayesiancomputationbook.com/)

BMCP is written by PyMC core developers, and every example is PyMC and ArviZ code — the exact
libraries this course uses. It is the closest thing to a purpose-built text for the last three
lectures: Chapter 1 covers priors and prior predictive checks, Chapter 2 the MCMC diagnostics you
will be asked to report, §4.5–4.6 pooling and hierarchical models, and §6.4.3 is *Bayesian
Structural Time Series* by name.

**A note on level.** BMCP is mathematically heavier than FPP in places, and **you are not
responsible for the derivations**. Read the assigned sections for the ideas and the code, run the
examples, and treat the rest as reference. The slides define what you are accountable for; BMCP
shows you the same ideas in working PyMC.

### Three further free resources

- **XGBoost, "Introduction to Boosted Trees"** —
  [xgboost.readthedocs.io](https://xgboost.readthedocs.io/en/stable/tutorials/model.html). Derives
  the regularized objective and the split-gain formula covered in Lecture 5.
- **Molnar, *Interpretable Machine Learning*** —
  [christophm.github.io](https://christophm.github.io/interpretable-ml-book/). Ch. 9 (decision
  trees), Ch. 19 (partial dependence), Ch. 23 (permutation feature importance). Useful for Lectures
  3 through 5.
- **McElreath, *Statistical Rethinking* (lecture series)** — free on YouTube. BMCP does not cover
  causal graphs, so this is where to go further on the **DAGs** in Lecture 12. It is the standard
  treatment.

Per-week readings are in the **weekly schedule** on the [course home page](https://www.luozijun.com/forecasting-course/).

---

## On difficulty and pacing

This course covers a wide range of methods — classical statistics, deep learning, Bayesian
inference. Some weeks will be more familiar than others depending on your background, and which
weeks those are differs from person to person. Work with each other so you can complement each
other.

**The methods build on each other.** Falling behind in one week makes the next harder, and the
effect compounds. Use office hours early rather than after you are stuck.

---

## How the material is organized

| | |
|---|---|
| **Slides** | One deck per lecture, published as interactive HTML rather than PDF — HTML carries real headings, alt text, table headers and readable maths |
| **Labs** | One executable lab per lecture except the synthesis, designed to be run alongside the deck. Each carries a **suggested time** — that is the classroom pace, not a target. Working through one on your own reasonably takes longer, and taking longer is not a sign you are behind |
| **Assignments** | Eleven submissions across seven assignments, each with its own prompt budget |
| **[The book](https://www.luozijun.com/forecasting-course/book/index.html)** | Every lecture as continuous prose — the slides plus what was said about them |
| **[The workbook](https://www.luozijun.com/forecasting-course/workbook/index.html)** | Every lab and assignment in one searchable volume, code shown rather than run |

The weekly schedule numbers the **teaching weeks** in order. It carries no calendar dates,
because those belong to a particular term and section; enrolled students get the calendar,
including any break weeks, from Canvas.
