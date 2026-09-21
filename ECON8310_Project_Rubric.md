# ECON 8310 Final Project — Grading Rubric
### 100 points · 25% of the course grade

[← Course website](https://www.luozijun.com/forecasting-course/) · [About this course](https://www.luozijun.com/forecasting-course/files/about.html)

Groups of no more than three. Four graded deliverables, broken down here. Read this before the proposal, not after — the proposal is worth 20 points and most
of what it is graded on is decided before you write any code.

| Deliverable | Due | Points |
|---|---|---|
| Group formation | Week 4 | 5 |
| Project proposal | Week 9 | 20 |
| Project presentation | Week 16 | 25 |
| Final report | after the presentations | 50 |

*The week numbers above match the course schedule. Calendar dates for your term come from your
course calendar — they are not the same in every offering.*

---

## 1. Group Formation — 5 points

| Points | Standard |
|---|---|
| 5 | Group of 1–3 submitted by the deadline, with a named contact and a one-line statement of the intended dataset or domain. |
| 3 | Late, or missing the dataset/domain line. |
| 0 | Not submitted; I assign you to a group. |

This is free credit for meeting a deadline. Take it.

---

## 2. Project Proposal — 20 points

One to two pages. The purpose is to catch a doomed project while there is still time to change it.

| Criterion | Pts | What earns full marks |
|---|---|---|
| **Business question** | 5 | A specific, decidable question tied to an action someone would take — "forecast weekly demand for category X to set reorder points," not "predict sales." States who the decision-maker is. |
| **Dataset** | 5 | Source named and accessible; at least 100 time periods; frequency stated; you have actually loaded it. Known limitations named honestly. |
| **Planned methods** | 5 | At least three methods from two or more parts of the course, plus the benchmark you will use. A sentence on *why those* suit this series — seasonality, sample size, exogenous drivers. |
| **Preliminary EDA** | 5 | At least one plot of the target over time, plus a stated observation about trend, seasonality, or breaks that will shape the modeling. |

**Common ways to lose points here:** a question that no decision follows from; a dataset you have
not yet opened; three methods from a single part of the course; no benchmark named.

### Choosing a dataset

**Bring your own if you have one.** A series you care about — from work, from a public agency, from
a hobby — almost always produces a better project, because you can tell when an answer is
implausible and a stranger's dataset gives you no such instinct.

**If you do not have one, two are ready for you.**

**The online store.** Weekly sales from a small online retailer that is still trading —
454 weeks, 2018 to 2026, already in your repository at
`data/processed/onlinestore_weekly.csv`. No account, no download. Revenue is an index
(November 2025 = 100) rather than dollars, which changes nothing about forecasting it. Two
things make it worth choosing: **nobody has ever modelled it**, so there is no published answer
to lean on, and **the owner is reachable through me** — you can ask the business questions the
data cannot answer, which is what you would do on a real engagement. It is small and lumpy, and
deciding what should not be modelled at all is part of the work. Full description in
[the datasets guide](https://www.luozijun.com/forecasting-course/files/datasets.html).

**Favorita.** Daily sales for Corporación Favorita, a large Ecuadorian
grocery chain, 2013–2017, with promotions, store metadata, national holidays and a daily oil price:
[Kaggle: Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/c/favorita-grocery-sales-forecasting)
(a Kaggle account is required). It is rich enough for serious work and different enough from the
M5 spine that you cannot reuse lecture code unchanged — which is the point. It also contains a
documented natural experiment: the magnitude 7.8 earthquake of **16 April 2016**, after which sales
spiked for weeks. A group that wants to study a genuine structural break rather than simulate one
has a clean case waiting.

**What you may not use:** the M5 weekly panel exactly as the homework uses it. Reframing it — a
different aggregation, a different question, a different horizon — is fine and sometimes strong.
Re-running Homework 4 is not a project.

---

## 3. Project Presentation — 25 points

Roughly 10–12 minutes plus questions. Every group member presents some part. The format —
in person, live online, or recorded — differs by section; your instructor sets it.

| Criterion | Pts | What earns full marks |
|---|---|---|
| **Problem framing** | 5 | The audience understands the business question and why it matters within the first two minutes. |
| **Method and validation** | 8 | What you fit, how you compared, and evidence that the comparison was time-aware. The benchmark appears on the same axis as the models. |
| **Results and interpretation** | 7 | Findings stated in business units with uncertainty attached. You say what the number means for the decision, not only what it is. |
| **Delivery and Q&A** | 5 | Within time, legible slides, and answers that engage the question asked. "We didn't test that" is a fine answer; inventing one is not. |

---

## 4. Final Report — 50 points

A rendered Quarto HTML file (`.qmd` + `.html`), submitted through the course repository. All code,
outputs, and written analysis in one self-contained document. No page limit; a well-organized
15–20 pages beats a disorganized 40.

| Criterion | Pts | What earns full marks |
|---|---|---|
| **Methods implemented** | 12 | Three or more methods, drawn from **at least two** of the four parts of the course, each correctly specified and actually fitted. Three variants of one family does not satisfy this. |
| **Benchmark** | 6 | A benchmark computed on your own data — seasonal naive or another one-line rule — reported alongside every model. A report without one cannot show that any of its work was worth doing, and caps at 6 lost points here plus the credibility cost throughout. |
| **Walk-forward validation** | 8 | Time-aware validation used for **all** model comparisons. No random splits, no `KFold`, no scaling fitted outside the fold, no unshifted rolling features. |
| **Business recommendation** | 10 | Written for a non-technical decision-maker: actionable, jargon-free, in the units of the business, with uncertainty stated. A reader who skipped the modeling section can act on it. |
| **Methods reflection** | 8 | Which model performed best, and **what about your specific dataset** made it win. Connects the result to a property of the series — sample size, seasonal structure, exogenous drivers, noise level. |
| **Reproducibility and craft** | 6 | Renders top to bottom from a clean restart. Random seeds set. Figures labeled. Sources cited. AI use logged per the course policy. |

### The four parts of the course

| Part | Lectures | Examples |
|---|---|---|
| Classical time series | 1–3 | ETS, ARIMA, VAR, Prophet, pyGAM |
| Trees and regularization | 4–6 | Random forest, XGBoost, LASSO, Elastic Net |
| Deep learning | 7–9 | FFN, 1D CNN, RNN, LSTM, Transformer |
| Bayesian | 10–12 | Structural time series, hierarchical, Bayesian regression |

---

## What Moves a Report from Good to Excellent

None of this is extra credit. It is what separates the top band from the middle one within the
criteria above.

- **A result that went against you, reported plainly.** Tuning that did not help. The model you
  expected to win and did not. Every scoreboard in this course had one, and the assignments were
  built so the obvious answer is frequently wrong. A report where every result confirms the
  author's expectation is not a report anyone should act on.
- **A reason, not just a ranking.** "XGBoost won" is a computation. "XGBoost won because our
  series has strong exogenous drivers and only 180 observations, which is too few for the
  sequence models to earn their parameters" is the work.
- **Evidence you checked the thing that could silently be wrong.** A leakage check, a planted
  noise control, a held-out validation of a claim, a convergence diagnostic actually reported.
- **Honest scope.** Naming what your forecast cannot support is worth more than overclaiming what
  it can.

---

## Academic Integrity and AI Use

The course AI policy applies in full. You may use AI assistants for code. **Analytical
interpretation, the business recommendation, and the methods reflection must be written by you** —
those three are where the grade actually lives, and using AI to write them defeats the purpose of
the assignment.

Group members receive the same grade unless a member's contribution is materially absent, in
which case I will adjust individually after speaking with the group.

---

*Enrolled students: ask about this rubric on the course discussion board, so that everyone sees
the answer.*
