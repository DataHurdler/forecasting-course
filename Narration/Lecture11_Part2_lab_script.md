# Lab 11 Part 2 — Recording Script

**ECON 8310: Business Forecasting · Partial Pooling — and the Question That Comes Before It**

Lab: `Labs/Lecture11_Part2_lab.qmd` (6 steps) · Measured runtime: **~28 minutes** of narration
(the in-room version is budgeted at 45)

---

## How to use this document

- **`▶ STEP n — Title`** matches the lab's own headings exactly.
- *Italic parentheticals* are stage directions. **[pause]** is a beat; **[STOP — learner works]**
  is where you tell the viewer to pause the video and do something.
- `random_seed=42` throughout; six fits, most under 30 seconds. The whole lab runs in about
  three minutes.
- **This is the most important lab in the Bayesian block.** It is the only place in the course
  where a model passes every diagnostic, produces a textbook-looking result, and is wrong.

### Correction (2026-10-13): Step 6's numbers are 43 / 1.003 / 966, as the lab prints

An earlier version of this note (2026-08-31) reported that the centered fit gives **127
divergences, R-hat 1.044, ESS 413**, and the deck was changed to match. **That run does not
reproduce.** The lab's own code with seed 42, on the pinned environment (PyMC 6.3.1, ArviZ 1.3.0),
gives **43 divergences, R-hat 1.003, ESS 966** centered and **0 / 1.003 / 1,719** non-centered —
which is also what the published lab page shows (rendered 2026-09-01).
`scripts/figures/lecture11_part2_figures.py` reproduces it and caches it in
`Figures/Lecture11_Part2/measured.json`. The same goes for the other 2026-08-31 values below (8.2×,
0.049, `mu_b` 0.077, `sd_b` 0.066, 11.5×): the lab prints 8.1×, 0.048, 0.078, 0.065 and 11.9×, and
this script now carries those.

Why two runs of "the same code" can disagree: NUTS trajectories are chaotic in the last bit.
Reading the training rows back through a CSV (a last-digit change in 343 of 1,425 values) moves
seed 42 from 43 divergences to 49. Over ten other seeds the centered form gives 15–285 divergences
and the non-centered form 0–11; R-hat or ESS objected in 2 of those 11 runs, divergences in all 11.
So the deck's claim is now the one every run supports: **R-hat and ESS pass; only the divergences
object.**

### In-room language that needs replacing

Step 2's "before you go on" note, Step 4's and Step 5's "your turn" blocks, and the four closing
**Discuss** blocks.

---

# ▶ OPENING

*(Screen: rendered lab, top of document.)*

Lab 11, Part 2 — and this is the lab I would keep if I could keep only one.

A hierarchical model lets thirty series share information. Each gets its own effect, drawn from a
common distribution that the data also estimates. Data-poor series borrow from data-rich ones, and
nobody has to choose the weights by hand. That is genuinely powerful, and in the second half of
today you will watch it work beautifully.

**[pause]**

In the first half you will watch it produce a confidently wrong answer. Same data. Same code.
Clean diagnostics, textbook shrinkage, a defensible-sounding conclusion — and no warning of any
kind.

The difference between the two runs is not in the code. It is a decision made **before any code
was written**: which series belong in the same hierarchy.

That is the lab. Everything else is scaffolding for it.

---

# ▶ SETUP

*(Screen: run setup.)*

```
30 series | 8,310 rows | 22.1% of weeks are high-SNAP
```

The question: **does a high-SNAP week lift units, and by how much in each store?**

*(Point at `empirical_effect`.)*

And note this helper, because it is what makes the whole lab possible. It computes the SNAP effect
actually observed in a set of weeks — the difference in mean log units between high-SNAP weeks and
the rest — with **no model involved.** Later we will hold weeks back, fit a model without them,
and then use this function to ask what really happened in those weeks.

Most of the time you cannot grade a model this way. Today you can, because we are going to create
the missing data ourselves.

---

# ▶ STEP 1 — Build the imbalance

*(Screen: run the thinning.)*

```
train 5,620 rows | held-out 2,690 rows
10 thin series (8 weeks each), 20 full series
```

Real panels are lopsided — some series have years of history, others are new stores with a handful
of weeks. We manufacture that imbalance: every third series cut down to **8 weeks**.

The weeks we remove are not discarded. They go into `heldout`, the model never sees them, and in
Step 3 they become the answer key.

**[pause]**

Remember the rule: `j % 3 == 0`. Every third series. It looks arbitrary and it is about to matter
enormously.

---

# ▶ STEP 2 — Fit it, and admire the shrinkage

*(Screen: run both fits and the diagnostics.)*

```
Diagnostics
  unpooled       r_hat 1.003  ess  6100  divergences    0
  hierarchical   r_hat 1.003  ess  5077  divergences    0
```

Both models sampled cleanly. R-hat near one, effective sample sizes in the thousands, not one
divergence. If you were checking your work the way Lecture 10 taught you, you would stop here
satisfied.

*(Run the shrinkage table.)*

```
              unpooled spread   after pooling   mean move
  thin (8w)       0.106           0.000          0.111
  full (277w)     0.012           0.001          0.009

  ratio: thin series moved 11.9x further
  mu_b 0.011   sd_b 0.007
```

**[pause]**

And this is textbook. The thin series — 8 weeks each — moved **11.9 times further** toward the
group mean than the full series did. Nobody told the model which series were starved. It worked
that out from how much each one had to say.

*(Point at the shrinkage figure.)*

The picture makes it obvious: the red lines, the thin series, collapse onto the dashed group mean.
The blue lines barely move.

**[STOP — learner works]** — *(replaces the "before you go on" note)*

Pause the video and write one sentence: what would you now tell the business about SNAP weeks?
Keep it where you can see it. Step 3 is going to ask whether it was right.

*(Resume — but plant this before moving on.)*

Before we grade it, look at one number I have not commented on. **`sd_b` is 0.007.**

That is the model's estimate of how much stores differ from one another. Seven thousandths of a
log unit — which is to say, the model has concluded that all thirty of these series respond to
SNAP **essentially identically**.

Hold that up against what you know about the world. Thirty series spanning three states, three
product categories, stores of different sizes. Is "they are all the same" plausible?

It is not a diagnostic. Nothing flags it. But it is a **claim about the world**, and claims about
the world can be sanity-checked in a way that R-hat cannot.

---

# ▶ STEP 3 — Grade it against the weeks it never saw

*(Screen: run the grading table.)*

```
    series  unpooled  hierarchical  held_out_truth
CA_1_FOODS     0.298         0.011           0.087
CA_2_FOODS     0.014         0.011           0.011
CA_3_FOODS     0.037         0.011           0.105
CA_4_FOODS     0.126         0.011           0.039
TX_1_FOODS     0.092         0.011           0.082
TX_2_FOODS     0.172         0.011           0.067
TX_3_FOODS     0.133         0.011           0.074
WI_1_FOODS    -0.016         0.011           0.033
WI_2_FOODS     0.292         0.011           0.209
WI_3_FOODS     0.017         0.011           0.177

  RMSE vs truth:  unpooled 0.103   hierarchical 0.098
  mean held-out effect across these series: 0.089
```

**[pause]** — *let the table sit. Do not talk over it. This is the moment of the lab.*

Read the middle column first. **0.011, ten times.** The hierarchical model reported essentially the
same effect for every one of these ten series.

Now read the last column. The held-out weeks — hundreds of them per series, weeks the model never
saw — say the true effects average **0.089**, and range from 0.011 up to 0.209.

The model was not merely imprecise. It moved all ten series to a number roughly **eight times too
small**, and reported tight intervals around it.

**[pause]**

And look at the RMSE line, because it takes away the last consolation. Unpooled 0.103,
hierarchical 0.098. All of that shrinkage — the beautiful collapsing lines, the 11.9× ratio —
bought a **five percent** improvement over just fitting each series separately.

*(Screen: the callout.)*

Now the sentence that matters most in this lecture. **The diagnostics could never have caught
this.** R-hat, ESS and divergences ask whether the sampler explored the posterior of *the model
you wrote*. They cannot ask whether that was the right model to write. Every diagnostic passed,
and every diagnostic was answering a different question from the one you cared about.

---

# ▶ STEP 4 — Find out what went wrong

*(Screen: print the thinned series.)*

Two things went wrong, and they compound.

```
thinned series: ['CA_1_FOODS', 'CA_2_FOODS', 'CA_3_FOODS', 'CA_4_FOODS', 'TX_1_FOODS',
                 'TX_2_FOODS', 'TX_3_FOODS', 'WI_1_FOODS', 'WI_2_FOODS', 'WI_3_FOODS']
```

**[pause]**

Every thinned series is a **FOODS** series. All ten. That is not a coincidence and it is not bad
luck — the panel is ordered store-major with three categories each, so `j % 3 == 0` selects the
same category every time.

*(Run the category table.)*

```
Empirical high-SNAP effect by category (all data, no model):
  FOODS      n=10  mean +0.089  sd 0.058
  HOBBIES    n=10  mean +0.006  sd 0.010
  HOUSEHOLD  n=10  mean +0.011  sd 0.014
```

And there it is. **SNAP is a food assistance program.** FOODS responds at +0.089. HOBBIES and
HOUSEHOLD are at +0.006 and +0.011 — indistinguishable from nothing.

So put the two together, because neither alone is fatal.

We told the model that thirty series were **thirty interchangeable draws from one distribution.**
They are not: twenty of them have no effect and ten do. The group mean therefore landed near zero,
dominated by the twenty. And then — the second error — we starved **precisely the ten series that
had something to say**, so those ten had almost no data with which to resist being pulled toward a
mean built from series unlike them.

That is why `sd_b` came out at 0.007. Averaged across a group where two thirds genuinely have no
effect, "stores are all the same" is the best summary of a badly-posed question.

*(Screen: the exchangeability warning box.)*

**[STOP — learner works]** — *(replaces the "your turn" string)*

Pause and answer the question in the code cell: **would a bigger sample have fixed this?**

*(Resume.)*

No — and this is worth being clear about, because "get more data" is the reflex. More weeks per
series would sharpen each estimate, and the hierarchy would still be pulling FOODS series toward a
mean computed from HOBBIES and HOUSEHOLD. The error is in the *structure* we imposed, and no
quantity of data corrects a structural mistake. It just makes you more confident about it.

**Exchangeability is a modeling assumption and it is yours to defend.** Groups are exchangeable
when, before seeing the data, you have no reason to expect one to differ from another in a
particular direction. Ten FOODS series across ten stores: plausible. FOODS against HOBBIES: you
knew the answer before you looked.

Nothing in PyMC checks this. No diagnostic reports it. It is subject-matter knowledge, applied
before the model is written.

---

# ▶ STEP 5 — Do it on the exchangeable set

*(Screen: rebuild on FOODS only, then run the fits.)*

```
10 FOODS series | 5 thinned to 8 weeks
train 1,425 rows | held-out 1,345 rows
```

Same code. Same priors. Same thinning trick — every other series cut to 8 weeks. The **only**
change is which series are allowed into the hierarchy.

```
       mean     sd hdi94_lb hdi94_ub ess_bulk ess_tail r_hat
mu_b  0.078  0.035    0.017     0.15     1544     1588  1.00
sd_b  0.065  0.038   0.0097     0.16     1027     1153  1.00
```

**[pause]** — *the comparison to make out loud.*

`mu_b` is **0.078** — a real SNAP effect, where before it was 0.011. And `sd_b` is **0.065**, where
before it was 0.007. Nine times larger.

That second number is the interesting one. The model now says stores genuinely differ from one
another, because on a defensible group they do. The earlier 0.007 was not a finding about stores;
it was an artifact of asking about the wrong group.

*(Run the shrinkage table.)*

```
              unpooled spread   after pooling   mean move
  thin (8w)       0.108           0.010          0.120
  full (277w)     0.055           0.037          0.015

  ratio: thin moved 8.1x further
```

Compare the middle column with Step 2's. There, **everything** collapsed — thin spread went to
0.000, full to 0.001. Here the thin series collapse to 0.010 but the full series **keep most of
their spread**: 0.055 down to 0.037, about two-thirds retained.

That is what partial pooling is supposed to look like. The model discarded variation that 8 weeks
could not support, and kept variation that 277 weeks could. In Step 2 it discarded everything,
which should have been the tell.

*(Run the grading table.)*

```
    series  unpooled  hierarchical  held_out_truth
CA_1_FOODS     0.286         0.097           0.087
CA_3_FOODS     0.105         0.082           0.102
TX_1_FOODS     0.052         0.075           0.082
TX_3_FOODS     0.219         0.091           0.072
WI_2_FOODS     0.338         0.103           0.207

  RMSE vs truth:  unpooled 0.126   hierarchical 0.048   (2.6x better)
```

**[pause]**

Now read the middle column against the last one. 0.097 against 0.087. 0.082 against 0.102. 0.091
against 0.072. These are *close*, and each one came from a series with **eight weeks of data.**

RMSE against the truth: 0.126 unpooled, **0.048** hierarchical. An error 2.6 times smaller.

**That is borrowing strength, and now it is real.** Same code, same priors, same diagnostics — and
the thin series are estimated well because the other five exist. The only thing that changed was
which series were allowed into the hierarchy.

**[STOP — learner works]**

Fill in the comparison in the last cell: what percentage did pooling improve RMSE in Step 3, and
what percentage here? *(Roughly 5% there against 61% here.)*

---

# ▶ STEP 6 — The trap in the code

*(Screen: the centered form, then run.)*

One more failure, and this one **is** mechanical rather than conceptual.

The natural way to write a hierarchy reads exactly like the equation:
`b = pm.Normal("b", mu_b, sd_b, shape=J)`. Everybody writes it that way first. It is called the
centered form.

```
Diagnostics
  centered       r_hat 1.003  ess   966  divergences   43
  non-centered   r_hat 1.003  ess  1719  divergences    0
```

**[pause]**

Same model. Same data. The same posterior, if you could sample it — but the centered form makes the
sampler walk a **funnel**: when `sd_b` is small the individual `b_j` must crowd into a narrow
neck, and a sampler tuned for the wide part of the funnel cannot get into the neck.

*(Screen: the callout.)*

Now read the three numbers, because only one of them objects — and that is the lesson.

**ESS passes.** 966, more than twice the threshold. You would not look twice.

**R-hat passes** — 1.003 against 1.01. On those two numbers alone, the two Lecture 10 taught you to
check, you would have shipped this model.

**43 divergences against a threshold of exactly zero.** There is no version of "close to zero"
that gets you to forty-three. It admits no conversation.

*(Screen: the second paragraph of the callout.)*

If your number differs from this one, that is expected: divergence counts move with the seed and
even the machine. Over ten other seeds we saw 15 to 285 centered and 0 to 11 non-centered. What
held every time was the pattern — the centered form diverged on all eleven runs, and R-hat or ESS
caught it on only two.

So the rule: on hierarchical models, **divergences fire first and least ambiguously.** Do not wait
for R-hat to decide for you.

---

# ▶ BEFORE YOU LEAVE

**[STOP — learner works]** — *(replaces the in-room "Discuss")*

Four questions for the board.

**The failure.** Step 2 had clean diagnostics, textbook shrinkage and a defensible conclusion.
Step 3 showed it was wrong. What kind of check caught it, and why could no amount of sampler
diagnostics have done so?

**[pause]** — *the answer to land: held-out data. Diagnostics audit the computation; only data the
model never saw can audit the model.*

**The assumption.** You are asked to pool weekly demand across the ten stores in one chain. Name
one variable that, if it differed across those stores, would break exchangeability the way
category did here — and say how you would handle it **without** abandoning the hierarchy.

**[pause]** — *the good answer is a level the hierarchy can absorb: put the offending variable in
as a predictor, or nest the hierarchy — stores within regions — rather than giving up on pooling.*

**The judgment.** In Step 5, `sd_b` came out around 0.065, so stores genuinely differ. The
unpooled fit put the thin stores' range at roughly 5% to 40%. Given both numbers, what do you actually
recommend the business do for the next SNAP week?

**The thread.** This is the sixth time this semester a model improved by pulling estimates toward
a common value. Name the other five, and say what is different about the shrinkage strength here.

**[pause]** — *ridge, LASSO, the bagged ensemble, the random forest, boosting's learning rate —
and the difference is that here the strength is* estimated from the data *rather than chosen by
you and tuned on a validation set. `sd_b` is the shrinkage parameter, and the model infers it.*

Solutions for the coded parts go up on Canvas after the deadline. Next week: Bayesian linear
regression — and a preprocessing step you have used all semester that silently destroys the
answer.

---

# Appendix — expected output

`random_seed=42` throughout; six fits, whole lab ≈ 3 minutes.

| Quantity | Value |
|---|---|
| Panel | 30 series · 8,310 rows · **22.1%** high-SNAP weeks |
| Split | train 5,620 · held-out 2,690 · 10 thin (8 wks) + 20 full |
| **Step 2** diagnostics | unpooled 1.003 / 6,100 / 0 · hierarchical 1.003 / 5,077 / **0** — all clean |
| Step 2 shrinkage | thin 0.106 → **0.000** (move 0.111) · full 0.012 → 0.001 (move 0.009) · ratio **11.9×** |
| Step 2 hierarchy | `mu_b` **0.011** · `sd_b` **0.007** ← implausibly small; the substantive tell |
| **Step 3** grading | hierarchical reports **0.011 for all ten**; held-out truth averages **0.089** |
| Step 3 RMSE | unpooled 0.103 · hierarchical 0.098 — pooling bought **~5%** |
| **Step 4** by category | FOODS **+0.089** (sd 0.058) · HOBBIES +0.006 · HOUSEHOLD +0.011 |
| Step 4 thinning | `j % 3 == 0` selects **all ten FOODS series** |
| **Step 5** hierarchy | `mu_b` **0.078** [0.02, 0.15] · `sd_b` **0.065** — 9× Step 2's |
| Step 5 shrinkage | thin 0.108 → 0.010 · full 0.055 → **0.037** (two-thirds retained) · ratio **8.1×** |
| Step 5 grading | unpooled 0.126 · hierarchical **0.048** — **2.6× better** |
| **Step 6** centered | R-hat 1.003 · ESS 966 · divergences **43** (seed-dependent: 15–285 over ten other seeds) |
| Step 6 non-centered | R-hat 1.003 · ESS 1,719 · divergences **0** (0–11 over ten other seeds) |

**Three things to know before recording.**

*Step 3 is the whole lab; give the table silence.* Ten identical 0.011s beside a truth column
ranging 0.011 to 0.209 is the single most persuasive exhibit in the Bayesian block. Do not narrate
over it.

*Plant `sd_b = 0.007` at the end of Step 2, before the grading.* It is the one clue available
*before* the answer key, and it is not a diagnostic — it is a claim about the world ("all thirty
stores respond identically") that a modeller can reject on subject-matter grounds. Students who
learn to read it have learned something no threshold gives them.

*Do not let "get more data" stand as the answer to Step 4.* The error is structural: more weeks
would sharpen estimates while the hierarchy went on pulling FOODS toward a mean built from
HOBBIES. More data makes a structural mistake more confident, not less wrong.
