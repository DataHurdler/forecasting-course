# Lab 11 Part 2 — Recording Script

**ECON 8310: Business Forecasting · Partial Pooling — and the Question That Comes Before It**

Lab: `Labs/Lecture11_Part2_lab.qmd` (6 steps) · Measured runtime: **~28 minutes** of narration
(the in-room version is budgeted at 45)

---

## How to use this document

- **`▶ STEP n — Title`** matches the lab's own headings exactly.
- *Italic parentheticals* are stage directions. **[pause]** is a beat; **[STOP — learner works]**
  is where you tell the viewer to pause the video and do something.
- `random_seed=42` throughout; five fits, most under 30 seconds. The whole lab runs in about
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

*(Show the rendered lab, at the top of the document.)*

This is Lab 11, Part 2, and if I could keep only one lab from this course, it would be this one.

A hierarchical model lets thirty series share information. Each series gets its own effect, drawn
from a common distribution that the data also estimates. Series that are poor in data borrow from
series that are rich in it, and nobody has to choose the weights by hand. That's genuinely
powerful, and in the second half of today, you'll watch it work beautifully.

**[pause]**

In the first half, you'll watch it produce a confidently wrong answer. It's the same data and the
same code, with clean diagnostics, textbook shrinkage and a conclusion that sounds defensible, and
there's no warning of any kind.

The difference between the two runs isn't in the code. It's a decision made **before any code was
written**, about which series belong in the same hierarchy.

That's the lab, and everything else is scaffolding for it.

---

# ▶ SETUP

*(Run the setup cell.)*

```
30 series | 8,310 rows | 22.1% of weeks are high-SNAP
```

The question is this: **does a high-SNAP week lift units, and by how much in each store?**

*(Point at `empirical_effect`.)*

Take note of this helper, because it's what makes the whole lab possible. It computes the SNAP
effect actually observed in a set of weeks, which is the difference in mean log units between
high-SNAP weeks and the rest, with **no model involved.** Later we'll hold some weeks back, fit a
model without them, and then use this function to ask what really happened in those weeks.

Most of the time, you can't grade a model this way. Today you can, because we're going to create
the missing data ourselves.

---

# ▶ STEP 1 — Build the imbalance

*(Run the thinning cell.)*

```
train 5,620 rows | held-out 2,690 rows
10 thin series (8 weeks each), 20 full series
```

Real panels are lopsided. Some series have years of history, and others are new stores with a
handful of weeks. So we manufacture that imbalance: every third series gets cut down to **eight
weeks**.

The weeks we remove aren't thrown away. They go into `heldout`, the model never sees them, and in
Step 3 they become the answer key.

**[pause]**

Remember the rule in the code: j modulo three equals zero, which picks every third series. It
looks arbitrary, and it's about to matter enormously.

---

# ▶ STEP 2 — Fit it, and admire the shrinkage

*(Run both fits and the diagnostics.)*

```
Diagnostics
  unpooled       r_hat 1.003  ess  6100  divergences    0
  hierarchical   r_hat 1.003  ess  5077  divergences    0
```

Both models sampled cleanly. R-hat is near one, the effective sample sizes are in the thousands,
and there isn't a single divergence. If you were checking your work the way Lecture 10 taught you,
two weeks ago, you'd stop here, satisfied.

*(Run the shrinkage table.)*

```
              unpooled spread   after pooling   mean move
  thin (8w)       0.106           0.000          0.111
  full (277w)     0.012           0.001          0.009

  ratio: thin series moved 11.9x further
  mu_b 0.011   sd_b 0.007
```

**[pause]**

And this is textbook. The thin series, with eight weeks each, moved **eleven point nine times
further** toward the group mean than the full series did. Nobody told the model which series were
starved of data. It worked that out from how much each one had to say.

*(Point at the shrinkage figure.)*

The picture makes it obvious. The red lines, which are the thin series, collapse onto the dashed
group mean, and the blue lines barely move.

**[STOP — learner works]** — *(replaces the "before you go on" note)*

Pause the video, and write one sentence: what would you now tell the business about SNAP weeks?
Keep it where you can see it, because Step 3 is going to ask whether it was right.

*(Resume, but plant this before moving on.)*

Before we grade it, look at one number I haven't commented on yet. **`sd_b` is zero point zero
zero seven.**

That's the model's estimate of how much stores differ from one another. Seven thousandths of a log
unit means the model has concluded that all thirty of these series respond to SNAP in **essentially
the same way**.

Hold that up against what you know about the world. These are thirty series spanning three
states, three product categories and stores of different sizes. Is "they're all the same"
plausible?

It isn't a diagnostic, and nothing flags it. But it is a **claim about the world**, and you can
check a claim about the world against what you know, which is something you can never do with
R-hat.

---

# ▶ STEP 3 — Grade it against the weeks it never saw

*(Run the grading table.)*

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

*(Let the table sit, and don't talk over it. This is the moment of the lab.)*

**[pause]**

Read the middle column first. It says **zero point zero one one, ten times over.** The hierarchical
model reported essentially the same effect for every one of these ten series.

Now read the last column. The held-out weeks, hundreds of them per series, weeks the model never
saw, say that the true effects average **zero point zero eight nine**, and that they range from
zero point zero one one up to zero point two zero nine.

So the model wasn't merely imprecise. It moved all ten series to a number roughly **eight times too
small**, and it reported tight intervals around it.

**[pause]**

Now look at the RMSE line, because it takes away the last consolation. It's zero point one zero
three unpooled and zero point zero nine eight hierarchical. All of that shrinkage, the beautiful
collapsing lines and the ratio of eleven point nine, bought a **five percent** improvement over
simply fitting each series on its own.

*(Show the callout.)*

Now here's the sentence that matters most in this lab. **The diagnostics could never have caught
this.** R-hat, the effective sample size and the divergences all ask whether the sampler explored
the posterior of *the model you wrote*. They can't ask whether that was the right model to write.
Every diagnostic passed, and every one of them was answering a different question from the one you
cared about.

---

# ▶ STEP 4 — Find out what went wrong

*(Print the thinned series.)*

Two things went wrong, and they compound.

```
thinned series: ['CA_1_FOODS', 'CA_2_FOODS', 'CA_3_FOODS', 'CA_4_FOODS', 'TX_1_FOODS',
                 'TX_2_FOODS', 'TX_3_FOODS', 'WI_1_FOODS', 'WI_2_FOODS', 'WI_3_FOODS']
```

**[pause]**

Every thinned series is a **FOODS** series, all ten of them. That isn't a coincidence, and it isn't
bad luck. The panel is ordered store by store, with three categories in each store, so taking
every third series selects the same category every time.

*(Run the category table.)*

```
Empirical high-SNAP effect by category (all data, no model):
  FOODS      n=10  mean +0.089  sd 0.058
  HOBBIES    n=10  mean +0.006  sd 0.010
  HOUSEHOLD  n=10  mean +0.011  sd 0.014
```

And there it is. **SNAP is a food assistance program.** FOODS responds, at plus zero point zero
eight nine. HOBBIES and HOUSEHOLD sit at plus zero point zero zero six and plus zero point zero one
one, which is indistinguishable from nothing.

So put the two together, because neither one alone is fatal.

We told the model that thirty series were **thirty interchangeable draws from one distribution.**
They aren't. Twenty of them have no effect, and ten do. So the group mean landed near zero,
dominated by the twenty. Then came the second error. We starved **precisely the ten series that
had something to say**, so those ten had almost no data with which to resist being pulled toward a
mean built from series unlike them.

That's why `sd_b` came out at zero point zero zero seven. Averaged across a group where two-thirds
of the series genuinely have no effect, "the stores are all the same" is the best summary of a
badly posed question.

*(Show the exchangeability warning box.)*

**[STOP — learner works]** — *(replaces the "your turn" string)*

Pause the video, and answer the question in the code cell: **would a bigger sample have fixed
this?**

*(Resume.)*

No, it wouldn't, and this is worth being clear about, because "get more data" is the reflex. More
weeks per series would sharpen each estimate, and the hierarchy would still be pulling the FOODS
series toward a mean computed from HOBBIES and HOUSEHOLD. The error is in the *structure* we
imposed, and no quantity of data corrects a structural mistake. It just makes you more confident
about it.

**Exchangeability is a modeling assumption, and it's yours to defend.** Groups are exchangeable
when, before seeing the data, you have no reason to expect one to differ from another in a
particular direction. Ten FOODS series across ten stores is plausible. FOODS against HOBBIES isn't,
because you knew the answer before you looked.

Nothing in PyMC checks this, and no diagnostic reports it. It's subject-matter knowledge, applied
before the model is written.

---

# ▶ STEP 5 — Do it on the exchangeable set

*(Rebuild on FOODS only, then run the fits.)*

```
10 FOODS series | 5 thinned to 8 weeks
train 1,425 rows | held-out 1,345 rows
```

It's the same code, the same priors and the same thinning trick, this time with every other series
cut to eight weeks. The **only** change is which series are allowed into the hierarchy.

```
       mean     sd hdi94_lb hdi94_ub ess_bulk ess_tail r_hat
mu_b  0.078  0.035    0.017     0.15     1544     1588  1.00
sd_b  0.065  0.038   0.0097     0.16     1027     1153  1.00
```

*(Make this comparison out loud.)*

**[pause]**

`mu_b` is **zero point zero seven eight**, a real SNAP effect, where before it was zero point zero
one one. And `sd_b` is **zero point zero six five**, where before it was zero point zero zero seven.
That's nine times larger.

That second number is the interesting one. The model now says that stores genuinely differ from
one another, because on a group you can defend, they do. The earlier zero point zero zero seven
wasn't a finding about stores. It was an artifact of asking about the wrong group.

*(Run the shrinkage table.)*

```
              unpooled spread   after pooling   mean move
  thin (8w)       0.108           0.010          0.120
  full (277w)     0.055           0.037          0.015

  ratio: thin moved 8.1x further
```

Compare the middle column with the one in Step 2. There, **everything** collapsed: the thin spread
went to zero point zero zero zero, and the full spread to zero point zero zero one. Here, the thin
series collapse to zero point zero one zero, but the full series **keep most of their spread**,
going from zero point zero five five down to zero point zero three seven. That's about two-thirds
of it retained.

That's what partial pooling is supposed to look like. The model discarded variation that eight
weeks couldn't support, and kept variation that two hundred seventy-seven weeks could. In Step 2 it
discarded everything, and that should have been the tell.

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

Now read the middle column against the last one. It's zero point zero nine seven against zero
point zero eight seven, then zero point zero eight two against zero point one zero two, then zero
point zero nine one against zero point zero seven two. These are *close*, and each one came from a
series with **eight weeks of data.**

The RMSE against the truth is zero point one two six unpooled and **zero point zero four eight**
hierarchical, an error two point six times smaller.

**That's borrowing strength, and now it's real.** It's the same code, the same priors and the same
diagnostics, and the thin series are estimated well because the other five exist. The only thing
that changed was which series were allowed into the hierarchy.

**[STOP — learner works]**

Pause the video, and fill in the comparison in the last cell. By what percentage did pooling
improve the RMSE in Step 3, and by what percentage here?

*(Resume.)*

In Step 3, pooling took the RMSE from zero point one zero three to zero point zero nine eight,
which is an improvement of about five percent. Here it went from zero point one two six to zero
point zero four eight, which is an improvement of about sixty-two percent.

---

# ▶ STEP 6 — The trap in the code

*(Show the centered form, then run it.)*

Here's one more failure, and this one **is** mechanical rather than conceptual.

The natural way to write a hierarchy reads exactly like the equation. You declare b directly as a
Normal, with mean `mu_b`, standard deviation `sd_b` and shape J. Everybody writes it that way
first, and it's called the centered form.

```
Diagnostics
  centered       r_hat 1.003  ess   966  divergences   43
  non-centered   r_hat 1.003  ess  1719  divergences    0
```

**[pause]**

It's the same model and the same data. It would even give the same posterior, if you could sample
it. But the centered form makes the sampler walk a **funnel**. When `sd_b` is small, the individual
b-j have to crowd into a narrow neck, and a sampler tuned for the wide part of the funnel can't get
into the neck.

*(Show the callout.)*

Now read the three numbers, because only one of them objects, and that's the lesson.

The effective sample size **passes**. At nine hundred sixty-six, it's more than twice the
threshold, and you wouldn't look twice.

R-hat **passes** too, at one point zero zero three against one point zero one. Those are two of the
three numbers Lecture 10 taught you to check, and on those two alone, you would have shipped this
model.

Then there are **forty-three divergences, against a threshold of exactly zero.** There's no version
of "close to zero" that gets you to forty-three, and there's nothing to argue about.

*(Show the second paragraph of the callout.)*

If your number differs from this one, that's expected, because divergence counts move with the
seed and even with the machine. Over ten other seeds, we saw fifteen to two hundred eighty-five
divergences in the centered form, and zero to eleven in the non-centered form. What held every time
was the pattern. The centered form diverged on all eleven runs, and R-hat or the effective sample
size caught it on only two.

So here's the rule. On hierarchical models, **divergences fire first, and least ambiguously.**
Don't wait for R-hat to decide for you.

---

# ▶ BEFORE YOU LEAVE

**[STOP — learner works]** — *(replaces the in-room "Discuss")*

Here are four questions to work through on your own before you go.

The first is about the failure. Step 2 had clean diagnostics, textbook shrinkage and a conclusion
that sounded defensible, and Step 3 showed it was wrong. What kind of check caught it, and why
could no amount of sampler diagnostics have done so?

**[pause]**

The answer is held-out data. Diagnostics audit the computation, and only data the model never saw
can audit the model.

The second is about the assumption. Suppose you're asked to pool weekly demand across the ten
stores in one chain. Name one variable that, if it differed across those stores, would break
exchangeability the way category did here, and say how you'd handle it **without** abandoning the
hierarchy.

**[pause]**

A good answer puts that variable at a level the hierarchy can absorb. You can put the offending
variable into the model as a predictor, or you can nest the hierarchy, with stores within regions,
rather than giving up on pooling.

The third is about judgment. In Step 5, `sd_b` came out around zero point zero six five, so stores
genuinely differ. The unpooled fit put the thin stores' range at roughly five percent to forty
percent. Given both numbers, what do you actually recommend the business do for the next SNAP
week?

**[pause]**

The answer the lecture gave comes in three parts. Plan on an eight percent FOODS uplift at every
store, which is Step 5's `mu_b` of zero point zero seven eight. Allow for real but modest variation
between stores around it, because `sd_b` says they do differ. And don't act on any one store's own
estimate until that store has the weeks to support one. Step 5's grading showed how far eight weeks
of a store's own data can mislead you.

The last one is about the thread running through the course. This is the sixth time this semester
that a model has improved by pulling estimates toward a common value. Name the other five, and say
what's different about the shrinkage strength here.

**[pause]**

The other five are the smoothing penalty on GAMs in Lecture 3, tree pruning in Lecture 4,
`reg_lambda` in Lecture 5, Ridge and LASSO in Lecture 6, and weight decay in Lecture 7. What's
different here is that the strength is *estimated from the data*, rather than chosen by you and
tuned on a validation set. `sd_b` is the shrinkage parameter, and the model infers it.

Solutions for the coded parts go up on Canvas after class. Next week, in Lecture 12, we turn to
Bayesian linear regression, and to a preprocessing step you've used all semester that silently
destroys the answer.

---

# Appendix — expected output

`random_seed=42` throughout; five fits, whole lab ≈ 3 minutes.

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
stores respond identically") that a modeler can reject on subject-matter grounds. Students who
learn to read it have learned something no threshold gives them.

*Do not let "get more data" stand as the answer to Step 4.* The error is structural: more weeks
would sharpen estimates while the hierarchy went on pulling FOODS toward a mean built from
HOBBIES. More data makes a structural mistake more confident, not less wrong.
