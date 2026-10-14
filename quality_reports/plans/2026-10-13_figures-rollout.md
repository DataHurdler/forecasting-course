# Figures rollout to every remaining deck, plus the content-error corrections

**Status:** COMPLETED 2026-10-13 (not committed; owner to review). APPROVED 2026-10-13. The owner liked two lists ("Geometry instead of algebra" and "The
first figure I'd make in each deck") and said: "Implement them." Mid-turn they added: "For
'Content errors found along the way', make sure you double check them, and then implement
correction."
**Roadmap and specs:** [visuals-and-geometry audit](2026-10-13_visuals-and-geometry-audit.md), with
one per-deck section and the content-errors table. **Worked precedents:**
[L09 Part 1](2026-10-13_lecture09-part1-visuals.md) and [L09 Part 2](2026-10-13_lecture09-part2-visuals.md),
with their scripts at `scripts/figures/lecture09_part1_figures.py` and `lecture09_part2_figures.py`.

## What gets built

★ = the first-figure list. G1–G5 = the five recurring geometric pictures (audit plan section
"Recurring geometric ideas").

| Builder | Deck | Figures (plus the word diet on every frame) | Content errors to verify, then fix if confirmed |
|---|---|---|---|
| A | L10 Bayesian foundations | ★ "The Update, in Pictures": prior, likelihood and posterior curves plus the balance point (G1) | none listed; check what you touch |
| A | L12 Bayesian regression | ★ Confounding: pooled slope −2.8 vs within-series −0.49 (remove series means *and* trend) | elasticity boundary −0.5 → −1 (`:101, 135–136, 238–239`); "XGBoost best tabular accuracy" (`:265`); ○ `:118` "half a percent either way" (lower HDI end ≈ −1.1%); ○ time as a second confounder (`:188–190`); ○ `:59–60` ridge/lasso ↔ prior holds at the posterior mode |
| B1 | L11 Part 1 Bayesian TS | ★ 94% band over the 52 held-out weeks, widening like √h; G5 Fourier pairs as arrows (amplitude, phase) | ○ "±30%" is an average; the width grows with h, so the slide should say "on average" |
| B2 | L11 Part 2 Hierarchical | ★ Shrinkage arrows (G1), thin series crimson, held-out truth as rings; G5 the funnel, centred vs non-centred | 8.2× (`:267`) vs 8.1× (`:359`): find the true number from the lab and use it everywhere |
| C | L13 Synthesis | ★ Scoreboard as a dot plot against the seasonal-naive line; G2 Christmas closures as squares (each closure about 18× the average area) | **Owner decision: the scoreboard moves to 52-week numbers** (RNN 805, LSTM 998, Transformer 990; CNN 1,032 is a 26-week fit and must be labelled). Never mix in 26-week numbers. Also ○ `:102` growth figures: 2011→2012 was +28.6%, say "since 2012". The book preface's claims (`book/index.qmd`, checked by `scripts/check-book-claims.py`) read this scoreboard: keep them true with a minimal factual edit |
| D1 | L01 ETS and evaluation | ★ SES on CA_1 FOODS: the level closing α of the gap (G1); G2 accuracy as squares (RMSE ≥ MAE); G4 SES weights as a staircase (equal weights frame); G4 the damped-trend ceiling (Holt frame) | ETS(A,A,M) listed among the 15 (`:271–274`) → ETS(M,A,M); ○ the fitted φ = 0.80 sits at statsmodels' lower bound (Lab 1 Step 5 prose) |
| D2 | L02 ARIMA and VAR | ★ Real ACF of levels vs first difference, and the 2×2 ACF/PACF regenerated from ArmaProcess on a labelled axis (G4); G4 shock responses and half-lives (unit-roots frame) | ARIMA box writes the ARMA on y_t (`:259`) → y′_t; the MA(1) sketch draws ρ₁ ≈ 0.67 (impossible, max 0.5): fixed by the regenerated 2×2 |
| D2 | L03 GAMs | ★ The smoothing penalty: three fits, (f″)² shaded as area (G3) | the PDP TikZ (`:268–300`): band outside its axes, bare `scale=`, keybox describing a falling curve no panel shows. Fix it (real PDP or corrected TikZ plus matching keybox) |
| E | L04 Decision trees | ★ The CART split: cut, leaf means, residual sticks, SSE against threshold (G2); G3 cost-complexity pruning, a line of slope −α touching the lower hull | ○ the flowchart's monthly numbers (≤38,000 → 61,300) are invented; CA_1 FOODS weekly runs 12k–27k. ○ Lab 4 Step 2 says "Lecture 4 drew this as a curve", but the deck doesn't |
| E | L05 Tree ensembles | ★ Root-split tally across 300 trees, bagging vs `sqrt`, with each one's test RMSE; G1 a forest pinned inside its training range (RSXFS) | OOB-noise reason (`:111–112`); `gamma` "the way ℓ1 does" (`:328–329`); ○ "last year's value always dominates" (`:131`); ○ Newton step "fewer trees" holds only for non-squared losses (`:315–317`). Also report whether the weekly tree and forest RMSEs (L04 `:300–304`, L05 `:195–200, 227–229`) reproduce from any script |
| F | L06 Regularization | ★ Diamond and disc on the real lag7/lag14 RSS ellipses (G3), caption honest about the budget | `:153` "likely to hit a corner first" holds only for small budgets (corner wins for t ≲ 0.18 on lag7/lag14) |
| F | L07 Neural nets | ★ The W-shaped electricity year: the best line vs a sum of four ReLU hinges at fixed knots | ○ `:46` "smooth, continuous" (ReLU nets are continuous and piecewise linear); ○ Lab 7 `:246` gradients accumulate "in that epoch" (without `zero_grad` they accumulate across all 60) |

Other audit candidates (the rest of each deck's ranked list) are **not** in scope this round.

## House rules (every builder)

**Figures**
- One generator per deck: `scripts/figures/lectureNN_figures.py` (`lecture11_part1_figures.py` and
  so on). Copy the header block (palette, `FONT`, `SCALE = 0.55`, `sz()`, `rcParams` with
  **`svg.fonttype: "path"`**, `save()`) from `lecture09_part2_figures.py`. Write
  `Figures/<Deck>/<name>.pdf` and `.svg`. Run with `.venv/bin/python` (absolute:
  `/Users/zijunluo/Documents/GitHub/Forecasting-Course/.venv/bin/python`).
- **Numbers come from code, not from the slide.** Recompute anything the figure prints, using the
  lab's own pipeline where one exists. Before printing it, reproduce the deck or lab number. If you
  cannot reproduce it, stop and report; do not quietly change a slide number.
- Anything trained or sampled: seed it, as the lab does, and cache what the figure needs in
  `Figures/<Deck>/measured.json`, with a `--retrain` flag (see the L09 scripts).
- **Legible at slide size.** Text in a figure should land at about 7 pt or more in Beamer. That
  needs `display height ÷ natural height ≥ 0.6`, so draw at roughly display size: keep `figsize`
  small and fonts 11–13. Check every figure as a PNG
  (`~/.venvs/slide-qa/bin/python -c "import pymupdf; ..."` renders a PDF page) and fix every
  collision: label on label, label on line, text outside its box. The precedents went through
  four or five redraws each; expect the same.
- Geometry claims must be exactly true for what is drawn. Caption the limits honestly: "two
  coefficients, constrained form"; "a 2-D cartoon of a 16-D head"; "given the variances".

**Fewer words on every slide (owner, mid-turn 2026-10-13)**

> "Another point I wanted to make about your slides: they tend to have too many words. For
> example, page 4 'Why Convolution?' in Lecture 8. You could sensibly break that slide into two, no?"

- Measure with `python3 scripts/slide_word_counts.py <Deck>`, which counts running text only
  (tables, math, code, TikZ and figures excluded). The current median is 91 words a frame. **Aim
  for 90 or fewer.** Above **120**, act: split the frame at a natural seam (two ideas become two
  frames), or cut, moving the detail into the narration. Do this for **every** frame in your deck,
  not only the ones you add figures to.
- **Splitting is allowed and encouraged** where a frame carries two ideas. Each new frame needs a
  title of its own that says what it is about. Don't produce a frame that holds one sentence.
- **Key Takeaways** frames: trim each point to about one line instead of splitting the frame.
- Where a figure goes into a frame, the figure carries the idea: cut the prose it makes redundant.
- **When the frame count changes:**
  - mirror it in the Quarto file, so `python3 scripts/check_mirror_parity.py` still shows equal
    counts;
  - renumber every `# ▶ SLIDE n` header and every "slide n" cross-reference in the narration;
  - update the narration's page count in its `Deck:` line;
  - split the narration between the two new slides.
- Words are not the only measure. Never cut a caveat that keeps a claim true; move it to the
  narration or a muted line instead.

**Slides**
- Beamer first: `cd Slides && TEXINPUTS=../Preambles:$TEXINPUTS xelatex -interaction=nonstopmode
  <deck>.tex` ×3, with `BIBINPUTS=..:$BIBINPUTS bibtex <deck>` after the first pass. **0 errors,
  0 overfull.** View every changed page as a PNG.
- Then the Quarto mirror in `Quarto/<deck>.qmd`: `![](../Figures/<Deck>/<name>.svg){fig-align="center" height="NNNpx"}`.
  Render only your own deck: `quarto render Quarto/<deck>.qmd`. Then run
  `~/.venvs/slide-qa/bin/python scripts/slide-qa.py Quarto/<deck>.qmd`: **0 overflow**. Clipped
  code blocks were already there and are fine. Look at the screenshots of the slides you changed.
- `python3 scripts/quality_score.py Quarto/<deck>.qmd` ≥ 80.

**Narration** (`Narration/<LectureNN>_script.md`)
- Update every changed slide so it walks the figure, in the existing voice: `*(stage directions)*`
  and `**[pause]**`, plain spoken numbers. Fix any numbers your corrections change. Keep the
  `# ▶ SLIDE n — Title` headers matching the frame titles exactly. Then run
  `python3 scripts/build_book.py <LectureNN…>`: every heading should still join its narration.
  If a lab's prose changes, update `Narration/<…>_lab_script.md` to match.

**Content errors**
- **Verify each listed error yourself** against the source and the data before changing anything.
  ✔ means verified once by hand; ○ means a reviewer reported it and nobody has checked. Fix what
  you confirm, in the deck, its mirror, its narration and the lab prose. Report anything you refute,
  with the evidence.
- If an assignment in `Homework/` or the student repo repeats an error, **do not edit it**: report
  it. Those are copied to students, and changing one mid-term is the owner's call.

**Hands off (other builders are working in the same tree at the same time)**
- Do not run `scripts/sync_to_docs.sh` or touch `docs/`. The coordinator publishes.
- Do not edit `CLAUDE.md`, the audit plan, `Bibliography_base.bib` (report any citation you need),
  `Homework/`, any other deck's files, or the L09 work.
- Do not render labs. Edit lab prose if an error requires it, and report which labs changed.
- **No git commands that change anything:** no add, commit, stash, checkout or reset. Read-only git
  is fine.

## Report (each builder returns this)

Per deck:
1. Figures added: frame, what is drawn, and what it replaced. Frames split or cut, with words per
   frame before and after (from `slide_word_counts.py`).
2. Numbers measured: did each reproduce the deck or lab?
3. Content errors: confirmed and fixed, refuted (with evidence), or reported only (assignments).
4. Verification: Beamer errors and overfull; slide-qa overflow; parity; quality score; book join.
5. Files changed. Labs edited (the coordinator re-renders them).
6. Anything unresolved.

## Outcome (2026-10-13, coordinator)

All eight builders reported; every deck verified by the coordinator afterwards:
- **Beamer:** all 15 decks compile with 0 errors, 0 overfull and 0 undefined.
- **HTML:** `sync_to_docs.sh slides` re-rendered every deck, and slide-qa finds 0 overflow in all 15 (clipped items are code blocks that were already clipped).
- **Consistency:** parity holds in all 15 decks, every narration slide count matches its PDF, and every narration joins its book chapter.
- **Gates:** `check-book-claims` (10/10), `check_links` and `backtest.sh` all pass.
- **Labs:** 1, 3, 4, 5, 7, 11 Part 2 and 12 were re-rendered and published.
- **Words per frame:** 181 frames, median 92, max 183, 53 over 120 → 193 frames, median 77, max ~105, **0 over 120**.

**Coordinator fixes during review:**
- `check-book-claims.py` `_ranks_match` now compares the pair as a set, so a 52-week swap of
  LSTM and Transformer doesn't fail a claim that names only the pair's places.
- The word counter treated `\%` as a comment; fixed.
- British spellings in new text corrected (labelled, colour, neighbouring, coloured).
- CLAUDE.md's L11 Part 2 numbers now match the reproducible lab (8.1×, 0.048, 43 → 0 divergences).

**Numbers that did not reproduce, corrected to the code's output:**
- L11 Part 2 divergences, R-hat and ESS.
- L12 P(e < −0.5) is 0.46, not 0.446.
- L11 Part 1 centred ESS is 274.
- L05 weekly tree/forest RMSEs: 1,456 / 1,115 against the slide's 1,415 / 1,020. No script ever produced the old pair.
- L04 snap_days importance is 0.05.
- XGBoost 781 came out at 779–792 here, inside the seed spread, so the slide keeps 781.

**Owner calls (reported, nothing edited):**
- **Assignments and keys repeating corrected numbers or claims:**
  - HW06 Part 1 assignment and key: ESS 252, RMSE 1,587, ±30%.
  - HW06 Part 2 key: 127 divergences, 1.044, 8.2×, 0.049.
  - HW07 solutions narration: the −0.5 boundary.
  - HW05 Part 3 key: "XGBoost (tuned) 781".
  - HW01 Part 1 key: φ ≈ 0.74 (impossible under the default bounds).
  - HW04: credits `reg_lambda` to Lecture 6.
  - HW04 Q4: claims scaling is refit inside each fold.
- **The `shuffle=True` claim** in L07, Lab 7 and HW05 Part 1 ("destroys the ordering" / "sees the
  future"): untrue under a time-based split.
- **L01 MASE:** now scaled by the training-data seasonal-naive MAE (the standard definition). Say if
  you want the old ratio back.
- **Instructor repo:** canvas/week-13-summary.md and the README carry the old L11 Part 2 numbers.
