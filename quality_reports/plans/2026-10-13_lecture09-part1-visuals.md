# Lecture 9 Part 1: figures and fixes (taught ~2026-10-22)

**Status:** COMPLETED 2026-10-13, not committed (owner to review). APPROVED 2026-10-13. Owner: "Your recommendation is sound. Follow the teaching
calendar." Source: [visuals-and-geometry audit](2026-10-13_visuals-and-geometry-audit.md).
**Standing decision (owner, 2026-10-13):** the **52-week window is canonical** for the L9 Part 2
and L13 scoreboards. That work belongs to the Part 2 deck (next week), not this one.

## Constraints

- **No new frames.** Figures go into existing frames and replace tables or repeated equations.
  Frame count, mirror parity and narration slide numbers all stay the same.
- **One generator:** `scripts/figures/lecture09_part1_figures.py`, in the L08 house style.
  It trains with the lab's own `Recurrent` class and `build()`, so its numbers are the lab's
  numbers. It runs on CPU in about 3 minutes.
- **Beamer first, then the Quarto mirror, then the narration.** Lecture scripts feed the book.

## Text fixes

1. `:151–152` table: 0.7^26 = 0.000 094 (not 0.000 002); 0.7^52 ≈ 10⁻⁸ (not 10⁻⁹).
   The takeaways (`:460`) and the narration (slides 9 and 22) carry the same numbers.
2. `:178` the forget gate is Gers, Schmidhuber & Cummins (2000), not Hochreiter (1997).
   Add a bib entry. On the cell-state frame, say that the 1997 cell path was exactly ResNet's
   identity, and that the forget gate made it learnable. This makes the ResNet callback more
   precise, not less.
3. `:404` "≈ 0 at every lead" → the lab prints −0.13 at t+2 and +0.12 at t+4. Both come from
   the SNAP calendar repeating, not from a lead. Check the explanation before writing it. Fix the
   lab prose (`:350`) and both narration scripts as well.
4. `:126` "above 1, it explodes" → "it can explode".

## Figures

| Frame | Figure | Replaces |
|---|---|---|
| Vanilla RNN: One Equation, Repeated (`:88`) | `unrolled_rnn`: a real 26-week window feeding a row of identical cells; h runs left to right; h₂₆ → linear → ŷ | prose |
| Why a Vanilla RNN Forgets (`:137`) | `gradient_decay`: log-scale gradient surviving k steps back. The 0.7 example, our trained RNN (52-week window, measured), and the same RNN untrained. Straight line on a log axis = the same factor every step | the table |
| Why the Cell State Fixes the Gradient (`:239`) | `cell_lanes`: backward lanes. RNN hops pass through ×W_h·tanh′; cell hops through ⊙f and +. Note the trained forget gates as measured | the repeated display equation |
| Check 1: Was It a Fair Fight? (`:356`) | `budget_curve`: test RMSE vs epochs at all 7 checkpoints, two-seed means, dashed line at 30, gap bracketed | the table |

Check 2 (30 series and ACF) is deferred: that frame is already full.

## Verification

- The figure script reproduces the deck's Check 1 numbers (1,099/842/828/830 and
  1,002/987/926/821). If it doesn't, stop and find out why before printing anything.
- Beamer: 3-pass XeLaTeX, 0 errors, 0 overfull. Every changed page viewed as an image.
- Quarto: render, slide-qa 0 overflow, figures viewed in the HTML; parity 17/17.
- `quality_score.py` ≥ 80 on the qmd.
- No commit or push until the owner asks.

## Outcome (2026-10-13)

- **The numbers reproduce the lab exactly.** The figure script retrains with the lab's code and
  gets RNN 1,099/922/842/834/828/833/830 and LSTM 1,002/1,033/987/953/926/885/821 at the seven
  checkpoints, and 805/998 at 52 weeks. Every printed number is in
  `Figures/Lecture09_Part1/measured.json`.
- **Measured, new:** the trained RNN (52-week windows) keeps ×0.79 of the gradient per step
  (×0.55 untrained): 0.36% at 26 steps back, 0.001% at 51. The median trained LSTM forget gate is
  0.52 (90th percentile 0.61); the cell path keeps 7% after four weeks.
- **SNAP leads verified:** the weekly `snap_days` autocorrelation is −0.84 at lag 2 and +0.82 at
  lag 4. Multiplied by the same-week +0.14, that reproduces the −0.13 and +0.12 leads.
- **Frames:** four figures in, two tables and one repeated display equation out, text cut to
  fit. The citation moved off the gates frame title because the double cite wrapped it to two
  lines.
- **Also fixed: SVG text is drawn as outlines now (`svg.fonttype: "path"`), in both figure
  scripts.** As live text, the cell-lanes prime rendered as "0" (cmsy10's code point for a
  prime). Lecture 8's residual block put ℱ at a STIX private-use code point (U+E232), which shows
  as a box on any machine without STIX fonts, and every label assumed Helvetica Neue. Lecture 8's
  SVGs were regenerated and its deck re-rendered. Its PDFs are unchanged (restored from HEAD).
- **Verification:**
  - Beamer: 3-pass, 0 errors, 0 overfull, 23 pages.
  - HTML: slide-qa reports 0 overflow and 1 clipped (the pre-existing PyTorch code block); L08 also 0 overflow.
  - Parity 17/17. `quality_score` ≥ 95. `backtest.sh` PASSED.
  - Lab re-rendered.
  - The narration (lecture and lab) is updated for every changed slide.

## Not done / found along the way

- Check 2 figure (30 series + ACF): deferred as planned.
- **The book is stale and will not show the new figures.** `docs/book` was last rebuilt
  2026-09-03. `build_book.py` copies the mirror's `![](../Figures/…)` paths unchanged into
  `book/chapters/`, where `../Figures` does not resolve. Fix the path in the generator before the
  next `sync_to_docs.sh books`.
- In headless screenshots, MathJax renders `\mathcal{F}` in L09's keybox as a box. The same
  markup was in the deck before this change, and MathJax loads its fonts from a CDN at view time,
  so this is probably the sandbox's network. Not verified in a normal browser.
