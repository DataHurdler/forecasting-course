# Lecture 9 Part 2: figures and the 52-week fix (taught ~2026-10-29, week 10)

**Status:** COMPLETED 2026-10-13, not committed (owner to review). APPROVED 2026-10-13. Owner: "Work on Lecture 9 Part 2." Standing decisions: the 52-week
window is canonical for this scoreboard, and figures are built in teaching-calendar order.
Source: [visuals-and-geometry audit](2026-10-13_visuals-and-geometry-audit.md), L09 Part 2 section.

## Constraints (as for Part 1)

- **No new frames.** Figures replace tables or display equations; frame count stays at 14.
- **One generator:** `scripts/figures/lecture09_part2_figures.py`, in the house style
  (`svg.fonttype: path`). It trains with the lab's own code and caches what it measures to
  `Figures/Lecture09_Part2/measured.json`.
- Beamer first, then the Quarto mirror, then the narration. The lab changes only where its prose
  is wrong.

## Text fixes: the window

The lab runs **52-week** windows (6,750 windows, 5,160 training). The deck says 26 weeks and 5,940
examples, and its scoreboard pairs the Transformer's 990 (52-week) with the RNN's 842 and the LSTM's
987 (26-week). The fixes:

- `:111` and `:364` "fine at T = 26" → 52.
- `:225` and `:296` "5,940" → 5,160 training windows; "26-week window" → 52.
- `:258` scoreboard → RNN **805**, LSTM **998** (Part 1 `:336`, the lab's own board). Order:
  LASSO 744, XGBoost 781, RNN 805, RF 899, Transformer 990, LSTM 998, CNN 1,032, naive 2,152. The
  CNN number is a 26-week fit (Lab 8): label it, don't silently mix.
- `:279` "990 against 987" → 990 against 998. The conclusion stands: a tie within the seed spread.
- `:304` "unordered bag of 26 weeks" → 52.
- Lab `:221` "5,900 windows" → 5,160 training windows.

## Figures

| Frame | Figure | Replaces / demotes |
|---|---|---|
| Scaled Dot-Product Attention (`:69`) | `attention_geometry`: (a) one query and five keys of different lengths, each key's shadow on the query; (b) the softmax weights as bars summing to 1; (c) the output as a point inside the values' hull. A 2-D cartoon of a 16-dimensional head, labelled as such | the two paragraphs that explain QKᵀ and softmax in words |
| What That Buys a Forecaster (`:91`) | `attention_heatmap`: where the trained model actually looks. 4 heads × 52 weeks back, averaged over the test windows, with lag 52 marked and the uniform level 1/52 as reference. A December row if it adds anything | the hypothetical "can place weight on last December". The text says what it *did* |
| The Encoder Block (`:150`) | `encoder_block`: x → multi-head attention → ⊕ → LayerNorm → feed-forward → ⊕ → LayerNorm, × N. Attention in crimson as the only new part; skip paths in L08's style | the four-row table |
| Positional Encoding (`:178`) | `positional_clocks`: 4 frequency pairs as clock dials, the 52 weeks as dots; plus PE(t)·PE(t′) as a 52×52 matrix that is constant along its diagonals | the display formula (kept small) |

Geometric facts to state exactly:
- q·k = |q| × (the signed length of k's shadow on q). It is "angle only" only for unit vectors,
  so draw the keys at different lengths.
- Softmax weights are positive and sum to 1, so a head's output (before W^O) lies inside the convex
  hull of the values. A head blends values; it cannot extrapolate past them.
- Each sin/cos pair is a point on a circle at angle ω_i·t. A shift by Δ rotates *that pair* by
  ω_i·Δ, so PE(t)·PE(t+Δ) = Σ_i cos(ω_i·Δ) depends only on Δ. It is not one rotation of the whole
  vector, and the similarity is not monotone in Δ.

## Verification

- The figure script reproduces the lab: Transformer seed 1 → 1,013, seed 2 → 967 (mean 990). If it
  doesn't, stop.
- Before reading any attention weights, the hand-computed forward pass must match `model(x)`.
- Beamer: 3-pass, 0 errors, 0 overfull; every changed page viewed. HTML: slide-qa 0 overflow.
  Parity 14/14. quality_score ≥ 80. The narration is updated for every changed slide.
- Commit and push only when the owner asks.

## Outcome (2026-10-13)

- **The lab reproduces exactly:** seed 1 → 1,013, seed 2 → 967, mean 990. The hand-rebuilt encoder
  matches `model(x)` before any attention weight is read. Every printed number is in
  `Figures/Lecture09_Part2/measured.json`.
- **What attention actually did (new, measured):**
  - In both runs, the forecasting position mostly attends to the last 2–7 weeks.
  - One or two heads are near-uniform (entropy ratio 0.97–1.00).
  - Run 1: no head gives the same week last year more than about an even share (max 0.023 against
    1/52 = 0.019).
  - Run 2: layer-1 head 2 puts **0.114** on it, about 6× an even share.
  - So the audit reviewer's "no annual-cycle head" held for seed 1 only. The slide now says what
    happened in each run, and that attention weights show where the model looked, not what mattered.
- **Text fixes:**
  - Every 26-week reference and the 5,940-example count now matches the lab's 52-week windows
    (5,160 training).
  - Scoreboard: RNN 805, LSTM 998. The CNN is labelled as a 26-week fit.
  - The Transformer is now 5th of 8 (990 vs LSTM 998, a tie within the seed spread).
  - The narration's "more capacity bought worse forecasts every single time" is softened to
    "never once bought a meaningfully better forecast", since 67k parameters tied 18k.
  - Also fixed:
    - "next slide but one" on the positional-encoding slide; the measurement is five slides later.
    - "a Tuesday in March" in the narration; the windows are weekly.
    - The lab's "5,900 windows".
- **Frames:** 4 figures. The definition box on the attention slide became a plain display
  equation, which saved space; the encoder table became a diagram; the positional-encoding formula
  shrank to inline. Frame count is unchanged at 14.
- **Verification:**
  - Beamer: 3-pass, 0 errors, 0 overfull, 0 undefined, 20 pages.
  - HTML: slide-qa reports 0 overflow and 1 clipped (the pre-existing PyTorch code block).
  - Parity 14/14. Quality score ≥ 95. The narration joins every heading.

## Left for later

- The Lecture 13 scoreboard still shows the 26-week RNN and LSTM numbers (842 and 987). The owner's
  decision moves it to 52 weeks, scheduled with Lecture 13 (week 16). The book preface's claims
  about where the attention models finished (`check-book-claims.py`) read that scoreboard and will
  need checking at the same time.
- Audit items not built: the recurrence-vs-attention path figure (`:39`).
