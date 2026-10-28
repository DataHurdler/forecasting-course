# Apply the narration-flag recommendations

**Status:** COMPLETED 2026-10-28; committed locally, not pushed. Owner: "As recommended." Applies every recommendation in
`quality_reports/2026-10-27_narration_flags.md` and `forecasting-instructor/narration/REWRITE_FLAGS_2026-10-27.md`
that does not first need the owner to check something. Push only on the owner's word.

**Course repo**
- Lab 8 callout → reshape-vs-transpose (verified: swapped axes raise a RuntimeError; `.reshape` is silent);
  Lab 8 script, Lecture 8 slide 18, Lab 9P2 and Lab 10 scripts follow. Re-render Lab 8.
- Lab 1 Step 4 "Step 2" → "Step 3"; Lab 5 "two blanks" → "the blank". Re-render both.
- Lecture 12 slide 15: footnote 781 → pinned 792 (Beamer + Quarto + narration); compile, parity, render.
- Narration wording: L11P1 (four percent ×2, coupling, "not the first time", ETS intervals), L11P2
  ("one of the most common", judgment answer), L12 lab closing, L01 slide 20 + how-to bullet,
  L04 re-time, L05 "again", L06 (124 as illustration, single equation), L07 "46%", L08 Q2 answer.
- L03: re-run the solved Step 5 to confirm 4,269 / 3,358.

**Instructor repo:** HW02 1,717; next-week closings HW02/03/04; HW01P2 XGBoost line; HW06 P1 thirty;
HW05 P3 annual signal; HW07 Q6 check pair + sign-flip paragraph; HW06 P2 `sd_b` check. Rebuild PDFs,
sync gitignored copies.

**Not applied (owner's call):** lab-solutions timing vs the syllabus; skipped HW questions.
