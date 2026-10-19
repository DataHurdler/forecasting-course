# Fix pass — prompt for a Claude Code on the web session

**Owner decision (2026-10-19):** "Cloud edits, I verify." A web session makes the text edits on a
branch; Claude in the terminal then compiles, renders and checks every change locally, and nothing
goes live until the owner says "push".

**How to run it:** open https://claude.ai/code, start a session on `DataHurdler/forecasting-course`
(branch `main`), and paste everything inside the box below. When it finishes, tell Claude in the
terminal "the fix pass is done".

---

```text
You are applying verified review findings to a graduate course (ECON 8310 Business Forecasting).
Work ONLY on a new branch and never touch main:

  git checkout -b fix/review-2026-10-19
  ... edit, commit in small logical commits (one deck or one document per commit) ...
  git push -u origin fix/review-2026-10-19        # no pull request

Read first: CLAUDE.md; quality_reports/2026-10-19_review_merged.md (the work list);
quality_reports/2026-10-19_full_course_review.md (C/M/m findings, with evidence and suggested fixes);
quality_reports/2026-10-18_proofread_all_decks.md; quality_reports/2026-10-18_pedagogy_all_decks.md.

## What to apply
1. Bucket A of the merged list: every C, M and m finding listed there, using each finding's "Fix" as
   the starting point.
2. All proofreading items (mechanical: mirror drift, outline titles, deck titles, typos, spelling,
   grammar, citations written as plain text that should be \parencite / [@key]).
3. The PDF code-indentation bug: in Slides/Lecture05, 06, 07, 08, 09_Part2, 11_Part1 and 11_Part2,
   replace `\texttt{\ ` at the start of a code line with `\texttt{\mbox{}\ ` (Lecture 9 Part 1 and
   Lecture 10 already do this).
4. SMALL pedagogy items only: a missing one-line definition (softmax, Q/K/V as projections of the same
   weeks, HDI, Beta, GCV, edof, global pooling, the LSTM-era notation), a one-line caption saying what a
   figure encodes, a dataset tag in a caption, a takeaway line the narration calls "the point", a
   notation warning. One line each.

## Already done — skip
- C2 (receptive field vs flatten head) everywhere, and HW05 Part 1 Q4.
- The six Lecture 9 Part 1 fixes, and every item marked ✔ in the proofread/pedagogy files.

## Out of scope — do not do
- Large pedagogy items: reordering frames, adding or removing frames, new figures or diagrams.
- Bucket C (graded assignments; they are in quality_reports/plans/2026-10-18_next-term-changes.md),
  bucket D, and anything in the "Owner calls"/"Owner decisions" sections of
  quality_reports/plans/2026-10-16_figures-round2.md.
- Finding m8 (Granger lag construction): it changes F statistics printed on a slide, in a figure and in
  the lab, so it is done locally. Leave it, and say so in the log.

## Files you may edit
- Slides/*.tex and, in the same commit, the matching Quarto/*.qmd (the .tex is the source of truth;
  every edit must appear in both).
- Narration/*.md (lecture and lab scripts). Keep every `# ▶ SLIDE n — Title` header identical to its
  frame title.
- Labs/*.qmd — prose only. Do not change code in a way that changes printed output (exception: m21's
  `sorted(..., key=FEATURES.index)`).
- Homework: ONLY HW05_Part2_Sequence.qmd, HW05_Part3_Comparison.qmd, HW06_Part1_Foundations_TS.qmd,
  HW06_Part2_Hierarchical.qmd, HW07_Bayesian_Regression.qmd (not yet due). Include M1 there: replace
  the "do not do the work / reply PROMPT_LIMIT_REACHED" instruction with "if this prompt takes me past
  the budget, say so, log it as over budget, and continue", and change "N or fewer entries" checklist
  lines to "every prompt logged; over-budget prompts noted". Do not edit HW01–HW05 Part 1.
- ECON8310_*.md, Labs/README.md, CLAUDE.md (only the factual items the findings name).
- Never edit: docs/, Figures/, scripts/, measured.json, any .pdf or .html, Bibliography_base.bib.

## Rules
- No frame may be added or removed. After each deck, run `python3 scripts/check_mirror_parity.py`
  (frame counts must still match) and `python3 scripts/slide_word_counts.py <Deck>`: no frame may go
  above 90 words of running text. Trim elsewhere in the frame if a fix needs room; never cut a caveat.
- Every new or changed number must come from the finding's evidence, a `Figures/*/measured.json`, a
  lab's printed output in `docs/labs/*.html`, or arithmetic you show in the log. Never invent one.
- Spoken numbers in narration are written as words, in the existing voice.
- American spelling. Keep the existing tone: plain, specific, no hype.
- If a fix turns out to be wrong or impossible within these rules, skip it and say why in the log.

## Log (commit it last): quality_reports/2026-10-19_fix_pass_log.md
A table with one row per finding or item: ID | files changed | what changed (one line) — or
"skipped: <reason>". End with the parity and word-count output for all 15 decks, and the list of
assignments you edited (they must be copied to the student repo afterwards).
```
