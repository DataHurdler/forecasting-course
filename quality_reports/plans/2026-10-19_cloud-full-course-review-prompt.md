# Full-course review — prompt for a Claude Code on the web session

**Why this file exists:** the first attempt ran on the owner's Mac (the "remote" agent option fell
back to a local worktree) and stopped at the weekly usage limit. The owner chose to run it on
Claude Code on the web instead, so it draws on cloud credits.

**How to run it:** open https://claude.ai/code, start a session on `DataHurdler/forecasting-course`
(branch `main`), and paste everything inside the box below as the first message. When it finishes,
tell Claude in the terminal "the cloud review is done"; it will fetch the branch, re-check every
finding, and bring you one short list.

---

```text
You are auditing a complete graduate course (ECON 8310 Business Forecasting) for correctness.
Do NOT change any course file. Your only output is one report file on its own branch:

  1. git checkout -b review/full-course-2026-10-19
  2. write quality_reports/2026-10-19_full_course_review.md
  3. commit only that file and push the branch. Do not open a pull request. Do not touch main.

## The course (read CLAUDE.md first)
- Slides/LectureNN_*.tex — Beamer decks, the SOURCE OF TRUTH (L01–L08, L09 Part 1/2, L10, L11 Part 1/2, L12, L13).
- Quarto/*.qmd — HTML mirrors (structure parity is already checked; skip it).
- Narration/LectureNN_script.md and *_lab_script.md — spoken scripts (numbers as words: "sixteen sixty" = 1,660).
- Labs/LectureNN_lab.qmd — executable labs that produce many numbers.
- scripts/figures/lectureNN_figures.py + Figures/<Deck>/measured.json — figure generators and the numbers they measured (seed 42).
- Homework/HW*.qmd — assignments students are doing now.
- ECON8310_*.md (about, datasets, project rubric, troubleshooting) and scripts/site.yml (schedule).
- data/processed/*.csv — all course data is in the repo.

## Environment (keep it light)
python3 -m venv .venv && .venv/bin/pip install pandas numpy scipy statsmodels scikit-learn matplotlib pyyaml
Install torch, pymc, prophet or pygam only if a specific check truly needs them. Do NOT retrain neural
networks or re-run MCMC: compare those numbers against measured.json and the labs' printed outputs, and
mark them "checked against cache".

## What to find, in priority order
1. Wrong numbers students see: a number on a slide, in lab prose, or in an assignment that the code does
   not produce. Re-run cheap computations yourself.
2. Wrong or overstated claims about methods (statistics, econometrics, ML), held to the standard of a
   careful textbook referee. Past examples, already fixed: "a full tree has zero bias" (it has zero
   training error); "−0.5 is the elastic/inelastic boundary" (it is −1).
3. The same quantity stated differently in two places: deck vs narration, deck vs lab, deck vs
   CLAUDE.md, deck vs deck, deck vs assignment, schedule vs documents.
4. Student-facing logistics that contradict each other (due weeks, dataset names, prompt budgets).

## Already known — do not report these again
- L08: two stacked width-3 filters see 5 weeks, not 6.
- L09 Part 1: "recurrence beats convolution decisively" also attached to the LSTM (987 vs 1,032);
  the gradient-decay figure is the 52-week RNN while the slide implies the 26-week one; Check 2's
  "nothing needs carrying across a gap" vs the RNN improving 878 → 842 → 805 with window length.
- L09 Part 2: "shuffling leaves every output unchanged" (attention is permutation-equivariant).
- L10: divergences required in the diagnostics table although the MCMC slide teaches Metropolis.
- L11 Part 1: "calibrated" vs 52/52 inside a 94% band (conservative).
- L05: "63%" vs "about 64%" a few lines apart.
- Everything in quality_reports/plans/2026-10-18_next-term-changes.md and in the "Owner calls" and
  "Owner decisions" sections of quality_reports/plans/2026-10-16_figures-round2.md.
Out of scope: wording, typos, notation style, slide layout, pedagogy (other reviews cover these).

## Standard of evidence (strict)
Every finding needs: file:line, the exact quoted text, and what is actually true, with evidence
(the command you ran and its output, or the conflicting file:line). Before reporting, try to refute
each finding (a different window, seed, series or horizon than you assumed?). Drop anything you cannot
evidence. Rounding within the stated precision, and seed variation the deck already acknowledges, are
not findings.

## Report format (the file you push)
1. Summary: what you checked (counts of decks, labs, assignments, documents), what you could not check
   and why, totals by severity.
2. Findings, most severe first. For each: ID; severity (critical = wrong number or false claim students
   see; major = inconsistency or overstated claim; minor = other); where (file:line, plus every other
   place the same text appears); what it says (quote); what is true (evidence); one-line fix;
   confidence (high/medium); verified by re-computation or against cache.
A short verified list beats a long speculative one.
```
