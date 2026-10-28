# Narration rewrite — prompt for a Claude Code on the web session

**Owner request (2026-10-26):** "Take time to rewrite all the narration scripts. Something like
'Five parts.' shouldn't happen. The script should be written naturally to be read from."
Scope: lecture and lab scripts (owner's choice). Approach: pilot first, then the rest. The pilot is
`Narration/Lecture09_Part2_script.md`, written to `Narration/STYLE_GUIDE.md`.

**Before sending:** the style guide and the pilot must be on `main` on GitHub, because the web
session clones from there. The owner approves the pilot's voice and says "push"; then paste the box
below into a new session at https://claude.ai/code on `DataHurdler/forecasting-course`. When it
finishes, tell Claude in the terminal "the narration rewrite is done". Nothing is merged or
published until it has been checked locally and the owner says "push".

---

```text
You are rewriting the recording scripts of a graduate course (ECON 8310 Business Forecasting) so
they read naturally aloud. The instructor reads them word for word into a microphone. Work ONLY on
a new branch and never touch main:

  git checkout -b narration/rewrite-2026-10-26
  ... one commit per script ...
  git push -u origin narration/rewrite-2026-10-26        # no pull request

Read first, in this order:
1. Narration/STYLE_GUIDE.md. It is the standard. Every rule applies.
2. The pilot, Narration/Lecture09_Part2_script.md, and its previous version, to see the change:
     git log -p -1 -- Narration/Lecture09_Part2_script.md
   Match its voice: whole sentences, spoken transitions, numbers in words, no headline fragments
   ("Five parts."), no bold labels opening a paragraph, no "One." "Two." takeaways.
3. CLAUDE.md, for the course and its measured numbers.

## The 28 scripts to rewrite (everything in Narration/ except the pilot and the style guide)
Lecture scripts: Lecture01, 02, 03, 04, 05, 06, 07, 08, 09_Part1, 10, 11_Part1, 11_Part2, 12, 13
  (Narration/LectureNN_script.md).
Lab scripts: Lecture01, 02, 03, 04, 05, 06, 07, 08, 09_Part1, 09_Part2, 10, 11_Part1, 11_Part2, 12
  (Narration/LectureNN_lab_script.md).
Suggested order: the course order, lecture script then its lab script.

## What to do in each script
1. Rewrite every spoken line to the style guide. Keep the content and the detail: the narration
   carries what the slide no longer shows. Aim for the same length, within about 15 percent.
2. Sync it to the current slide or lab. Since 2026-10-18 several slides changed without their
   narration being updated (see `git log --since=2026-10-18 -p -- Slides/` and
   quality_reports/2026-10-19_fix_pass_log.md). For a lecture script, read each frame in
   Slides/<Deck>.tex and make the narration agree with what the frame says now. For a lab script,
   read Labs/<Lab>.qmd. Where the narration and the slide disagree, the slide wins.
3. Fix only what is wrong. If a number in the narration does not match the slide, a
   Figures/*/measured.json, or a lab's printed output in docs/labs/*.html, use the source's number
   and log it. If you cannot tell which is right, keep the old wording and flag it in the log. Never
   invent a number, a claim or a citation.

## Never change
- Any header line: `# ▶ SLIDE n — Title`, `# ▶ STEP n — Title`, `# ▶ OPENING` and the like. The
  book builder joins narration to slides by these titles.
- The `Deck:` / `Lab:` line, the "How to use this document" block, correction notes and any
  "Appendix — expected output" (they are notes to the instructor, not spoken). Fix a number there
  only if it is wrong, and log it.
- `**[pause]**` and `**[STOP — learner works]**` markers may be moved or dropped (at most one or
  two pauses per slide), but STOP markers must stay wherever the learner has to act.
- Every file outside Narration/ except the log. Do not edit Slides/, Quarto/, Labs/, Homework/,
  docs/, Figures/, scripts/ or CLAUDE.md. If a slide itself looks wrong, write it in the log.

## Checks before each commit
- Headers unchanged:   git diff main -- Narration/<file> | grep -E '^[-+]# ▶'    must print nothing.
- Book still joins:    python3 scripts/build_book.py <Deck>   (lecture scripts) reports the same
  "N headings, M with narration" as on main.
- Spelling and numbers: American spelling; spoken numbers in words; no digits in spoken text
  except in code, file names and model names (1D CNN, M5, Lecture 7, Homework 4).

## Log (commit it last): quality_reports/2026-10-26_narration_rewrite_log.md
1. A table, one row per script: file | words before | words after | slides or steps synced to the
   current source (slide n: what changed, one line each) | flags.
2. A section "Flags for the instructor": every number or claim you could not verify, every slide
   that looks wrong, and every place you kept old wording because you were unsure.
3. The output of `python3 scripts/build_book.py` for all decks, and
   `python3 scripts/check-book-claims.py`.
```
