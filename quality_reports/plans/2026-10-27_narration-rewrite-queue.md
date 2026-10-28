# Narration rewrite — the queue

**Status:** IN PROGRESS (2026-10-27). Owner: "rewrite all narrations using the new style" —
lecture scripts, lab scripts, and the homework solution scripts.

**Standard:** `Narration/STYLE_GUIDE.md` (20 rules). Owner-approved models: Lecture 9 Part 2
lecture (`f2ce154`) and lab (`349f41a`), Lecture 10 lecture and lab (`5e33c7e`).

**Method:** one background agent per lecture (lecture + lab script), about five at a time, all in
the same working tree with no state-changing git. Each returns words before/after, sync changes,
corrections with sources, and flags. Claude then checks headers (`git diff | grep '^[-+]# ▶'`),
`build_book.py` counts, STOP-marker counts, spot-checks the corrections against the source, and
commits one lecture per commit. Owner decisions on flags are collected, not guessed. Push on the
owner's word.

**Solution scripts** live in the private instructor repo (`forecasting-instructor/narration/`,
tracked) with gitignored copies in `Forecasting-Course/Narration/`. Rewrite the instructor copies,
rebuild `narration/pdf/` with `build_narration_pdfs.py`, then copy the `.md` files over the local
gitignored copies. (HW03 and HW05 Part 2 differed only by two British spellings, fixed locally.)

| Item | Status |
|---|---|
| L09P2 lecture + lab | done, pushed (`f2ce154`, `349f41a`) |
| L10 lecture + lab | done, pushed (`5e33c7e`) |
| L11P1 lecture + lab | agent running |
| L11P2, L12, L13, L09P1 | wave 1, agents running |
| L01, L02, L03, L04, L05 | wave 2 |
| L06, L07, L08, HW01–HW04 solutions, HW05–HW07 solutions | wave 3 |

## Video (side task, same session)

Owner chose the Kokoro voice **Michael** (`am_michael`), ~1 s before a slide change and ~1.1 s
before the next slide's narration. Tool: `scripts/make_lecture_video.py` + `scripts/video_requirements.txt`
(own venv `.venv-video`); output to `~/Movies/ECON8310/<Deck>.mp4`, never the repo.
