# Narration rewrite — flags for the owner

Collected from each lecture's rewrite (2026-10-27). Corrections made against a source are already
in the scripts and listed in each commit message; these are the items that need the owner's call.
Nothing here has been changed unless the line says so.

**Resolved 2026-10-28 (owner: "As recommended").** Every *Recommend* below was applied, except
where it says "leave" or "keep" (left as is). Specifics worth knowing:

- **Lab 8:** the callout is now "Transpose, never reshape". Swapped axes raise a `RuntimeError`
  (re-tested); the silent bug is `.reshape(n, 4, W)`. Lab 8's script, Lecture 8 slide 18, and the
  Lab 9 Part 2 *and* Lab 10 scripts (which also listed "a transposed axis") now say "a reshape where
  a transpose belonged". The Lab 8 script also gains a spoken answer to question 2.
- **Lecture 12 slide 15:** footnote added in Beamer and Quarto, same wording as Lecture 13.
- **Lecture 6:** the 124 is now said as an illustration ("of, say, one hundred twenty-four"); the
  standardized diamond in `lecture06_figures.py` can't be its source.
- **Lecture 1:** the damped Holt row (426 / 351, phi 0.800) and **Lecture 3**'s single-harmonic row
  (4,269 / 3,358) were both reproduced by running the commented-out cells.
- **Lecture 4:** re-timed to about 33 minutes from the word count at the narrated video's pace
  (4,677 words in 30.3 minutes). By the same measure the Lecture 2 and 3 headers (34 and 24 minutes)
  are also low, at about 39 and 33; not changed.

**Still open:** lab-solution timing (Lecture 2 item 3): the syllabus calls labs self-paced while
every lab page and script says solutions go up "after class".

## Lecture 11 Part 1 (commit after `769ce09`)

1. **"Worth paying four percent for"** (slide 17; lab Step 4). The four percent is a *gain* over
   seasonal naive, not a cost. Kept. *Recommend:* "worth far more than the four percent".
2. **Lab, closing question 2** now has a short spoken answer built from Step 4 and the lecture's
   stockout question (no note recorded one). *Recommend:* keep; strike it for a pure exercise.
3. **Lab: "independent in the posterior geometry"** overstates it (independent in the prior, much
   less coupled in the posterior), and the lab calls the problem a "funnel" despite zero
   divergences. Kept, because the lab page says it too. *Recommend:* "much less coupled", in the
   lab and the script together.
4. **Lab Step 3: "the third time this course has said so"** — a count that could not be verified.
   Kept. *Recommend:* "not the first time this course has said so".
5. **Lab appendix 4.28%** comes from rounded RMSEs; exact is 4.26%. Both round to 4%. Left.
6. **Slide 4: ETS intervals "derived from asymptotic theory"** is loose. Kept. *Recommend:* "from
   the model's own error formula".

## Lecture 13 (`fda5f3c`)

1. Slides 11 and 19, section dividers, had no narration; each now says its subtitle in one sentence
   (done, so the video does not sit silent on them).
2. Slide 16's Ecuador earthquake and Favorita mention are in the narration only, not on the slide.
   Accurate, kept.
3. Slide 7's "(If someone asks.)" aside on the damped Holt-Winters variants is reworded so it also
   works when a video reads it every time. *Recommend:* keep.

## Lecture 11 Part 2

1. Slide 12 gains two sentences: Lecture 10 promised divergences "for real" here, and this is the
   mirror image of Part 1 (R-hat and ESS failed there with zero divergences). *Recommend:* keep.
2. The lab's "judgment" question has no spoken answer (no note records one); slide 17 of the deck
   holds one. *Recommend:* add it from slide 17.
3. "Before You Leave" keeps three pauses, one before each spoken answer — one over the guide's
   limit. *Recommend:* keep.
4. Slide 13: "the single most common failure in applied hierarchical modeling" is unsourced.
   *Recommend:* "one of the most common".

## Lecture 12

1. **XGBoost 781 on slide 15.** The narration keeps 781 because the slide says it, but the pinned
   xgboost 3.4.1 gives 792, and Lecture 13 footnotes this. *Recommend:* add the same footnote to
   Lecture 12's slide 15.
2. **Lab closing: every failure "was caught by holding something out and checking."** This lab's
   own confounder was caught by the DAG, not a holdout. Kept. *Recommend:* "caught by checking
   something the model couldn't check for itself".

## Lecture 9 Part 1

1. Instructor notes fixed (factual): the lab has four closing Discuss blocks, not three; the
   lecture note now says the LSTM loses "at thirty epochs (they tie by a hundred)". Done.
2. The tree aside's "five thousand training rows" matches Homework 4's 5,160; the RNN trained on
   5,940 windows. Kept. *Recommend:* leave it.
3. The lab's closing questions are now the lab's current four (result, budget, practice,
   Homework 5); the budget and practice questions stay as exercises (no recorded answers).

## Lecture 1

1. **Slide 20's narration mentions a "test script in the course repository"** that does not exist
   (the student repo's scripts are data prep and submission checkers). Kept. *Recommend:* point to
   the package list in STUDENT_QUICKSTART instead.
2. **The lab's how-to bullet says every number is printed on the rendered page**, but the damped
   Holt row (426 / 351, phi 0.800) is in a commented-out block; a re-run reproduces it.
   *Recommend:* amend the bullet.
3. **Lab .qmd Step 4 asks about "the model you expected in Step 2"**; the ordering is predicted in
   Step 3, and the script says Step 3. *Recommend:* fix the .qmd (a one-word lab prose edit).
4. The timing guide's 4,985-word count is stale (spoken text grew about 9%). Left as an
   instructor note.
5. Lab Step 3's claim that Holt-Winters won't fit three years of weekly data was not verified.
   Kept.

## Lecture 3

1. Slide 13 says Sunday runs "about twenty-five hundred below"; Prophet measured 2,458. Fine
   rounded. *Recommend:* leave it.
2. The lab's rendered page prints three rows in the final table because Step 5 is commented out;
   the single-harmonic numbers (4,269 / 3,358) come only from the Appendix. *Recommend:* re-run
   the solved version once to confirm them.
3. Slide 20 now names FPP §7.4 and §7.7, the syllabus's Week 3 reading.

## Lecture 4

1. The lecture header says about 19 minutes; the timing table adds to about 24, and the script is
   now longer. *Recommend:* re-time and update the header.
2. Solutions "after class" (applied, as you decided for Lecture 10: the narration matches the lab).
3. The milk-versus-canned-goods question has no recorded answer; left as an exercise.
4. CART is spelled out at first mention ("classification and regression trees"); after that the
   voice says "C-A-R-T". *Recommend:* fine as is.

## Lecture 2

1. Under rule 20, the macro discussion question now has a spoken answer built from the appendix
   note ("COVID supplies one enormous shared shock") and slide 20. It sits a little awkwardly with
   "I want both in the Week 2 thread." *Recommend:* keep for the video.
2. Lab Step 3 told the narrator to "say your own number"; a word-for-word reading or the video said
   none. It now speaks the placeholder's result from the narrator note: "With the placeholder
   order, one, one, one, the fifty-two-week RMSE comes out at five hundred sixty-nine." Done.
3. The syllabus says Lab 2 is self-paced, yet the lab says solutions go up "after class" (and the
   narration now matches the lab). *Recommend:* decide once for all labs; if labs are self-paced,
   "after the deadline" in every lab page and script.

## Lecture 5

1. Slide 20 says regularization appears "a fourth time in Lecture 7"; counting Lecture 3's
   smoothing penalty, it is the fifth. Kept. *Recommend:* "again in Lecture 7".
2. Lab 5 Step 5's code comment says "fill in the two blanks", but the cell has one. *Recommend:*
   fix the comment in the lab .qmd (one word).
3. The lecture script is 15.1% longer, just over the guide's 15%, all from numbers in words. Fine.

## Lecture 6

1. **Slide 14 quotes "the coefficient on lag seven was one hundred twenty-four"**; the number is in
   neither the lab nor measured.json. Kept. *Recommend:* check it, or say it as an illustration.
2. The lab closing calls the eleven-feature LASSO "the only model of the five you could write on a
   whiteboard", yet Lab 4's tree is read aloud. *Recommend:* "the only one you could write as a
   single equation".
3. The Step 4 fill-in (the one-standard-error index) stays an exercise; no answer is recorded
   outside the code.

## Lecture 7

1. The lab appendix says "45% loss to the LASSO"; 1,084 against 744 is 45.7% worse. Left.
   *Recommend:* "about 46%".
2. **Code read aloud.** This script writes code as it is said ("zero-grad", "loss dot backward",
   "n-n dot Linear"); others keep backticked identifiers. Both read fine to a person; the video
   tool respells the common backticked ones. *Recommend:* leave the mix.

## Lecture 8

1. **The lab's transpose warning is wrong — the most important flag in this list.** Lab 8's callout
   says that if you get `(batch, time, channels)` backwards, "nothing crashes". Tested at
   `2897f67` with the lab's own first layer, `Conv1d(4, 32, 5)`: a `(batch, 26, 4)` input raises
   `RuntimeError: ... expected input[8, 26, 4] to have 4 channels, but got 26`. The narration was
   kept matching the lab, and the claim travels: the Lecture 9 Part 2 lab script counts "a
   transposed axis in Lab 8" among the course's silent failures. The genuinely silent version is
   using `.reshape`/`.view` instead of `.transpose`: the shapes fit and the numbers are scrambled
   across time and channels. *Recommend:* rewrite the callout around reshape-versus-transpose,
   then update the Lab 8 script, slide 18's narration and the Lecture 9 Part 2 lab script to match.
2. Lab "Before You Leave" questions 2 and 3 have no recorded answers; left as exercises.
   *Recommend:* add an answer to question 2 (the flatten head's extra weights) to the lab notes.
3. Slides 9 and 20 had no narration; each now speaks its subtitle (done). The lab header's
   ~22-minute runtime is now about two minutes low. Left.
