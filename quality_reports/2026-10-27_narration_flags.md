# Narration rewrite — flags for the owner

Collected from each lecture's rewrite (2026-10-27). Corrections made against a source are already
in the scripts and listed in each commit message; these are the items that need the owner's call.
Nothing here has been changed unless the line says so.

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
