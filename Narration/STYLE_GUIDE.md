# How the recording scripts are written

The scripts in this folder are read aloud, word for word, into a microphone. Write them the way a
good lecturer talks to a room: in whole sentences, in a natural order, with the next idea always
connected to the last one. If a line would sound odd said out loud, it is wrong on the page.

Owner's rule (2026-10-26): *"Something like 'Five parts.' shouldn't happen. The script should be
written naturally to be read from."*

## Sentences

1. **Every line is a complete sentence someone would actually say.** No headline fragments, no
   telegraphic notes, no labels standing in for sentences.
   - Not: *"Five parts. What recurrence could not do — two specific constraints."*
   - But: *"We'll take this in five parts. First, the two things recurrence could not do."*
2. **No list labels in bold at the start of a paragraph.** Turn the label into the sentence.
   - Not: *"**Parallelism.** No score waits for another."*
   - But: *"The first is parallelism. No score has to wait for another one."*
3. **Takeaways are spoken, not numbered.** Use "First," "Second," "The last one," not "One." "Two."
4. **Connect the ideas.** Each paragraph should follow from the one before: "Here's why that
   matters," "So the question becomes," "That brings us to." A slide change gets a sentence that
   tells the listener where we are going.
5. **Short, clear sentences, but not choppy ones.** Vary the length the way speech does. Avoid
   strings of three-word sentences used for effect ("Score, normalize, average. That's it.");
   say it once, plainly: "So attention does three things: it scores, it normalizes, and it averages."

## Words and numbers

6. **Write numbers the way they are said**, in words: "eight forty-two," "one thousand thirty-two,"
   "zero point eight three," "ninety-one percent," "twenty twenty-six." Keep the same spoken form
   for the same number throughout a script.
7. **Read math the way you would say it**: "Q times K-transpose, divided by the square root of
   d-k," not symbols.
8. **Nothing that is awkward to say**: no parentheses, no slashes, no "e.g." or "i.e.," few
   semicolons, no em-dash chains. Say the aside as its own sentence.
9. **Contractions are fine** ("it's," "we'll," "doesn't"). Plain words over jargon; when jargon is
   needed, say what it means the first time.
10. **American spelling.**

## Directions to the speaker

11. **Stage directions only where the speaker has to do something** (point at a panel, advance,
    run a cell). Keep them short, in italics and parentheses, on their own line:
    *(Point at the left panel.)*
12. **Use [pause] sparingly**, at most once or twice a slide, where a pause genuinely helps. Drop
    *(Beat.)* and similar cues.
13. **Bold is for the word you lean on**, at most once or twice a paragraph, never for labels.

## Content

14. **Every number, claim and caveat must match the current slide and the code.** Never add a new
    number or a new claim. If the old script and the slide disagree, the slide wins; if a number
    looks wrong, flag it rather than guess.
15. **Keep the detail.** The narration carries what the slide no longer shows. Rewriting is for
    the voice, not for cutting content; the length should stay about the same.
16. **Keep the structure the tools rely on:** the header block, the `Deck:` or `Lab:` line, every
    `# ▶ SLIDE n — Exact Frame Title` or `# ▶ STEP n — Title` header unchanged, `---` between
    sections, `**[STOP — learner works]**` markers, and any timing table. Keep every "slide n" and
    "Step n" cross-reference correct.
17. **Only the spoken text is rewritten.** The notes to the instructor ("How to use this document",
    correction notes, an "Appendix — expected output") are read, not recorded; leave them as they
    are unless a number in them is wrong.

## A worked example

Before:

> Five parts. What recurrence could not do — two specific constraints. Then attention itself,
> which is a weighted lookup and much less mysterious than the formula makes it look.

After:

> We'll take this in five parts. First, the two things recurrence could not do, both of them built
> into reading a sequence one step at a time. Then attention itself, which turns out to be a
> weighted lookup and a lot less mysterious than its formula makes it look.

The finished model of the voice is `Lecture09_Part2_script.md`, the first script rewritten to this
guide (2026-10-26).
