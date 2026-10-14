#!/usr/bin/env python3
"""Words of running text on each Beamer frame: the number to keep down.

    python3 scripts/slide_word_counts.py                       # every deck, heaviest frames first
    python3 scripts/slide_word_counts.py Lecture08_CNNs        # one deck, in frame order
    python3 scripts/slide_word_counts.py --over 120            # only frames above a threshold

Counts the prose a student has to read. Tables, display math, code boxes, TikZ pictures and
included figures are excluded, because they are not paragraphs. The owner's guidance
(2026-10-13): the slides carry too many words. Aim for about 90 words of running text or fewer;
above about 120, split the frame at a natural seam or move the detail into the narration.
This is a report, not a gate.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = {"Lecture Outline", "References"}


def words(body: str) -> int:
    b = re.sub(r"(?<!\\)%.*", "", body)        # a comment, not an escaped \%
    for env in ("tabular", "examplebox", "tikzpicture", "equation", "align"):
        b = re.sub(rf"\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}", " ", b, flags=re.S)
    b = re.sub(r"\\\[.*?\\\]", " ", b, flags=re.S)
    b = re.sub(r"\\(includegraphics|vspace|hspace|parencite|textcite|cite)\*?(\[[^\]]*\])?\{[^}]*\}", " ", b)
    b = re.sub(r"\\[a-zA-Z]+\*?", " ", b)
    b = re.sub(r"[{}$\\&~]", " ", b)
    return sum(1 for w in b.split() if re.search(r"[A-Za-z0-9]", w))


def frames(tex: Path):
    s = tex.read_text()
    for m in re.finditer(r"\\begin\{frame\}(?:\[[^\]]*\])?\{((?:[^{}]|\{[^{}]*\})*)\}(.*?)\\end\{frame\}", s, re.S):
        title = re.sub(r"\\parencite\{[^}]*\}", "", m.group(1)).strip()
        if title not in SKIP:
            yield title, words(m.group(2))


def main(argv):
    over = 0
    if "--over" in argv:
        over = int(argv[argv.index("--over") + 1])
        argv = [a for i, a in enumerate(argv) if a != "--over" and argv[i - 1] != "--over"]
    decks = argv or sorted(p.stem for p in (ROOT / "Slides").glob("Lecture*.tex"))
    rows = []
    for d in decks:
        for i, (t, n) in enumerate(frames(ROOT / "Slides" / f"{d}.tex"), 1):
            if n > over:
                rows.append((n, d, i, t))
    if len(decks) > 1:
        rows.sort(reverse=True)
    for n, d, i, t in rows:
        flag = "  <- split or cut" if n > 120 else ("  <- trim" if n > 90 else "")
        print(f"{n:4d}  {d:30s} {i:2d}. {t}{flag}")


if __name__ == "__main__":
    main(sys.argv[1:])
