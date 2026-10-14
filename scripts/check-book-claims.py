#!/usr/bin/env python3
"""check-book-claims.py — the two hand-written pages in two generated books.

Everything in the book and the workbook is assembled by `build_book.py` /
`build_workbook.py` from the decks, labs, assignments and narration, so a number
in a chapter cannot disagree with the material it came from — it is never copied,
it is read on every build. The prefaces are the exception. They are authorial
prose, written once by hand, and they make claims about the volume they open.

That is not a reason to stop writing them by hand: the specific numbers are what
make a reader keep going, and a preface with nothing checkable in it is duller and
no more true. It is a reason to check what they claim. book/index.qmd has already
carried one stale sentence.

So each claim below is bound to its source on disk. A claim whose regex no longer
matches is a FAILURE, not a skip — a preface reworded until it says nothing is
exactly how coverage disappears quietly.

    python3 scripts/check-book-claims.py

Exit 0 = every claim matched and every claim is true. Runs in ./scripts/backtest.sh.
"""
from __future__ import annotations

import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOK = "book/index.qmd"
WORKBOOK = "workbook/index.qmd"

_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
          "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
          "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
          "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50}
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
             "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}


def read(p: str) -> str:
    try:
        with open(os.path.join(ROOT, p), encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return ""


def flat(p: str) -> str:
    """A surface with its line wrapping removed.

    The prefaces are hard-wrapped prose, so a claim routinely straddles a newline
    ("ahead of\nfour neural architectures"). Patterns that assume a single space
    silently stop matching the day a sentence is re-wrapped — and an unmatched
    pattern here is a FAILURE, so that would surface as a false alarm rather than
    silent loss, but it is still noise nobody should have to debug. Collapse the
    whitespace once, and let every pattern below be written the way the sentence
    reads.
    """
    return re.sub(r'\s+', ' ', read(p))


def num(tok: str | None) -> int | None:
    if tok is None:
        return None
    t = tok.strip().lower()
    if t.isdigit():
        return int(t)
    return _WORDS.get(t, _ORDINALS.get(t))


# --- sources of truth ---------------------------------------------------------

def scoreboard() -> list[tuple[str, int]]:
    """Lecture 13's scoreboard — the one table that ranks every model in the course.

    Parsed from the deck's Quarto mirror rather than restated here: the preface's
    claims about the ordering must answer to the table students actually see.
    """
    t = read("Quarto/Lecture13_Synthesis.qmd")
    m = re.search(r'^\| Model \| Test RMSE \| Where \|\n\|[-: |]+\|\n((?:\|.*\n)+)', t, re.M)
    if not m:
        return []
    rows = []
    for line in m.group(1).strip().split("\n"):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        rmse = re.sub(r'[^\d]', '', cells[1])
        if rmse:
            rows.append((cells[0], int(rmse)))
    return rows


def n_chapters() -> int:
    return len(re.findall(r'^\s*-\s*chapters/', read("book/_quarto.yml"), re.M))


def n_decks() -> int:
    """Chapters build_book.py is configured to produce — its own CHAPTERS list."""
    t = read("scripts/build_book.py")
    m = re.search(r'^CHAPTERS\s*=\s*\[(.*?)^\]', t, re.S | re.M)
    return len(re.findall(r'\(\s*"', m.group(1))) if m else 0


def n_labs() -> int:
    return len(glob.glob(os.path.join(ROOT, "Labs", "Lecture*_lab.qmd")))


def n_assignments() -> int:
    return len(glob.glob(os.path.join(ROOT, "Homework", "HW*.qmd")))


def n_neural(rows: list[tuple[str, int]]) -> int:
    kw = ("rnn", "lstm", "transformer", "cnn", "neural", "ffn")
    return sum(1 for name, _ in rows if any(k in name.lower() for k in kw))


def rank_of(rows: list[tuple[str, int]], needle: str) -> int | None:
    """Where a model finishes — ranked by its RMSE, not by its row in the table.

    Row position would make this claim self-confirming: re-measure the LSTM at
    640, leave it on line five because nobody re-sorted the deck, and a check
    reading positions still says "fifth" and passes. The preface is claiming an
    ORDERING, so the ordering is what gets recomputed. `scoreboard_is_sorted`
    below then catches the other half — a table whose rows no longer match the
    numbers in them.
    """
    for i, (name, _) in enumerate(sorted(rows, key=lambda r: r[1]), start=1):
        if needle.lower() in name.lower():
            return i
    return None


def scoreboard_is_sorted(rows: list[tuple[str, int]]) -> tuple[bool, str]:
    vals = [v for _, v in rows]
    if vals == sorted(vals):
        return True, f"{len(rows)} rows, ascending RMSE"
    bad = next(i for i in range(1, len(vals)) if vals[i] < vals[i - 1])
    return False, (f"row {bad + 1} ({rows[bad][0]}, {rows[bad][1]}) sorts above "
                   f"row {bad} ({rows[bad - 1][0]}, {rows[bad - 1][1]})")


# --- the claims ---------------------------------------------------------------
# (surface, label, regex capturing what is claimed, a callable returning the truth)
# The regex MUST match; a claim that has been reworded away fails.

def claims():
    rows = scoreboard()
    top = rows[0][0] if rows else ""
    top_coef = num(re.search(r'(\d+)', top).group(1)) if re.search(r'(\d+)', top) else None

    return [
        (BOOK, "one chapter per lecture",
         r'Every chapter is one lecture',
         lambda g: (n_chapters() == n_decks() and n_chapters() > 0,
                    f"{n_chapters()} chapters in book/_quarto.yml, "
                    f"{n_decks()} in build_book.py's CHAPTERS list")),

        (BOOK, "the generator it names",
         r'`(scripts/build_book\.py)`',
         lambda g: (os.path.exists(os.path.join(ROOT, g)), f"{g} exists")),

        (BOOK, "panel size",
         r'panel of ([a-z]+|\d+) Walmart store-category series',
         lambda g: (num(g) == panel_size(),
                    f"preface says {num(g)}, Lecture 13's scoreboard header says {panel_size()}")),

        (BOOK, "the top of the scoreboard is a linear model",
         r'a (\d+)-coefficient linear model finishes at the top',
         lambda g: (int(g) == top_coef and _is_linear(top),
                    f"preface says {g}-coefficient linear; scoreboard row 1 is {top!r}")),

        (BOOK, "how many neural architectures it beat",
         r'ahead of ([a-z]+|\d+) neural architectures',
         lambda g: (num(g) == n_neural(rows),
                    f"preface says {num(g)}, scoreboard has {n_neural(rows)}")),

        (BOOK, "where the two attention models finished",
         r'the two with the most attention behind them finish (\w+) and (\w+)',
         lambda g: (_ranks_match(rows, g),
                    f"preface says {g[0]}/{g[1]}; LSTM is {rank_of(rows,'lstm')}, "
                    f"Transformer is {rank_of(rows,'transformer')}")),

        # The preface's ranking claims are only meaningful if the table students
        # read is itself in the order it appears to be in.
        (BOOK, "the scoreboard it ranks against is sorted",
         r'finishes at the top',
         lambda g: scoreboard_is_sorted(rows)),

        (WORKBOOK, "lab count",
         r'the ([a-z]+|\d+) in-class labs',
         lambda g: (num(g) == n_labs(), f"preface says {num(g)}, {n_labs()} on disk")),

        (WORKBOOK, "assignment count",
         r'([a-z]+|\d+) assignments, in one place',
         lambda g: (num(g) == n_assignments(),
                    f"preface says {num(g)}, {n_assignments()} on disk")),

        # Not a count, and the most important one. The workbook is published to a
        # PUBLIC site; solution keys are gitignored precisely so they cannot reach
        # one. This asserts the promise the preface makes to students is still
        # true of what the build actually produced.
        (WORKBOOK, "solutions are not in it",
         r'Solutions are not here and never will be',
         lambda g: _no_solutions()),
    ]


def _is_linear(name: str) -> bool:
    return any(k in name.lower() for k in ("lasso", "ridge", "elastic", "linear", "ols"))


def _ranks_match(rows, g) -> bool:
    # The preface names the PAIR's places ("finish fifth and sixth"), not which model took which,
    # so compare as a set. On 52-week windows the Transformer (990) edges the LSTM (998).
    want = (num(g[0]), num(g[1]))
    got = (rank_of(rows, "lstm"), rank_of(rows, "transformer"))
    return None not in want and None not in got and sorted(want) == sorted(got)


def panel_size() -> int | None:
    m = re.search(r'same (\d+) series', read("Quarto/Lecture13_Synthesis.qmd"))
    return int(m.group(1)) if m else None


def _no_solutions():
    """No generated workbook chapter carries solution material."""
    bad = []
    for f in sorted(glob.glob(os.path.join(ROOT, "workbook", "chapters", "*.qmd"))):
        body = open(f, encoding="utf-8").read()
        if re.search(r'(?im)^#+\s*(solution|answer key)\b|_solutions_script|Homework/solutions/', body):
            bad.append(os.path.relpath(f, ROOT))
    return (not bad, "no solution material in workbook/chapters/"
            if not bad else "solution material found in: " + ", ".join(bad))


def main() -> int:
    problems, checked = [], 0
    for surface, label, pattern, verdict in claims():
        text = flat(surface)
        if not text:
            problems.append(f"{surface}: file missing")
            continue
        m = re.search(pattern, text)
        if not m:
            problems.append(
                f"{surface}: NO CLAIM MATCHED for '{label}' — the preface was reworded "
                f"and this claim is no longer gated. Update the pattern in "
                f"scripts/check-book-claims.py, or drop the row if the claim is gone.")
            continue
        checked += 1
        g = m.groups() if len(m.groups()) > 1 else m.group(1) if m.groups() else m.group(0)
        ok, detail = verdict(g)
        status = "ok" if ok else "MISMATCH"
        print(f"  {surface:<20} {label:<44} {status}   ({detail})")
        if not ok:
            problems.append(f"{surface}: {label} — {detail}")

    if problems:
        print(f"\ncheck-book-claims: {len(problems)} problem(s)\n")
        for p in problems:
            print(f"  {p}")
        return 1
    print(f"\ncheck-book-claims: {checked} preface claim(s) match their sources on disk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
