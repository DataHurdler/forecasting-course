#!/usr/bin/env python3
"""Show prose edits as before/after paragraph pairs, for reading.

`git diff` reports a rewrapped paragraph as ten changed lines, which is how a one-word change
hides. This diffs paragraphs instead and prints each change as a pair: what the file said, and
what it says now. Works on any file in any of the course's repos, so narration, assignments,
slides, and Canvas pages all read the same way.

    python scripts/review_diff.py Narration/Lecture09_Part2_script.md
    python scripts/review_diff.py --days 1 Narration/                  # a day's work, no hash to find
    python scripts/review_diff.py --words --since HEAD~5 Homework/     # bracket the changed words
"""
import argparse
import difflib
import re
import subprocess
import sys
from pathlib import Path

PROSE = (".qmd", ".md", ".tex", ".html")


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)


def paragraphs(text: str) -> list[str]:
    """Blank-line separated blocks with internal wrapping flattened.

    A table row and a list item are edited one at a time, so each is its own paragraph.
    """
    out = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip("\n")
        if not block.strip():
            continue
        if re.match(r"^\s*(\||[-*] |\d+\. )", block) and "\n" in block:
            out.extend(l for l in block.split("\n") if l.strip())
        else:
            out.append(" ".join(block.split()))
    return out


def mark(a: str, b: str) -> tuple[str, str]:
    """Bracket the words that differ, so the eye lands on them."""
    aw, bw = a.split(), b.split()
    A, B = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, aw, bw).get_opcodes():
        if tag == "equal":
            A.extend(aw[i1:i2])
            B.extend(bw[j1:j2])
        else:
            if i1 != i2:
                A.append("[[" + " ".join(aw[i1:i2]) + "]]")
            if j1 != j2:
                B.append("[[" + " ".join(bw[j1:j2]) + "]]")
    return " ".join(A), " ".join(B)


def default_since(repo: Path, rel: Path) -> str:
    """HEAD if the file has uncommitted edits; otherwise the state before the last commit to it."""
    if git(repo, "diff", "--quiet", "HEAD", "--", str(rel)).returncode:
        return "HEAD"
    last = git(repo, "log", "-1", "--format=%H", "--", str(rel)).stdout.strip()
    return f"{last}~1" if last else "HEAD"


def days_since(repo: Path, rel: Path, days: int) -> str:
    """The last commit to this file from before the last N days."""
    before = git(repo, "log", "-1", "--format=%H", f"--before={days}.days.ago", "--", str(rel)).stdout.strip()
    return before or default_since(repo, rel)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="files, or directories to search for prose files")
    ap.add_argument("--since", help="revision to compare against (default: see default_since)")
    ap.add_argument("--days", type=int, metavar="N", help="everything that changed in the last N days")
    ap.add_argument("--words", action="store_true", help="bracket the changed words in each pair")
    a = ap.parse_args()

    files = []
    for p in a.paths:
        q = Path(p).resolve()
        if not q.exists():
            print(f"{p}: no such file or directory", file=sys.stderr)
            return 2
        files.extend(sorted(f for f in q.rglob("*") if f.suffix in PROSE) if q.is_dir() else [q])

    total = 0
    for f in files:
        top = git(f.parent, "rev-parse", "--show-toplevel").stdout.strip()
        if not top:
            print(f"{f}: not in a git repository", file=sys.stderr)
            continue
        repo = Path(top).resolve()
        rel = f.relative_to(repo)
        since = a.since or (days_since(repo, rel, a.days) if a.days else default_since(repo, rel))
        old = git(repo, "show", f"{since}:{rel}")
        if old.returncode:
            # A directory walk meets untracked and brand-new files; a named file should say so.
            if len(files) == 1:
                print(f"{rel}: cannot read it at {since}", file=sys.stderr)
            continue
        before, after = paragraphs(old.stdout), paragraphs(f.read_text())
        changes = [(" ".join(before[i1:i2]), " ".join(after[j1:j2]))
                   for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes()
                   if tag != "equal"]
        if not changes:
            continue
        print(f"\n{'=' * 78}\n{rel}: {len(changes)} change(s) since {since}\n")
        for n, (x, y) in enumerate(changes, 1):
            if a.words and x and y:
                x, y = mark(x, y)
            print(f"--- {n} " + "-" * 70)
            print(f"  WAS: {x or '(nothing)'}\n")
            print(f"  NOW: {y or '(removed)'}\n")
        total += len(changes)
    print(f"\n{total} change(s) across {len(files)} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
