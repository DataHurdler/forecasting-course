# ECON 8310 — Troubleshooting

[← Course website](https://www.luozijun.com/forecasting-course/) · [Setup and Quarto guide](https://www.luozijun.com/forecasting-course/files/setup-guide.html)

The failures this course actually produces, with the fix. Read this before emailing — most of
what goes wrong here goes wrong for everybody, and the answer is usually one line.

---

## Running code

### `Running cells with 'Python 3.12.x' requires the ipykernel package`

**VS Code is pointed at a different Python than the one you installed the packages into.** This is
the most common first-week error, and installing `ipykernel` is usually the *wrong* fix — it clears
this message and leaves you missing `statsmodels` on the next cell.

Point VS Code at the right interpreter instead:

> **⌘⇧P** (macOS) or **Ctrl+Shift+P** → `Python: Select Interpreter` → pick the environment where
> you ran `pip install`.

If you are not sure which that is, run this in the terminal where `pip install` worked:

```bash
python -c "import sys; print(sys.executable)"
```

That path is the interpreter to select. If you genuinely have only one Python and want to install
into it, `pip install ipykernel` is then correct.

### `ModuleNotFoundError: No module named 'statsmodels'` (or pandas, torch, pymc…)

Same cause as above nine times out of ten: the packages went into one Python and you are running
another. Check with the command above before installing anything twice.

If the interpreter is right, install everything the course needs in one command, from the top of
your clone:

```bash
pip install -r requirements.txt
```

### `FileNotFoundError: Could not find data/processed/m5_weekly.csv`

The prepared data is **already committed** to the course environment repository, so this usually
means you are running from the wrong folder rather than that the file is missing.

```bash
ls data/processed/          # from the top of your clone — you should see m5_weekly.csv and the other course .csv files
```

If they are there, open the repository *root* in your editor, not the `assignments/` folder. If
they are genuinely absent, rebuild them:

```bash
python scripts/prep_m5.py            # about 48 MB, a few minutes
python scripts/prep_fred.py
python scripts/prep_electricity.py
```

### `import xgboost` fails on macOS

XGBoost needs the OpenMP runtime, which macOS does not ship.

```bash
brew install libomp
```

Once, before Week 5. Nothing earlier needs it.

---

## Rendering

### `quarto render` fails with a kernel or jupyter error

Quarto runs your Python through a Jupyter kernel even though nothing in your code imports it.

```bash
pip install jupyter
```

Into the same interpreter your code runs in — see the first entry above.

### It runs in the editor but fails when I render

**That is the whole point of rendering**, and it is worth understanding rather than working
around. Cells you ran interactively kept variables alive in an order you no longer remember. A
render starts from nothing and runs top to bottom.

Before you submit, always:

> **Restart the kernel, then run everything from the top.**

If it survives that, it will render. **The single most common way to lose points in this course is
submitting a `.qmd` that does not render.**

### My `.html` is older than my `.qmd`

You edited and forgot to re-render. `python scripts/check_my_submission.py` catches this
specifically — run it before every submission.

### Google Colab will not open my file

Colab cannot open `.qmd` files, and this course is built on them. Use **VS Code** (with the Quarto
extension) or **Positron** (Quarto built in). Both are free.

---

## GitHub and submitting

### `git push` → `permission denied` / `403`

**Expected.** `forecasting-env` is the instructor's repository and you do not have write access to
it. It is your *working folder*, not your submission.

**Submit on Canvas.** Upload `HWxx.html`, `HWxx.qmd`, `PROMPT_LOG.md` and `INITIAL_PROMPT.md`
(plus `REPORT.md` where the assignment asks for one), individually or zipped.

### Then why clone it at all?

For `git pull`. When an assignment is corrected during the term, pulling brings you the fix
without re-downloading anything. Local commits also work perfectly well with nowhere to push to.

### Can I keep my work on GitHub?

Yes, and it is optional. If you do, **make your repository private** — a fork of a public
repository is always public, and a public copy puts your work and your prompt log in front of the
whole class.

### `check_my_submission.py` says "No submissions/ folder found"

Run it from the **top of your clone**, not from inside a subfolder:

```bash
cd path/to/forecasting-env
python scripts/check_my_submission.py
```

It takes no arguments and finds your folders itself.

---

## Results and grading

### My numbers do not match the lecture's

**Expected, and not graded.** Results depend on the random seed, your library versions, your
hardware and your thread count. Two people running identical, correct code get different numbers.

Quote **your own** output, and interpret what you actually got. Tuning toward a number you saw on
a slide is the one way to turn this into a wrong answer.

One caveat: the broad *ordering* of models usually survives. If yours comes out backwards from the
lecture, say so in a sentence and offer a reason — either you found something real about your run,
or there is a bug worth hunting. Both earn more than quietly reporting a number you do not believe.

### I went over the prompt budget

That is allowed. The budget is a **target, not a limit** — log the extra prompts and add a line
saying where you got stuck. What costs you is a log that does not match the work you submit.

---

## Still stuck

Post to the course discussion board — if you hit it, someone else did too, and the answer helps
them. Include the **exact error text**, not a description of it, and say what you had already
tried.
