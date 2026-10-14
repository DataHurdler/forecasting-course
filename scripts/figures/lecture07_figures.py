#!/usr/bin/env python3
"""Figures for Lecture 7 (Neural networks) — one script, every figure, both formats.

    .venv/bin/python scripts/figures/lecture07_figures.py                       # all
    .venv/bin/python scripts/figures/lecture07_figures.py relu_hinges           # one, by name

Writes Figures/Lecture07_NeuralNets/<name>.pdf (Beamer) and <name>.svg (Quarto), both vector,
and Figures/Lecture07_NeuralNets/measured.json with every number the figure prints.

relu_hinges: weekday electricity load (data/processed/electricity_daily.csv, PJM East 2002-2018,
weekends dropped) against day of year, a W-shaped year. Left, the best straight line: what any
stack of linear layers reduces to. Right, four ReLU units with one input, y = b + sum_k v_k
max(0, x - knot_k), at knots fixed in advance; only the output weights v_k and b are fitted, by
least squares on the four hinge features. That fit is exact, so nothing is trained or sampled
and the figure needs no seed. The knots are the best four on a 10-day grid (searched below, in
about ten seconds), so the right panel is the best this family can do up to that grid, not a
lucky draw. A trained network would place its own knots.
"""
from __future__ import annotations

import itertools
import json
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Figures" / "Lecture07_NeuralNets"
MEASURED = OUT / "measured.json"

# The deck's palette — Preambles/header.tex and Quarto/uno-clean.scss.
CRIMSON, BLACK, GRAY = "#D71920", "#1A1A1A", "#525252"
GREEN, AMBER, BLUE = "#15803D", "#B45309", "#005CA9"
LIGHT_CRIMSON, LIGHT_GRAY, LIGHT_BLUE = "#FDECEA", "#F4F4F5", "#E8F0FA"

_fonts = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ("Source Sans Pro", "Source Sans 3", "Helvetica Neue", "Arial")
             if f in _fonts), "DejaVu Sans")
# Drawn near display size, as in Lecture 8: see the SCALE note in lecture08_figures.py.
SCALE = 0.55


def sz(w: float, h: float) -> tuple[float, float]:
    return (w * SCALE, h * SCALE)


# SVG text is drawn as outlines; see lecture09_part1_figures.py for why live text breaks.
plt.rcParams.update({"font.family": FONT, "font.size": 13, "svg.fonttype": "path",
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.05,
                    transparent=True, dpi=200)
    plt.close(fig)
    print(f"  {name}.pdf / .svg")


# ================================================================== the data and the two fits
@lru_cache(maxsize=None)
def weekdays() -> tuple[np.ndarray, np.ndarray]:
    e = pd.read_csv(ROOT / "data" / "processed" / "electricity_daily.csv")
    wd = e[e.is_weekend == 0]
    return wd.doy.values.astype(float), wd.mw_mean.values.astype(float)


def hinge_design(x: np.ndarray, knots) -> np.ndarray:
    """Columns: 1, then max(0, x - knot) for each knot (one ReLU unit each, input weight 1)."""
    return np.column_stack([np.ones_like(x)] + [np.maximum(0.0, x - k) for k in knots])


def lstsq_rmse(H: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    w, *_ = np.linalg.lstsq(H, y, rcond=None)
    return float(np.sqrt(np.mean((y - H @ w) ** 2))), w


@lru_cache(maxsize=None)
def fits() -> dict:
    x, y = weekdays()
    line_rmse, line_w = lstsq_rmse(np.column_stack([np.ones_like(x), x]), y)
    grid = np.arange(10, 361, 10)
    best = min(itertools.combinations(grid, 4),
               key=lambda kn: lstsq_rmse(hinge_design(x, kn), y)[0])
    knots = [int(k) for k in best]
    hinge_rmse, w = lstsq_rmse(hinge_design(x, knots), y)
    # the best any function of day-of-year can do here: the mean load on each day of the year
    doy_mean = pd.Series(y).groupby(x).transform("mean").values
    ceiling = float(np.sqrt(np.mean((y - doy_mean) ** 2)))
    return dict(n=int(len(y)), line_rmse=line_rmse, line_w=line_w.tolist(), knots=knots,
                hinge_rmse=hinge_rmse, hinge_w=w.tolist(), ceiling_rmse=ceiling)


# ================================================================== the figure
MONTHS = [(1, "Jan"), (91, "Apr"), (182, "Jul"), (274, "Oct")]


def relu_hinges() -> None:
    x, y = weekdays()
    f = fits()
    assert round(f["line_rmse"]) == 4513, f"line RMSE {f['line_rmse']:.0f}, audit had 4,513"
    xs = np.linspace(1, 366, 800)
    b, v = f["hinge_w"][0], f["hinge_w"][1:]
    H = hinge_design(xs, f["knots"])
    total = H @ np.array(f["hinge_w"])
    line = f["line_w"][0] + f["line_w"][1] * xs

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=sz(11.4, 4.3), sharey=True)
    fig.subplots_adjust(wspace=0.06, left=0.09, right=0.99, top=0.88, bottom=0.12)
    YL = (19, 53)
    for ax in (ax1, ax2):
        ax.scatter(x, y / 1000, s=1.6, color="#B8B8BC", lw=0, rasterized=True, zorder=1)
        ax.set_ylim(*YL); ax.set_xlim(-4, 370)
        ax.set_yticks([20, 30, 40, 50])
        ax.set_xticks([m for m, _ in MONTHS]); ax.set_xticklabels([n for _, n in MONTHS])
        ax.tick_params(labelsize=12)

    ax1.plot(xs, line / 1000, color=BLUE, lw=2.4, zorder=3)
    ax1.set_ylabel("weekday load\n(thousand MW)", fontsize=12)
    ax1.set_title("any stack of linear layers", fontsize=13, pad=6, color=BLUE)
    ax1.text(362, 51, f"RMSE {f['line_rmse']:,.0f} MW", ha="right", va="center", fontsize=12,
             color=BLUE, bbox=dict(facecolor="white", edgecolor="none", pad=0.8), zorder=5)

    # four ReLU units: each switches on at its knot and changes the slope by its weight v_k.
    # Dashed: the line as it was before that unit switched on, i.e. the sum without unit k.
    knots = f["knots"]
    w = np.array(f["hinge_w"])
    for i, k in enumerate(knots):
        before = hinge_design(xs, knots[:i]) @ w[:i + 1]          # b + units 1..i-1
        m_ = (xs >= k) & (xs <= k + 34)
        ax2.plot(xs[m_], before[m_] / 1000, color=CRIMSON, lw=1.5, ls=(0, (3.5, 2.5)), zorder=2)
    ax2.plot(xs, total / 1000, color=CRIMSON, lw=2.4, zorder=3)
    kv = hinge_design(np.array(knots, float), knots) @ w
    ax2.plot(knots, kv / 1000, "o", ms=7, color=CRIMSON, mec="white", mew=1.2, zorder=4)
    ax2.set_title("what 4 ReLU units can represent", fontsize=13, pad=6, color=CRIMSON)
    ax2.text(362, 51, f"RMSE {f['hinge_rmse']:,.0f} MW", ha="right", va="center", fontsize=12,
             color=CRIMSON, bbox=dict(facecolor="white", edgecolor="none", pad=0.8), zorder=5)
    ax2.tick_params(labelleft=False)
    fig.text(0.54, -0.03, "knots fixed at days " + ", ".join(str(k) for k in knots) +
             ", weights by least squares\ndashed: the line before each unit switches on",
             ha="center", va="top", fontsize=12, color=GRAY, linespacing=1.2)
    save(fig, "relu_hinges")

    m = {"note": "Written by scripts/figures/lecture07_figures.py: every number the Lecture 7 "
                 "ReLU-hinge figure prints. Weekday days only, x = day of year.", **f}
    MEASURED.write_text(json.dumps(m, indent=1) + "\n")
    print(f"  line RMSE {f['line_rmse']:,.0f}; 4 hinges at {f['knots']} RMSE {f['hinge_rmse']:,.0f};"
          f" day-of-year ceiling {f['ceiling_rmse']:,.0f}; n = {f['n']:,}")


FIGURES = {"relu_hinges": relu_hinges}

if __name__ == "__main__":
    pick = [a for a in sys.argv[1:] if not a.startswith("--")] or list(FIGURES)
    print(f"Lecture 7 figures -> {OUT.relative_to(ROOT)}   (font: {FONT})")
    for name in pick:
        FIGURES[name]()
