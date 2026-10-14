#!/usr/bin/env python3
"""Figures for Lecture 6 (Regularization) — one script, every figure, both formats.

    .venv/bin/python scripts/figures/lecture06_figures.py                     # all
    .venv/bin/python scripts/figures/lecture06_figures.py diamond_and_disc    # one, by name

Writes Figures/Lecture06_Regularization/<name>.pdf (Beamer) and <name>.svg (Quarto), both vector,
and Figures/Lecture06_Regularization/measured.json with every number the figure prints.

diamond_and_disc: the RSS ellipses for a two-predictor regression of daily units on lag7 and
lag14, built with Lab 6's own feature pipeline and training window (Labs/Lecture06_lab.qmd,
Setup: CA_1 FOODS daily, the 365 test days held out). Predictors and target are standardized, as
the lab standardizes before every penalized fit. Nothing is trained or sampled: OLS, the ridge
path (R + lambda I)^-1 c and the LASSO path (LARS) are exact, so the figure needs no cache.

The geometry, exactly. With standardized columns, RSS(b) = RSS(b_ols) + n (b - b_ols)' R (b - b_ols),
where R is the 2 x 2 correlation matrix. So the contours are ellipses centred on OLS, the same for
both panels. The LASSO keeps lag14 at exactly zero while lag7's correlation with the residual
exceeds lag14's: c1 - t > c2 - rho t, i.e. for budgets t < (c1 - c2) / (1 - rho).
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Polygon
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Figures" / "Lecture06_Regularization"
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


# ================================================================== the lab, verbatim
# Labs/Lecture06_lab.qmd, Setup. The lag and rolling columns decide which rows survive dropna,
# so they are all built even though only lag7 and lag14 are used here.
@lru_cache(maxsize=None)
def lab_train() -> pd.DataFrame:
    daily = (pd.read_csv(ROOT / "data" / "processed" / "m5_daily.csv", parse_dates=["date"])
               .sort_values("date"))
    u = daily.set_index("date")["units"]
    for L in list(range(1, 29)) + [35, 42, 49, 56, 364, 365, 366]:
        daily[f"lag{L}"] = u.shift(L).values
    for W in (7, 14, 28, 91, 364):
        daily[f"roll{W}_mean"] = u.shift(1).rolling(W).mean().values
        daily[f"roll{W}_std"] = u.shift(1).rolling(W).std().values
    df = daily.dropna(subset=[c for c in daily.columns if c.startswith(("lag", "roll"))])
    return df.iloc[:-365].copy()                      # N_TEST = 365, as in the lab


@lru_cache(maxsize=None)
def two_lags() -> dict:
    """R, c, OLS, the corner budget, and both solution paths for standardized lag7 and lag14."""
    tr = lab_train()
    X = tr[["lag7", "lag14"]].values.astype(float)
    X = (X - X.mean(0)) / X.std(0)                     # StandardScaler (ddof = 0)
    y = tr["units"].values.astype(float)
    y = (y - y.mean()) / y.std()
    n = len(y)
    R, c = X.T @ X / n, X.T @ y / n
    rho = float(R[0, 1])
    b_ols = np.linalg.solve(R, c)
    assert c[0] > c[1] > 0, "lag7 should enter the LASSO first"
    t_corner = float((c[0] - c[1]) / (1 - rho))        # lag14 = 0 for every budget below this

    # cross-check against sklearn's LARS path: same knot, same end point
    from sklearn.linear_model import lars_path
    _, _, coefs = lars_path(X, y, method="lasso")
    knots = coefs.T
    assert np.allclose(knots[1], [t_corner, 0], atol=1e-6) and np.allclose(knots[-1], b_ols, atol=1e-6)

    lams = np.logspace(4, -4, 600)
    ridge_path = np.array([np.linalg.solve(R + lam * np.eye(2), c) for lam in lams])
    return dict(n=n, R=R, c=c, rho=rho, b_ols=b_ols, t_corner=t_corner,
                l1_ols=float(np.abs(b_ols).sum()), ridge_path=ridge_path,
                lasso_knots=knots)


def ridge_at(s: float) -> np.ndarray:
    """The ridge solution whose l2 norm is s (bisection on lambda)."""
    d = two_lags()
    lo, hi = 1e-8, 1e6
    for _ in range(200):
        lam = np.sqrt(lo * hi)
        b = np.linalg.solve(d["R"] + lam * np.eye(2), d["c"])
        lo, hi = (lam, hi) if np.linalg.norm(b) > s else (lo, lam)
    return b


# ================================================================== the diamond and the disc
BUDGET = 0.10          # inside the corner regime (t < t_corner = 0.155), clear of its kink


def diamond_and_disc() -> None:
    d = two_lags()
    R, b0 = d["R"], d["b_ols"]
    s = BUDGET
    assert s < d["t_corner"]
    lasso_pt = np.array([s, 0.0])
    ridge_pt = ridge_at(s)

    def Q(b):
        e = np.asarray(b) - b0
        return float(e @ R @ e)

    # contour grid
    gx, gy = np.meshgrid(np.linspace(-0.3, 0.75, 700), np.linspace(-0.3, 0.6, 600))
    E1, E2 = gx - b0[0], gy - b0[1]
    QQ = R[0, 0] * E1**2 + 2 * R[0, 1] * E1 * E2 + R[1, 1] * E2**2

    XL, YL = (-0.15, 0.63), (-0.15, 0.44)
    fig, axes = plt.subplots(1, 2, figsize=sz(10.4, 4.9))
    fig.subplots_adjust(wspace=0.2, left=0.02, right=0.93, top=0.9, bottom=0.1)

    for ax, kind in zip(axes, ("ridge", "lasso")):
        col, light = (BLUE, LIGHT_BLUE) if kind == "ridge" else (CRIMSON, LIGHT_CRIMSON)
        ax.set_aspect("equal")
        ax.set_xlim(*XL); ax.set_ylim(*YL)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.plot(XL, [0, 0], color=GRAY, lw=0.9, zorder=1)
        ax.plot([0, 0], YL, color=GRAY, lw=0.9, zorder=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(XL[1] + 0.012, 0, "lag7", ha="left", va="center", fontsize=12, color=GRAY)
        ax.text(-0.015, YL[1], "lag14", ha="right", va="top", fontsize=12, color=GRAY)

        contact = ridge_pt if kind == "ridge" else lasso_pt
        q = Q(contact)
        ax.contour(gx, gy, QQ, levels=[q * 0.012, q * 0.045], colors=[GRAY],
                   linewidths=0.8, alpha=0.75, zorder=2)
        ax.contour(gx, gy, QQ, levels=[q], colors=[BLACK], linewidths=1.2, zorder=3)

        if kind == "ridge":
            ax.add_patch(Circle((0, 0), s, facecolor=light, edgecolor=col, lw=1.8, zorder=2))
            p = d["ridge_path"]
        else:
            ax.add_patch(Polygon([(s, 0), (0, s), (-s, 0), (0, -s)], closed=True,
                                 facecolor=light, edgecolor=col, lw=1.8, zorder=2))
            p = d["lasso_knots"]
        ax.plot(p[:, 0], p[:, 1], color=col, lw=1.7, zorder=4, solid_capstyle="round")

        ax.plot(*b0, "o", ms=6, color=BLACK, zorder=5)
        ax.text(b0[0] - 0.02, b0[1] + 0.025, "OLS", ha="right", va="bottom", fontsize=12,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6), zorder=5)
        ax.plot(*contact, "o", ms=7.5, color=col, mec="white", mew=1.0, zorder=6)

    # labels, placed against the computed geometry: right of each contact point, clear of the
    # contact ellipse (which runs from upper left to lower right through it)
    ax = axes[0]
    ax.annotate(f"({ridge_pt[0]:.2f}, {ridge_pt[1]:.2f}):\nboth kept", xy=ridge_pt,
                xytext=(0.205, 0.035), fontsize=12, color=BLUE, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.9, shrinkB=4))
    ax = axes[1]
    ax.annotate(f"({s:.2f}, 0): lag14 = 0", xy=lasso_pt, xytext=(0.27, -0.105), fontsize=12,
                color=CRIMSON, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=CRIMSON, lw=0.9, shrinkB=4))
    tk = d["t_corner"]
    ax.plot([tk], [0], marker="|", ms=7, mew=1.8, color=CRIMSON, zorder=5)
    ax.annotate(f"t = {tk:.2f}:\nlag14 enters", xy=(tk, 0.0), xytext=(0.32, 0.035),
                fontsize=12, color=CRIMSON, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=CRIMSON, lw=0.9, shrinkA=2, shrinkB=5))

    axes[0].set_title("Ridge: a disc", fontsize=13, pad=4)
    axes[1].set_title("LASSO: a diamond", fontsize=13, pad=4)
    for a_ in axes:
        a_.apply_aspect()
    y_cap = axes[0].get_position().y0 - 0.04
    fig.text(0.5, y_cap, "two coefficients, constrained form; each λ corresponds to some budget t\n"
             "colored line: the solution as the budget t grows", ha="center", va="top",
             fontsize=11, color=GRAY, linespacing=1.25)
    save(fig, "diamond_and_disc")

    m = {"note": "Written by scripts/figures/lecture06_figures.py: every number the Lecture 6 "
                 "diamond-and-disc figure prints. Lab 6 training window, standardized lag7/lag14.",
         "n_train": d["n"], "rho_lag7_lag14": d["rho"], "corr_with_units": d["c"].tolist(),
         "ols": d["b_ols"].tolist(), "ols_l1": d["l1_ols"], "t_corner": d["t_corner"],
         "t_corner_share_of_ols_l1": d["t_corner"] / d["l1_ols"], "budget_drawn": s,
         "ridge_contact": ridge_pt.tolist(), "lasso_contact": lasso_pt.tolist()}
    MEASURED.write_text(json.dumps(m, indent=1) + "\n")
    print(f"  OLS {np.round(d['b_ols'], 3)}, rho {d['rho']:.3f}, corner wins for t < "
          f"{d['t_corner']:.3f} ({d['t_corner'] / d['l1_ols']:.0%} of OLS l1 {d['l1_ols']:.3f})")


FIGURES = {"diamond_and_disc": diamond_and_disc}

if __name__ == "__main__":
    pick = [a for a in sys.argv[1:] if not a.startswith("--")] or list(FIGURES)
    print(f"Lecture 6 figures -> {OUT.relative_to(ROOT)}   (font: {FONT})")
    for name in pick:
        FIGURES[name]()
